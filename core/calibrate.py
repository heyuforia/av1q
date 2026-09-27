"""Sample-to-full calibration: per-file measurements and the cross-file
cohort prior (rolling averages with shrinkage)."""

import json
import time

from .constants import (
    DEFAULT_BITRATE_DECAY, SCENE_OFFSET_PRIOR, VMAF_SLOPE_MAX, VMAF_SLOPE_MIN,
)
from .util import atomic_write_json, clamp


def load_global_calibration(cache_dir):
    """Load cross-file rolling averages used as defaults for new files."""
    path = cache_dir / "_global_calibration.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# Pseudo-count for shrinking the cohort VMAF offset toward its structural
# center. The cohort average of n files is blended as if K additional
# center-valued files were observed, so a near-empty cohort can't fully
# steer new files: one outlier first file otherwise mispredicts every
# following file by its whole offset, costing an extra full encode each
# (the "cohort n=1" failure). Trust ramps with evidence: n=1 → 33%,
# n=10 → 83%, n=50 (N_CAP) → 96%.
COHORT_SHRINK_K = 2


# Cohort key names per sampling mode. Scene-selected and evenly-spaced
# samples measure two different populations: the scene cohort's whole
# content is complexity-selection bias, which by construction does not
# exist in an evenly-spaced sample. Mixing them mis-aims both, so each
# mode keeps its own rolling average under its own keys. The scene keys
# keep their original names, so a cohort file written before the split
# stays valid and keeps steering scene-sampled files; the evenly-spaced
# keys are purely additive.
#
# Evenly-spaced files used to be excluded from the cohort outright —
# never rolled in, never read back — which pinned them to the fixed
# EVEN_SAMPLE_MARGIN guess forever. A real 34-minute file measured a
# true sample→full ratio of 1.04 against that guess's implied 0.95: the
# floor threshold came out 9% high, the search capped a quarter-step
# early, and the refine loop spent a second full encode recovering the
# headroom. Their own cohort is what retires that tax permanently.
_SCENE_KEYS = {
    "offset": ("vmaf_offset", "n_offset"),
    "ratio": ("ratio", "n_ratio"),
}
_EVEN_KEYS = {
    "offset": ("even_vmaf_offset", "n_even_offset"),
    "ratio": ("even_ratio", "n_even_ratio"),
}

# Mini-plan runs (a few 2s clips from a short source) are a third and
# fourth population. Their clips cover a large share of a short file, so
# their ratio sits near the file's own, while a standard plan's clips are
# a sliver of a long one: on one real batch the mini runs averaged ~1.00
# (even) and ~0.93 (scene), the long files ~0.96 and ~0.87. Steered by
# the mini-heavy average, two long files (one of each mode) landed ~4%
# under the floor and bought a second full encode of 34 and 63 minutes.
# The standard plan keeps the original keys; these are purely additive.
_MINI_SCENE_KEYS = {
    "offset": ("mini_vmaf_offset", "n_mini_offset"),
    "ratio": ("mini_ratio", "n_mini_ratio"),
}
_MINI_EVEN_KEYS = {
    "offset": ("mini_even_vmaf_offset", "n_mini_even_offset"),
    "ratio": ("mini_even_ratio", "n_mini_even_ratio"),
}


def cohort_keys(quantity, even, mini=False):
    """(value_key, count_key) in the cohort file for one quantity."""
    if mini:
        return (_MINI_EVEN_KEYS if even else _MINI_SCENE_KEYS)[quantity]
    return (_EVEN_KEYS if even else _SCENE_KEYS)[quantity]


def calibration_offset(per_file_cal, global_cal, prior_center=0.0, even=False,
                       mini=False):
    """Pick the sample→full VMAF offset used to aim the sample search.

    Per-file calibration is a direct measurement of this exact file and is
    trusted as-is. The cohort average is indirect evidence (other files'
    offsets), so it's shrunk toward prior_center by n/(n+COHORT_SHRINK_K).

    prior_center is the structural expectation for the sampling mode:
    complexity-selected samples are the file's hardest scenes and read
    systematically LOW against the full encode (SCENE_OFFSET_PRIOR),
    while evenly-spaced samples are representative and center on 0.
    Shrinking scene-sampled cohorts toward 0 under-corrected every
    early file by most of its real offset — one wasted full re-encode
    each; with no cohort at all the center itself is the best estimate
    and is returned directly.

    `even` and `mini` pick which cohort to read (see cohort_keys): the
    sampling modes and plans measure different populations and never
    share an average.

    Returns (offset, source_label); (None, None) when neither source has
    a usable value and the center is 0. Values outside ±3.0 are treated
    as corrupt and skipped.
    """
    if isinstance(per_file_cal, dict):
        o = per_file_cal.get("vmaf_offset")
        if isinstance(o, (int, float)) and -3.0 <= o <= 3.0:
            return float(o), "per-file"
    if isinstance(global_cal, dict):
        k_off, k_n = cohort_keys("offset", even, mini)
        g_off = global_cal.get(k_off)
        if isinstance(g_off, (int, float)) and -3.0 <= g_off <= 3.0:
            n = global_cal.get(k_n)
            if not isinstance(n, int) or n < 1:
                n = 1
            w = n / (n + COHORT_SHRINK_K)
            shrunk = g_off * w + prior_center * (1 - w)
            label = f"cohort n={n}"
            if abs(g_off - shrunk) >= 0.05:
                label += f", blended from {g_off:+.2f}"
            return shrunk, label
    if prior_center:
        return prior_center, "cold-start prior"
    return None, None


def scene_offset_center(bias, reference_bias):
    """Structural sample→full VMAF offset expected for THIS file.

    SCENE_OFFSET_PRIOR is the offset a fully complexity-biased sample
    reads — but how biased a given file's selection actually came out
    varies, and it is measurable up front (sampling.complexity_bias).
    A file whose hottest scenes barely clear its own average is
    effectively evenly sampled, and its offset is near 0; charging it the
    full structural prior aims the whole search low and buys a full
    re-encode. Scale linearly between the two: no bias, no offset;
    reference bias, the full prior.

    `reference_bias` is the cold-start belief about scene-selection bias
    (cfg["bitrate_margin"]) — the same number complexity_bias_margin uses
    as its upper bound, so both halves of the sample→full prediction are
    anchored to one scale rather than two that can drift apart.

    The law is linear because one field point anchors it (bias 1.03 read
    a true offset of +0.04) and the -0.75 end is assumed to sit at the
    reference bias; the clamp is what makes that assumption safe, since
    the result can only ever shrink the correction, never exceed it.
    Returns SCENE_OFFSET_PRIOR unchanged when the bias is unmeasurable.
    """
    if bias is None or not reference_bias or reference_bias <= 1.0:
        return SCENE_OFFSET_PRIOR
    return SCENE_OFFSET_PRIOR * clamp(
        (bias - 1.0) / (reference_bias - 1.0), 0.0, 1.0
    )


# Sanity range for a bitrate-decay slope d(log kbps)/d(quantizer). Real
# SVT-AV1 content measures ~0.04-0.2 per step; values outside are treated
# as corrupt and skipped.
DECAY_MIN, DECAY_MAX = 0.02, 0.5


def decay_prior(per_file_cal, global_cal, default=DEFAULT_BITRATE_DECAY):
    """Pick the starting bitrate-decay slope for the search's floor model.

    Mirrors calibration_offset: a per-file measured decay is a direct
    measurement of this file and trusted as-is; the cohort average is
    blended toward `default` — the engine's own cold-start decay, what
    the search would otherwise assume — by n/(n+COHORT_SHRINK_K).

    `default` is per engine (Engine.default_decay) because the quantity
    is encoder physics: Essential's CRF curve is roughly half as steep as
    the generic ln2/6, so blending its cohort toward that generic value
    held every early file's estimate high and cost 1-2 extra probes each,
    for as long as the cohort stayed small.

    Returns (decay, source_label); (None, None) when neither source has a
    usable value (the search then uses `default` itself).
    """
    if isinstance(per_file_cal, dict):
        d = per_file_cal.get("decay")
        if isinstance(d, (int, float)) and DECAY_MIN <= d <= DECAY_MAX:
            return float(d), "per-file"
    if isinstance(global_cal, dict):
        g = global_cal.get("decay")
        if isinstance(g, (int, float)) and DECAY_MIN <= g <= DECAY_MAX:
            n = global_cal.get("n_decay")
            if not isinstance(n, int) or n < 1:
                n = 1
            w = n / (n + COHORT_SHRINK_K)
            shrunk = g * w + default * (1 - w)
            label = f"cohort n={n}"
            if abs(g - shrunk) >= 0.005:
                label += f", blended from {g:.3f}"
            return shrunk, label
    return None, None


def vmaf_slope_prior(per_file_cal, enc_tag):
    """The VMAF slope this file's last search measured, for a refine loop
    that runs without a search this time (a resumed file), or None.

    Per file only, never a cohort average: how fast VMAF falls per
    quantizer step is a property of this source's content. Trusted only
    when the block was written under the same encode settings, because
    preset, grain and crop change the slope too.
    """
    if not isinstance(per_file_cal, dict) or per_file_cal.get("enc_tag") != enc_tag:
        return None
    s = per_file_cal.get("vmaf_slope")
    if isinstance(s, (int, float)) and VMAF_SLOPE_MIN <= s <= VMAF_SLOPE_MAX:
        return float(s)
    return None


# Sample→full bitrate ratio range. Samples cut from the hottest scenes
# usually encode richer than the full file, so the ratio (full ÷ sample)
# is typically < 1 — but not always: when the bitrate is carried by
# something scene-independent (film grain everywhere, uniform detail),
# the sampled peaks barely exceed the file's average and the measured
# ratio runs slightly above 1 (1.13 observed on grain-heavy film). Those
# measurements are real signal for the floor model — clamping them to
# 1.0 silently over-caps the search — so the range admits them; 1.3
# still rejects nonsense from a mis-measured probe. Mirrors the gates in
# effective_sample_floor and the pipeline's calibration recorder.
RATIO_MIN, RATIO_MAX = 0.5, 1.3


def ratio_prior(per_file_cal, global_cal, margin, even=False, mini=False):
    """Pick the sample→full bitrate ratio for the search's floor threshold.

    Mirrors decay_prior and calibration_offset: a per-file measured ratio
    is a direct measurement of this file and trusted as-is; the cohort
    average is shrunk toward the margin-implied ratio (1/margin — what
    effective_sample_floor would otherwise assume) by n/(n+COHORT_SHRINK_K).

    This is the cross-file half of the sample→full bitrate calibration. The
    cohort already learns the ratio after every file
    (update_global_calibration); this is what finally feeds it back into the
    next file's floor search, instead of every fresh file re-paying the
    conservative-margin tax (the search over-predicting the video bitrate,
    capping the quantizer too low, and shipping a video well over the floor
    that the refine loop then has to climb back down with a second full
    encode).

    `even` picks which cohort to read (see cohort_keys) and, with it, the
    shrink target. For scene-selected samples that target is the
    margin-implied ratio — what effective_sample_floor would otherwise
    assume. Evenly-spaced samples are representative by construction, so
    their structural center is 1.0: EVEN_SAMPLE_MARGIN is a cold-start
    cushion against ratio noise, not a belief about the ratio, and
    shrinking toward it would hold every even-sampled file's threshold
    ~5% above the truth no matter how much evidence accumulated. `mini`
    picks the mini-plan cohort of the same mode; the centers are the same.

    Returns (ratio, source_label); (None, None) when neither source has a
    usable value (the search then falls back to the raw margin).
    """
    if isinstance(per_file_cal, dict):
        r = per_file_cal.get("ratio")
        if isinstance(r, (int, float)) and RATIO_MIN <= r <= RATIO_MAX:
            return float(r), "per-file"
    if isinstance(global_cal, dict):
        k_rat, k_n = cohort_keys("ratio", even, mini)
        g = global_cal.get(k_rat)
        if isinstance(g, (int, float)) and RATIO_MIN <= g <= RATIO_MAX:
            n = global_cal.get(k_n)
            if not isinstance(n, int) or n < 1:
                n = 1
            if even:
                implied = 1.0
            else:
                implied = 1.0 / margin if margin and margin > 0 else 1.0
            w = n / (n + COHORT_SHRINK_K)
            shrunk = g * w + implied * (1 - w)
            label = f"cohort n={n}"
            if abs(g - shrunk) >= 0.005:
                label += f", blended from {g:.2f}"
            return shrunk, label
    return None, None


def update_global_calibration(cache_dir, vmaf_offset=None, ratio=None,
                              decay=None, even=False, mini=False):
    """Roll new measurements into the cohort calibration cache.

    Per-file calibration only helps on re-runs of the same file. The
    cohort cache gives new files an informed starting point so first-
    encounter sample-vs-full mispredict is corrected up front, avoiding
    a wasted second full encode. n is capped so the average stays
    responsive to drift (e.g. encoder/preset changes).

    `even` and `mini` route the offset and ratio into that sampling
    mode's and plan's own keys (see cohort_keys). Decay is skipped by the
    split on purpose: it
    measures how this ENGINE's quantizer maps to bitrate, which is the
    same physics however the file was sampled, so both modes feed and
    read one shared average.
    """
    N_CAP = 50
    g = load_global_calibration(cache_dir)

    def roll(key, n_key, val):
        if val is None:
            return
        prev = g.get(key)
        n = g.get(n_key, 0)
        if not isinstance(prev, (int, float)) or not isinstance(n, int) or n <= 0:
            g[key] = float(val)
            g[n_key] = 1
            return
        n_new = min(n + 1, N_CAP)
        weight = 1.0 / n_new
        g[key] = prev * (1 - weight) + float(val) * weight
        g[n_key] = n_new

    roll(*cohort_keys("offset", even, mini), vmaf_offset)
    roll(*cohort_keys("ratio", even, mini), ratio)
    roll("decay", "n_decay", decay)
    g["t"] = time.time()

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "_global_calibration.json"
    atomic_write_json(path, g)
