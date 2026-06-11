"""Sample-to-full calibration: per-file measurements and the cross-file
cohort prior (rolling averages with shrinkage)."""

import json
import time

from .constants import DEFAULT_BITRATE_DECAY
from .util import atomic_write_json


def load_global_calibration(cache_dir):
    """Load cross-file rolling averages used as defaults for new files."""
    path = cache_dir / "_global_calibration.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# Pseudo-count for shrinking the cohort VMAF offset toward 0 (= "sample
# predicts full exactly"). The cohort average of n files is blended as if
# K additional zero-offset files were observed, so a near-empty cohort
# can't fully steer new files: one outlier first file otherwise mispredicts
# every following file by its whole offset, costing an extra full encode
# each (the "cohort n=1" failure). Trust ramps with evidence: n=1 → 33%,
# n=10 → 83%, n=50 (N_CAP) → 96%.
COHORT_SHRINK_K = 2


def calibration_offset(per_file_cal, global_cal):
    """Pick the sample→full VMAF offset used to aim the sample search.

    Per-file calibration is a direct measurement of this exact file and is
    trusted as-is. The cohort average is indirect evidence (other files'
    offsets), so it's shrunk toward 0 by n/(n+COHORT_SHRINK_K).

    Returns (offset, source_label); (None, None) when neither source has a
    usable value. Values outside ±3.0 are treated as corrupt and skipped.
    """
    if isinstance(per_file_cal, dict):
        o = per_file_cal.get("vmaf_offset")
        if isinstance(o, (int, float)) and -3.0 <= o <= 3.0:
            return float(o), "per-file"
    if isinstance(global_cal, dict):
        g_off = global_cal.get("vmaf_offset")
        if isinstance(g_off, (int, float)) and -3.0 <= g_off <= 3.0:
            n = global_cal.get("n_offset")
            if not isinstance(n, int) or n < 1:
                n = 1
            shrunk = g_off * n / (n + COHORT_SHRINK_K)
            label = f"cohort n={n}"
            if abs(g_off - shrunk) >= 0.05:
                label += f", shrunk from {g_off:+.2f}"
            return shrunk, label
    return None, None


# Sanity range for a bitrate-decay slope d(log kbps)/d(quantizer). Real
# SVT-AV1 content measures ~0.04-0.2 per step; values outside are treated
# as corrupt and skipped.
DECAY_MIN, DECAY_MAX = 0.02, 0.5


def decay_prior(per_file_cal, global_cal):
    """Pick the starting bitrate-decay slope for the search's floor model.

    Mirrors calibration_offset: a per-file measured decay is a direct
    measurement of this file and trusted as-is; the cohort average is
    shrunk toward DEFAULT_BITRATE_DECAY (what the search would otherwise
    assume) by n/(n+COHORT_SHRINK_K). This is how each engine learns how
    its nominal quantizer maps to bitrate — Essential's CRF encodes
    noticeably richer than mainline's CQ at equal numbers, which a shared
    cold-start constant can't know, so its first jump toward the floor
    fell short and cost 1-2 extra probes per file.

    Returns (decay, source_label); (None, None) when neither source has a
    usable value (the search then uses DEFAULT_BITRATE_DECAY itself).
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
            shrunk = g * w + DEFAULT_BITRATE_DECAY * (1 - w)
            label = f"cohort n={n}"
            if abs(g - shrunk) >= 0.005:
                label += f", shrunk from {g:.3f}"
            return shrunk, label
    return None, None


def update_global_calibration(cache_dir, vmaf_offset=None, ratio=None,
                              decay=None):
    """Roll new measurements into the cohort calibration cache.

    Per-file calibration only helps on re-runs of the same file. The
    cohort cache gives new files an informed starting point so first-
    encounter sample-vs-full mispredict is corrected up front, avoiding
    a wasted second full encode. n is capped so the average stays
    responsive to drift (e.g. encoder/preset changes).
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

    roll("vmaf_offset", "n_offset", vmaf_offset)
    roll("ratio", "n_ratio", ratio)
    roll("decay", "n_decay", decay)
    g["t"] = time.time()

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "_global_calibration.json"
    atomic_write_json(path, g)
