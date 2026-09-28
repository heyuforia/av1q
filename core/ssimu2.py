"""SSIMULACRA2 measurement via FFVship for the info column.

Display-only second opinion printed next to VMAF scores. It never gates
or refines anything, and without an FFVship binary the column simply
doesn't appear — FFVship is not a requirement of either pipeline.

One shared runner serves both engines' entry points, which differ only
in where their scratch files go and in essential's --metric-every:
  * measure_ssimu2_display — av1q's, scratch under the shared cache root.
  * ssimu2_info — av1q-essential's, scratch under its own cache root.
A skip, whether deliberate or FFVship failing, prints one line per file
and returns None. When no persistent ref_index is given, a temp source
index is created and deleted so FFVship never writes index files next
to the videos.
"""

import json
import math
import subprocess

from .probe import frame_geometry
from .tools import ffprobe_exe, find_ffvship_optional
from .ui import DIM, RESET, label
from .util import (
    _temp_files, ascii_dir, ascii_path, make_temp_log, scan_budget,
    suppress_win_error_dialog,
)

# Per-file probe results, memoized so the source of a long file is only
# inspected once across its verify/refine/final measurements. Keyed on
# identity, not just path, so a rewritten file is never read stale.
_frame_counts = {}
_geometries = {}

# Skip reasons already announced. A source that can't be measured says so
# ONCE per file: the reason is a property of the source, while _run_ffvship
# is called per probe, so announcing it every time buries the search rows
# it sits between under four copies of the same sentence.
_announced = set()


def _skip(ref, reason):
    """Announce a skip once per file and reason, in the label column
    every other line uses, and return None for the caller to hand back."""
    key = (_file_key(ref), reason)
    if key not in _announced:
        _announced.add(key)
        print(f"{label('ssimu2')}{DIM}skipped: {reason}{RESET}")
    return None


def _file_key(path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_size, int(st.st_mtime))


def _geometry(path):
    """frame_geometry(), memoized per file identity."""
    key = _file_key(path)
    if key is None:
        return None
    if key not in _geometries:
        _geometries[key] = frame_geometry(path)
    return _geometries[key]


def _video_frame_count(path, duration=None):
    """Video packet count via demux only (packets stand in for frames,
    same as the keyframe/complexity scans). None when uncountable.
    `duration` sizes the whole-file pass's budget (scan_budget).

    On field-coded interlaced H.264 each field is a packet, so the gate
    refuses a pair that is really aligned ("N vs 2N frames"). Accepted:
    the column is display only, and a true frame count would decode the
    whole file."""
    key = _file_key(path)
    if key is None:
        return None
    if key in _frame_counts:
        return _frame_counts[key]
    try:
        r = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "default=nw=1:nk=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            timeout=scan_budget(duration),
        )
        if r.returncode != 0:
            return None
        n = int(r.stdout.strip())
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None
    _frame_counts[key] = n
    return n


def ffvship_crop_args(crop, src_w, src_h):
    """FFVship per-edge source-crop flags for a 'W:H:X:Y' crop.

    Crop applies to the SOURCE only: the encode is already cropped, so
    cropping it again would shave real picture (same rule as av1q's
    measure_vmaf reference-only crop chain).
    """
    if not crop:
        return []
    w, h, x, y = (int(v) for v in crop.split(":"))
    edges = (
        ("--cropLeftSource", x),
        ("--cropTopSource", y),
        ("--cropRightSource", src_w - w - x),
        ("--cropBottomSource", src_h - h - y),
    )
    args = []
    for flag, v in edges:
        if v > 0:
            args += [flag, str(v)]
    return args


def parse_ssimu2_json(path):
    """Per-frame FFVship JSON ([[score], ...]) -> {'mean', 'p5'}."""
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    scores = [
        float(r[0]) for r in rows
        if r and isinstance(r[0], (int, float)) and math.isfinite(r[0])
    ]
    if not scores:
        return {"mean": float("nan"), "p5": float("nan")}
    mean = sum(scores) / len(scores)
    s = sorted(scores)
    p5 = s[max(0, int(len(s) * 5 / 100) - 1)]
    return {"mean": mean, "p5": p5}


def _comparability_gap(ref, dist, meta):
    """Why FFVship cannot meaningfully compare these two files, or None.

    FFVship reads both sides through FFMS2 and pairs frame i with frame
    i. It never errors on a mismatch: unequal frame counts just compare
    the overlap, and unequal geometry makes it RESCALE the encode to the
    source's dimensions. Either way the numbers come back at exit code 0,
    so the only defense is refusing to ask the question — for an info
    column, no number beats a wrong one.

    The four ways the two sides can disagree, cheapest check first:

      orientation  ffmpeg auto-rotates on decode, so a source carrying a
                   display matrix was encoded upright; FFMS2 reads that
                   matrix as a property and hands back the picture
                   untouched. The comparison is then upright-vs-rotated
                   (or mirrored) — every frame wrong.
      stream       every other stage here is pinned to v:0, but FFMS2
                   picks its own video track, so a source with a second
                   one may be read off a different picture entirely.
      geometry     the catch-all the first two can't see: whatever the
                   cause, if the reference (after the crop applied to
                   its side) isn't the encode's size, FFVship rescales
                   and the scores are meaningless.
      frames       the source's timing slipped past the VFR gate and the
                   CFR feed duplicated or dropped frames, so pairing
                   drifts after the first divergence.

    Frame count is checked last: it demuxes the whole file, the others
    read headers.
    """
    ref_geo = _geometry(ref)
    if ref_geo:
        if ref_geo["transformed"]:
            return "source is rotated — FFVship's reader reads it unrotated"
        if ref_geo["n_video"] > 1:
            return (f"source has {ref_geo['n_video']} video streams —"
                    f" FFVship may read the wrong one")

    dist_geo = _geometry(dist)
    if ref_geo and dist_geo:
        want_w, want_h = ref_geo["w"], ref_geo["h"]
        if meta.get("crop"):
            # The crop is applied to the REFERENCE side only (the encode
            # is already cropped), so that is the size FFVship compares.
            try:
                want_w, want_h = (int(v) for v in meta["crop"].split(":")[:2])
            except (ValueError, TypeError):
                pass
        if (want_w, want_h) != (dist_geo["w"], dist_geo["h"]):
            return (f"{want_w}x{want_h} source vs {dist_geo['w']}x"
                    f"{dist_geo['h']} encode — FFVship would rescale")

    n_ref = _video_frame_count(ref, meta.get("duration"))
    n_dist = _video_frame_count(dist, meta.get("duration"))
    if n_ref is not None and n_dist is not None and n_ref != n_dist:
        return (f"{n_ref} vs {n_dist} frames — pairing would drift")
    return None


def _run_ffvship(ref, dist, meta, cache_dir, exe, ref_index=None, every=1):
    """Run FFVship and parse its per-frame JSON. Returns {'mean', 'p5'}
    or None on any failure (empty/non-finite scores included).

    Comparability is gated first (_comparability_gap) — FFVship answers
    with a number whether or not the question makes sense.

    The crop applies to the SOURCE side only, same rule as measure_vmaf's
    reference chain. `ref_index` names a persistent FFMS2 index for the
    reference so a source measured repeatedly (search probes, verify,
    refine) is only indexed once; the distorted index is per-encode.
    Index files live under <cache_dir>/_ffindex, never next to videos.
    """
    if not ref.exists() or not dist.exists():
        return None
    # Every skip prints its reason once per file, deliberate or not: a
    # column that vanishes without saying why reads as a broken FFVship,
    # and the reason is usually something about the source worth knowing.
    gap = _comparability_gap(ref, dist, meta)
    if gap:
        return _skip(ref, gap)

    # FFVship reads every path argument as ANSI argv, so a non-ASCII one
    # arrives '?'-mangled and can't be opened. Its own files (log,
    # indexes, hardlinks) go under the cache root's ASCII spelling, and
    # each video gets an ASCII spelling (8.3 alias or a hardlink).
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if ref_index:
            ref_index.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return _skip(ref, f"cache folder unwritable ({type(e).__name__})")
    scratch = ascii_dir(cache_dir)
    idx_home = ascii_dir(ref_index.parent) if ref_index else scratch
    if scratch is None or idx_home is None:
        return _skip(ref, "the cache folder has no ASCII path")
    safe_ref, ref_link = ascii_path(ref, scratch)
    safe_dist, dist_link = ascii_path(dist, scratch)
    if safe_ref is None or safe_dist is None:
        for lk in (ref_link, dist_link):
            if lk:
                try:
                    lk.unlink()
                except OSError:
                    pass
        return _skip(ref, f"no ASCII path for '{ref.name}'")

    log = make_temp_log(scratch, "ssimu2", "json")
    idx_dir = scratch / "_ffindex"
    if ref_index:
        src_idx = idx_home / ref_index.name
    else:
        src_idx = make_temp_log(idx_dir, "src", "ffindex")
    dst_idx = make_temp_log(idx_dir, "dist", "ffindex")
    cmd = [
        str(exe), "--source", str(safe_ref), "--encoded", str(safe_dist),
        "-m", "SSIMULACRA2", "--json", str(log),
        "-t", "2", "-g", "3",
        "--cache-index", "--source-index", str(src_idx),
        "--encoded-index", str(dst_idx),
    ]
    if every > 1:
        cmd += ["--every", str(every)]
    cmd += ffvship_crop_args(meta.get("crop"), meta["w"], meta["h"])
    try:
        # A broken FFVship (missing runtime, GPU/driver mismatch) fails to
        # initialize and Windows pops a modal error box that hangs the batch
        # until dismissed. Suppress it so the crash stays a silent non-zero
        # exit — SSIMU2 is a display-only column and must never block the run.
        with suppress_win_error_dialog():
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
            )
        if r.returncode != 0:
            # The exit code only: the output tail names the per-probe
            # encode, which would defeat the once-per-file announcement.
            return _skip(ref, f"FFVship failed (exit {r.returncode})")
        result = parse_ssimu2_json(log)
        if not math.isfinite(result["mean"]):
            return _skip(ref, "FFVship returned no finite scores")
        return result
    except OSError as e:
        # Either FFVship could not start, or it exited 0 without writing
        # its score log. The error type only, for the same reason.
        return _skip(ref, f"FFVship failed ({type(e).__name__})")
    except ValueError:
        return _skip(ref, "FFVship wrote an unreadable score log")
    finally:
        cleanup = [log, dst_idx] if ref_index else [log, src_idx, dst_idx]
        cleanup += [lk for lk in (ref_link, dist_link) if lk]
        for p in cleanup:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
            _temp_files.discard(p)


def measure_ssimu2_display(ref, dist, meta, cache_dir, ref_index=None):
    """SSIMULACRA2 of dist vs ref for av1q's info column.

    Returns {'mean', 'p5'} or None. Display only, never used in
    decisions: no FFVship binary means no column, silently; a failed
    measurement means no column, with its reason printed once per file.
    """
    exe = find_ffvship_optional()
    if not exe:
        return None
    return _run_ffvship(ref, dist, meta, cache_dir, exe, ref_index=ref_index)


def ssimu2_info(ref, dist, meta, cfg, ref_index=None):
    """SSIMULACRA2 of dist vs ref for av1q-essential's info column: the
    same contract as measure_ssimu2_display, with scratch under the
    essential cache root and --metric-every honored. Uncached on purpose
    (informational, and FFVship is fast on the GPU).
    """
    exe = find_ffvship_optional()
    if not exe:
        return None
    return _run_ffvship(
        ref, dist, meta, cfg["e_cache_dir"], exe,
        ref_index=ref_index, every=cfg.get("metric_every", 1),
    )
