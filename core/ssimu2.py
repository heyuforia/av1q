"""SSIMULACRA2 measurement via FFVship for the info column.

Display-only second opinion printed next to VMAF scores. It never gates
or refines anything, and without an FFVship binary the column simply
doesn't appear — FFVship is not a requirement of either pipeline.

One shared runner serves both engines' wrappers:
  * measure_ssimu2_display — av1q's flavor: finds the binary itself,
    fails silently (no binary or broken measurement -> None).
  * ssimu2_info — av1q-essential's flavor: binary comes from cfg
    (discovered once in engine setup), failures print an error line,
    honors --metric-every.
When no persistent ref_index is given, a temp source index is created
and deleted so FFVship never writes index files next to the videos.
"""

import json
import math
import subprocess

from .tools import find_ffvship_optional
from .ui import RED, RESET
from .util import _temp_files, make_temp_log


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


def _run_ffvship(ref, dist, meta, cache_dir, exe,
                 ref_index=None, every=1, verbose=False):
    """Run FFVship and parse its per-frame JSON. Returns {'mean', 'p5'}
    or None on any failure (empty/non-finite scores included).

    The crop applies to the SOURCE side only, same rule as measure_vmaf's
    reference chain. `ref_index` names a persistent FFMS2 index for the
    reference so a source measured repeatedly (search probes, verify,
    refine) is only indexed once; the distorted index is per-encode.
    Index files live under <cache_dir>/_ffindex, never next to videos.
    """
    log = make_temp_log(cache_dir, "ssimu2", "json")
    idx_dir = cache_dir / "_ffindex"
    if ref_index:
        idx_dir.mkdir(parents=True, exist_ok=True)
        src_idx = ref_index
    else:
        src_idx = make_temp_log(idx_dir, "src", "ffindex")
    dst_idx = make_temp_log(idx_dir, "dist", "ffindex")
    cmd = [
        str(exe), "--source", str(ref), "--encoded", str(dist),
        "-m", "SSIMULACRA2", "--json", str(log),
        "-t", "2", "-g", "3",
        "--cache-index", "--source-index", str(src_idx),
        "--encoded-index", str(dst_idx),
    ]
    if every > 1:
        cmd += ["--every", str(every)]
    cmd += ffvship_crop_args(meta.get("crop"), meta["w"], meta["h"])
    try:
        r = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        if r.returncode != 0:
            if verbose:
                tail = "\n".join(
                    ((r.stderr or "") + (r.stdout or "")).splitlines()[-40:]
                )
                print(f" {RED}SSIMU2 error: FFVship exit {r.returncode}\n{tail}{RESET}")
            return None
        result = parse_ssimu2_json(log)
        if not math.isfinite(result["mean"]):
            return None
        return result
    except (OSError, json.JSONDecodeError, ValueError) as e:
        if verbose:
            print(f" {RED}SSIMU2 error: {e}{RESET}")
        return None
    finally:
        cleanup = [log, dst_idx] if ref_index else [log, src_idx, dst_idx]
        for p in cleanup:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
            _temp_files.discard(p)


def measure_ssimu2_display(ref, dist, meta, cache_dir, ref_index=None):
    """SSIMULACRA2 of dist vs ref for av1q's info column.

    Returns {'mean', 'p5'} or None on any failure — a missing binary or
    a broken measurement must never affect the pipeline.
    """
    exe = find_ffvship_optional()
    if not exe:
        return None
    return _run_ffvship(ref, dist, meta, cache_dir, exe, ref_index=ref_index)


def ssimu2_info(ref, dist, meta, cfg, ref_index=None):
    """SSIMULACRA2 of dist vs ref for av1q-essential's info column.

    Returns {'mean', 'p5'} or None. Display only — never used in
    decisions: no FFVship binary means no column, a failed measurement
    means no column. Uncached on purpose (informational, and FFVship
    is fast on the GPU).
    """
    if not cfg.get("ffvship_exe") or not dist.exists() or not ref.exists():
        return None
    return _run_ffvship(
        ref, dist, meta, cfg["e_cache_dir"], cfg["ffvship_exe"],
        ref_index=ref_index, every=cfg.get("metric_every", 1), verbose=True,
    )
