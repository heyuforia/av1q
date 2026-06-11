#!/usr/bin/env python3
"""
av1q — Intelligent VMAF-targeted AV1 video encoding.

Automatically finds the optimal CQ value for each video to hit a target VMAF
quality score, using scene-based sampling for fast quality estimation.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # don't litter core/ with __pycache__

# The implementation lives in the core/ package. av1q.py remains the
# launcher and the stable public import surface: av1q-crop.py and
# av1q-essential.py import the shared helper names from here.
from core.ui import (
    GREEN, ORANGE, PURPLE, RED, RESET, BOLD, DIM, CHECK, CROSS, SEP, MIDDOT,
    fmt_time, fmt_size, vmaf_pass_color, fmt_s2,
)
from core.constants import (
    VIDEO_EXTENSIONS, INTRA_ONLY_CODECS, TARGET_VMAF_BY_RES,
    FALLBACK_MAXRATE, MIN_BITRATE_KBPS, VMAF_OVERSHOOT,
)
from core.util import (
    _temp_files, run_cmd, cleanup_temp, atomic_write_json, make_temp_log,
    escape_filter_path, partial_hash, clamp,
)
from core.probe import detect_hwaccel, probe_video, get_fps, res_tier
from core.bitrate import (
    calc_kbps, video_kbps, measured_kbps, effective_sample_floor,
)
from core.analyze import detect_scenes, analyze_complexity, get_keyframes
from core.sampling import sampling_plan, select_samples, extract_samples
from core.vmaf import measure_vmaf
from core.tools import _http_download, find_ffvship_optional
from core.ssimu2 import measure_ssimu2_display
from core.cache import load_cache
from core.calibrate import (
    COHORT_SHRINK_K, calibration_offset, load_global_calibration,
    update_global_calibration,
)
from core.crop import (
    crop_token, load_crop_sidecar, detect_crop_window, aggregate_crops,
    detect_crop_for_file,
)
from core import search as core_search
from core import pipeline as core_pipeline
from core import vmaf as core_vmaf
from core.engines.svt_ffmpeg import SvtAv1FfmpegEngine, enc_signature, encode_av1
from core.search import initial_cq_seed

_ENGINE = SvtAv1FfmpegEngine()


# ── VMAF Measurement ─────────────────────────────────────────


def vmaf_cached(ref, dist, meta, cq, cache, cache_path, threads, tag=None):
    """Compute VMAF with file-based caching.

    Compat wrapper over core.vmaf.vmaf_cached with av1q's frozen cache
    layout (entries keyed by str(cq), scores under 'full'/'sample_full').
    The measure closure resolves this module's globals at call time so
    monkeypatching av1q.measure_vmaf keeps working.
    """
    return core_vmaf.vmaf_cached(
        ref, dist, meta, cq, cache, cache_path, tag=tag,
        threads=threads, log_dir=cache_path.parent,
        key_base="full", q_key=str(cq),
        measure=lambda *a: measure_vmaf(*a),
    )


# ── CQ Search ────────────────────────────────────────────────


def search_cq(source, meta, target, cache, cache_path,
              enc_func, threads, cfg, tag=None):
    """Find the optimal CQ that hits the target VMAF using adaptive search.

    Compat wrapper over the shared brain (core.search.search) with av1q's
    integer-CQ engine. The measurement closures resolve this module's
    globals at call time, so monkeypatching av1q.vmaf_cached /
    av1q.probe_video keeps working.
    """
    # Persistent FFMS2 index for the search source (SSIMU2 info column
    # only). Stem + size keeps it stable across probes of one search but
    # distinct when the underlying file changes (e.g. re-extracted samples).
    try:
        s2_ref_index = (
            cache_path.parent / "_ffindex"
            / f"{source.stem}_{source.stat().st_size}.ffindex"
        )
    except OSError:
        s2_ref_index = None
    return core_search.search(
        source, meta, target, cache, cache_path, enc_func, cfg, _ENGINE,
        tag=tag,
        measure_fn=lambda ref, dist, q: vmaf_cached(
            ref, dist, meta, q, cache, cache_path, threads, tag=tag),
        probe_fn=lambda f: probe_video(f),
        s2_fn=lambda ref, dist, m, ri: measure_ssimu2_display(
            ref, dist, m, cache_path.parent, ref_index=ri),
        s2_ref_index=s2_ref_index,
    )


# ── Main Processing ──────────────────────────────────────────


def process_videos(cfg):
    """Process every video under cfg's input dir with av1q's engine
    (mainline SVT-AV1 via ffmpeg). The pipeline lives in core.pipeline."""
    return core_pipeline.process_videos(cfg, _ENGINE)


# ── CLI ──────────────────────────────────────────────────────


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="av1q — VMAF-targeted AV1 encoding with intelligent sampling",
    )
    parser.add_argument(
        "-i", "--input", type=Path,
        default=script_dir / "Video Input",
        help="Input directory (default: ./Video Input)",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=script_dir / "AV1 Output",
        help="Output directory (default: ./AV1 Output)",
    )
    parser.add_argument(
        "--vmaf", type=float, default=None,
        help="Target VMAF score (default: auto by resolution)",
    )
    parser.add_argument(
        "--preset", type=int, default=4,
        help="SVT-AV1 preset 0-10, lower=slower+better (default: 4)",
    )
    parser.add_argument(
        "--min-cq", type=int, default=18,
        help="Minimum CQ / highest quality (default: 18)",
    )
    parser.add_argument(
        "--max-cq", type=int, default=38,
        help="Maximum CQ / lowest quality (default: 38)",
    )
    parser.add_argument(
        "--film-grain", type=int, default=24,
        help="Film grain synthesis level 0-50 (default: 24)",
    )
    parser.add_argument(
        "--no-10bit", action="store_true",
        help="Disable forced 10-bit encoding",
    )
    parser.add_argument(
        "--no-recurse", action="store_true",
        help="Don't process subdirectories",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-encode even if output exists",
    )
    parser.add_argument(
        "--samples", type=int, default=8,
        help="Number of sample segments for estimation (default: 8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Find optimal CQ but skip final encoding",
    )
    parser.add_argument(
        "--no-crops", action="store_true",
        help="Ignore <file>.crop.json sidecars (otherwise auto-applied when present "
             "and confidence=high)",
    )
    parser.add_argument(
        "--auto-crop", action="store_true",
        help="Detect letterbox/pillarbox crop inline for each file before encoding "
             "(skips files that already have a sidecar)",
    )
    parser.add_argument(
        "--seed-cq", type=int, default=None,
        help="Starting CQ for the search (default: auto from source bitrate; "
             "prompted interactively when run in a terminal)",
    )

    args = parser.parse_args()

    if args.min_cq > args.max_cq:
        parser.error("--min-cq must be <= --max-cq")
    if not 0 <= args.preset <= 10:
        parser.error("--preset must be 0-10 (SVT-AV1 v3 removed presets above 10)")
    if args.seed_cq is not None and not args.min_cq <= args.seed_cq <= args.max_cq:
        parser.error("--seed-cq must be within --min-cq..--max-cq")

    cfg = {
        "input_dir": args.input,
        "output_dir": args.output,
        "cache_dir": script_dir / "_cache",
        "container": ".mkv",
        "recurse": not args.no_recurse,
        "skip_existing": not args.overwrite,
        "preset": args.preset,
        "min_cq": args.min_cq,
        "max_cq": args.max_cq,
        "film_grain": args.film_grain,
        "force_10bit": not args.no_10bit,
        "maxrate_factor": 1.6,
        "target_vmaf": args.vmaf,
        "vmaf_tolerance": 0.1,
        "bitrate_margin": 1.20,
        "dry_run": args.dry_run,
        "use_crops": not args.no_crops,
        "auto_crop": args.auto_crop,
        "seed_cq": args.seed_cq,
        "sample_count": args.samples,
        "sample_duration": 6.0,
        "min_scene_duration": 2.0,
        "short_threshold": 48,
        "scene_threshold": 3,
    }

    return process_videos(cfg)


if __name__ == "__main__":
    try:
        code = main() or 0
    except KeyboardInterrupt:
        cleanup_temp()
        code = 0
    try:
        input("\nPress Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(code)
