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
    fmt_time, fmt_size, vmaf_pass_color, fmt_s2, label,
)
from core.constants import (
    VIDEO_EXTENSIONS, INTRA_ONLY_CODECS, TARGET_VMAF_BY_RES,
    FALLBACK_MAXRATE, MIN_BITRATE_KBPS, VMAF_OVERSHOOT,
    OUTPUT_CONTAINER, VMAF_TOLERANCE, BITRATE_MARGIN, SAMPLE_DURATION,
    MIN_SCENE_DURATION, SHORT_THRESHOLD, SCENE_THRESHOLD,
)
from core.util import (
    _temp_files, run_cmd, cleanup_temp, atomic_write_json, make_temp_log,
    escape_filter_path, partial_hash, clamp, run_launcher,
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
    SCAN_WINDOWS, WINDOW_DURATION, LIMIT_SDR, LIMIT_HDR, ROUND,
    MIN_KEEP_RATIO, AGREE_RATIO,
    crop_scan_cfg, crop_token, load_crop_sidecar, read_crop_sidecar,
    sidecar_crop, sidecar_path, detect_crop_window, aggregate_crops,
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
    av1q.probe_video / av1q.measure_ssimu2_display keeps working.
    """
    return core_search.search(
        source, meta, target, enc_func, cfg, _ENGINE,
        tag=tag,
        measure_fn=lambda ref, dist, q: vmaf_cached(
            ref, dist, meta, q, cache, cache_path, threads, tag=tag),
        probe_fn=lambda f: probe_video(f),
        s2_fn=lambda ref, dist, m, ri: measure_ssimu2_display(
            ref, dist, m, cfg["cache_dir"], ref_index=ri),
        s2_ref_index=core_search.search_ref_index(_ENGINE, cfg, source, tag),
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
        help="Encode every file again, finished ones included, reusing "
             "no encode an earlier run made",
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
        help="Ignore crop sidecars in _cache/_crop (otherwise auto-applied "
             "when present and confidence=high)",
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
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Disable resumable segmented encoding for long sources "
             "(by default, interrupted full encodes resume at the last "
             "finished segment instead of restarting)",
    )
    parser.add_argument(
        "--force-cq", type=int, default=None,
        help="Encode every file at exactly this CQ and skip everything "
             "else: no sampling, no search, no VMAF measurement, no "
             "refinement. Independent of --min-cq/--max-cq. Outputs at "
             "different forced values coexist, and a result larger than "
             "the source is kept.",
    )

    args = parser.parse_args()

    if args.min_cq > args.max_cq:
        parser.error("--min-cq must be <= --max-cq")
    # 1-63: ffmpeg's libsvtav1 wrapper clamps -crf to 63 and treats 0 as
    # "unset" — out-of-range bounds would silently encode at a different
    # CQ than the one the search records in the cache.
    if not 1 <= args.min_cq <= 63 or not 1 <= args.max_cq <= 63:
        parser.error("CQ bounds must be within 1-63")
    if not 0 <= args.preset <= 10:
        parser.error("--preset must be 0-10 (SVT-AV1 4.0 capped presets at 10)")
    if not 0 <= args.film_grain <= 50:
        parser.error("--film-grain must be 0-50")
    if args.samples < 1:
        parser.error("--samples must be >= 1")
    # `0 <` also rejects nan; without this a --vmaf 0 would silently fall
    # through to the automatic per-resolution target.
    if args.vmaf is not None and not 0 < args.vmaf <= 100:
        parser.error("--vmaf must be above 0 and at most 100")
    if args.seed_cq is not None and not args.min_cq <= args.seed_cq <= args.max_cq:
        parser.error("--seed-cq must be within --min-cq..--max-cq")
    if args.force_cq is not None:
        # Same 1-63 wrapper-clamp rationale as the search bounds; checked
        # against the encoder's real range, not --min/--max-cq — those
        # bound the search, and --force-cq replaces it.
        if not 1 <= args.force_cq <= 63:
            parser.error("--force-cq must be within 1-63")
        if args.vmaf is not None:
            parser.error("--force-cq has no VMAF target; drop --vmaf")
        if args.seed_cq is not None:
            parser.error("--force-cq replaces the search; drop --seed-cq")
        if args.dry_run:
            parser.error("--force-cq has no search to dry-run; drop --dry-run")

    cfg = {
        "input_dir": args.input,
        "output_dir": args.output,
        "cache_dir": script_dir / "_cache",
        "learned_dir": script_dir / "_learned",
        "container": OUTPUT_CONTAINER,
        "recurse": not args.no_recurse,
        "skip_existing": not args.overwrite,
        "preset": args.preset,
        "min_cq": args.min_cq,
        "max_cq": args.max_cq,
        "film_grain": args.film_grain,
        "force_10bit": not args.no_10bit,
        "target_vmaf": args.vmaf,
        "vmaf_tolerance": VMAF_TOLERANCE,
        "bitrate_margin": BITRATE_MARGIN,
        "dry_run": args.dry_run,
        "use_crops": not args.no_crops,
        "auto_crop": args.auto_crop,
        "seed_cq": args.seed_cq,
        "force_q": args.force_cq,
        "resume_encodes": not args.no_resume,
        "sample_count": args.samples,
        "sample_duration": SAMPLE_DURATION,
        "min_scene_duration": MIN_SCENE_DURATION,
        "short_threshold": SHORT_THRESHOLD,
        "scene_threshold": SCENE_THRESHOLD,
    }

    return process_videos(cfg)


if __name__ == "__main__":
    run_launcher(main)
