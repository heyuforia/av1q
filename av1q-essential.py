#!/usr/bin/env python3
"""
av1q-essential — VMAF-targeted AV1 encoding with SVT-AV1-Essential.

Side-script to av1q.py, the way av1q-crop.py is: av1q's exact pipeline
brain — scene-sampled adaptive search, per-resolution VMAF targets,
bitrate floors, sample↔full calibration — with only the encoder swapped.
Encoding runs through the standalone SVT-AV1-Essential binary (ffmpeg
pipes 10-bit Y4M into it) on its quarter-step CRF grid.

SSIMULACRA2 (FFVship, GPU) is measured and shown next to every VMAF
score as information only — it never drives a decision. Without an
FFVship binary the info column simply disappears.

The shared brain lives in the core/ package (core.search, core.pipeline)
and is driven here through the Essential engine (core/engines/essential.py);
av1q.py is imported for the shared helper names, exactly the pattern
av1q-crop.py uses. The two pipelines keep separate per-file caches
(different encoders must never share calibration), while the
extracted-sample cache is shared.

Tools expected under ./tools (any subfolder):
  SvtAv1EncApp*  — SVT-AV1-Essential encoder binary (required)
  FFVship*       — Vship standalone CLI (optional — GPU SSIMULACRA2 info)
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True  # don't litter the script dir with __pycache__
import av1q

from av1q import (
    VIDEO_EXTENSIONS, INTRA_ONLY_CODECS, MIN_BITRATE_KBPS,
    TARGET_VMAF_BY_RES, VMAF_OVERSHOOT,
    GREEN, ORANGE, PURPLE, RED, RESET, BOLD, DIM, CHECK, CROSS, SEP, MIDDOT,
    run_cmd, cleanup_temp, atomic_write_json, make_temp_log,
    partial_hash, res_tier, calc_kbps, video_kbps, measured_kbps,
    effective_sample_floor, crop_token, initial_cq_seed, clamp,
    fmt_time, fmt_size, vmaf_pass_color, fmt_s2,
    probe_video, get_fps, detect_scenes, analyze_complexity, get_keyframes,
    select_samples, extract_samples, measure_vmaf,
    load_cache, load_global_calibration,
    calibration_offset, update_global_calibration,
    load_crop_sidecar, detect_crop_for_file,
)
from core import search as core_search
from core import pipeline as core_pipeline
from core import vmaf as core_vmaf
from core.engines.essential import (
    EssentialEngine,
    CRF_STEP, qcrf, crf_str, crf_range, enc_signature_e,
    SVT_PRIMARIES, SVT_TRANSFER, SVT_MATRIX, SVT_RANGE, build_color_args,
    encode_essential,
)
from core.probe import _ratval, is_vfr, probe_hdr_metadata
from core.sampling import clean_sample_source
from core.ssimu2 import ffvship_crop_args, parse_ssimu2_json, ssimu2_info
from core.tools import find_encoder

_ENGINE = EssentialEngine()


# ── Constants ────────────────────────────────────────────────

SIG = "avqe-c1"


SCRIPT_DIR = Path(__file__).resolve().parent


# ── Tool discovery ───────────────────────────────────────────


def find_ffvship():
    """FFVship is optional — it only powers the SSIMULACRA2 info column,
    so a missing binary returns None instead of raising. Delegates to
    av1q's helper, which auto-downloads the GPU-matched build into
    tools/FFVship/ on first run."""
    return av1q.find_ffvship_optional()


# ── VMAF caching (quarter-step grid) ─────────────────────────


def vmaf_cached_e(ref, dist, meta, crf, cache, cache_path, cfg, tag=None):
    """av1q.vmaf_cached generalized to the quarter-step CRF grid.

    Entries are keyed by crf_str and the scores stored under 'vmaf' /
    'sample_vmaf' — deliberately NOT the 'full' / 'sample_full' keys the
    SSIMU2-era cache used, so a stale avqe-c1 entry from before the
    metric switch can never be misread as a VMAF score (the sig stays
    fixed by policy; key separation does the invalidation instead).

    Compat wrapper over core.vmaf.vmaf_cached. The measure closure
    resolves this module's globals at call time so monkeypatching
    this module's measure_vmaf keeps working.
    """
    return core_vmaf.vmaf_cached(
        ref, dist, meta, crf, cache, cache_path, tag=tag,
        threads=cfg.get("vmaf_threads") or (os.cpu_count() or 4),
        log_dir=cfg["e_cache_dir"],
        key_base="vmaf", q_key=crf_str(crf),
        measure=lambda *a: measure_vmaf(*a),
    )


# ── CRF Search ───────────────────────────────────────────────


def search_crf(source, meta, target, cache, cache_path, enc_func, cfg, tag=None):
    """Find the optimal CRF that hits the target VMAF using adaptive
    search on the quarter-step grid.

    Compat wrapper over the shared brain (core.search.search) with the
    Essential engine. The measurement closures resolve this module's
    globals at call time, so monkeypatching this module's vmaf_cached_e /
    probe_video keeps working.
    """
    return core_search.search(
        source, meta, target, cache, cache_path, enc_func, cfg, _ENGINE,
        tag=tag,
        measure_fn=lambda ref, dist, q: vmaf_cached_e(
            ref, dist, meta, q, cache, cache_path, cfg, tag=tag),
        probe_fn=lambda f: probe_video(f),
        s2_fn=lambda ref, dist, m, ri: ssimu2_info(
            ref, dist, m, cfg, ref_index=ri),
        s2_ref_index=cfg.get("_search_ref_index"),
    )


# ── Main Processing ──────────────────────────────────────────


def process_videos(cfg):
    """Process every video under cfg's input dir with the Essential
    engine (Y4M pipe into SvtAv1EncApp). The pipeline lives in
    core.pipeline."""
    return core_pipeline.process_videos(cfg, _ENGINE)


# ── CLI ──────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="av1q-essential — VMAF-targeted AV1 encoding with SVT-AV1-Essential",
    )
    parser.add_argument(
        "-i", "--input", type=Path,
        default=SCRIPT_DIR / "Video Input",
        help="Input directory (default: ./Video Input)",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=SCRIPT_DIR / "AV1 Output",
        help="Output directory (default: ./AV1 Output)",
    )
    parser.add_argument(
        "--vmaf", type=float, default=None,
        help="Target VMAF score (default: auto by resolution — "
             "94 HD, 93 SD, 90 4K, same tiers as av1q)",
    )
    parser.add_argument(
        "--preset", type=int, default=4,
        help="Encoder preset 0-10, lower=slower+better (default: 4)",
    )
    parser.add_argument(
        "--min-crf", type=float, default=18.0,
        help="Minimum CRF / highest quality, 0.25 steps (default: 18)",
    )
    parser.add_argument(
        "--max-crf", type=float, default=38.0,
        help="Maximum CRF / lowest quality, 0.25 steps (default: 38)",
    )
    parser.add_argument(
        "--film-grain", type=int, default=24,
        help="Film grain synthesis level 0-50 (default: 24, matching av1q — "
             "won the eye test against the fork docs' suggested 8; lighter "
             "grain reads as waxy skin even though SSIMU2 scores it higher)",
    )
    parser.add_argument(
        "--tune", type=int, default=1,
        help="Encoder tune (default: 1, the fork default; 0=VQ, 2=SSIM, "
             "3=IQ, 4=SSIMU2-optimized — 4 inflates the measured target)",
    )
    parser.add_argument(
        "--metric-every", type=int, default=1,
        help="Measure every Nth frame in the FFVship SSIMU2 info column "
             "(default: 1 = all frames)",
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
        help="Find optimal CRF but skip final encoding",
    )
    parser.add_argument(
        "--no-crops", action="store_true",
        help="Ignore <file>.crop.json sidecars",
    )
    parser.add_argument(
        "--auto-crop", action="store_true",
        help="Detect letterbox/pillarbox crop inline before each encode",
    )
    parser.add_argument(
        "--seed-crf", type=float, default=None,
        help="Starting CRF for the search (default: auto from source bitrate; "
             "prompted interactively when run in a terminal)",
    )

    args = parser.parse_args()

    if args.min_crf > args.max_crf:
        parser.error("--min-crf must be <= --max-crf")
    if not 1 <= args.min_crf <= 70 or not 1 <= args.max_crf <= 70:
        parser.error("CRF bounds must be within 1-70")
    if not 0 <= args.preset <= 10:
        parser.error("--preset must be 0-10")
    if not 0 <= args.tune <= 4:
        parser.error("--tune must be 0-4")
    if not 0 <= args.film_grain <= 50:
        parser.error("--film-grain must be 0-50")
    if args.metric_every < 1:
        parser.error("--metric-every must be >= 1")
    if args.seed_crf is not None and not args.min_crf <= args.seed_crf <= args.max_crf:
        parser.error("--seed-crf must be within --min-crf..--max-crf")

    cache_dir = SCRIPT_DIR / "_cache"
    cfg = {
        "input_dir": args.input,
        "output_dir": args.output,
        # _cache is shared with av1q (sample extraction + temp logs);
        # everything encoder/metric-specific lives under _essential so
        # the two pipelines never clobber each other's per-file caches.
        "cache_dir": cache_dir,
        "e_cache_dir": cache_dir / "_essential",
        "container": ".mkv",
        "recurse": not args.no_recurse,
        "skip_existing": not args.overwrite,
        "preset": args.preset,
        "min_crf": qcrf(args.min_crf),
        "max_crf": qcrf(args.max_crf),
        "film_grain": args.film_grain,
        "tune": args.tune,
        "target_vmaf": args.vmaf,
        "vmaf_tolerance": 0.1,
        "bitrate_margin": 1.20,
        "metric_every": args.metric_every,
        "dry_run": args.dry_run,
        "use_crops": not args.no_crops,
        "auto_crop": args.auto_crop,
        "seed_crf": qcrf(args.seed_crf) if args.seed_crf is not None else None,
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
