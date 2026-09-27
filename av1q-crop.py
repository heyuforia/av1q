#!/usr/bin/env python3
"""
av1q-crop — Batch letterbox/pillarbox crop detection for av1q.

Writes a sidecar JSON per source under _cache/_crop, named after the
source, so input folders stay clean. av1q reads these automatically
(use --no-crops to ignore). For a one-step workflow,
run av1q.py --auto-crop instead — it does the same detection inline
before each encode.

Conservative by design: only marks high-confidence crops for auto-apply.
Ambiguous results (dark sources, mixed aspect ratios, rotated sources)
are written with confidence="low" for manual review — never silently
applied.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True  # don't litter the script dir with __pycache__
from av1q import (
    VIDEO_EXTENSIONS, SCENE_THRESHOLD, SHORT_THRESHOLD,
    SCAN_WINDOWS, WINDOW_DURATION, LIMIT_SDR, LIMIT_HDR, ROUND,
    MIN_KEEP_RATIO, AGREE_RATIO,
    PURPLE, RESET, BOLD, DIM, CHECK, CROSS, SEP, label,
    atomic_write_json,
    cleanup_temp,
    crop_scan_cfg,
    partial_hash,
    probe_video,
    sidecar_path,
    detect_crop_for_file,
    run_launcher,
)


def process_file(source, cfg):
    file_hash = partial_hash(source)
    sidecar = sidecar_path(cfg["cache_dir"], source, file_hash)
    if sidecar.exists() and not cfg["force"]:
        print(f"{label('skip')}{DIM}sidecar exists (--force rewrites it){RESET}")
        return

    try:
        meta = probe_video(source)
    except Exception as e:
        print(f" {CROSS} probe failed: {e}")
        return

    # The same sources the encoders pass over: an audio-only container
    # has no picture to scan, and an AV1 source is never encoded, so a
    # sidecar for it would only ever be read by hand.
    if not meta["w"] or not meta["h"]:
        print(f" {CROSS} No video stream, skipping")
        return
    if meta["codec"] == "av1":
        print(f" {CHECK} Already AV1, skipping")
        return

    sidecar_data = detect_crop_for_file(source, meta, cfg, file_hash)

    if cfg["dry_run"]:
        print(f"{label('dry-run')}{DIM}sidecar not written{RESET}")
        return

    sidecar.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(sidecar, sidecar_data, indent=2)


def main():
    script_dir = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="av1q-crop — detect letterbox/pillarbox crop for av1q",
    )
    p.add_argument(
        "-i", "--input", type=Path,
        default=script_dir / "Video Input",
        help="Input dir or single file (default: ./Video Input)",
    )
    p.add_argument("--no-recurse", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="rewrite existing sidecars")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sample-count", type=int, default=SCAN_WINDOWS,
                   help=f"windows scanned per file (default: {SCAN_WINDOWS})")
    p.add_argument("--window-duration", type=float, default=WINDOW_DURATION,
                   help=f"seconds of cropdetect per window (default: {WINDOW_DURATION:g})")
    p.add_argument("--limit-sdr", type=int, default=LIMIT_SDR,
                   help="cropdetect darkness threshold for SDR, as an 8-bit "
                        f"code value 0-255, scaled to the source's bit depth "
                        f"(default: {LIMIT_SDR})")
    p.add_argument("--limit-hdr", type=int, default=LIMIT_HDR,
                   help="cropdetect darkness threshold for HDR, as a 10-bit "
                        f"code value 0-1023 (default: {LIMIT_HDR})")
    p.add_argument("--round", type=int, default=ROUND,
                   help=f"output dim divisibility (2=accurate, 16=codec-friendly; "
                        f"default: {ROUND})")
    p.add_argument("--min-keep-ratio", type=float, default=MIN_KEEP_RATIO,
                   help="absolute floor — refuse high confidence if cropped area "
                        f"below this (default {MIN_KEEP_RATIO}, catches "
                        "catastrophic misdetect only)")
    p.add_argument("--agree-ratio", type=float, default=AGREE_RATIO,
                   help=f"fraction of windows that must agree (default {AGREE_RATIO})")

    args = p.parse_args()

    cfg = crop_scan_cfg(
        {
            "cache_dir": script_dir / "_cache",
            "scene_threshold": SCENE_THRESHOLD,
            "short_threshold": SHORT_THRESHOLD,
        },
        force=args.force,
        dry_run=args.dry_run,
        sample_count=max(1, args.sample_count),
        window_duration=max(0.5, args.window_duration),
        limit_sdr=args.limit_sdr,
        limit_hdr=args.limit_hdr,
        round=max(2, args.round),
        min_keep_ratio=args.min_keep_ratio,
        agree_ratio=args.agree_ratio,
    )
    cfg["cache_dir"].mkdir(parents=True, exist_ok=True)

    print(f"{PURPLE}{BOLD}av1q-crop{RESET}\n{SEP}")

    if args.input.is_file():
        files = [args.input]
    else:
        # A missing folder is created, as the encode launchers do, so a
        # first double-click leaves the place to drop files into; a
        # missing FILE (a typo'd video name) is an error, not a folder.
        if (not args.input.exists()
                and args.input.suffix.lower() in VIDEO_EXTENSIONS):
            print(f"{CROSS} input not found: {args.input}")
            return 1
        args.input.mkdir(parents=True, exist_ok=True)
        pattern = "**/*" if not args.no_recurse else "*"
        files = sorted(
            f for f in args.input.glob(pattern)
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

    if not files:
        print(f"{CROSS} no videos found in {args.input}")
        return 1

    total = len(files)
    t_start = time.time()
    for idx, f in enumerate(files, 1):
        if idx > 1:
            print(SEP)
        print(f"{PURPLE}{BOLD}[{idx}/{total}]{RESET} {PURPLE}{f.name}{RESET}")
        try:
            process_file(f, cfg)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f" {CROSS} {e}")
        finally:
            cleanup_temp()

    print(SEP)
    elapsed = time.time() - t_start
    print(f"{CHECK} Scanned {BOLD}{total}{RESET} files in {BOLD}{elapsed:.1f}s{RESET}")
    print(f"{CHECK} Done")
    return 0


if __name__ == "__main__":
    run_launcher(main)
