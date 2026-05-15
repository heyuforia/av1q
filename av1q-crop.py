#!/usr/bin/env python3
"""
av1q-crop — Batch letterbox/pillarbox crop detection for av1q.

Writes a sidecar JSON (<file>.crop.json) beside each source. av1q reads
these automatically (use --no-crops to ignore). For a one-step workflow,
run av1q.py --auto-crop instead — it does the same detection inline
before each encode.

Conservative by design: only marks high-confidence crops for auto-apply.
Ambiguous results (dark sources, mixed aspect ratios) are written with
confidence="low" for manual review — never silently applied.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True  # don't litter the script dir with __pycache__
from av1q import (
    VIDEO_EXTENSIONS,
    PURPLE, RESET, BOLD, DIM, CHECK, CROSS, SEP,
    cleanup_temp,
    partial_hash,
    probe_video,
    detect_crop_for_file,
)


def process_file(source, cfg):
    sidecar = source.with_suffix(source.suffix + ".crop.json")
    if sidecar.exists() and not cfg["force"]:
        print(f"  {DIM}sidecar exists — skipping (use --force to rewrite){RESET}")
        return

    try:
        meta = probe_video(source)
    except Exception as e:
        print(f"  {CROSS} probe failed: {e}")
        return

    if meta["w"] <= 0 or meta["h"] <= 0 or meta["duration"] <= 0:
        print(f"  {CROSS} invalid source dimensions or duration")
        return

    file_hash = partial_hash(source)
    sidecar_data = detect_crop_for_file(
        source, meta, cfg, file_hash, label_prefix="  ",
    )

    if cfg["dry_run"]:
        print(f"  {DIM}(dry-run, sidecar not written){RESET}")
        return

    tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    tmp.write_text(json.dumps(sidecar_data, indent=2), encoding="utf-8")
    tmp.replace(sidecar)


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
    p.add_argument("--sample-count", type=int, default=8)
    p.add_argument("--window-duration", type=float, default=2.0,
                   help="seconds of cropdetect per window (default: 2.0)")
    p.add_argument("--limit-sdr", type=int, default=24,
                   help="cropdetect darkness threshold for SDR (0-255)")
    p.add_argument("--limit-hdr", type=int, default=128,
                   help="cropdetect darkness threshold for HDR (0-255)")
    p.add_argument("--round", type=int, default=2,
                   help="output dim divisibility (2=accurate, 16=codec-friendly)")
    p.add_argument("--min-keep-ratio", type=float, default=0.10,
                   help="absolute floor — refuse high confidence if cropped area "
                        "below this (default 0.10, catches catastrophic misdetect only)")
    p.add_argument("--agree-ratio", type=float, default=0.75,
                   help="fraction of windows that must agree (default 0.75)")

    args = p.parse_args()

    cfg = {
        "cache_dir": script_dir / "_cache",
        "force": args.force,
        "dry_run": args.dry_run,
        "sample_count": max(1, args.sample_count),
        "window_duration": max(0.5, args.window_duration),
        "limit_sdr": args.limit_sdr,
        "limit_hdr": args.limit_hdr,
        "round": max(2, args.round),
        "min_keep_ratio": args.min_keep_ratio,
        "agree_ratio": args.agree_ratio,
        "scene_threshold": 3,
        "short_threshold": 48,
    }
    cfg["cache_dir"].mkdir(parents=True, exist_ok=True)

    print(f"{PURPLE}{BOLD}av1q-crop{RESET}\n{SEP}")

    if args.input.is_file():
        files = [args.input]
    else:
        if not args.input.exists():
            print(f"{CROSS} input not found: {args.input}")
            return 1
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
            print(f"  {CROSS} {e}")
        finally:
            cleanup_temp()

    print(SEP)
    elapsed = time.time() - t_start
    print(f"{CHECK} Scanned {BOLD}{total}{RESET} files in {BOLD}{elapsed:.1f}s{RESET}")
    print(f"{CHECK} Done")
    return 0


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
