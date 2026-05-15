#!/usr/bin/env python3
"""
av1q-crop — Detect letterbox/pillarbox crop for av1q.

Writes a sidecar JSON (<file>.crop.json) beside each source. av1q reads
these via --use-crops to remove black bars during encoding.

Conservative by design: only marks high-confidence crops for auto-apply.
Ambiguous results (dark sources, mixed aspect ratios) are written with
confidence="low" for manual review — never silently applied.
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True  # don't litter the script dir with __pycache__
from av1q import (
    VIDEO_EXTENSIONS,
    GREEN, ORANGE, PURPLE, RED, RESET, BOLD, DIM, CHECK, CROSS, SEP,
    _temp_files,
    cleanup_temp,
    partial_hash,
    probe_video,
    detect_hwaccel,
    detect_scenes,
    analyze_complexity,
    get_keyframes,
    select_samples,
)


def detect_crop_window(source, start, duration, limit, round_to, cache_dir):
    """Run cropdetect on a single time window. Returns (w, h, x, y) or None."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = cache_dir / f"crop_{os.getpid()}_{int(time.time() * 1_000_000)}.txt"
    _temp_files.add(log)
    # ffmpeg's filter graph parser uses ':' as the option separator, so
    # Windows drive colons in the metadata file path must be double-escaped
    # (one level for the graph, one for the option value).
    log_path = log.as_posix().replace(":", "\\\\:")

    try:
        hw = detect_hwaccel()
        attempts = [hw, None] if hw else [None]

        last_err = ""
        for accel in attempts:
            cmd = ["ffmpeg", "-hide_banner", "-v", "error"]
            if accel:
                cmd += ["-hwaccel", accel]
            cmd += [
                "-ss", f"{start:.3f}",
                "-i", str(source),
                "-t", f"{duration:.3f}",
                "-an", "-sn",
                "-vf",
                f"cropdetect=limit={limit}:round={round_to}:reset_count=0,"
                f"metadata=mode=print:file={log_path}",
                "-f", "null", "-",
            ]
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=120,
            )
            if r.returncode == 0:
                break
            last_err = r.stderr
        else:
            return None

        if not log.exists():
            return None

        w = h = x = y = None
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if "lavfi.cropdetect.w=" in line:
                try:
                    w = int(line.split("=")[-1])
                except ValueError:
                    pass
            elif "lavfi.cropdetect.h=" in line:
                try:
                    h = int(line.split("=")[-1])
                except ValueError:
                    pass
            elif "lavfi.cropdetect.x=" in line:
                try:
                    x = int(line.split("=")[-1])
                except ValueError:
                    pass
            elif "lavfi.cropdetect.y=" in line:
                try:
                    y = int(line.split("=")[-1])
                except ValueError:
                    pass

        if None in (w, h, x, y) or w <= 0 or h <= 0:
            return None
        return (w, h, x, y)

    finally:
        try:
            if log.exists():
                log.unlink()
        except OSError:
            pass
        _temp_files.discard(log)


def aggregate_crops(windows, frame_w, frame_h, min_keep_ratio, agree_ratio):
    """Take mode across windows, apply confidence rules.

    Window agreement is the primary trust signal: misdetects happen on
    specific frames (fades, dark scenes) so they make windows disagree.
    A crop that every window returns identically is real even when
    aggressive (e.g. 9:16 vertical content in a 16:9 frame keeps only
    ~32% of pixels but is detected unanimously).

    Mode (not median) handles outlier windows naturally — HandBrake uses
    median and over-crops on dark/mixed-aspect content (long-known bug).
    """
    valid = [c for c in windows if c is not None]
    n_total = len(windows)
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "crop": None, "confidence": "low",
            "reason": "no windows returned crop values (source too dark or unreadable)",
        }

    if n_valid < n_total * 0.7:
        return {
            "crop": None, "confidence": "low",
            "reason": (
                f"only {n_valid}/{n_total} windows returned valid crops "
                f"(likely many dark scenes)"
            ),
        }

    counter = collections.Counter(valid)
    top, top_count = counter.most_common(1)[0]
    w, h, x, y = top

    if w >= frame_w and h >= frame_h:
        return {
            "crop": None, "confidence": "none",
            "reason": "full frame — no letterbox/pillarbox detected",
        }

    agreement = top_count / n_valid
    keep_ratio = (w * h) / (frame_w * frame_h)

    # Windows disagreeing is the strongest unreliability signal — likely
    # mixed aspect ratios (IMAX-style scenes) or scattered misdetects.
    if agreement < agree_ratio:
        return {
            "crop": top, "confidence": "low",
            "reason": (
                f"{len(counter)} distinct crop values across {n_valid} windows; "
                f"top supported by only {top_count}/{n_valid} "
                f"(likely mixed aspect ratios)"
            ),
        }

    # Catastrophic-misdetect floor. Below this, even unanimous agreement
    # can't be trusted (e.g. an entirely dark video where every window
    # misdetects identically into a tiny region). Set well below the
    # most-aggressive legitimate case (9:16 vertical in 16:9 ≈ 32%).
    if keep_ratio < min_keep_ratio:
        return {
            "crop": top, "confidence": "low",
            "reason": (
                f"detected crop keeps only {keep_ratio:.0%} of frame; "
                f"below safety floor of {min_keep_ratio:.0%} "
                f"(rerun with --min-keep-ratio if intentional)"
            ),
        }

    return {
        "crop": top, "confidence": "high",
        "reason": (
            f"agreed on {top_count}/{n_valid} windows "
            f"({keep_ratio:.0%} of frame kept)"
        ),
    }


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

    is_hdr = meta["hdr"] or "10le" in meta["pix_fmt"]
    limit = cfg["limit_hdr"] if is_hdr else cfg["limit_sdr"]
    print(
        f"  {ORANGE}{'probe':<10}{RESET}"
        f"{meta['w']}x{meta['h']} "
        f"{DIM}{'HDR' if is_hdr else 'SDR'} · limit={limit}{RESET}"
    )

    sample_cfg = {
        "scene_threshold": cfg["scene_threshold"],
        "cache_dir": cfg["cache_dir"],
        "short_threshold": cfg["short_threshold"],
        "sample_duration": cfg["window_duration"],
        "min_scene_duration": 2.0,
    }

    safe_start = meta["duration"] * 0.05
    safe_end = meta["duration"] * 0.95

    scenes = []
    complexity = []
    keyframes = []
    if meta["duration"] >= cfg["short_threshold"]:
        scenes = detect_scenes(source, sample_cfg)
        complexity = analyze_complexity(source)
        keyframes = get_keyframes(source)

    samples = None
    if scenes:
        scoped = [s for s in scenes if safe_start <= s["time"] <= safe_end]
        if scoped:
            samples = select_samples(
                scoped, complexity, meta["duration"],
                cfg["sample_count"], keyframes, sample_cfg,
            )

    if not samples:
        n = cfg["sample_count"]
        span = max(0.0, safe_end - safe_start)
        if span <= 0:
            n = 1
            span = max(meta["duration"], 1.0)
            safe_start = 0.0
        samples = [
            {"time": safe_start + span * (i + 0.5) / n,
             "duration": cfg["window_duration"]}
            for i in range(n)
        ]
        print(f"  {DIM}using fixed grid ({n} windows){RESET}")
    else:
        print(
            f"  {DIM}{len(samples)} scene-windows · "
            f"{cfg['window_duration']:.0f}s each{RESET}"
        )

    crops = []
    for i, s in enumerate(samples):
        c = detect_crop_window(
            source, s["time"], cfg["window_duration"],
            limit, cfg["round"], cfg["cache_dir"],
        )
        crops.append(c)
        marker = CHECK if c else CROSS
        cstr = f"{c[0]}:{c[1]}:{c[2]}:{c[3]}" if c else "—"
        print(
            f"  {DIM}window {i + 1}/{len(samples)} @ "
            f"{s['time']:.0f}s {marker} {cstr}{RESET}"
        )

    result = aggregate_crops(
        crops, meta["w"], meta["h"],
        cfg["min_keep_ratio"], cfg["agree_ratio"],
    )

    sidecar_data = {
        "version": 1,
        "source_hash": partial_hash(source),
        "source_name": source.name,
        "frame_width": meta["w"],
        "frame_height": meta["h"],
        "hdr": is_hdr,
        "limit": limit,
        "round": cfg["round"],
        "confidence": result["confidence"],
        "reason": result["reason"],
        "windows": [
            {
                "time": round(s["time"], 2),
                "crop": (f"{c[0]}:{c[1]}:{c[2]}:{c[3]}" if c else None),
            }
            for s, c in zip(samples, crops)
        ],
        "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if result["crop"]:
        w, h, x, y = result["crop"]
        sidecar_data.update({"width": w, "height": h, "x": x, "y": y})

    conf = result["confidence"]
    color = GREEN if conf == "high" else (ORANGE if conf == "low" else DIM)
    if result["crop"]:
        w, h, x, y = result["crop"]
        out = f"crop={w}:{h}:{x}:{y}"
    else:
        out = "no crop"
    print(
        f"  {color}{BOLD}{conf:<10}{RESET}{out}  "
        f"{DIM}({result['reason']}){RESET}"
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
