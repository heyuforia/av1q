"""Content analysis: scene detection, packet-stat complexity sampling,
and keyframe listing (demux only — no decode)."""

import json
import subprocess

from .probe import detect_hwaccel
from .util import _temp_files, clamp, escape_filter_path, make_temp_log


def detect_scenes(source, cfg, duration=None):
    """Detect scene changes using ffmpeg's scdet filter.

    scdet decodes the whole file (downscaled to 640px) frame by frame, so
    its wall time scales with runtime — measured ~13x realtime for 1080p
    H.264, slower for 4K/HEVC. A fixed short cap silently killed the scan
    on feature-length sources and dropped the pipeline to evenly-spaced
    samples on exactly the long films that most benefit from complexity-
    biased selection. The scan budget therefore scales with duration (a
    1x-realtime allowance — many times the measured throughput, so it
    only trips on a stalled decode), floored for short files and ceiled so
    a genuinely hung process still can't block forever. A timeout is a
    graceful miss: it returns [] like any other failure and the caller
    falls back to even sampling.
    """
    cache_dir = cfg["cache_dir"]
    log = make_temp_log(cache_dir, "scdet", "txt")
    log_path = escape_filter_path(log)
    scan_timeout = (
        int(clamp(duration, 300, 3600)) if duration and duration > 0 else 300
    )

    try:
        hw = detect_hwaccel()
        attempts = [hw, None] if hw else [None]

        for accel in attempts:
            cmd = ["ffmpeg", "-hide_banner"]
            if accel:
                cmd += ["-hwaccel", accel]
            cmd += [
                "-i", str(source), "-an",
                "-vf", f"scale=640:-2,scdet=t={cfg['scene_threshold']},"
                       f"metadata=mode=print:file={log_path}",
                "-f", "null", "-",
            ]
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                timeout=scan_timeout,
            )
            if r.returncode == 0:
                break
        else:
            if log.exists():
                log.unlink()
            _temp_files.discard(log)
            return []

        scenes = []
        if log.exists():
            for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "lavfi.scd.time" in line:
                    try:
                        scenes.append({
                            "time": float(line.split("=")[1].strip()),
                            "score": 10.0,
                        })
                    except (ValueError, IndexError):
                        pass
            log.unlink()
            _temp_files.discard(log)

        for i, sc in enumerate(scenes):
            sc["duration"] = (
                scenes[i + 1]["time"] - sc["time"]
                if i + 1 < len(scenes) else 10.0
            )
        return scenes

    except Exception:
        return []


def analyze_complexity(source):
    """Analyze per-window frame complexity by packet sizes.

    Reads packets (demux only — no decode), so it runs at I/O speed even
    on long 4K sources. Keyframe packets (flag K) stand in for I-frames,
    everything else for inter frames; the avg-size + inter/intra ratio
    heuristic only needs that split, not exact pict_types.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time,dts_time,size,flags",
             "-of", "json", str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        if r.returncode != 0:
            return []

        packets = json.loads(r.stdout or "{}").get("packets", [])
        if not packets:
            return []

        windows = {}
        for pk in packets:
            try:
                t = float(pk.get("pts_time") or pk.get("dts_time") or 0)
                sz = int(pk.get("size", 0))
            except (TypeError, ValueError):
                continue
            is_key = "K" in (pk.get("flags") or "")
            idx = int(t / 5)
            if idx not in windows:
                windows[idx] = {"time": idx * 5, "i": [], "p": []}
            if is_key:
                windows[idx]["i"].append(sz)
            else:
                windows[idx]["p"].append(sz)

        results = []
        for idx in sorted(windows):
            w = windows[idx]
            i, p = w["i"], w["p"]
            avg = (
                sum(i) / len(i) / 1000 if i
                else sum(p) / len(p) / 1000 if p
                else 0
            )
            ratio = (
                (sum(p) / len(p)) / (sum(i) / len(i))
                if i and p else 0.5
            )
            results.append({
                "time": w["time"],
                "complexity": avg * 0.7 + ratio * 100 * 0.3,
            })
        return results

    except Exception:
        return []


def get_keyframes(source):
    """Get sorted list of keyframe timestamps.

    Reads packet headers (demux only) — the K flag marks keyframes, so no
    decode is needed, unlike the -skip_frame nokey frame walk this replaces.
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time,dts_time,flags",
             "-of", "json", str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=600,
        )
        if r.returncode != 0:
            return []
        keyframes = []
        for pk in json.loads(r.stdout or "{}").get("packets", []):
            if "K" not in (pk.get("flags") or ""):
                continue
            try:
                keyframes.append(float(pk.get("pts_time") or pk["dts_time"]))
            except (TypeError, ValueError, KeyError):
                continue
        return sorted(keyframes)
    except Exception:
        return []
