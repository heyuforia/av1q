#!/usr/bin/env python3
"""
av1q — Intelligent VMAF-targeted AV1 video encoding.

Automatically finds the optimal CRF value for each video to hit a target VMAF
quality score, using scene-based sampling for fast quality estimation.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from colorama import init as colorama_init
    colorama_init()
except ImportError:
    if os.name == "nt":
        os.system("")

# ── ANSI Colors ──────────────────────────────────────────────

GREEN = "\033[38;5;46m"
ORANGE = "\033[38;5;208m"
PURPLE = "\033[38;5;141m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CHECK = f"{GREEN}✓{RESET}"
CROSS = f"{RED}✗{RESET}"
SEP = f"{DIM}{'─' * 48}{RESET}"

# ── Constants ────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".mov", ".m4v", ".ts", ".avi", ".webm"}
INTRA_ONLY_CODECS = {"prores", "dnxhd", "mjpeg", "rawvideo", "ffv1", "jpeg2000", "cfhd"}

TARGET_VMAF_BY_RES = {0: 93.0, 720: 94.0, 2160: 90.0}

FALLBACK_MAXRATE = {
    0: 8_000_000, 720: 12_000_000, 1080: 25_000_000,
    1440: 35_000_000, 2160: 45_000_000, 4320: 60_000_000,
}

MIN_BITRATE_KBPS = {0: 0, 720: 1000, 1080: 1500, 1440: 2500, 2160: 4000, 4320: 8000}

_temp_files = set()
_hwaccel = None
_hwaccel_checked = False

# ── Utilities ────────────────────────────────────────────────


def run_cmd(cmd):
    """Run a command and return the result. Raises on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        tail = "\n".join((p.stderr or "").splitlines()[-80:])
        raise RuntimeError(
            f"exit {p.returncode}\n"
            f"{subprocess.list2cmdline(cmd) if os.name == 'nt' else ' '.join(map(shlex.quote, cmd))}"
            f"\n{tail}"
        )
    return p


def cleanup_temp():
    for path in list(_temp_files):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
        _temp_files.discard(path)


def partial_hash(filepath, block=1 << 16):
    """Fast file identity hash: size + first/last 64KB."""
    h = hashlib.sha256()
    st = filepath.stat()
    h.update(st.st_size.to_bytes(8, "little"))
    with open(filepath, "rb") as f:
        h.update(f.read(block))
        if st.st_size > block * 2:
            f.seek(-block, 2)
            h.update(f.read(block))
    return h.hexdigest()


def res_tier(w, h):
    """Resolution tier based on the short dimension (handles vertical video)."""
    short = min(w, h)
    for t in (4320, 2160, 1440, 1080, 720):
        if short >= t:
            return t
    return 0


def calc_kbps(size_bytes, duration):
    if duration < 1.0:
        return None
    return int((size_bytes * 8) / 1000 / duration)


def video_kbps(filepath, duration=None):
    """Video-only bitrate by summing video packet sizes.

    File-size / duration counts muxed audio + subs, which breaks floor
    comparisons against sample bitrates (samples are -an video-only).
    """
    try:
        if duration is None:
            duration = probe_video(filepath)["duration"]
        if not duration or duration < 1.0:
            return None
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=size", "-of", "csv=p=0", str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=300,
        )
        if r.returncode != 0:
            return None
        total = sum(int(l) for l in r.stdout.splitlines() if l.strip())
        if total <= 0:
            return None
        return int(total * 8 / 1000 / duration)
    except Exception:
        return None


def effective_sample_floor(min_kbps, margin, calibration=None):
    """Sample-bitrate threshold that predicts full video clears min_kbps.

    Samples are cut from max-complexity scenes, so they encode at a higher
    bitrate than the full video at the same CQ. When a measured per-file
    sample→full ratio is cached, use it; otherwise fall back to the default
    margin.
    """
    if calibration:
        r = calibration.get("ratio")
        if isinstance(r, (int, float)) and 0.5 <= r <= 1.0:
            return min_kbps / r
    return min_kbps * margin


def initial_cq_seed(source_kbps, floor_kbps, min_cq, max_cq, default_cq=30):
    """Starting CQ tuned to source bitrate headroom over the floor.

    High source/floor ratio = lots of compression headroom → start lower.
    Low ratio or unknown data → fall back to default_cq. Seeding slightly
    below the optimal CQ is preferred: it costs marginal bitrate, while
    seeding above costs a full extra encode to step down.
    """
    lo = max(min_cq, min(min_cq + 2, max_cq))
    hi = max(lo, max_cq - 2)
    if not source_kbps or not floor_kbps or source_kbps <= 0 or floor_kbps <= 0:
        return max(min_cq, min(default_cq, max_cq))
    ratio = source_kbps / floor_kbps
    if ratio < 1.5:
        return max(min_cq, min(default_cq, max_cq))
    cq = round(36 - 4 * math.log2(ratio))
    return max(lo, min(cq, hi))


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def fmt_time(seconds):
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


# ── Hardware Acceleration ────────────────────────────────────


def detect_hwaccel():
    """Detect available hardware decoder. Cached after first call."""
    global _hwaccel, _hwaccel_checked
    if _hwaccel_checked:
        return _hwaccel

    candidates = {
        "Darwin": ["videotoolbox"],
        "Windows": ["cuda", "d3d11va"],
        "Linux": ["cuda", "vaapi"],
    }.get(platform.system(), [])

    for hw in candidates:
        try:
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccel", hw,
                 "-f", "lavfi", "-i", "nullsrc=s=16x16:d=0.01",
                 "-f", "null", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
            )
            if r.returncode == 0:
                _hwaccel = hw
                break
        except (subprocess.TimeoutExpired, OSError):
            pass

    _hwaccel_checked = True
    return _hwaccel


# ── Video Analysis ───────────────────────────────────────────


def probe_video(filepath):
    """Extract video metadata via ffprobe."""
    r = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,bit_rate,pix_fmt,color_primaries,"
        "color_transfer,color_space,color_range,codec_name",
        "-show_entries", "format=duration,bit_rate",
        "-of", "json", str(filepath),
    ])
    data = json.loads(r.stdout or "{}")
    s = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    bitrate = None
    for v in (fmt.get("bit_rate"), s.get("bit_rate")):
        if v:
            try:
                bitrate = int(v)
                break
            except (ValueError, TypeError):
                pass
    cp = (s.get("color_primaries") or "").lower()
    ct = (s.get("color_transfer") or "").lower()
    cs = (s.get("color_space") or "").lower()
    cr = (s.get("color_range") or "").lower()
    codec = (s.get("codec_name") or "").lower()
    pf = s.get("pix_fmt") or ""
    hdr = ct in {"smpte2084", "arib-std-b67"} or cp == "bt2020"

    return {
        "w": int(s.get("width") or 0),
        "h": int(s.get("height") or 0),
        "pix_fmt": pf, "bitrate": bitrate,
        "duration": float(fmt.get("duration") or 0),
        "cp": cp, "ct": ct, "cs": cs, "cr": cr,
        "codec": codec, "hdr": hdr,
    }


def get_fps(filepath):
    """Get frame rate as a rational string to avoid VMAF frame misalignment."""
    try:
        r = run_cmd([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=nw=1:nk=1", str(filepath),
        ])
        v = r.stdout.strip()
        if not v or v in ("0/0", "N/A"):
            return None
        if "/" in v:
            a, b = v.split("/", 1)
            if float(b) == 0 or float(a) / float(b) <= 0:
                return None
        elif float(v) <= 0:
            return None
        return v
    except (RuntimeError, ValueError):
        return None


def detect_scenes(source, cfg):
    """Detect scene changes using ffmpeg's scdet filter."""
    cache_dir = cfg["cache_dir"]
    log = cache_dir / f"scdet_{os.getpid()}_{int(time.time() * 1000)}.txt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _temp_files.add(log)

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
                       f"metadata=mode=print:file={log.as_posix()}",
                "-f", "null", "-",
            ]
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=300,
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
    """Analyze per-window frame complexity by packet sizes."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "frame=pict_type,pkt_size,pts_time",
             "-of", "json", str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=600,
        )
        if r.returncode != 0:
            return []

        frames = json.loads(r.stdout or "{}").get("frames", [])
        if not frames:
            return []

        windows = {}
        for fr in frames:
            t = float(fr.get("pts_time", 0))
            sz = int(fr.get("pkt_size", 0))
            pt = fr.get("pict_type", "")
            idx = int(t / 5)
            if idx not in windows:
                windows[idx] = {"time": idx * 5, "i": [], "p": []}
            if pt == "I":
                windows[idx]["i"].append(sz)
            elif pt == "P":
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
    """Get sorted list of keyframe timestamps."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
             "-of", "csv=p=0", str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=600,
        )
        return sorted(float(l.strip()) for l in r.stdout.splitlines() if l.strip())
    except Exception:
        return []


# ── Sample Selection ─────────────────────────────────────────


def select_samples(scenes, complexity, duration, count, keyframes, cfg):
    """Select representative sample segments for quality estimation."""
    if duration < cfg["short_threshold"]:
        return None

    sample_dur = cfg["sample_duration"]

    if not scenes:
        if not keyframes:
            return [
                {"time": duration * (i + 1) / (count + 1), "duration": sample_dur}
                for i in range(count)
            ]

        seg = duration / count
        selected = []
        for i in range(count):
            start, end = i * seg, (i + 1) * seg
            cands = [k for k in keyframes if start <= k < end]
            best = min(cands or keyframes, key=lambda k: abs(k - (start + end) / 2))
            if best not in [s["time"] for s in selected]:
                selected.append({"time": best, "duration": sample_dur})

        if len(selected) >= count // 2:
            return selected
        return [
            {"time": duration * (i + 1) / (count + 1), "duration": sample_dur}
            for i in range(count)
        ]

    comp_map = {int(c["time"] / 5) * 5: c["complexity"] for c in complexity}
    scored = [
        {
            "time": sc["time"],
            "duration": sc["duration"],
            "complexity": comp_map.get(int(sc["time"] / 5) * 5, 50),
        }
        for sc in scenes
        if sc["duration"] >= cfg["min_scene_duration"]
    ]

    if not scored:
        return select_samples([], complexity, duration, count, keyframes, cfg)

    seg = duration / count
    selected = []
    used = set()
    for i in range(count):
        start, end = i * seg, (i + 1) * seg
        cands = (
            [s for s in scored if start <= s["time"] < end and s["time"] not in used]
            or [s for s in scored if s["time"] not in used]
        )
        if cands:
            best = max(cands, key=lambda x: x["complexity"])
            selected.append({
                "time": best["time"],
                "duration": min(best["duration"], sample_dur),
            })
            used.add(best["time"])

    return selected or None


def extract_samples(source, scenes, keyframes, cfg, file_hash=None):
    """Extract and concatenate sample clips from the source video."""
    if not scenes:
        return None

    sample_dir = cfg["cache_dir"] / "_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tag = file_hash or f"{os.getpid()}_{int(time.time() * 1000)}"
    concat_out = sample_dir / f"samples_{tag}.mkv"
    if concat_out.exists() and concat_out.stat().st_size > 0:
        print(f"{'':>11}{DIM}Samples: {concat_out.stat().st_size / 1e6:.1f}MB (cached){RESET}")
        return concat_out

    ts = int(time.time() * 1000)
    clips = []
    if keyframes is None:
        keyframes = get_keyframes(source)

    for i, sc in enumerate(scenes):
        clip = sample_dir / f"sample_{ts}_{i}.mkv"
        _temp_files.add(clip)
        try:
            start = (
                min(keyframes, key=lambda k: abs(k - sc["time"]))
                if keyframes else sc["time"]
            )
            run_cmd([
                "ffmpeg", "-y", "-hide_banner", "-v", "error",
                "-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{sc['duration']:.3f}",
                "-c", "copy", "-an", "-avoid_negative_ts", "make_zero",
                str(clip),
            ])
            if clip.exists() and clip.stat().st_size > 0:
                clips.append(clip)
        except RuntimeError as e:
            print(f" {RED}Clip error: {e}{RESET}")

    if not clips:
        print(f" {RED}No clips extracted{RESET}")
        return None

    concat_list = sample_dir / f"concat_{ts}.txt"
    _temp_files.add(concat_list)
    concat_list.write_text(
        "\n".join(f"file '{c.as_posix()}'" for c in clips), encoding="utf-8"
    )

    try:
        run_cmd([
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", str(concat_out),
        ])
        for c in clips:
            try:
                c.unlink()
                _temp_files.discard(c)
            except OSError:
                pass
        concat_list.unlink()
        _temp_files.discard(concat_list)

        if concat_out.exists():
            print(f"{'':>11}{DIM}Samples: {concat_out.stat().st_size / 1e6:.1f}MB{RESET}")
            return concat_out
    except RuntimeError as e:
        print(f" {RED}Concat error: {e}{RESET}")
    return None


# ── VMAF Measurement ─────────────────────────────────────────


def measure_vmaf(ref, dist, meta, subsample, threads, cache_dir):
    """Compute VMAF score between reference and distorted video."""
    filters = []
    fps = get_fps(dist)

    filters.append("setpts=PTS-STARTPTS")  # normalize MP4 edit lists
    if fps:
        filters.append(f"fps={fps}")

    if meta["hdr"] or "10le" in meta["pix_fmt"]:
        filters += ["zscale=t=bt709:m=bt709:r=tv:p=709", "tonemap=hable:desat=0"]
    filters.append("format=yuv420p")

    pf = ",".join(filters)
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = cache_dir / f"vmaf_{os.getpid()}_{int(time.time() * 1000)}.json"
    _temp_files.add(log)

    th = f":n_threads={threads}" if threads > 1 else ""
    model = "vmaf_4k_v0.6.1" if meta["h"] >= 2160 else "vmaf_v0.6.1"
    log_esc = log.as_posix().replace("\\", "/").replace("'", "\\'").replace(":", "\\:")

    try:
        hw = detect_hwaccel()
        attempts = [hw, None] if hw else [None]

        for accel in attempts:
            cmd = ["ffmpeg", "-v", "error", "-hide_banner"]
            if accel:
                cmd += ["-hwaccel", accel]
            cmd += ["-i", str(ref)]
            if accel:
                cmd += ["-hwaccel", accel]
            cmd += ["-i", str(dist)]
            cmd += [
                "-filter_complex",
                f"[0:v]{pf}[r];[1:v]{pf}[d];"
                f"[d][r]libvmaf=model=version={model}:"
                f"n_subsample={subsample}{th}:"
                f"log_fmt=json:log_path='{log_esc}'",
                "-f", "null", "-",
            ]
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            if r.returncode == 0:
                break
        else:
            tail = "\n".join((r.stderr or "").splitlines()[-80:])
            raise RuntimeError(f"VMAF ffmpeg failed (exit {r.returncode})\n{tail}")

        with open(log, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        mean = data.get("pooled_metrics", {}).get("vmaf", {}).get("mean")
        scores = sorted(
            fr.get("metrics", {}).get("vmaf", 0)
            for fr in data.get("frames", [])
            if fr.get("metrics", {}).get("vmaf") is not None
        )
        p5 = scores[max(0, int(len(scores) * 5 / 100) - 1)] if scores else mean

        if log.exists():
            log.unlink()
        _temp_files.discard(log)
        return {
            "mean": float(mean) if mean else float("nan"),
            "p5": float(p5) if p5 else float("nan"),
        }

    except RuntimeError as e:
        print(f" {RED}VMAF error: {e}{RESET}")
        try:
            if log.exists():
                log.unlink()
        except OSError:
            pass
        _temp_files.discard(log)
        return {"mean": float("nan"), "p5": float("nan")}


def vmaf_cached(ref, dist, meta, cq, cache, cache_path, threads, tag=None):
    """Compute VMAF with file-based caching."""
    if not dist.exists() or not ref.exists():
        return {"mean": float("nan"), "p5": float("nan")}
    try:
        dist_size = dist.stat().st_size
    except OSError:
        return {"mean": float("nan"), "p5": float("nan")}

    key = f"{tag}_full" if tag else "full"
    entry = cache["entries"].get(str(cq))

    if entry and key in entry and entry.get("size") == dist_size:
        return {
            "mean": float(entry[key]),
            "p5": float(entry.get(f"{key}_p5", entry[key])),
        }

    result = measure_vmaf(ref, dist, meta, 1, threads, cache_path.parent)

    if math.isfinite(result["mean"]) and 0 <= result["mean"] <= 100:
        cache["entries"].setdefault(str(cq), {}).update({
            key: result["mean"], f"{key}_p5": result["p5"],
            "size": dist_size,
            "t": time.time(),
        })
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(cache_path)

    return result


# ── CQ Search ────────────────────────────────────────────────


def search_cq(source, meta, target, cache, cache_path,
              enc_func, threads, cfg, tag=None):
    """Find the optimal CQ that hits the target VMAF using adaptive search."""
    min_cq, max_cq = cfg["min_cq"], cfg["max_cq"]
    tol = cfg["vmaf_tolerance"]
    slope = 0.5
    enc_time = vmaf_time = 0.0
    tested = {}
    tested_paths = {}

    min_kbps = MIN_BITRATE_KBPS.get(res_tier(meta["w"], meta["h"]), 0)
    src_duration = 0.0
    if min_kbps:
        try:
            src_duration = probe_video(source)["duration"]
        except Exception:
            pass

    floor_cap = max_cq
    bitrate_points = {}
    enc_tag = f"p{cfg['preset']}g{cfg['film_grain']}"

    def eff_floor():
        # Sample path converts the video floor into a sample-bitrate threshold.
        # Full path already measures video-only kbps, so compare raw.
        if not tag:
            return min_kbps
        return effective_sample_floor(
            min_kbps, cfg["bitrate_margin"], cache.get("calibration")
        )

    def estimate_max_cq_for_floor():
        """Highest CQ whose sample bitrate predicts full video at/above floor.

        Log-linear model: log(bitrate) ≈ a - b * CQ. With two+ points uses
        measured decay; otherwise falls back to ±6 CRF ≈ 2x bitrate. The
        floor comparison runs on the calibration-adjusted sample threshold,
        so this extrapolates in both directions (down when samples already
        undershoot, up when they're above).
        """
        if not min_kbps or not bitrate_points:
            return max_cq

        cqs = sorted(bitrate_points)
        default_decay = math.log(2) / 6

        if len(cqs) >= 2:
            c1, c2 = cqs[0], cqs[-1]
            b1, b2 = bitrate_points[c1], bitrate_points[c2]
            if b1 > 0 and b2 > 0 and c1 != c2:
                measured = math.log(b1 / b2) / (c2 - c1)
                decay = measured if measured > 0 else default_decay
            else:
                decay = default_decay
        else:
            decay = default_decay

        if decay <= 0:
            return cqs[0]

        ref_cq = cqs[0]
        ref_kbps = bitrate_points[ref_cq]
        delta = math.log(ref_kbps / eff_floor()) / decay
        return int(math.floor(ref_cq + delta))

    def test(cq, measure_vmaf=True):
        nonlocal enc_time, vmaf_time, floor_cap
        cq = clamp(cq, min_cq, max_cq)
        if cq in tested:
            # Upgrade a previously-skipped VMAF measurement if now needed
            if measure_vmaf and not math.isfinite(
                tested[cq].get("mean", float("nan"))
            ) and cq in tested_paths and tested_paths[cq].exists():
                t0 = time.time()
                tested[cq] = vmaf_cached(
                    source, tested_paths[cq], meta, cq,
                    cache, cache_path, threads, tag=tag,
                )
                vmaf_time += time.time() - t0
            return cq, tested[cq]

        t0 = time.time()
        dst = enc_func(cq)
        enc_time += time.time() - t0

        if measure_vmaf:
            t0 = time.time()
            vm = vmaf_cached(
                source, dst, meta, cq, cache, cache_path, threads, tag=tag
            )
            vmaf_time += time.time() - t0
        else:
            vm = {"mean": float("nan"), "p5": float("nan")}

        tested[cq] = vm
        tested_paths[cq] = dst
        sz_mb = dst.stat().st_size / 1e6 if dst.exists() else 0
        # Samples are -an (extract_samples strips audio), so calc_kbps is
        # already video-only. Full encodes carry muxed audio/subs — use
        # video_kbps to isolate the video stream for floor comparisons.
        if src_duration > 1:
            kbps = calc_kbps(dst.stat().st_size, src_duration) if tag else video_kbps(dst, src_duration)
        else:
            kbps = None
        kbps_str = f" {kbps}kbps" if kbps else ""
        vmaf_field = (
            f"  VMAF {BOLD}{vm['mean']:.2f}{RESET}  P5 {BOLD}{vm['p5']:.2f}{RESET}"
            if math.isfinite(vm["mean"])
            else f"  {DIM}VMAF skipped{RESET}"
        )
        print(
            f" {ORANGE}{'search':<10}{RESET}CQ {BOLD}{cq}{RESET}"
            f"{vmaf_field}"
            f"  {DIM}{sz_mb:.1f}MB{kbps_str}{RESET}"
        )

        if kbps:
            bitrate_points[cq] = kbps
            if tag:
                cache["entries"].setdefault(str(cq), {})[
                    f"{tag}_kbps_{enc_tag}"
                ] = kbps
                tmp = cache_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(cache), encoding="utf-8")
                tmp.replace(cache_path)

        if min_kbps and src_duration > 1 and kbps:
            ef = eff_floor()
            if kbps <= ef:
                floor_cap = min(floor_cap, cq - 1)
                label = "sample" if tag else "video"
                print(
                    f" {ORANGE}{'bitrate':<10}{RESET}{kbps}kbps {label} at CQ {cq}"
                    f" below {min_kbps}kbps floor"
                    f" (threshold {int(ef)}kbps), capping at CQ {BOLD}{floor_cap}{RESET}"
                )

        return cq, vm

    # Seed the first CQ from source bitrate instead of a hardcoded 30.
    # High source/floor ratio has more compression headroom, so we start
    # closer to the answer. Falls back to 30 when source bitrate or floor
    # is unknown.
    src_kbps_hint = None
    if meta.get("bitrate"):
        src_kbps_hint = int(meta["bitrate"] / 1000)
    seed_cq = initial_cq_seed(src_kbps_hint, min_kbps, min_cq, max_cq)
    if seed_cq != 30:
        print(
            f" {ORANGE}{'seed':<10}{RESET}CQ {BOLD}{seed_cq}{RESET}"
            f" {DIM}(source {src_kbps_hint or '?'}kbps vs floor {min_kbps or '-'}kbps){RESET}"
        )

    cq, vm = test(seed_cq)
    if not math.isfinite(vm["mean"]):
        return None, None, enc_time, vmaf_time

    # Floor-bound detection: VMAF comfortably above target AND bitrate well
    # below floor. In this regime VMAF is not binding — it's a pure bitrate
    # targeting problem. Skip VMAF on intermediate probes; verify once on
    # the final candidate. Monotonicity (lower CQ → higher VMAF) keeps this
    # safe as long as the seed test already cleared target.
    floor_bound = bool(
        min_kbps and vm["mean"] > target + 2.0
        and cq in bitrate_points
        and bitrate_points[cq] < eff_floor() * 0.80
    )
    if floor_bound:
        print(
            f" {ORANGE}{'mode':<10}{RESET}floor-bound "
            f"{DIM}(skipping VMAF on intermediate probes){RESET}"
        )

    # Proactive bitrate jump: go straight to the extrapolated floor CQ.
    if (min_kbps and (floor_bound or vm["mean"] >= target - tol)
            and cq in bitrate_points
            and bitrate_points[cq] < eff_floor() * 1.10
            and cq - 1 >= min_cq):
        floor_cq = clamp(estimate_max_cq_for_floor(), min_cq, cq - 1)
        if floor_cq != cq and floor_cq not in tested:
            prev_cq, prev_vm = cq, vm
            cq, vm = test(floor_cq, measure_vmaf=not floor_bound)
            if (math.isfinite(vm["mean"]) and math.isfinite(prev_vm["mean"])
                    and prev_cq != cq):
                slope = clamp(
                    abs(prev_vm["mean"] - vm["mean"]) / abs(prev_cq - cq),
                    0.1, 1.5,
                )

    if floor_bound:
        # Bitrate-only convergence: keep picking the estimated floor CQ
        # until we bracket it; accept when at the ceiling with floor met.
        for _ in range(4):
            effective_max = min(max_cq, floor_cap, estimate_max_cq_for_floor())
            current_kbps = bitrate_points.get(cq, 0)
            cq_violates_floor = (
                cq in bitrate_points and bitrate_points[cq] < eff_floor()
            )
            if (cq >= effective_max and current_kbps >= min_kbps
                    and not cq_violates_floor):
                print(
                    f" {ORANGE}{'accept':<10}{RESET}bitrate floor met at"
                    f" CQ {BOLD}{cq}{RESET}"
                )
                break

            next_cq = clamp(estimate_max_cq_for_floor(), min_cq, effective_max)
            if next_cq == cq and current_kbps < min_kbps and cq - 1 >= min_cq:
                next_cq = cq - 1
            if next_cq == cq or next_cq in tested:
                break

            cq, vm = test(next_cq, measure_vmaf=False)
            if cq not in bitrate_points:
                break
    else:
        for _ in range(4):
            if target - tol <= vm["mean"] <= target + 1.0:
                break
            delta = (vm["mean"] - target) / slope
            effective_max = min(max_cq, floor_cap, estimate_max_cq_for_floor())

            # At the CQ ceiling (can't go higher without violating floor),
            # accept any overshoot as long as VMAF meets target and this
            # CQ's bitrate isn't already below the predicted floor.
            cq_violates_floor = (
                min_kbps and cq in bitrate_points
                and bitrate_points[cq] < eff_floor()
            )
            if (vm["mean"] >= target - tol
                    and cq >= effective_max and not cq_violates_floor):
                print(
                    f" {ORANGE}{'accept':<10}{RESET}VMAF passes and CQ"
                    f" {BOLD}{cq}{RESET} is at bitrate ceiling"
                )
                break

            next_cq = clamp(int(round(cq + delta)), min_cq, effective_max)
            if next_cq == cq:
                next_cq = cq + (1 if vm["mean"] > target else -1)
            next_cq = clamp(next_cq, min_cq, effective_max)
            if next_cq == cq or next_cq in tested:
                break

            prev_cq, prev_vm = cq, vm
            cq, vm = test(next_cq)
            if not math.isfinite(vm["mean"]):
                break
            if prev_cq != cq:
                slope = clamp(
                    abs(prev_vm["mean"] - vm["mean"]) / abs(prev_cq - cq), 0.1, 1.5
                )

    def valid_cq(c):
        vm_c = tested[c]
        if math.isfinite(vm_c["mean"]) and vm_c["mean"] < target - tol:
            return False
        if min_kbps and src_duration > 1 and c in tested_paths:
            path = tested_paths[c]
            kbps = calc_kbps(path.stat().st_size, src_duration) if tag else video_kbps(path, src_duration)
            if kbps and kbps < eff_floor():
                return False
        return True

    best = max((c for c in tested if valid_cq(c)), default=None)
    if best is None:
        best = max(
            (c for c in tested if math.isfinite(tested[c]["mean"])),
            key=lambda c: tested[c]["mean"], default=None,
        )

    # Guarantee a VMAF measurement on the returned candidate (floor-bound
    # path may have skipped it). Monotonicity makes a failure here very
    # unlikely — floor-bound only triggers when the seed already cleared
    # target by 2, and every subsequent probe is at lower CQ (higher VMAF).
    if (best is not None and not math.isfinite(tested[best]["mean"])
            and best in tested_paths and tested_paths[best].exists()):
        t0 = time.time()
        tested[best] = vmaf_cached(
            source, tested_paths[best], meta, best,
            cache, cache_path, threads, tag=tag,
        )
        vmaf_time += time.time() - t0

    return best, tested.get(best), enc_time, vmaf_time


# ── Cache ────────────────────────────────────────────────────


def load_cache(cache_dir, file_hash, sig):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{file_hash}.json"
    if cp.exists():
        with open(cp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("sig") == sig:
            return data, cp
    return {"sig": sig, "entries": {}}, cp


# ── Encoding ─────────────────────────────────────────────────


def encode_av1(source, dest, meta, cq, cfg):
    """Encode video to AV1 using SVT-AV1 via ffmpeg."""
    pix = (
        "yuv420p10le"
        if meta["hdr"] or "10le" in meta["pix_fmt"] or cfg["force_10bit"]
        else "yuv420p"
    )

    tmp = dest.with_suffix(".tmp.mkv")
    _temp_files.add(tmp)
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass

    color_args = []
    if meta["cp"] and meta["ct"]:
        color_args += ["-color_primaries", meta["cp"], "-color_trc", meta["ct"]]
        if meta["cs"]:
            color_args += ["-colorspace", meta["cs"]]
    if meta["cr"]:
        color_args += ["-color_range", meta["cr"]]

    bitrate = meta.get("bitrate") or FALLBACK_MAXRATE[res_tier(meta["w"], meta["h"])]
    maxrate = min(int(bitrate * cfg["maxrate_factor"]), 100_000_000)
    crf = clamp(cq, 0, 63)

    threads = os.cpu_count() or 1
    tiles = str(max(0, min(6, int(math.floor(math.log2(threads))) - 1)))
    fg = cfg["film_grain"]
    svt_params = f"tune=0:film-grain={fg}:film-grain-denoise=0:enable-tf=0:enable-overlays=1:scd=1"

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-v", "error", "-nostats",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a?", "-map", "0:s?",
        "-c:a", "copy", "-c:s", "copy",
        "-pix_fmt", pix,
        "-c:v", "libsvtav1",
        "-preset", str(cfg["preset"]),
        "-crf", str(crf),
        "-tiles", tiles,
        "-svtav1-params", svt_params,
        "-g", str(cfg["gop"]),
        "-threads", str(threads),
        "-maxrate", str(maxrate),
        "-bufsize", str(maxrate * 2),
        "-fps_mode", "passthrough",
        *color_args, str(tmp),
    ]

    run_cmd(cmd)

    if dest.exists():
        dest.unlink()
    tmp.rename(dest)
    _temp_files.discard(tmp)


# ── Main Processing ──────────────────────────────────────────


def process_videos(cfg):
    input_dir = cfg["input_dir"]
    output_dir = cfg["output_dir"]
    cache_dir = cfg["cache_dir"]
    ext = cfg["container"]
    vmaf_threads = os.cpu_count() or 4

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        print(f"{CROSS} ffmpeg/ffprobe not found in PATH")
        return 1

    LBL = 10  # label column width for aligned output
    def lbl(tag):
        return f" {ORANGE}{tag:<{LBL}}{RESET}"

    print(f"{PURPLE}{BOLD}av1q{RESET}\n{SEP}")
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p in output_dir.rglob("*.tmp.mkv"):
        try:
            p.unlink()
        except OSError:
            pass

    pattern = "**/*" if cfg["recurse"] else "*"
    files = [
        f for f in input_dir.glob(pattern)
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    total = len(files)

    stats = {
        "proc": 0, "vmaf_sum": 0.0, "vmaf_n": 0,
        "saved": 0, "orig": 0, "deleted": 0,
    }
    t_start = time.time()

    for idx, filepath in enumerate(files, 1):
        sample_src = None
        _file_error = False
        try:
            rel = filepath.parent.relative_to(input_dir)
            out_dir = output_dir / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            dst_path = lambda cq: out_dir / f"{filepath.stem}_CQ{cq}{ext}"

            file_hash = partial_hash(filepath)
            cache, cp = load_cache(cache_dir, file_hash, "svt4")

            if cfg["skip_existing"]:
                verified = False
                for c in range(cfg["min_cq"], cfg["max_cq"] + 1):
                    d = dst_path(c)
                    if d.exists():
                        entry = cache["entries"].get(str(c))
                        if (entry and "full" in entry
                                and entry.get("size") == d.stat().st_size):
                            verified = True
                            break
                if verified:
                    print(f" {PURPLE}{filepath.name:<30}{RESET} {CHECK} exists")
                    continue

            if idx > 1:
                print(SEP)
            print(f"{PURPLE}{BOLD}[{idx}/{total}]{RESET} {PURPLE}{filepath.name}{RESET}")

            meta = probe_video(filepath)

            # File info line
            in_sz = filepath.stat().st_size
            res_str = f"{meta['w']}x{meta['h']}" if meta["w"] and meta["h"] else "?"
            codec_str = (meta["codec"] or "?").upper()
            sz_str = f"{in_sz / 1e9:.2f}GB" if in_sz >= 1e9 else f"{in_sz / 1e6:.1f}MB"
            src_kbps = f"{meta['bitrate'] // 1000}kbps" if meta.get("bitrate") else ""
            dur_str = fmt_time(meta["duration"]) if meta["duration"] > 0 else ""
            hdr_str = "HDR" if meta["hdr"] else ""
            info_parts = [p for p in [res_str, codec_str, sz_str, src_kbps, dur_str, hdr_str] if p]
            print(f"      {DIM}{' \u00b7 '.join(info_parts)}{RESET}")

            if meta["codec"] == "av1":
                print(f" {CHECK} Already AV1, skipping")
                continue

            tier = max(k for k in TARGET_VMAF_BY_RES if min(meta["w"], meta["h"]) >= k)
            target = cfg.get("target_vmaf") or TARGET_VMAF_BY_RES[tier]
            min_p5 = target - cfg["vmaf_p5_margin"]

            existing_cq = None
            for c in range(cfg["max_cq"], cfg["min_cq"] - 1, -1):
                if dst_path(c).exists():
                    existing_cq = c
                    break

            if existing_cq is None:
                rec = cache.get("recommended")
                if (rec and rec.get("target") == target
                        and rec.get("min_cq") == cfg["min_cq"]
                        and rec.get("max_cq") == cfg["max_cq"]
                        and rec.get("preset") == cfg["preset"]
                        and rec.get("film_grain") == cfg["film_grain"]):
                    existing_cq = rec["cq"]
                    print(f"{lbl('resume')}CQ {BOLD}{existing_cq}{RESET} from previous search")

            sample_scenes = sample_src = None

            if existing_cq is None and meta["duration"] >= cfg["short_threshold"]:
                if meta["codec"] in INTRA_ONLY_CODECS:
                    print(f"{lbl('skip')}Intra-only codec ({meta['codec']}), using even samples")
                    scenes = []
                    complexity = []
                    keyframes = []
                else:
                    scene_cfg = {"scene_threshold": cfg["scene_threshold"]}
                    if (all(k in cache for k in ("scenes", "complexity", "keyframes"))
                            and cache.get("scene_cfg") == scene_cfg):
                        print(f"{lbl('cache')}Using cached scene data")
                        scenes = cache["scenes"]
                        complexity = cache["complexity"]
                        keyframes = cache["keyframes"]
                    else:
                        print(f"{lbl('analyze')}Detecting scenes...")
                        scenes = detect_scenes(filepath, cfg)
                        complexity = analyze_complexity(filepath)
                        keyframes = get_keyframes(filepath)
                        cache.update(scenes=scenes, complexity=complexity,
                                     keyframes=keyframes, scene_cfg=scene_cfg)
                        tmp = cp.with_suffix(".json.tmp")
                        tmp.write_text(json.dumps(cache), encoding="utf-8")
                        tmp.replace(cp)

                sample_scenes = select_samples(
                    scenes, complexity, meta["duration"],
                    cfg["sample_count"], keyframes, cfg,
                )
                if sample_scenes:
                    info = (
                        f"samples from {BOLD}{len(scenes)}{RESET} scenes"
                        if scenes else "evenly-spaced samples"
                    )
                    print(f"{lbl('scenes')}{BOLD}{len(sample_scenes)}{RESET} {info}")
                    print(f"{lbl('extract')}Extracting samples...")
                    sample_src = extract_samples(filepath, sample_scenes, keyframes, cfg, file_hash=file_hash)
                    if not sample_src:
                        print(f"{lbl('fallback')}Extraction failed, using full encode")
                        sample_scenes = None
                else:
                    print(f"{lbl('scenes')}Using full VMAF")
            elif existing_cq is None:
                print(f"{lbl('short')}<{cfg['short_threshold']}s, full VMAF")

            t_enc = t_vmaf = 0.0
            sample_enc_dir = cache_dir / "_sample_enc"
            sample_enc_dir.mkdir(parents=True, exist_ok=True)
            sample_enc_cache = {}
            enc_tag = f"p{cfg['preset']}g{cfg['film_grain']}"

            def do_enc_sample(cq):
                nonlocal t_enc
                cq = clamp(cq, cfg["min_cq"], cfg["max_cq"])
                if cq in sample_enc_cache:
                    return sample_enc_cache[cq]
                if not sample_src or not sample_src.exists():
                    raise RuntimeError("Sample source missing")
                d = sample_enc_dir / f"sample_enc_{file_hash[:8]}_{enc_tag}_{cq}.mkv"
                if d.exists() and d.stat().st_size > 0:
                    sample_enc_cache[cq] = d
                    return d
                t0 = time.time()
                encode_av1(sample_src, d, meta, cq, cfg)
                t_enc += time.time() - t0
                if not d.exists():
                    raise RuntimeError("Encoding failed")
                sample_enc_cache[cq] = d
                return d

            def do_enc_full(cq):
                nonlocal t_enc
                cq = clamp(cq, cfg["min_cq"], cfg["max_cq"])
                d = dst_path(cq)
                if not d.exists():
                    t0 = time.time()
                    encode_av1(filepath, d, meta, cq, cfg)
                    t_enc += time.time() - t0
                return d

            floor_kbps = MIN_BITRATE_KBPS.get(res_tier(meta["w"], meta["h"]), 0)
            floor_str = f" {DIM}\u00b7{RESET} floor {BOLD}{floor_kbps}kbps{RESET}" if floor_kbps else ""
            print(
                f"{lbl('target')}VMAF {BOLD}{target:.1f}{RESET}"
                f" (P5 >= {BOLD}{min_p5:.1f}{RESET}){floor_str}"
            )

            if existing_cq is not None:
                best_cq = existing_cq
                print(f"{lbl('reuse')}Found existing CQ {BOLD}{existing_cq}{RESET} encode")
            elif sample_src:
                best_cq, _, _, vt = search_cq(
                    sample_src, meta, target, cache, cp,
                    do_enc_sample, vmaf_threads, cfg, tag="sample",
                )
                t_vmaf += vt
                if best_cq is not None:
                    cache["recommended"] = {
                        "cq": best_cq, "target": target,
                        "min_cq": cfg["min_cq"], "max_cq": cfg["max_cq"],
                        "preset": cfg["preset"], "film_grain": cfg["film_grain"],
                    }
                    tmp = cp.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(cache), encoding="utf-8")
                    tmp.replace(cp)
                for p in sample_enc_cache.values():
                    try:
                        p.unlink()
                    except OSError:
                        pass
            else:
                best_cq, best_vmaf, _, vt = search_cq(
                    filepath, meta, target, cache, cp,
                    do_enc_full, vmaf_threads, cfg,
                )
                t_vmaf += vt

            if best_cq is None:
                print(f" {CROSS} No valid CQ found")
                continue

            if cfg["dry_run"]:
                entry = cache["entries"].get(str(best_cq), {})
                vmaf_str = ""
                sv = entry.get("sample_full") or entry.get("full")
                if isinstance(sv, dict):
                    vmaf_str = f" VMAF {BOLD}{sv['mean']:.2f}{RESET}  P5 {BOLD}{sv['p5']:.2f}{RESET}"
                elif isinstance(sv, (int, float)):
                    vmaf_str = f" VMAF {BOLD}{sv:.2f}{RESET}"
                print(f" {CHECK} Recommended CQ {BOLD}{best_cq}{RESET}{vmaf_str}")
                print(f"   Run without --dry-run to encode")
                continue

            if sample_src or existing_cq is not None:
                if not dst_path(best_cq).exists():
                    print(f"{lbl('encode')}CQ {BOLD}{best_cq}{RESET} final encode...")
                    t0 = time.time()
                    encode_av1(filepath, dst_path(best_cq), meta, best_cq, cfg)
                    t_enc += time.time() - t0
                else:
                    print(f"{lbl('reuse')}CQ {BOLD}{best_cq}{RESET} encode exists")

                print(f"{lbl('verify')}Full VMAF...")
                t0 = time.time()
                best_vmaf = vmaf_cached(
                    filepath, dst_path(best_cq), meta, best_cq,
                    cache, cp, vmaf_threads,
                )
                t_vmaf += time.time() - t0
                print(
                    f"{'':>{LBL + 1}}VMAF {BOLD}{best_vmaf['mean']:.2f}{RESET}"
                    f"  P5 {BOLD}{best_vmaf['p5']:.2f}{RESET}"
                )

                for _ in range(3):
                    if best_vmaf["mean"] >= target - cfg["vmaf_tolerance"]:
                        break
                    if best_cq <= cfg["min_cq"]:
                        break
                    try_cq = best_cq - 1
                    print(f"{lbl('adjust')}Trying CQ {BOLD}{try_cq}{RESET}")
                    if not dst_path(try_cq).exists():
                        t0 = time.time()
                        encode_av1(filepath, dst_path(try_cq), meta, try_cq, cfg)
                        t_enc += time.time() - t0
                    t0 = time.time()
                    adj = vmaf_cached(
                        filepath, dst_path(try_cq), meta, try_cq,
                        cache, cp, vmaf_threads,
                    )
                    t_vmaf += time.time() - t0
                    if math.isfinite(adj["mean"]) and adj["mean"] > best_vmaf["mean"]:
                        best_cq, best_vmaf = try_cq, adj
                    else:
                        break

            for _ in range(3):
                if best_vmaf["p5"] >= min_p5 or best_cq <= cfg["min_cq"]:
                    break
                try_cq = best_cq - 1
                print(f"{lbl('safety')}P5 low, trying CQ {BOLD}{try_cq}{RESET}")
                do_enc_full(try_cq)
                t0 = time.time()
                safe = vmaf_cached(
                    filepath, dst_path(try_cq), meta, try_cq,
                    cache, cp, vmaf_threads,
                )
                t_vmaf += time.time() - t0
                if math.isfinite(safe["p5"]) and safe["p5"] > best_vmaf["p5"]:
                    best_cq, best_vmaf = try_cq, safe
                else:
                    break

            min_kbps = MIN_BITRATE_KBPS.get(res_tier(meta["w"], meta["h"]), 0)
            if min_kbps and dst_path(best_cq).exists():
                actual_kbps = video_kbps(dst_path(best_cq), meta["duration"])

                enc_tag_cfg = f"p{cfg['preset']}g{cfg['film_grain']}"
                sample_kbps_key = f"sample_kbps_{enc_tag_cfg}"
                sample_pts = {
                    int(k): v[sample_kbps_key]
                    for k, v in cache.get("entries", {}).items()
                    if isinstance(v, dict) and sample_kbps_key in v
                }
                sample_at_best = sample_pts.get(best_cq)
                if sample_at_best and actual_kbps and sample_at_best > 0:
                    ratio = actual_kbps / sample_at_best
                    if 0.5 <= ratio <= 1.0:
                        cache["calibration"] = {
                            "ratio": ratio, "at_cq": best_cq,
                            "enc_tag": enc_tag_cfg, "t": time.time(),
                        }
                        tmp = cp.with_suffix(".json.tmp")
                        tmp.write_text(json.dumps(cache), encoding="utf-8")
                        tmp.replace(cp)
                        print(
                            f"{lbl('calibr')}sample {sample_at_best}kbps ->"
                            f" video {actual_kbps}kbps (ratio {ratio:.2f})"
                        )

                default_decay = math.log(2) / 6
                for _ in range(5):
                    if not actual_kbps or actual_kbps >= min_kbps:
                        break
                    if best_cq <= cfg["min_cq"]:
                        break

                    decay = default_decay
                    if len(sample_pts) >= 2:
                        cqs_s = sorted(sample_pts)
                        c1, c2 = cqs_s[0], cqs_s[-1]
                        b1, b2 = sample_pts[c1], sample_pts[c2]
                        if b1 > 0 and b2 > 0 and c1 != c2:
                            m = math.log(b1 / b2) / (c2 - c1)
                            if m > 0:
                                decay = m
                    needed = math.log(min_kbps / actual_kbps) / decay
                    step = max(1, int(math.ceil(needed)))
                    try_cq = max(cfg["min_cq"], best_cq - step)

                    print(
                        f"{lbl('bitrate')}video {actual_kbps}kbps"
                        f" < {min_kbps}kbps floor,"
                        f" trying CQ {BOLD}{try_cq}{RESET}"
                    )
                    do_enc_full(try_cq)
                    t0 = time.time()
                    adj = vmaf_cached(
                        filepath, dst_path(try_cq), meta, try_cq,
                        cache, cp, vmaf_threads,
                    )
                    t_vmaf += time.time() - t0
                    best_cq, best_vmaf = try_cq, adj
                    actual_kbps = video_kbps(dst_path(best_cq), meta["duration"])

            final = dst_path(best_cq)
            if not final.exists():
                print(f" {CROSS} Final encode missing")
                continue

            for c in range(cfg["min_cq"], cfg["max_cq"] + 1):
                if c != best_cq and dst_path(c).exists():
                    try:
                        dst_path(c).unlink()
                    except OSError:
                        pass

            out_sz = final.stat().st_size

            if out_sz >= in_sz:
                final.unlink()
                stats["deleted"] += 1
                print(
                    f" {CROSS} Larger ({BOLD}{out_sz / 1e6:.1f}MB{RESET}"
                    f" vs {BOLD}{in_sz / 1e6:.1f}MB{RESET}) - deleted"
                )
                continue

            saved = (1.0 - out_sz / in_sz) * 100
            out_kbps = calc_kbps(final.stat().st_size, meta["duration"])
            kbps_str = f" ({BOLD}{out_kbps}kbps{RESET})" if out_kbps else ""
            in_str = f"{in_sz / 1e9:.2f}GB" if in_sz >= 1e9 else f"{in_sz / 1e6:.1f}MB"
            out_str = f"{out_sz / 1e9:.2f}GB" if out_sz >= 1e9 else f"{out_sz / 1e6:.1f}MB"
            print(SEP)
            print(
                f" {CHECK} CQ {BOLD}{best_cq}{RESET}"
                f"  VMAF {BOLD}{best_vmaf['mean']:.2f}{RESET}"
                f"  P5 {BOLD}{best_vmaf['p5']:.2f}{RESET}"
            )
            print(
                f" {CHECK} {in_str} -> {BOLD}{out_str}{RESET}{kbps_str}"
                f" saved {BOLD}{saved:.1f}%{RESET}"
            )
            print(f"   {DIM}Enc {fmt_time(t_enc)} \u00b7 VMAF {fmt_time(t_vmaf)}{RESET}")

            stats["proc"] += 1
            if math.isfinite(best_vmaf["mean"]):
                stats["vmaf_sum"] += best_vmaf["mean"]
                stats["vmaf_n"] += 1
            stats["saved"] += in_sz - out_sz
            stats["orig"] += in_sz

        except KeyboardInterrupt:
            _file_error = True
            raise
        except Exception as e:
            _file_error = True
            print(f" {CROSS} {e}")
        finally:
            cleanup_temp()
            if not _file_error and sample_src:
                try:
                    if sample_src.exists():
                        sample_src.unlink()
                except OSError:
                    pass

    if total > 0:
        print(SEP)
    if stats["proc"] > 0:
        avg = stats["vmaf_sum"] / stats["vmaf_n"] if stats["vmaf_n"] else 0
        pct = stats["saved"] / stats["orig"] * 100 if stats["orig"] else 0
        print(f"{CHECK} Processed: {BOLD}{stats['proc']}{RESET}")
        print(f"{CHECK} Avg VMAF: {BOLD}{avg:.2f}{RESET}")
        print(
            f"{CHECK} Saved: {BOLD}{stats['saved'] / 1e9:.2f}GB{RESET}"
            f" ({BOLD}{pct:.1f}%{RESET})"
        )
        if stats["deleted"] > 0:
            print(f"{ORANGE} Deleted: {BOLD}{stats['deleted']}{RESET}")
        print(f"{CHECK} Time: {BOLD}{fmt_time(time.time() - t_start)}{RESET}")
    else:
        print(f"{CHECK} No files processed")

    print(f"{SEP}\n{CHECK} Done")
    return 0


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

    args = parser.parse_args()

    if args.min_cq > args.max_cq:
        parser.error("--min-cq must be <= --max-cq")
    if not 0 <= args.preset <= 10:
        parser.error("--preset must be 0-10 (SVT-AV1 v3 removed presets above 10)")

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
        "gop": 250,
        "film_grain": args.film_grain,
        "force_10bit": not args.no_10bit,
        "maxrate_factor": 1.6,
        "target_vmaf": args.vmaf,
        "vmaf_p5_margin": 5.0,
        "vmaf_tolerance": 0.1,
        "bitrate_margin": 1.20,
        "dry_run": args.dry_run,
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
