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

TARGET_VMAF_BY_RES = {0: 93.0, 720: 94.0, 2160: 90.0}

FALLBACK_MAXRATE = {
    0: 8_000_000, 720: 12_000_000, 1080: 25_000_000,
    1440: 35_000_000, 2160: 45_000_000, 4320: 60_000_000,
}

VMAF_SUBSAMPLE = {0: 15, 720: 20, 1080: 30, 1440: 40, 2160: 60, 4320: 80}

MIN_BITRATE_KBPS = {0: 0, 720: 1000, 1080: 1500, 1440: 2500, 2160: 4000, 4320: 8000}

# ── Global State ─────────────────────────────────────────────

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


def res_tier(height):
    for t in (4320, 2160, 1440, 1080, 720):
        if height >= t:
            return t
    return 0


def calc_kbps(size_bytes, duration):
    if duration < 1.0:
        return None
    return int((size_bytes * 8) / 1000 / duration)


def video_kbps(filepath, duration):
    """Get video-only bitrate in kbps via ffprobe, excluding audio/subs."""
    if duration < 1.0:
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=bit_rate",
             "-of", "default=nw=1:nk=1", str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
        )
        val = r.stdout.strip()
        if val and val not in ("N/A", "0"):
            return int(int(val) / 1000)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    # Fallback: total file bitrate (close enough if probe fails)
    return calc_kbps(filepath.stat().st_size, duration)


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

    bitrate = next(
        (int(v) for v in (fmt.get("bit_rate"), s.get("bit_rate")) if v), None
    )
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
    """Get average frame rate as a rational string (e.g., '24000/1001').

    Returns the raw fraction from ffprobe to avoid float truncation
    that can cause frame misalignment in VMAF comparisons.
    """
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
                "-i", str(source),
                "-vf", f"scdet=t={cfg['scene_threshold']},"
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

    # Deterministic name so samples survive interrupts and can be reused
    tag = file_hash or f"{os.getpid()}_{int(time.time() * 1000)}"
    concat_out = sample_dir / f"samples_{tag}.mkv"
    if concat_out.exists() and concat_out.stat().st_size > 0:
        print(f" {DIM}Samples: {concat_out.stat().st_size / 1e6:.1f}MB (cached){RESET}")
        return concat_out

    ts = int(time.time() * 1000)
    clips = []
    keyframes = keyframes or get_keyframes(source)

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

    # concat_out defined above with deterministic name; not added to
    # _temp_files so it survives interrupts for reuse on next run
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
            print(f" {DIM}Samples: {concat_out.stat().st_size / 1e6:.1f}MB{RESET}")
            return concat_out
    except RuntimeError as e:
        print(f" {RED}Concat error: {e}{RESET}")
    return None


# ── VMAF Measurement ─────────────────────────────────────────


def measure_vmaf(ref, dist, meta, scenes, subsample, threads, cache_dir):
    """Compute VMAF score between reference and distorted video."""
    filters = []
    fps = get_fps(dist)

    if scenes:
        # Scene-based: fps normalizes rate (using original timestamps),
        # select picks scenes, setpts resets PTS for contiguous output
        if fps:
            filters.append(f"fps={fps}")
        expr = "+".join(
            f"between(t,{s['time']:.3f},{s['time'] + s['duration']:.3f})"
            for s in scenes
        )
        filters.append(f"select='{expr}',setpts=PTS-STARTPTS")
    else:
        # Full-file: normalize start PTS to zero first (handles MP4 edit
        # lists / non-zero start times), then normalize frame rate
        filters.append("setpts=PTS-STARTPTS")
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
        run_cmd([
            "ffmpeg", "-v", "error", "-hide_banner",
            "-i", str(ref), "-i", str(dist),
            "-filter_complex",
            f"[0:v]{pf}[r];[1:v]{pf}[d];"
            f"[d][r]libvmaf=model=version={model}:"
            f"n_subsample={subsample}{th}:"
            f"log_fmt=json:log_path='{log_esc}'",
            "-f", "null", "-",
        ])

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


def vmaf_cached(ref, dist, meta, cq, scenes, cache, cache_path, threads, tag=None):
    """Compute VMAF with file-based caching."""
    if not dist.exists() or not ref.exists():
        return {"mean": float("nan"), "p5": float("nan")}
    try:
        dist_size = dist.stat().st_size
    except OSError:
        return {"mean": float("nan"), "p5": float("nan")}

    base_key = "full" if scenes is None else f"S{len(scenes)}"
    key = f"{tag}_{base_key}" if tag else base_key
    entry = cache["entries"].get(str(cq))

    if entry and key in entry and entry.get("size") == dist_size:
        return {
            "mean": float(entry[key]),
            "p5": float(entry.get(f"{key}_p5", entry[key])),
        }

    tier = res_tier(meta["h"])
    sub = 1 if scenes is None else (VMAF_SUBSAMPLE.get(tier) or 30)
    result = measure_vmaf(ref, dist, meta, scenes, sub, threads, cache_path.parent)

    if math.isfinite(result["mean"]) and 0 <= result["mean"] <= 100:
        cache["entries"].setdefault(str(cq), {}).update({
            key: result["mean"], f"{key}_p5": result["p5"],
            "size": dist_size,
            "kbps": calc_kbps(dist_size, meta["duration"]),
            "t": time.time(),
        })
        tmp = cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache), encoding="utf-8")
        tmp.replace(cache_path)

    return result


# ── CQ Search ────────────────────────────────────────────────


def search_cq(source, meta, scenes, target, cache, cache_path,
              enc_func, threads, cfg, tag=None):
    """Find the optimal CQ that hits the target VMAF using adaptive search."""
    min_cq, max_cq = cfg["min_cq"], cfg["max_cq"]
    tol = cfg["vmaf_tolerance"]
    slope = 0.5
    enc_time = vmaf_time = 0.0
    tested = {}
    tested_paths = {}

    # Get bitrate floor and actual source duration for bitrate checks
    min_kbps = MIN_BITRATE_KBPS.get(res_tier(meta["h"]), 0)
    src_duration = 0.0
    if min_kbps:
        try:
            src_duration = probe_video(source)["duration"]
        except Exception:
            pass

    def test(cq):
        nonlocal enc_time, vmaf_time
        cq = clamp(cq, min_cq, max_cq)
        if cq in tested:
            return cq, tested[cq]

        t0 = time.time()
        dst = enc_func(cq)
        enc_time += time.time() - t0

        t0 = time.time()
        vm = vmaf_cached(
            source, dst, meta, cq, scenes, cache, cache_path, threads, tag=tag
        )
        vmaf_time += time.time() - t0

        tested[cq] = vm
        tested_paths[cq] = dst
        sz_mb = dst.stat().st_size / 1e6 if dst.exists() else 0
        kbps = video_kbps(dst, src_duration) if src_duration > 1 else None
        kbps_str = f" {kbps}kbps" if kbps else ""
        print(
            f" {ORANGE}search{RESET} CQ={BOLD}{cq}{RESET}"
            f" VMAF={BOLD}{vm['mean']:.2f}{RESET}"
            f" P5={BOLD}{vm['p5']:.2f}{RESET}"
            f" {DIM}{sz_mb:.1f}MB{kbps_str}{RESET}"
        )
        return cq, vm

    cq, vm = test(28)
    if not math.isfinite(vm["mean"]):
        return None, None, enc_time, vmaf_time

    for _ in range(4):
        if target - tol <= vm["mean"] <= target + 1.0:
            break
        delta = (vm["mean"] - target) / slope
        next_cq = clamp(int(round(cq + delta)), min_cq, max_cq)
        if next_cq == cq:
            next_cq = cq + (1 if vm["mean"] > target else -1)
        next_cq = clamp(next_cq, min_cq, max_cq)
        if next_cq in tested:
            break

        # If going higher CQ but bitrate is already at/below floor, search down instead
        if min_kbps and next_cq > cq and src_duration > 1 and cq in tested_paths:
            curr_kbps = video_kbps(tested_paths[cq], src_duration)
            if curr_kbps and curr_kbps <= min_kbps:
                next_cq = cq - 1
                if next_cq < min_cq or next_cq in tested:
                    break
                print(
                    f" {ORANGE}bitrate{RESET} {curr_kbps}kbps at CQ={cq}"
                    f" below floor ({min_kbps}kbps), trying CQ={BOLD}{next_cq}{RESET}"
                )

        prev_cq, prev_vm = cq, vm
        cq, vm = test(next_cq)
        if not math.isfinite(vm["mean"]):
            break
        if prev_cq != cq:
            slope = clamp(
                abs(prev_vm["mean"] - vm["mean"]) / abs(prev_cq - cq), 0.1, 1.5
            )

    # Pick highest CQ (smallest file) that meets target AND bitrate floor
    def valid_cq(c):
        if not math.isfinite(tested[c]["mean"]) or tested[c]["mean"] < target - tol:
            return False
        if min_kbps and src_duration > 1 and c in tested_paths:
            kbps = video_kbps(tested_paths[c], src_duration)
            if kbps and kbps < min_kbps:
                return False
        return True

    best = max((c for c in tested if valid_cq(c)), default=None)
    if best is None:
        best = max(
            (c for c in tested if math.isfinite(tested[c]["mean"])),
            key=lambda c: tested[c]["mean"], default=None,
        )
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

    # Color metadata
    color_args = []
    if meta["cp"] and meta["ct"]:
        color_args += ["-color_primaries", meta["cp"], "-color_trc", meta["ct"]]
        if meta["cs"]:
            color_args += ["-colorspace", meta["cs"]]
    if meta["cr"]:
        color_args += ["-color_range", meta["cr"]]

    # Rate control — map user-facing CQ (min_cq..max_cq) to SVT-AV1 CRF (18..38)
    bitrate = meta.get("bitrate") or FALLBACK_MAXRATE[res_tier(meta["h"])]
    maxrate = min(int(bitrate * cfg["maxrate_factor"]), 100_000_000)
    cq_range = cfg["max_cq"] - cfg["min_cq"]
    crf = clamp(
        18 + int(round((cq - cfg["min_cq"]) * 20 / cq_range)) if cq_range > 0 else cq,
        0, 63,
    )

    # Threading
    threads = os.cpu_count() or 1
    tiles = str(max(0, min(6, int(math.floor(math.log2(threads))) - 1)))
    fg = cfg["film_grain"]
    svt_params = f"tune=0:film-grain={fg}:film-grain-denoise=0:enable-tf=0"

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
            # Output path mirrors input subdirectory structure
            rel = filepath.parent.relative_to(input_dir)
            out_dir = output_dir / rel
            out_dir.mkdir(parents=True, exist_ok=True)
            dst_path = lambda cq: out_dir / f"{filepath.stem}_CQ{cq}{ext}"

            file_hash = partial_hash(filepath)
            cache, cp = load_cache(cache_dir, file_hash, "svt4")

            # Skip only if output exists AND has verified full VMAF in cache
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
                    print(f"{PURPLE}{filepath.name:<30}{RESET} {CHECK} exists")
                    continue

            if idx > 1:
                print(SEP)
            print(f"{PURPLE}{BOLD}[{idx}/{total}]{RESET} {PURPLE}{filepath.name}{RESET}")

            meta = probe_video(filepath)
            if meta["codec"] == "av1":
                print(f" {CHECK} Already AV1, skipping")
                continue

            tier = max(k for k in TARGET_VMAF_BY_RES if meta["h"] >= k)
            target = cfg.get("target_vmaf") or TARGET_VMAF_BY_RES[tier]
            min_p5 = target - cfg["vmaf_p5_margin"]

            # ── Check for existing output needing verification ──
            existing_cq = None
            for c in range(cfg["min_cq"], cfg["max_cq"] + 1):
                if dst_path(c).exists():
                    existing_cq = c
                    break

            # Resume from cached search result (interrupted after sample search)
            if existing_cq is None:
                rec = cache.get("recommended")
                if (rec and rec.get("target") == target
                        and rec.get("min_cq") == cfg["min_cq"]
                        and rec.get("max_cq") == cfg["max_cq"]):
                    existing_cq = rec["cq"]
                    print(f" {ORANGE}resume{RESET} Using CQ={BOLD}{existing_cq}{RESET} from previous search")

            # ── Scene analysis ──
            sample_scenes = sample_src = None

            if existing_cq is None and meta["duration"] >= cfg["short_threshold"]:
                scene_cfg = {"scene_threshold": cfg["scene_threshold"]}
                if (all(k in cache for k in ("scenes", "complexity", "keyframes"))
                        and cache.get("scene_cfg") == scene_cfg):
                    print(f" {ORANGE}cache{RESET} Using cached scene data")
                    scenes = cache["scenes"]
                    complexity = cache["complexity"]
                    keyframes = cache["keyframes"]
                else:
                    print(f" {ORANGE}analyze{RESET} Detecting scenes...")
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
                        if scenes else "keyframe-aligned samples"
                    )
                    print(f" {ORANGE}scenes{RESET} {BOLD}{len(sample_scenes)}{RESET} {info}")
                    print(f" {ORANGE}extract{RESET} Extracting samples...")
                    sample_src = extract_samples(filepath, sample_scenes, keyframes, cfg, file_hash=file_hash)
                    if not sample_src:
                        print(f" {ORANGE}fallback{RESET} Extraction failed, using full encode")
                        sample_scenes = None
                else:
                    print(f" {ORANGE}scenes{RESET} Using full VMAF")
            elif existing_cq is None:
                print(f" {ORANGE}short{RESET} <{cfg['short_threshold']}s, full VMAF")

            # ── CQ search ──
            t_enc = t_vmaf = 0.0
            sample_enc_dir = cache_dir / "_sample_enc"
            sample_enc_dir.mkdir(parents=True, exist_ok=True)
            sample_enc_cache = {}

            def do_enc_sample(cq):
                nonlocal t_enc
                cq = clamp(cq, cfg["min_cq"], cfg["max_cq"])
                if cq in sample_enc_cache:
                    return sample_enc_cache[cq]
                if not sample_src or not sample_src.exists():
                    raise RuntimeError("Sample source missing")
                d = sample_enc_dir / f"sample_enc_{os.getpid()}_{cq}.mkv"
                _temp_files.add(d)
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

            print(
                f" {ORANGE}target{RESET} VMAF={BOLD}{target:.1f}{RESET}"
                f" (P5>={BOLD}{min_p5:.1f}{RESET})"
            )

            if existing_cq is not None:
                best_cq = existing_cq
                print(f" {ORANGE}reuse{RESET} Found existing CQ={BOLD}{existing_cq}{RESET} encode")
            elif sample_src:
                best_cq, _, _, vt = search_cq(
                    sample_src, meta, None, target, cache, cp,
                    do_enc_sample, vmaf_threads, cfg, tag="sample",
                )
                t_vmaf += vt
                # Cache recommended CQ so interrupted runs can resume
                if best_cq is not None:
                    cache["recommended"] = {
                        "cq": best_cq, "target": target,
                        "min_cq": cfg["min_cq"], "max_cq": cfg["max_cq"],
                    }
                    tmp = cp.with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(cache), encoding="utf-8")
                    tmp.replace(cp)
                for p in sample_enc_cache.values():
                    try:
                        p.unlink()
                        _temp_files.discard(p)
                    except OSError:
                        pass
            else:
                best_cq, best_vmaf, _, vt = search_cq(
                    filepath, meta, sample_scenes, target, cache, cp,
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
                    vmaf_str = f" VMAF={BOLD}{sv['mean']:.2f}{RESET} P5={BOLD}{sv['p5']:.2f}{RESET}"
                elif isinstance(sv, (int, float)):
                    vmaf_str = f" VMAF={BOLD}{sv:.2f}{RESET}"
                print(f" {CHECK} Recommended CQ={BOLD}{best_cq}{RESET}{vmaf_str}")
                print(f"   Run without --dry-run to encode")
                continue

            # ── Full encode + verification ──
            if sample_src or existing_cq is not None:
                if not dst_path(best_cq).exists():
                    print(f" {ORANGE}encode{RESET} Final encode at CQ={BOLD}{best_cq}{RESET}...")
                    t0 = time.time()
                    encode_av1(filepath, dst_path(best_cq), meta, best_cq, cfg)
                    t_enc += time.time() - t0
                else:
                    print(f" {ORANGE}reuse{RESET} CQ={BOLD}{best_cq}{RESET} encode exists")

                print(f" {ORANGE}verify{RESET} Full VMAF...")
                t0 = time.time()
                best_vmaf = vmaf_cached(
                    filepath, dst_path(best_cq), meta, best_cq, None,
                    cache, cp, vmaf_threads,
                )
                t_vmaf += time.time() - t0
                print(
                    f" {ORANGE}full{RESET} VMAF={BOLD}{best_vmaf['mean']:.2f}{RESET}"
                    f" P5={BOLD}{best_vmaf['p5']:.2f}{RESET}"
                )

                for _ in range(3):
                    if best_vmaf["mean"] >= target - cfg["vmaf_tolerance"]:
                        break
                    if best_cq <= cfg["min_cq"]:
                        break
                    try_cq = best_cq - 1
                    print(f" {ORANGE}adjust{RESET} Trying CQ={BOLD}{try_cq}{RESET}")
                    if not dst_path(try_cq).exists():
                        t0 = time.time()
                        encode_av1(filepath, dst_path(try_cq), meta, try_cq, cfg)
                        t_enc += time.time() - t0
                    t0 = time.time()
                    adj = vmaf_cached(
                        filepath, dst_path(try_cq), meta, try_cq, None,
                        cache, cp, vmaf_threads,
                    )
                    t_vmaf += time.time() - t0
                    if math.isfinite(adj["mean"]) and adj["mean"] > best_vmaf["mean"]:
                        best_cq, best_vmaf = try_cq, adj
                    else:
                        break

            elif sample_scenes is not None:
                print(f" {ORANGE}verify{RESET} Full VMAF at CQ={BOLD}{best_cq}{RESET}...")
                t0 = time.time()
                full_vm = vmaf_cached(
                    filepath, dst_path(best_cq), meta, best_cq, None,
                    cache, cp, vmaf_threads,
                )
                t_vmaf += time.time() - t0
                print(
                    f" {ORANGE}full{RESET} VMAF={BOLD}{full_vm['mean']:.2f}{RESET}"
                    f" P5={BOLD}{full_vm['p5']:.2f}{RESET}"
                )

                for _ in range(3):
                    if full_vm["mean"] >= target - cfg["vmaf_tolerance"]:
                        break
                    if best_cq <= cfg["min_cq"]:
                        break
                    try_cq = best_cq - 1
                    print(f" {ORANGE}adjust{RESET} Trying CQ={BOLD}{try_cq}{RESET}")
                    do_enc_full(try_cq)
                    t0 = time.time()
                    adj = vmaf_cached(
                        filepath, dst_path(try_cq), meta, try_cq, None,
                        cache, cp, vmaf_threads,
                    )
                    t_vmaf += time.time() - t0
                    if math.isfinite(adj["mean"]) and adj["mean"] > full_vm["mean"]:
                        best_cq, full_vm = try_cq, adj
                    else:
                        break
                best_vmaf = full_vm

            # ── P5 safety check ──
            for _ in range(3):
                if best_vmaf["p5"] >= min_p5 or best_cq <= cfg["min_cq"]:
                    break
                try_cq = best_cq - 1
                print(f" {ORANGE}safety{RESET} P5 low, trying CQ={BOLD}{try_cq}{RESET}")
                do_enc_full(try_cq)
                t0 = time.time()
                safe = vmaf_cached(
                    filepath, dst_path(try_cq), meta, try_cq, None,
                    cache, cp, vmaf_threads,
                )
                t_vmaf += time.time() - t0
                if math.isfinite(safe["p5"]) and safe["p5"] > best_vmaf["p5"]:
                    best_cq, best_vmaf = try_cq, safe
                else:
                    break

            # ── Bitrate floor check ──
            min_kbps = MIN_BITRATE_KBPS.get(tier, 0)
            if min_kbps and dst_path(best_cq).exists():
                actual_kbps = video_kbps(dst_path(best_cq), meta["duration"])
                for _ in range(5):
                    if not actual_kbps or actual_kbps >= min_kbps:
                        break
                    if best_cq <= cfg["min_cq"]:
                        break
                    try_cq = best_cq - 1
                    print(
                        f" {ORANGE}bitrate{RESET} {actual_kbps}kbps"
                        f" < {min_kbps}kbps floor,"
                        f" trying CQ={BOLD}{try_cq}{RESET}"
                    )
                    do_enc_full(try_cq)
                    t0 = time.time()
                    adj = vmaf_cached(
                        filepath, dst_path(try_cq), meta, try_cq, None,
                        cache, cp, vmaf_threads,
                    )
                    t_vmaf += time.time() - t0
                    best_cq, best_vmaf = try_cq, adj
                    actual_kbps = video_kbps(
                        dst_path(best_cq), meta["duration"]
                    )

            # ── Finalize ──
            final = dst_path(best_cq)
            if not final.exists():
                print(f" {CROSS} Final encode missing")
                continue

            # Clean up non-optimal encodes
            for c in range(cfg["min_cq"], cfg["max_cq"] + 1):
                if c != best_cq and dst_path(c).exists():
                    try:
                        dst_path(c).unlink()
                    except OSError:
                        pass

            out_sz = final.stat().st_size
            in_sz = filepath.stat().st_size

            if out_sz >= in_sz:
                final.unlink()
                stats["deleted"] += 1
                print(
                    f" {CROSS} Larger ({BOLD}{out_sz / 1e6:.1f}MB{RESET}"
                    f" vs {BOLD}{in_sz / 1e6:.1f}MB{RESET}) - deleted"
                )
                continue

            saved = (1.0 - out_sz / in_sz) * 100
            out_kbps = video_kbps(final, meta["duration"])
            kbps_str = f" ({BOLD}{out_kbps}kbps{RESET})" if out_kbps else ""
            hdr_tag = f" {ORANGE}[HDR]{RESET}" if meta["hdr"] else ""
            print(
                f" {CHECK} CQ={BOLD}{best_cq}{RESET}"
                f" VMAF={BOLD}{best_vmaf['mean']:.2f}{RESET}"
                f" P5={BOLD}{best_vmaf['p5']:.2f}{RESET}"
            )
            print(
                f" {CHECK} Size={BOLD}{out_sz / 1e6:.1f}MB{RESET}{kbps_str}"
                f" Saved={BOLD}{saved:.1f}%{RESET}{hdr_tag}"
            )
            print(f" {DIM}Enc:{fmt_time(t_enc)} VMAF:{fmt_time(t_vmaf)}{RESET}")

            stats["proc"] += 1
            if math.isfinite(best_vmaf["mean"]):
                stats["vmaf_sum"] += best_vmaf["mean"]
                stats["vmaf_n"] += 1
            stats["saved"] += in_sz - out_sz
            stats["orig"] += in_sz

        except Exception as e:
            _file_error = True
            print(f" {CROSS} {e}")
        finally:
            cleanup_temp()
            # Clean up sample source after successful processing;
            # preserve on error/interrupt for reuse on next run
            if not _file_error and sample_src:
                try:
                    if sample_src.exists():
                        sample_src.unlink()
                except OSError:
                    pass

    # ── Summary ──
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
        help="SVT-AV1 preset 0-13, lower=slower+better (default: 4)",
    )
    parser.add_argument(
        "--min-cq", type=int, default=16,
        help="Minimum CQ / highest quality (default: 16)",
    )
    parser.add_argument(
        "--max-cq", type=int, default=40,
        help="Maximum CQ / lowest quality (default: 40)",
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
