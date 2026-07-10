"""Letterbox/pillarbox crop: per-window detection, the union+agreement
aggregation, the sidecar contract, and the filename token."""

import json
import subprocess
import time

from .analyze import analyze_complexity, detect_scenes, get_keyframes
from .probe import detect_hwaccel
from .sampling import select_samples
from .ui import BOLD, CHECK, CROSS, DIM, GREEN, ORANGE, RESET
from .util import _temp_files, escape_filter_path, make_temp_log


def crop_token(crop):
    """Filename/cache-key-safe token for a 'W:H:X:Y' crop ('' when none).

    Colons are illegal in Windows filenames, so the geometry is flattened
    with 'x' (e.g. '_c1920x800x0x140').
    """
    return f"_c{crop.replace(':', 'x')}" if crop else ""


def load_crop_sidecar(filepath, file_hash):
    """Read <file>.crop.json from av1q-crop. Returns 'W:H:X:Y' or None.

    Only confidence='high' sidecars are auto-applied; 'low' is for manual
    review. Hash mismatch means the file was replaced after detection.
    """
    sidecar = filepath.with_suffix(filepath.suffix + ".crop.json")
    if not sidecar.exists():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("confidence") != "high":
        return None
    if data.get("source_hash") and data["source_hash"] != file_hash:
        return None
    try:
        w, h, x, y = data["width"], data["height"], data["x"], data["y"]
    except KeyError:
        return None
    if not all(isinstance(v, int) and v >= 0 for v in (w, h, x, y)):
        return None
    if w <= 0 or h <= 0:
        return None
    return f"{w}:{h}:{x}:{y}"


def detect_crop_window(source, start, duration, limit, round_to, cache_dir):
    """Run cropdetect on a single time window. Returns (w, h, x, y) or None."""
    log = make_temp_log(cache_dir, "crop", "txt")
    log_path = escape_filter_path(log)

    try:
        hw = detect_hwaccel()
        attempts = [hw, None] if hw else [None]

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
                text=True, encoding="utf-8", errors="replace", timeout=120,
            )
            if r.returncode == 0:
                break
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
    """Aggregate window crops via bounding-box union with per-edge agreement.

    cropdetect on dark or noisy scenes typically returns crops SMALLER
    than the truth — it mistakes shadowed picture edges for bars. Older
    area-agreement conflated letterbox and pillarbox axes: horizontal
    noise from a few dark scenes in a clean letterboxed film would drop
    overall area below the threshold and falsely flag "mixed aspect".

    Per-edge agreement only checks the edges the union is actually
    cropping. A pure-letterbox film has stable top/bottom edges; left
    and right sit at the frame boundary and are skipped, so horizontal
    cropdetect noise from dark scenes is irrelevant. True mixed-aspect
    content disagrees on the cropped axis itself and still fails the
    agreement threshold.
    """
    EDGE_TOL = 4  # px tolerance per edge

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

    x_min = min(c[2] for c in valid)
    y_min = min(c[3] for c in valid)
    x_max = min(frame_w, max(c[2] + c[0] for c in valid))
    y_max = min(frame_h, max(c[3] + c[1] for c in valid))
    w = x_max - x_min
    h = y_max - y_min
    x = x_min
    y = y_min

    if w >= frame_w and h >= frame_h:
        return {
            "crop": None, "confidence": "none",
            "reason": "full frame — no letterbox/pillarbox detected",
        }

    edges = []
    if y_min > 0:
        m = sum(1 for c in valid if abs(c[3] - y_min) <= EDGE_TOL)
        edges.append(("top", m))
    if y_max < frame_h:
        m = sum(1 for c in valid if abs(c[3] + c[1] - y_max) <= EDGE_TOL)
        edges.append(("bottom", m))
    if x_min > 0:
        m = sum(1 for c in valid if abs(c[2] - x_min) <= EDGE_TOL)
        edges.append(("left", m))
    if x_max < frame_w:
        m = sum(1 for c in valid if abs(c[2] + c[0] - x_max) <= EDGE_TOL)
        edges.append(("right", m))

    if not edges:
        return {
            "crop": None, "confidence": "none",
            "reason": "full frame — no letterbox/pillarbox detected",
        }

    worst_name, worst_match = min(edges, key=lambda e: e[1])
    agreement = worst_match / n_valid

    if agreement < agree_ratio:
        return {
            "crop": (w, h, x, y), "confidence": "low",
            "reason": (
                f"only {worst_match}/{n_valid} windows agree on {worst_name} edge "
                f"(±{EDGE_TOL}px); likely mixed aspect ratios"
            ),
        }

    keep_ratio = (w * h) / (frame_w * frame_h)
    if keep_ratio < min_keep_ratio:
        return {
            "crop": (w, h, x, y), "confidence": "low",
            "reason": (
                f"detected crop keeps only {keep_ratio:.0%} of frame; "
                f"below safety floor of {min_keep_ratio:.0%} "
                f"(rerun with --min-keep-ratio if intentional)"
            ),
        }

    edge_summary = ", ".join(f"{n} {m}/{n_valid}" for n, m in edges)
    return {
        "crop": (w, h, x, y), "confidence": "high",
        "reason": (
            f"edges agree within ±{EDGE_TOL}px ({edge_summary}); "
            f"{keep_ratio:.0%} of frame kept"
        ),
    }


def detect_crop_for_file(source, meta, cfg, file_hash, label_prefix=" "):
    """Detect crop for one video. Returns sidecar dict; does NOT write it.

    Prints per-window progress and a confidence-marked summary using
    label_prefix for indentation (single space matches av1q's main loop,
    two spaces matches av1q-crop's batch output).
    """
    LBL = 10

    def lbl(tag):
        return f"{label_prefix}{ORANGE}{tag:<{LBL}}{RESET}"

    is_hdr = meta["hdr"] or "10le" in meta["pix_fmt"]
    limit = cfg["limit_hdr"] if is_hdr else cfg["limit_sdr"]

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
        scenes = detect_scenes(source, sample_cfg, meta["duration"])
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

    src_type = "HDR" if is_hdr else "SDR"
    print(
        f"{lbl('crop scan')}{BOLD}{len(samples)}{RESET} windows · "
        f"{cfg['window_duration']:.0f}s each · "
        f"{DIM}limit={limit} {src_type}{RESET}"
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
            f"{lbl('window')}{i + 1}/{len(samples)} @ "
            f"{s['time']:.0f}s {marker} {DIM}{cstr}{RESET}"
        )

    result = aggregate_crops(
        crops, meta["w"], meta["h"],
        cfg["min_keep_ratio"], cfg["agree_ratio"],
    )

    sidecar_data = {
        "version": 1,
        "source_hash": file_hash,
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
        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if result["crop"]:
        w, h, x, y = result["crop"]
        sidecar_data.update({"width": w, "height": h, "x": x, "y": y})

    conf = result["confidence"]
    color = GREEN if conf == "high" else (ORANGE if conf == "low" else DIM)
    if result["crop"]:
        w, h, x, y = result["crop"]
        out = f"{w}:{h}:{x}:{y}"
    else:
        out = "(none)"
    print(
        f"{lbl('crop')}{color}{BOLD}{conf}{RESET}  "
        f"{BOLD}{out}{RESET}  {DIM}({result['reason']}){RESET}"
    )

    return sidecar_data
