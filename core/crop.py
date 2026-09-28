"""Letterbox/pillarbox crop: per-window detection, the union+agreement
aggregation, the sidecar contract, and the filename token."""

import json
import subprocess
import time

from .analyze import scene_analysis
from .constants import MIN_SCENE_DURATION
from .probe import frame_geometry
from .sampling import select_samples
from .tools import ffmpeg_exe
from .ui import BOLD, CHECK, CROSS, DIM, GREEN, ORANGE, RESET, label
from .util import _temp_files, escape_filter_path, make_temp_log

# Scan policy: one home for av1q-crop's CLI defaults and the inline
# --auto-crop scan, so the two can never drift apart.
SCAN_WINDOWS = 8       # windows spread across the runtime
WINDOW_DURATION = 2.0  # seconds of cropdetect per window
# cropdetect's darkness threshold, as a code value at the depth each was
# tuned for: SDR 8-bit, HDR 10-bit. Both reach the filter as a fraction
# of full scale, so the DECODED depth sets the real threshold and a
# 10-bit SDR rip is judged at the same darkness as an 8-bit one (24 at
# 8 bits is 96 at 10) instead of at 24/1023. HDR sits higher because
# PQ spends a wide run of low code values on its darkest stops: shadow
# detail beside a bar edge reads as black at the SDR threshold.
LIMIT_SDR = 24
LIMIT_HDR = 128
ROUND = 2              # output-dimension divisibility; 16 is codec-friendly
# Confidence gates. A crop keeping less of the frame than the keep ratio
# is a misdetection whatever the windows agree on (the floor only catches
# the catastrophic kind); agreement is the share of valid windows that
# place each cropped edge within EDGE_TOL pixels of the union's.
MIN_KEEP_RATIO = 0.10
AGREE_RATIO = 0.75
EDGE_TOL = 4
# Fewer valid windows than this share of the scan means it mostly hit
# black and says nothing about where the bars are.
MIN_VALID_SHARE = 0.7
# Windows stay inside the middle of the runtime: studio logos and end
# credits put text boxes where the picture should be.
SAFE_MARGIN = 0.05
# One window's ceiling. A seek plus two seconds of decode is well under
# this even on 4K in software; hitting it means a stalled decode.
WINDOW_TIMEOUT = 120


def crop_scan_cfg(cfg, **overrides):
    """The crop scan's settings: the policy defaults above plus what the
    scan borrows from the launcher cfg (the temp-log dir and the scene
    and short-file thresholds). av1q-crop overrides from its CLI."""
    return {
        "cache_dir": cfg["cache_dir"],
        "scene_threshold": cfg["scene_threshold"],
        "short_threshold": cfg["short_threshold"],
        "sample_count": SCAN_WINDOWS,
        "window_duration": WINDOW_DURATION,
        "limit_sdr": LIMIT_SDR,
        "limit_hdr": LIMIT_HDR,
        "round": ROUND,
        "min_keep_ratio": MIN_KEEP_RATIO,
        "agree_ratio": AGREE_RATIO,
        **overrides,
    }


def crop_token(crop):
    """Filename/cache-key-safe token for a 'W:H:X:Y' crop ('' when none).

    Colons are illegal in Windows filenames, so the geometry is flattened
    with 'x' (e.g. '_c1920x800x0x140').
    """
    return f"_c{crop.replace(':', 'x')}" if crop else ""


def sidecar_crop(data, file_hash):
    """The crop a sidecar dict puts on the file, as (crop, note).

    crop is 'W:H:X:Y' from a high-confidence sidecar that still matches
    the file, else None. note is None when there is nothing to say (a
    'none' verdict found the full frame) and otherwise why a present
    sidecar is not applied — worth a line, because a file the user
    scanned is about to be encoded uncropped. A changed file never gets
    here: its new hash names another sidecar path. The hash inside
    still refuses a sidecar copied by hand under another file's name;
    a missing hash (hand-written sidecar) is trusted.
    """
    conf = data.get("confidence")
    if conf == "none":
        return None, None
    if conf == "low":
        why = data.get("reason")
        return None, (
            "sidecar confidence low, not applied"
            + (f" ({why})" if isinstance(why, str) and why else "")
        )
    if conf != "high":
        return None, "sidecar malformed, ignored"
    if data.get("source_hash") and data["source_hash"] != file_hash:
        return None, "sidecar belongs to another file, ignored"
    try:
        w, h, x, y = data["width"], data["height"], data["x"], data["y"]
    except KeyError:
        return None, "sidecar malformed, ignored"
    if not all(isinstance(v, int) and v >= 0 for v in (w, h, x, y)):
        return None, "sidecar malformed, ignored"
    if w <= 0 or h <= 0:
        return None, "sidecar malformed, ignored"
    fw, fh = data.get("frame_width"), data.get("frame_height")
    if (isinstance(fw, int) and isinstance(fh, int)
            and (x + w > fw or y + h > fh)):
        return None, "sidecar crop exceeds its own frame, ignored"
    return f"{w}:{h}:{x}:{y}", None


def sidecar_path(cache_dir, filepath, file_hash):
    """Where the crop sidecar for this file lives: the shared cache
    root, never the source's own folder, so input folders stay clean.
    The name carries the source's name for a reader looking for it by
    eye, and the identity hash so a changed file or a same-named file in
    another folder never reads another file's verdict."""
    return (
        cache_dir / "_crop" / f"{filepath.name}.{file_hash[:16]}.crop.json"
    )


def read_crop_sidecar(cache_dir, filepath, file_hash):
    """The sidecar's verdict on this file, as (crop, note) per
    sidecar_crop; (None, None) when there is no sidecar."""
    sidecar = sidecar_path(cache_dir, filepath, file_hash)
    if not sidecar.exists():
        return None, None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, "sidecar unreadable, ignored"
    if not isinstance(data, dict):
        return None, "sidecar malformed, ignored"
    return sidecar_crop(data, file_hash)


def load_crop_sidecar(cache_dir, filepath, file_hash):
    """'W:H:X:Y' from a high-confidence sidecar that still matches the
    file, else None (see sidecar_crop for the rules)."""
    return read_crop_sidecar(cache_dir, filepath, file_hash)[0]


def detect_crop_window(source, start, duration, limit, round_to, cache_dir):
    """cropdetect over one window: the bounding box of everything
    brighter than `limit` (a fraction of full scale) across all of its
    frames, as (w, h, x, y); None when the window is unreadable or
    entirely black.

    Software decode only. A hardware decoder that misreads a profile
    hands back garbage frames at exit code 0 (the trap core.vmaf guards
    against), and a crop read off garbage would be applied at high
    confidence to every frame of the encode. A few seconds of CPU
    decode per window is the whole price of never finding out that way.
    """
    log = make_temp_log(cache_dir, "crop", "txt")
    log_path = escape_filter_path(log)

    try:
        # Pin the first video stream, same as the scene scan: default
        # selection takes the LARGEST video stream, and detecting bars
        # on a second video track would write a sidecar whose crop
        # belongs to a different picture than the one encoded.
        cmd = [
            ffmpeg_exe(), "-hide_banner", "-v", "error",
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-map", "0:v:0",
            "-an", "-sn",
            "-vf",
            f"cropdetect=limit={limit:.6f}:round={round_to}:reset_count=0,"
            f"metadata=mode=print:file={log_path}",
            "-f", "null", "-",
        ]
        try:
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
                timeout=WINDOW_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"cropdetect stalled for {WINDOW_TIMEOUT}s at {start:.0f}s"
            )
        if r.returncode != 0 or not log.exists():
            return None

        # cropdetect prints its running box on every frame and never
        # resets it here, so the last frame's values are the union over
        # the whole window.
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
    valid = [c for c in windows if c is not None]
    n_total = len(windows)
    n_valid = len(valid)

    if n_valid == 0:
        return {
            "crop": None, "confidence": "low",
            "reason": "no windows returned crop values (source too dark or unreadable)",
        }

    if n_valid < n_total * MIN_VALID_SHARE:
        return {
            "crop": None, "confidence": "low",
            "reason": (
                f"only {n_valid}/{n_total} windows returned valid crops "
                f"(likely many dark scenes)"
            ),
        }

    # A box past the frame edge means the decoded picture is not the
    # size the probe reported (a rotated source seen upright, for one),
    # and every coordinate below would be measured against the wrong
    # frame. Clamping it into the frame used to turn exactly that into
    # a plausible-looking crop at high confidence.
    for w, h, x, y in valid:
        if x + w > frame_w or y + h > frame_h:
            return {
                "crop": None, "confidence": "low",
                "reason": (
                    f"a window's crop {w}:{h}:{x}:{y} exceeds the "
                    f"{frame_w}x{frame_h} frame (decoded picture differs "
                    f"from the probed size)"
                ),
            }

    x_min = min(c[2] for c in valid)
    y_min = min(c[3] for c in valid)
    x_max = max(c[2] + c[0] for c in valid)
    y_max = max(c[3] + c[1] for c in valid)
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


def _scan_windows(source, meta, cfg, file_hash):
    """Where to look: the most complex scenes of the middle 90% of the
    runtime (bright, detailed picture is where bar edges read cleanly),
    from the same stored scene analysis the sample stage uses, so the
    scan is paid once per source however many stages ask. Short
    sources, and any without usable scenes, get evenly spaced windows
    instead."""
    duration = meta["duration"]
    safe_start = duration * SAFE_MARGIN
    safe_end = duration * (1 - SAFE_MARGIN)

    samples = None
    if duration >= cfg["short_threshold"]:
        scenes, complexity, keyframes = scene_analysis(
            source, meta, cfg, file_hash
        )
        scoped = [s for s in scenes if safe_start <= s["time"] <= safe_end]
        if scoped:
            samples = select_samples(
                scoped, complexity, duration, cfg["sample_count"], keyframes,
                {
                    "short_threshold": cfg["short_threshold"],
                    "sample_duration": cfg["window_duration"],
                    "min_scene_duration": MIN_SCENE_DURATION,
                },
            )
    if samples:
        return samples

    n = cfg["sample_count"]
    span = max(0.0, safe_end - safe_start)
    if span <= 0:
        # Unknown duration: one window, just past the start.
        n = 1
        span = 1.0
        safe_start = 0.0
    return [
        {"time": safe_start + span * (i + 0.5) / n,
         "duration": cfg["window_duration"]}
        for i in range(n)
    ]


def detect_crop_for_file(source, meta, cfg, file_hash):
    """Detect crop for one video. Returns the sidecar dict; does NOT
    write it. Prints per-window progress and a confidence-marked
    summary in the pipeline's label column.
    """
    is_hdr = bool(meta["hdr"])
    src_type = "HDR" if is_hdr else "SDR"
    limit_code, scale = (
        (cfg["limit_hdr"], 1023) if is_hdr else (cfg["limit_sdr"], 255)
    )
    # A fraction of full scale, so cropdetect scales it to the decoded
    # bit depth (see LIMIT_SDR). The half step keeps the filter's float
    # truncation from landing one code value under the configured one at
    # the depth it was tuned for.
    limit = (limit_code + 0.5) / scale

    # ffmpeg auto-rotates on decode, so the picture cropdetect measures
    # is upright while the probe reports the stored (unrotated) size and
    # a stream-copied sample keeps whatever orientation its muxer carries.
    # No single crop geometry is right for all of those, so a source with
    # a display matrix is not scanned: an upright portrait measured
    # against landscape dimensions used to come out as a confident
    # square crop.
    geo = frame_geometry(source)
    if geo and geo["transformed"]:
        samples, crops = [], []
        result = {
            "crop": None, "confidence": "low",
            "reason": "source carries a rotation or flip; not scanned",
        }
    else:
        samples = _scan_windows(source, meta, cfg, file_hash)
        print(
            f"{label('crop scan')}{BOLD}{len(samples)}{RESET} windows · "
            f"{cfg['window_duration']:g}s each · "
            f"{DIM}limit={limit_code} {src_type}{RESET}"
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
                f"{label('window')}{i + 1}/{len(samples)} @ "
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
        "limit": limit_code,
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
        f"{label('crop')}{color}{BOLD}{conf}{RESET}  "
        f"{BOLD}{out}{RESET}  {DIM}({result['reason']}){RESET}"
    )

    return sidecar_data
