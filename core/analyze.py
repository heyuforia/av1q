"""Content analysis: scene detection (a decode at 640px), packet-stat
complexity and keyframe listing (one demux pass, no decode), and the
shared per-source store all of it lives in."""

import json
import subprocess

from .constants import (
    COMPLEXITY_WINDOW, INTRA_ONLY_CODECS, MIN_SCENE_DURATION,
)
from .probe import detect_hwaccel, hw_decode_unsafe
from .tools import ffmpeg_exe, ffprobe_exe
from .ui import DIM, RESET, label
from .util import (
    _temp_files, atomic_write_json, escape_filter_path, make_temp_log,
    scan_budget,
)


def _fail(what, r=None, timeout=None):
    """One line saying why a whole-file pass produced nothing, so an
    evenly-sampled file can be told from a scan that never finished."""
    if timeout is not None:
        why = f"timed out after {timeout}s"
    else:
        tail = (r.stderr or "").strip().splitlines()
        why = f"exit {r.returncode}" + (f": {tail[-1]}" if tail else "")
    print(f"{label('analyze')}{DIM}{what} failed ({why}){RESET}")


def detect_scenes(source, cfg, duration=None, profile=None):
    """Scene boundaries from ffmpeg's scdet filter, as a list of
    {time, duration} in seconds from the container's start time (ffmpeg
    rebases decoded frames there); None when the scan fails.

    scdet decodes the whole file (downscaled to 640px) frame by frame,
    so the wall-clock budget scales with runtime (scan_budget). The
    decode goes through the hardware decoder when there is one, except
    for the profiles it cannot be trusted on (hw_decode_unsafe): a
    mis-decoded stream yields garbage cuts at exit code 0, and the
    minutes saved are not worth sampling the wrong scenes.

    scdet reports cut times only. The opening stretch up to the first
    cut is as much a scene as any other, so it is listed at 0 whenever
    there is a cut at all; a file with no cuts is one scene, which the
    consumers sample evenly, and stays an empty list. The last scene
    runs to the end of the file (MIN_SCENE_DURATION stands in when the
    runtime is unknown, so it stays a candidate).

    A timeout or a failed decode returns None with the reason printed:
    the consumers fall back to even sampling either way, but a failure
    is not a fact about the source and must not be stored as one.
    """
    log = make_temp_log(cfg["cache_dir"], "scdet", "txt")
    log_path = escape_filter_path(log)
    timeout = scan_budget(duration)

    try:
        hw = None if hw_decode_unsafe(profile) else detect_hwaccel()
        attempts = [hw, None] if hw else [None]

        for accel in attempts:
            cmd = [ffmpeg_exe(), "-hide_banner", "-v", "error"]
            if accel:
                cmd += ["-hwaccel", accel]
            # Pin the first video stream. Default selection picks the
            # LARGEST video stream rather than v:0, so a source carrying
            # a second video track would have its scenes read off the
            # wrong picture while probe, sampling, encode and VMAF all
            # work on v:0. (ffmpeg already skips an attached cover image
            # here; a genuine second track it does not.)
            cmd += [
                "-i", str(source), "-map", "0:v:0", "-an",
                "-vf", f"scale=640:-2,scdet=t={cfg['scene_threshold']},"
                       f"metadata=mode=print:file={log_path}",
                "-f", "null", "-",
            ]
            try:
                r = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8", errors="replace",
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                _fail("scene scan", timeout=timeout)
                return None
            if r.returncode == 0:
                break
        else:
            _fail("scene scan", r)
            return None

        cuts = []
        if log.exists():
            for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "lavfi.scd.time" in line:
                    try:
                        cuts.append(float(line.split("=")[1].strip()))
                    except (ValueError, IndexError):
                        pass
        if not cuts:
            return []
        starts = [0.0] + cuts if cuts[0] > 0 else cuts
        end = (
            duration if duration and duration > starts[-1]
            else starts[-1] + MIN_SCENE_DURATION
        )
        return [
            {"time": t, "duration": nxt - t}
            for t, nxt in zip(starts, starts[1:] + [end])
        ]
    except OSError as e:
        print(f"{label('analyze')}{DIM}scene scan failed ({e}){RESET}")
        return None
    finally:
        try:
            if log.exists():
                log.unlink()
        except OSError:
            pass
        _temp_files.discard(log)


def read_packets(source, duration=None, start_time=0.0):
    """Every v:0 packet as (time, size, is_key), demux only — no decode,
    so it runs at I/O speed even on long 4K sources. None when the
    stream can't be read (the reason is printed).

    time is the packet's pts, or its dts when it has none (a decoder
    fills in the same way); a packet with neither is skipped. Keyframe
    packets (ffprobe's K flag) stand in for I-frames.

    ffprobe prints raw stream timestamps, but ffmpeg subtracts the
    container's start time (probe_video's start_time) from everything it
    reads: the scene scan's cut times and every -ss seek count from it.
    Subtracting it here puts every time this module stores on that one
    timeline. With raw times on a source that starts late (MPEG-TS
    commonly does), a clip's seek misses its keyframe by start_time, so
    stream copy keeps up to a GOP of pre-roll, and each scene is ranked
    by the complexity of the window start_time before it.
    """
    timeout = scan_budget(duration)
    try:
        r = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time,dts_time,size,flags",
             "-of", "csv=p=0", str(source)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _fail("packet scan", timeout=timeout)
        return None
    except OSError as e:
        print(f"{label('analyze')}{DIM}packet scan failed ({e}){RESET}")
        return None
    if r.returncode != 0:
        _fail("packet scan", r)
        return None

    packets = []
    for line in (r.stdout or "").splitlines():
        fields = line.split(",")
        if len(fields) < 4:
            continue
        pts, dts, size, flags = fields[:4]
        t = None
        for v in (pts, dts):
            try:
                t = float(v)
                break
            except ValueError:
                continue  # N/A: no timestamp of this kind
        try:
            size = int(size)
        except ValueError:
            continue
        if t is not None:
            packets.append((t - start_time, size, "K" in flags))
    return packets


def window_of(t):
    """Start time of the complexity window containing t. The one
    bucketing rule for every consumer of the per-window list."""
    return int(t / COMPLEXITY_WINDOW) * COMPLEXITY_WINDOW


def span_complexity(comp_map, start, stop):
    """Complexity of the picture in [start, stop), as (mean, seconds):
    each window it touches weighted by the seconds of the span inside
    it. comp_map maps window_of starts to complexity. A window with no
    entry (a gap in the stream) has no picture to count, so it adds no
    seconds; (None, 0.0) when no window of the span has a value.

    Read over the span, never at its start: a scene that starts late in
    a window would otherwise be ranked by the picture before it."""
    total = secs = 0.0
    w = window_of(start)
    while w < stop:
        v = comp_map.get(w)
        inside = min(stop, w + COMPLEXITY_WINDOW) - max(start, w)
        if isinstance(v, (int, float)) and v > 0 and inside > 0:
            total += v * inside
            secs += inside
        w += COMPLEXITY_WINDOW
    return (total / secs if secs else None), secs


def complexity_windows(packets):
    """Per-window complexity from a read_packets list, as
    [{time, complexity}] in window order.

    Complexity is the window's mean packet size in kilobytes: under the
    source encoder's own rate control, bytes per frame track how hard
    the picture was to code, and a window's mean averages that over
    every frame in it, so a keyframe landing in one window and not the
    next moves the number by a few percent instead of deciding it. The
    scale is arbitrary — consumers read it over a span (span_complexity)
    to rank scenes and take ratios (sampling.complexity_bias), never read
    the value on its own.
    """
    windows = {}
    for t, size, _ in packets:
        w = window_of(t)
        tot, n = windows.get(w, (0, 0))
        windows[w] = (tot + size, n + 1)
    return [
        {"time": w, "complexity": tot / n / 1000}
        for w, (tot, n) in sorted(windows.items())
    ]


def keyframe_times(packets):
    """Sorted keyframe timestamps from a read_packets list."""
    return sorted(t for t, _, key in packets if key)


def analyze_complexity(source, duration=None, start_time=0.0):
    """Per-window complexity of a source (see complexity_windows); []
    when the stream can't be read."""
    packets = read_packets(source, duration, start_time)
    return complexity_windows(packets) if packets else []


def get_keyframes(source, duration=None, start_time=0.0):
    """Sorted keyframe timestamps of a source; [] when the stream can't
    be read."""
    packets = read_packets(source, duration, start_time)
    return keyframe_times(packets) if packets else []


def analysis_path(cache_dir, file_hash):
    """Where a source's scene analysis lives: under the cache root both
    pipelines and av1q-crop share, keyed by the file's partial hash like
    the sample concats — it is a fact about the source, not about any
    engine's search."""
    return cache_dir / "_analysis" / f"{file_hash}.json"


def scene_analysis(source, meta, cfg, file_hash):
    """A source's scene boundaries, per-window complexity, and keyframes,
    as (scenes, complexity, keyframes), all in seconds from the
    container's start time: the timeline ffmpeg's -ss seeks on.

    Intra-only sources get three empty lists without a scan: every packet
    is a keyframe there and packet size follows the picture, not the
    cut structure, so there is nothing to rank and the consumers pick
    evenly spaced clips instead. Everything else is scanned once per
    source, ever: a stored analysis made under the same scene threshold
    and container start time is read back from the shared store
    (analysis_path), so the crop scan, the sample stage, and the other
    pipeline all ride one scdet decode. Only a complete scan is stored — a timed-out or failed pass
    returns empty lists for this run (even sampling) and is retried by
    the next.
    """
    if meta["codec"] in INTRA_ONLY_CODECS:
        return [], [], []
    scene_cfg = {"scene_threshold": cfg["scene_threshold"]}
    start_time = meta.get("start_time", 0.0)
    path = analysis_path(cfg["cache_dir"], file_hash)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            data = None
        # The packet times are stored relative to the container start
        # (read_packets), so a store made under another start time is on
        # another clock. A store without the field was written from raw
        # times, which only still match a source that starts at 0.
        if (isinstance(data, dict)
                and all(isinstance(data.get(k), list)
                        for k in ("scenes", "complexity", "keyframes"))
                and data.get("scene_cfg") == scene_cfg
                and data.get("start_time", 0.0) == start_time):
            print(f"{label('cache')}Using stored scene analysis")
            return data["scenes"], data["complexity"], data["keyframes"]
        if data is None:
            print(f"{label('cache')}{DIM}{path.name} unreadable, rescanning{RESET}")

    print(f"{label('analyze')}Detecting scenes...")
    scenes = detect_scenes(
        source, cfg, meta["duration"], profile=meta.get("profile")
    )
    packets = read_packets(source, meta["duration"], start_time)
    complexity = complexity_windows(packets) if packets else []
    keyframes = keyframe_times(packets) if packets else []
    if scenes is not None and packets is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {
            "scene_cfg": scene_cfg, "start_time": start_time,
            "scenes": scenes, "complexity": complexity,
            "keyframes": keyframes,
        })
    return scenes or [], complexity, keyframes
