"""Source inspection: ffprobe metadata (one call per source, timing facts
included), the VFR verdict, HDR10 static metadata, the cached
hardware-decode probe, and resolution tiering."""

import json
import math
import platform
import subprocess

from .tools import ffmpeg_exe, ffprobe_exe
from .util import run_cmd, scan_budget

_hwaccel = None
_hwaccel_checked = False

# Profile tags whose streams must never go through a hardware decoder.
# ffmpeg's hwaccel capability check asks the driver about codec, chroma
# and bit depth but never the profile, so a stream using profile tools
# the silicon lacks is CLAIMED supported and decodes to garbage at exit
# code 0 — scores and scene cuts read off it are silently wrong. x264's
# lossless mode tags High 4:4:4 Predictive even for 4:2:0 content and
# its transform-bypass frames mis-decode on Blackwell; 4:2:2 and the
# HEVC range extensions are the same trap.
HW_UNSAFE_PROFILES = ("4:4:4", "4:2:2", "444", "422", "rext")


def hw_decode_unsafe(profile):
    """True when a stream's ffprobe profile string is on the hw-unsafe
    list. Unknown or empty profiles are False: mainstream profiles are
    the overwhelmingly common case, and a hardware decode that errors
    out still falls back to the software attempt."""
    prof = (profile or "").lower()
    return any(t in prof for t in HW_UNSAFE_PROFILES)


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
                [ffmpeg_exe(), "-hide_banner", "-hwaccel", hw,
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


def parse_rate(v):
    """Frame rate as a float from ffprobe's rational or decimal spelling
    ('24000/1001', '25'); None for missing, 'N/A', '0/0', or anything
    not a positive finite number."""
    if not v:
        return None
    try:
        s = str(v)
        if "/" in s:
            a, b = s.split("/", 1)
            b = float(b)
            f = float(a) / b if b else 0.0
        else:
            f = float(s)
    except (ValueError, TypeError):
        return None
    return f if f > 0 and math.isfinite(f) else None


def _tag(tags, name):
    """A Matroska statistics tag by name. ffmpeg suffixes tag keys with
    their language ('NUMBER_OF_FRAMES-eng'), so match the stem."""
    for k, v in (tags or {}).items():
        if str(k).upper().split("-", 1)[0] == name:
            return v
    return None


def _parse_tag_duration(v):
    """mkvmerge's DURATION tag ('01:52:33.093000000') in seconds."""
    try:
        h, m, s = str(v).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, AttributeError):
        return None


def probe_video(filepath):
    """Extract video metadata via ffprobe: the one header read of a
    source, which every stage works from instead of re-probing.

    Alongside the dimension, color, bitrate, and codec facts (`profile`
    is the stream's profile string, lowercased, for the hardware-decode
    gate), it carries the timing facts the essential engine's VFR
    verdict and CFR feed need:
      fps       avg_frame_rate as a rational string, or None
      rfps      r_frame_rate as a rational string, or None: the nominal
                cadence, the finest the stream's timestamps fall on
      mean_fps  the exact whole-file mean frame rate when the container
                states its frame count (MP4/MOV sample tables; the
                NUMBER_OF_FRAMES statistics tag mkvmerge writes), else
                None. Not avg_frame_rate: ffmpeg derives that from the
                frame count only for MP4/MOV and takes the nominal
                DefaultDuration for Matroska (see is_vfr).
    """
    r = run_cmd([
        ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,bit_rate,pix_fmt,color_primaries,"
        "color_transfer,color_space,color_range,codec_name,profile,"
        "r_frame_rate,avg_frame_rate,nb_frames,duration"
        ":stream_tags:format=duration,bit_rate",
        "-of", "json", str(filepath),
    ])
    data = json.loads(r.stdout or "{}")
    s = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    tags = s.get("tags") if isinstance(s.get("tags"), dict) else {}

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
    profile = str(s.get("profile") or "").lower()
    pf = s.get("pix_fmt") or ""
    hdr = ct in {"smpte2084", "arib-std-b67"} or cp == "bt2020"
    duration = float(fmt.get("duration") or 0)

    fps = s.get("avg_frame_rate")
    rfps = s.get("r_frame_rate")

    frames = None
    for v in (s.get("nb_frames"), _tag(tags, "NUMBER_OF_FRAMES")):
        try:
            frames = int(v)
            break
        except (ValueError, TypeError):
            pass
    # The stream's own duration pairs with its frame count. The format
    # duration is the fallback; it can run a little long on a trailing
    # audio track, which only costs a short file the cheap verdict.
    vid_duration = None
    for v in (s.get("duration"),
              _parse_tag_duration(_tag(tags, "DURATION")), duration):
        try:
            if v is not None and float(v) > 0:
                vid_duration = float(v)
                break
        except (ValueError, TypeError):
            pass
    mean_fps = frames / vid_duration if frames and vid_duration else None

    return {
        "w": int(s.get("width") or 0),
        "h": int(s.get("height") or 0),
        "pix_fmt": pf, "bitrate": bitrate,
        "duration": duration,
        "cp": cp, "ct": ct, "cs": cs, "cr": cr,
        "codec": codec, "profile": profile, "hdr": hdr,
        "fps": fps if parse_rate(fps) else None,
        "rfps": rfps if parse_rate(rfps) else None,
        "mean_fps": mean_fps,
    }


def get_fps(filepath):
    """avg_frame_rate of v:0 as a rational string, or None.

    For encodes (the VMAF pair's distorted side): a source's rates ride
    in probe_video's meta, so no stage re-probes a source for them.
    """
    try:
        r = run_cmd([
            ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate",
            "-of", "default=nw=1:nk=1", str(filepath),
        ])
    except RuntimeError:
        return None
    v = r.stdout.strip()
    return v if parse_rate(v) else None


def res_tier(w, h):
    """Resolution tier based on the short dimension (handles vertical video)."""
    short = min(w, h)
    for t in (4320, 2160, 1440, 1080, 720):
        if short >= t:
            return t
    return 0


def _ratval(v):
    """Parse an ffprobe side-data value that may be '34000/50000' (older
    builds) or '0.680000' (newer builds). Returns float or None."""
    if v is None:
        return None
    s = str(v)
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            b = float(b)
            return float(a) / b if b else None
        return float(s)
    except (ValueError, TypeError):
        return None


def probe_hdr_metadata(filepath):
    """HDR10 static metadata from the first frame's side data.

    Returns (mastering_display_str, content_light_str), either may be None.
    Formats follow SvtAv1EncApp --color-help:
      G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)  and  "max_cll,max_fall".
    """
    try:
        r = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
             "-show_frames", "-read_intervals", "%+#1",
             "-show_entries", "frame=side_data_list",
             "-of", "json", str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        if r.returncode != 0:
            return None, None
        frames = json.loads(r.stdout or "{}").get("frames", [])
        side = frames[0].get("side_data_list", []) if frames else []
    except Exception:
        return None, None

    mastering = cll = None
    for sd in side:
        t = sd.get("side_data_type", "")
        if t == "Mastering display metadata":
            vals = {k: _ratval(sd.get(k)) for k in (
                "red_x", "red_y", "green_x", "green_y", "blue_x", "blue_y",
                "white_point_x", "white_point_y",
                "max_luminance", "min_luminance",
            )}
            if all(v is not None for v in vals.values()):
                mastering = (
                    f"G({vals['green_x']:.5f},{vals['green_y']:.5f})"
                    f"B({vals['blue_x']:.5f},{vals['blue_y']:.5f})"
                    f"R({vals['red_x']:.5f},{vals['red_y']:.5f})"
                    f"WP({vals['white_point_x']:.5f},{vals['white_point_y']:.5f})"
                    f"L({vals['max_luminance']:.4f},{vals['min_luminance']:.4f})"
                )
        elif t == "Content light level metadata":
            mc = sd.get("max_content")
            ma = sd.get("max_average")
            if isinstance(mc, int) and isinstance(ma, int):
                cll = f"{mc},{ma}"
    return mastering, cll


# A frame interval this far off the median is irregular. Container
# timestamp rounding jitters a CFR stream's intervals by one tick — 2.4%
# for 23.976fps in Matroska's 1ms ticks, 12% at 120fps — while a dropped
# or held frame at least doubles the interval; 25% separates the two
# with room on both sides.
VFR_INTERVAL_TOLERANCE = 0.25

# Up to this fraction of irregular frames — or, on the header fast path,
# of frames missing against the nominal cadence — is a CFR source with
# glitches, not a variable one: the CFR feed's dup/drop then touches at
# most one frame in a hundred.
VFR_IRREGULAR_MAX = 0.01


def _frame_intervals(filepath, duration):
    """Presentation-order frame intervals of v:0 in seconds, from every
    packet's pts (demux only, no decode); None when the timeline can't be
    read."""
    try:
        r = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time",
             "-of", "default=nw=1:nk=1", str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            timeout=scan_budget(duration),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    pts = []
    for tok in (r.stdout or "").split():
        try:
            pts.append(float(tok))
        except ValueError:
            continue  # N/A: a packet without a timestamp
    pts.sort()
    return [b - a for a, b in zip(pts, pts[1:])]


def is_vfr(filepath, meta):
    """True when the source's frame timing is genuinely variable.

    The Y4M pipe's CFR feed resamples every source onto its nominal
    cadence (fps=r_frame_rate): irregular frame intervals become dups
    and drops and the frame count changes, where av1q's ffmpeg path
    passes VFR timing through untouched. So the essential engine refuses
    such files and points at av1q.

    The container headers alone cannot say. ffmpeg derives
    avg_frame_rate from the exact frame count for MP4/MOV, but for
    Matroska it takes the nominal DefaultDuration and sets r_frame_rate
    to the same value: header agreement proves nothing there, and on
    MP4 the header mismatch IS the signal (a packet count compared
    against avg_frame_rate would only re-derive the header). Two stages:

      1. When the container states its frame count (meta["mean_fps"]),
         the exact mean cadence against the nominal r_frame_rate settles
         it without touching the packets: intervals never fall below the
         nominal one, so a mean within VFR_IRREGULAR_MAX of it leaves no
         room for irregular intervals. Interlaced streams report a
         field-rate r_frame_rate and fall through to the scan, which
         clears them.
      2. Otherwise (no count: Matroska without mkvmerge statistics tags,
         MPEG-TS; or a mismatch to explain) every packet's pts is read
         and the intervals judged directly: more than VFR_IRREGULAR_MAX
         of them off the median means variable timing.

    An unreadable timeline refuses the file only when the headers
    already disagreed; with nothing suspicious, a broken probe must not
    gate the file.
    """
    rf = parse_rate(meta.get("rfps"))
    mean = meta.get("mean_fps")
    header_mismatch = None
    if rf and mean:
        header_mismatch = abs(mean - rf) / rf > VFR_IRREGULAR_MAX
        if not header_mismatch:
            return False

    intervals = _frame_intervals(filepath, meta.get("duration"))
    if intervals is None:
        return bool(header_mismatch)
    if len(intervals) < 2:
        return False
    median = sorted(intervals)[len(intervals) // 2]
    if median <= 0:
        return False
    irregular = sum(
        1 for d in intervals
        if abs(d - median) > VFR_INTERVAL_TOLERANCE * median
    )
    return irregular / len(intervals) > VFR_IRREGULAR_MAX


def frame_geometry(filepath):
    """Picture geometry as a frame-indexing reader sees it.

    FFVship reads through FFMS2, which decodes without applying the
    display matrix and chooses its own video track, while ffmpeg — every
    other stage here — auto-rotates and is pinned to v:0. This reports
    what must agree before the two can be compared at all:

      w, h         v:0's display dimensions. Not coded_width/height: the
                   reader hands back the cropped picture (1080, not the
                   macroblock-padded 1088).
      transformed  a display matrix is attached, so ffmpeg's decode is
                   oriented differently than a reader that ignores it.
                   The demuxers suppress identity matrices, so a matrix
                   being present at all means a real rotation or flip.
      n_video      video streams, excluding attached cover art (ffmpeg
                   skips those in stream selection; a genuine second
                   track it does not).

    Returns None when the file can't be read.
    """
    try:
        r = run_cmd([
            ffprobe_exe(), "-v", "error", "-select_streams", "v",
            "-show_streams", "-of", "json", str(filepath),
        ])
        streams = json.loads(r.stdout or "{}").get("streams") or []
    except (RuntimeError, ValueError, OSError):
        # OSError included: a missing ffprobe must not take down the
        # display-only caller that asks for this.
        return None

    real = [
        s for s in streams
        if not (s.get("disposition") or {}).get("attached_pic")
    ]
    if not real:
        return None
    first = real[0]

    # Field names inside side_data_list have moved between ffprobe
    # versions, so match on the type string ffmpeg exports for a display
    # matrix and on the two field names that only a display matrix
    # carries — any of the three is the same fact.
    transformed = False
    for sd in first.get("side_data_list") or []:
        if not isinstance(sd, dict):
            continue
        if ("rotation" in sd or "displaymatrix" in sd
                or "Display Matrix" in (
                    str(v) for v in sd.values() if isinstance(v, str))):
            transformed = True
            break

    try:
        w, h = int(first.get("width") or 0), int(first.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return {"w": w, "h": h, "transformed": transformed, "n_video": len(real)}
