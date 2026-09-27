"""Source inspection: ffprobe metadata (one call per source, timing facts
included), the VFR verdict, HDR10 static metadata, the cached
hardware-decode probe, and resolution tiering."""

import json
import math
import platform
import re
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

# The spellings ffprobe gives a color tag the stream does not state
# ("" when the field is absent).
UNTAGGED = {"", "unknown", "unspecified", "reserved"}


def hw_decode_unsafe(profile):
    """True when a stream's ffprobe profile string is on the hw-unsafe
    list. Unknown or empty profiles are False: mainstream profiles are
    the overwhelmingly common case, and a hardware decode that errors
    out still falls back to the software attempt."""
    prof = (profile or "").lower()
    return any(t in prof for t in HW_UNSAFE_PROFILES)


def high_bit_depth(pix_fmt):
    """True when a pixel format carries more than 8 bits per sample.

    ffmpeg spells every deeper format with its depth right before the
    endianness suffix (yuv420p10le, yuv444p12le, p010le, gray10be,
    yuv444p10msble); 8-bit formats carry no suffix (yuv420p, nv12,
    rgb24). Matching "10le" alone misses 9, 12 and 16 bits and every
    big-endian spelling, and drops those sources to 8 bits. Packed raw
    RGB (rgb565le) reads as deep too; no library source uses it, and
    10 bits is the safe side.
    """
    m = re.search(r"(\d+)(?:msb)?[lb]e$", pix_fmt or "")
    return bool(m) and int(m.group(1)) > 8


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
    verdict needs:
      fps       avg_frame_rate as a rational string, or None
      rfps      r_frame_rate as a rational string, or None: the nominal
                cadence, the finest the stream's timestamps fall on
      mean_fps  the exact whole-file mean frame rate when the container
                states its frame count (MP4/MOV sample tables; the
                NUMBER_OF_FRAMES statistics tag mkvmerge writes), else
                None. Not avg_frame_rate: ffmpeg derives that from the
                frame count only for MP4/MOV and takes the nominal
                DefaultDuration for Matroska (see is_vfr).
      start_time  the container's start time in seconds, 0.0 when
                unstated. ffmpeg subtracts it from every timestamp it
                reads, so the scene scan and every -ss count from it;
                ffprobe's packet times do not (see read_packets).
    """
    r = run_cmd([
        ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,bit_rate,pix_fmt,color_primaries,"
        "color_transfer,color_space,color_range,codec_name,profile,"
        "r_frame_rate,avg_frame_rate,nb_frames,duration"
        ":stream_tags:format=duration,bit_rate,start_time",
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
    # HDR is a transfer fact: PQ or HLG (BT.2100). BT.2020 primaries
    # alone are wide gamut, and with a stated SDR transfer (bt709,
    # bt2020-10) the source is SDR; only with the transfer untagged do
    # they stand in for it. A false HDR flag tonemaps the VMAF chain and
    # moves the crop scan's darkness limit.
    hdr = ct in {"smpte2084", "arib-std-b67"} or (
        cp == "bt2020" and ct in UNTAGGED
    )
    duration = float(fmt.get("duration") or 0)
    try:
        start_time = float(fmt.get("start_time") or 0)
    except ValueError:
        start_time = 0.0  # 'N/A': ffmpeg applies no offset either

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
        "duration": duration, "start_time": start_time,
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


# The steps HDR10 mastering display metadata is stated in: HEVC's SEI
# and MP4's mdcv box count chromaticities in 1/50000 and luminances in
# 1/10000 cd/m², and ffmpeg's -mastering_display takes those counts.
MDCV_CHROMA_DEN = 50000
MDCV_LUMA_DEN = 10000

# Side-data fields of a mastering display, in the order both spellings
# state them: G, B, R, white point.
_MDCV_XY = (
    "green_x", "green_y", "blue_x", "blue_y",
    "red_x", "red_y", "white_point_x", "white_point_y",
)

# How many packets the HDR10 read looks through for the first keyframe
# when a capture starts between keyframes. 600 is ten seconds at 60 fps,
# longer than any broadcast or streaming GOP and than x265's default of
# 250 frames. Only keyframes decode on that read, so the window costs a
# demux plus one decode per keyframe inside it.
HDR_KEYFRAME_WINDOW = 600


def _keyframe_side_data(filepath, packets):
    """The side-data list of each keyframe ffprobe decodes among the
    first `packets` packets of v:0, in output order ([] when none
    decodes). Raises RuntimeError saying why when ffprobe gives no
    answer."""
    timeout = 120
    try:
        r = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-skip_frame", "nokey",
             "-read_intervals", f"%+#{packets}",
             "-select_streams", "v:0", "-show_frames",
             "-show_entries", "frame=side_data_list",
             "-of", "json", str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out after {timeout}s") from None
    except OSError as e:
        raise RuntimeError(f"ffprobe did not start: {e}") from None
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        raise RuntimeError(
            f"ffprobe exit {r.returncode}" + (f": {tail[-1]}" if tail else "")
        )
    try:
        frames = json.loads(r.stdout or "{}").get("frames") or []
        return [f.get("side_data_list") or [] for f in frames]
    except (ValueError, AttributeError, TypeError):
        raise RuntimeError("ffprobe output not readable") from None


def _steps(v, den):
    """A side-data value as a whole count of 1/den steps, or None."""
    f = _ratval(v)
    return round(f * den) if f is not None and math.isfinite(f) else None


def probe_hdr_metadata(filepath):
    """HDR10 static metadata as the first decoded keyframes state it.

    Returns (mastering, cll), each None when the source does not state
    it:
      mastering  G, B, R and white point x/y in 1/MDCV_CHROMA_DEN steps,
                 then max and min luminance in 1/MDCV_LUMA_DEN steps
      cll        (max_cll, max_fall) in cd/m²
    svt_mastering_display, ffmpeg_mastering_display and
    content_light_str spell them for each consumer.

    A failed read says nothing about the source and raises RuntimeError
    with the reason, which tells a failure a rerun may fix (a timeout)
    from one that returns on every run (no keyframe in the window).

    A mastering display the standard forbids (a chromaticity past 1, a
    minimum luminance above the maximum) or a light level past 16 bits
    is corrupt and dropped like an incomplete one: ffmpeg fails the
    whole command on a value past its own limits, which sit at or
    beyond these.

    Container-level metadata rides every decoded frame, and HEVC's SEI
    metadata every frame from the keyframe that carries it. Only
    keyframes are decoded, because a picture between keyframes carries
    no SEI, and whether the decoder emits one for it when the capture
    starts there depends on the codec and the ffmpeg version. So the
    first packet answers when it is a keyframe, one decode for a stream
    that starts on one and for an intra-only master, and a capture that
    starts between keyframes decodes nothing from it and reads on to its
    first keyframe. A keyframe without the metadata is the answer, not
    a miss. A window with no keyframe says nothing about the source and
    is a failed read.
    """
    frames = _keyframe_side_data(filepath, 1)
    if not frames:
        frames = _keyframe_side_data(filepath, HDR_KEYFRAME_WINDOW)
    if not frames:
        raise RuntimeError(
            f"no keyframe in the first {HDR_KEYFRAME_WINDOW} packets"
        )

    mastering = cll = None
    for side in frames:
        for sd in side:
            t = sd.get("side_data_type", "")
            if t == "Mastering display metadata" and mastering is None:
                xy = [_steps(sd.get(k), MDCV_CHROMA_DEN) for k in _MDCV_XY]
                hi = _steps(sd.get("max_luminance"), MDCV_LUMA_DEN)
                lo = _steps(sd.get("min_luminance"), MDCV_LUMA_DEN)
                if (all(v is not None and 0 <= v <= MDCV_CHROMA_DEN
                        for v in xy)
                        and None not in (hi, lo)
                        and 0 <= lo <= hi <= 0x7FFFFFFF):
                    mastering = (*xy, hi, lo)
            elif t == "Content light level metadata" and cll is None:
                mc = sd.get("max_content")
                ma = sd.get("max_average")
                if all(isinstance(v, int) and 0 <= v <= 0xFFFF
                       for v in (mc, ma)):
                    cll = (mc, ma)
    return mastering, cll


def svt_mastering_display(m):
    """A probe_hdr_metadata mastering display as SVT-AV1 spells it, the
    same string for SvtAv1EncApp's flag and -svtav1-params:
    G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min) in decimals, five and four
    places being exactly the two step sizes."""
    return (
        "G({:.5f},{:.5f})B({:.5f},{:.5f})R({:.5f},{:.5f})"
        "WP({:.5f},{:.5f})L({:.4f},{:.4f})"
    ).format(*(v / MDCV_CHROMA_DEN for v in m[:8]),
             *(v / MDCV_LUMA_DEN for v in m[8:]))


def ffmpeg_mastering_display(m):
    """The same as ffmpeg's -mastering_display spells it: the same
    layout in whole steps."""
    return "G({},{})B({},{})R({},{})WP({},{})L({},{})".format(*m)


def content_light_str(c):
    """A probe_hdr_metadata light level as SVT-AV1 and ffmpeg both spell
    it: 'max_cll,max_fall'."""
    return f"{c[0]},{c[1]}"


_SHOWINFO_CONFIG = re.compile(
    r"config in time_base: (\d+)/(\d+), frame_rate: (\d+)/(\d+)"
)
_SHOWINFO_FIRST = re.compile(r"\bn:\s*0\s+pts:\s*(-?\d+|NOPTS)")


def picture_timing(filepath):
    """How ffmpeg itself decodes v:0, read off its first decoded frame.

    Returns {"start", "rate"}:
      start  seconds from the container start to the first frame the
             decoder outputs, on the timeline every ffmpeg run here uses.
             Not the stream's stated start time: a capture whose leading
             frames do not decode starts its picture later than that.
      rate   ffmpeg's own frame-rate guess for the stream (the rate its
             filter graph and every -fps_mode cfr output run at) as
             'num/den', or None when it has none. Not r_frame_rate: an
             interlaced H.264 stream states its field rate there, and
             ffmpeg corrects it to the frame rate from the codec.

    showinfo's config line and first frame line carry both facts, and
    -frames:v 1 stops the decode there. Raises RuntimeError when ffmpeg
    fails or decodes no frame.
    """
    r = run_cmd([
        ffmpeg_exe(), "-hide_banner", "-nostats", "-v", "info",
        "-i", str(filepath), "-map", "0:v:0", "-vf", "showinfo",
        "-frames:v", "1", "-f", "null", "-",
    ])
    err = r.stderr or ""
    config = _SHOWINFO_CONFIG.search(err)
    first = _SHOWINFO_FIRST.search(err)
    if not config or not first:
        raise RuntimeError(f"No decodable video frame in {filepath.name}")
    tb_num, tb_den, fr_num, fr_den = (int(g) for g in config.groups())
    pts = first.group(1)
    start = 0.0
    if pts != "NOPTS" and tb_den:
        start = max(0.0, int(pts) * tb_num / tb_den)
    rate = f"{fr_num}/{fr_den}" if fr_num > 0 and fr_den > 0 else None
    return {"start": start, "rate": rate}


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

    The Y4M pipe's CFR feed resamples every source onto one constant
    rate: irregular frame intervals become dups and drops and the frame
    count changes, where av1q's ffmpeg path
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
