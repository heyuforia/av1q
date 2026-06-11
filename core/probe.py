"""Source inspection: ffprobe metadata, frame rate, the cached
hardware-decode probe, and resolution tiering."""

import json
import platform
import subprocess

from .util import run_cmd

_hwaccel = None
_hwaccel_checked = False


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
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_frames", "-read_intervals", "%+#1",
             "-show_entries", "frame=side_data_list",
             "-of", "json", str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120,
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


def is_vfr(filepath, meta):
    """True when the source is genuinely variable-frame-rate.

    Y4M is CFR-only, so piping a VFR source would silently desync audio
    AND misalign FFVship's frame pairing (it decodes source and encode
    independently). Header r_frame_rate vs avg_frame_rate mismatch alone
    is full of false positives, so a mismatch is confirmed by counting
    real packets against duration x avg_fps before rejecting a file.
    """
    try:
        r = run_cmd([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate",
            "-of", "json", str(filepath),
        ])
        s = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]

        def _fps(v):
            if not v or v in ("0/0", "N/A"):
                return None
            if "/" in v:
                a, b = v.split("/", 1)
                return float(a) / float(b) if float(b) else None
            return float(v)

        rf, af = _fps(s.get("r_frame_rate")), _fps(s.get("avg_frame_rate"))
        if not rf or not af or abs(rf - af) / af <= 0.01:
            return False

        duration = meta.get("duration") or 0
        if duration <= 1:
            return False
        c = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "default=nw=1:nk=1", str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=600,
        )
        if c.returncode != 0:
            return True  # suspicious header and uncountable: don't risk it
        n = int(c.stdout.strip() or 0)
        expected = duration * af
        return expected > 0 and abs(n - expected) / expected > 0.005
    except Exception:
        return False
