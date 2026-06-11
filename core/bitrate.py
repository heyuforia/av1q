"""Bitrate accounting: video-only kbps measurement and the conversion of
the configured floor into a sample-bitrate threshold."""

import subprocess

from .probe import probe_video


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


def measured_kbps(path, duration, tag):
    """Video-only bitrate for an encode, branching on sample vs full.

    Sample encodes (tag set) are extracted with -an, so the whole file is
    video and calc_kbps on its byte size is already video-only. Full encodes
    carry muxed audio/subs, so the video stream must be isolated with
    video_kbps. Floor comparisons rely on this distinction.
    """
    if tag:
        return calc_kbps(path.stat().st_size, duration)
    return video_kbps(path, duration)


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
