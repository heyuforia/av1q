"""Bitrate accounting: video-only kbps measurement and the conversion of
the configured floor into a sample-bitrate threshold."""

import subprocess
from pathlib import Path

from .calibrate import RATIO_MAX, RATIO_MIN
from .tools import ffprobe_exe
from .ui import DIM, RESET, label
from .util import scan_budget


def calc_kbps(size_bytes, duration):
    if duration < 1.0:
        return None
    return int((size_bytes * 8) / 1000 / duration)


def video_kbps(filepath, duration):
    """Video-only bitrate by summing video packet sizes, or None.

    File-size / duration counts muxed audio + subs, which breaks floor
    comparisons against sample bitrates (samples are -an video-only).
    A failed read prints its reason: None here switches the floor checks
    off for this encode, which must not happen silently.
    """
    if not duration or duration < 1.0:
        return None
    timeout = scan_budget(duration)
    why = None
    try:
        r = subprocess.run(
            [ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=size", "-of", "csv=p=0", str(filepath)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", timeout=timeout,
        )
        if r.returncode != 0:
            tail = (r.stderr or "").strip().splitlines()
            why = f"exit {r.returncode}" + (f": {tail[-1]}" if tail else "")
        else:
            total = sum(int(l) for l in r.stdout.splitlines() if l.strip())
            if total > 0:
                return int(total * 8 / 1000 / duration)
            why = "no video packets"
    except subprocess.TimeoutExpired:
        why = f"timed out after {timeout}s"
    except (OSError, ValueError) as e:
        why = str(e)
    print(
        f"{label('bitrate')}{DIM}video bitrate of {Path(filepath).name}"
        f" unreadable ({why}){RESET}"
    )
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


def effective_sample_floor(min_kbps, margin, calibration=None, ratio_prior=None):
    """Sample-bitrate threshold that predicts full video clears min_kbps.

    Samples are cut from max-complexity scenes, so they usually encode at
    a higher bitrate than the full video at the same quantizer (a
    measured ratio slightly above 1 — sample cooler than the file — is
    also valid; see RATIO_MIN/RATIO_MAX in core/calibrate.py). Sources of
    the sample→full ratio, in order of trust:

      1. a measured per-file ratio (this exact file under these settings,
         after one full encode; the caller hands over that block only)
      2. the cohort ratio prior (learned across files — see ratio_prior in
         core/calibrate.py; this is what lets a fresh file skip the
         conservative-margin tax once a few similar files have been seen)
      3. the cold-start margin

    The first two convert the floor by dividing; the margin multiplies.
    """
    if calibration:
        r = calibration.get("ratio")
        if isinstance(r, (int, float)) and RATIO_MIN <= r <= RATIO_MAX:
            return min_kbps / r
    if isinstance(ratio_prior, (int, float)) and RATIO_MIN <= ratio_prior <= RATIO_MAX:
        return min_kbps / ratio_prior
    return min_kbps * margin
