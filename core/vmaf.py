"""VMAF measurement via ffmpeg's libvmaf filter, plus the shared
file-cached wrapper both pipelines build on."""

import json
import math
import subprocess
import time

from .probe import UNTAGGED, detect_hwaccel, get_fps, hw_decode_unsafe
from .tools import ffmpeg_exe, ffprobe_exe
from .ui import RED, RESET
from .util import (
    _temp_files, atomic_write_json, escape_filter_path, make_temp_log, run_cmd,
)


def _sw_decode_only(path):
    """True when this stream's profile is on the hw-unsafe list (see
    probe.HW_UNSAFE_PROFILES): x264's lossless clean-sample reference
    collapsed sample VMAF to ~71 through a Blackwell decode that
    claimed the profile, while software decode is bit-exact. An
    unreadable profile is False, like an unknown one."""
    try:
        r = run_cmd([
            ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=profile",
            "-of", "default=nw=1:nk=1", str(path),
        ])
    except RuntimeError:
        return False
    return hw_decode_unsafe((r.stdout or "").strip())


def measure_vmaf(ref, dist, meta, subsample, threads, cache_dir):
    """Compute VMAF score between reference and distorted video.

    meta["vmaf_pair"] == "index" pairs frames by INDEX instead of by
    timestamp: both chains get identical synthetic timestamps so
    framesync matches frame n with frame n — the same pairing FFVship
    uses. vmaf_cached sets this for every sample measurement: the pair
    is frame-aligned by construction (the search source feeds the
    encoder 1:1), while its container timestamps inherit quirks from
    stream-copied cuts of irregular sources — which the timestamp/fps
    chain turned into one-sided dup/drops, collapsing sample VMAF (flat
    ~76 / P5 ~2 against a sane SSIMU2, even in software decode). The
    full path must keep timestamp pairing: there the encode-side CFR
    conversion changes the frame count, and mirroring that conversion
    on the reference is exactly what keeps full VMAF aligned.
    """
    index_pair = meta.get("vmaf_pair") == "index"
    fps = None if index_pair else get_fps(dist)

    def build_chain(with_crop):
        # Crop only applies to the ref chain: dist was already encoded with
        # crop applied, so its frames are pre-cropped. Cropping it again here
        # would shave off real picture content and silently tank VMAF.
        if index_pair:
            # Identical re-stamp on both chains -> framesync pairs 1:1
            # by index. The rate constant is arbitrary (VMAF scores
            # per frame; nothing is temporally resampled).
            f = ["settb=AVTB", "setpts=N/(25*TB)"]
        else:
            f = ["setpts=PTS-STARTPTS"]  # normalize MP4 edit lists
            if fps:
                f.append(f"fps={fps}")
        if with_crop and meta.get("crop"):
            f.append(f"crop={meta['crop']}")
        if meta["hdr"]:
            # Gate on HDR signaling, not bit depth: SDR 10-bit needs no
            # tonemap, and untagged SDR (common in screen-recording ProRes)
            # would make zscale fail with "no path between colorspaces".
            #
            # zscale must know the input transfer/primaries/matrix to
            # linearize, so fill in only the tags the stream is missing
            # with HDR defaults before the conversion.
            tags = []
            if meta["ct"] in UNTAGGED:
                tags.append("color_trc=smpte2084")
            if meta["cp"] in UNTAGGED:
                tags.append("color_primaries=bt2020")
            if meta["cs"] in UNTAGGED:
                tags.append("colorspace=bt2020nc")
            if tags:
                f.append("setparams=" + ":".join(tags))
            # tonemap expects linear-light RGB input: linearize first
            # (float RGB), convert primaries in linear space, tonemap, then
            # convert transfer/matrix/range back to SDR bt709.
            f += [
                "zscale=t=linear:npl=100",
                "format=gbrpf32le",
                "zscale=p=bt709",
                "tonemap=hable:desat=0",
                "zscale=t=bt709:m=bt709:r=tv",
            ]
        f.append("format=yuv420p")
        return ",".join(f)

    pf_ref = build_chain(with_crop=True)
    pf_dist = build_chain(with_crop=False)
    log = make_temp_log(cache_dir, "vmaf", "json")

    th = f":n_threads={threads}" if threads > 1 else ""
    model = "vmaf_4k_v0.6.1" if meta["h"] >= 2160 else "vmaf_v0.6.1"

    try:
        hw = detect_hwaccel()
        # Probed from each file, never read from meta: on the sample path
        # ref is the search source (essential's is the x264 clean sample),
        # not the file meta describes.
        if hw and (_sw_decode_only(ref) or _sw_decode_only(dist)):
            hw = None
        attempts = [hw, None] if hw else [None]

        for accel in attempts:
            cmd = [ffmpeg_exe(), "-v", "error", "-hide_banner"]
            if accel:
                cmd += ["-hwaccel", accel]
            cmd += ["-i", str(ref)]
            if accel:
                cmd += ["-hwaccel", accel]
            cmd += ["-i", str(dist)]
            cmd += [
                "-filter_complex",
                f"[0:v]{pf_ref}[r];[1:v]{pf_dist}[d];"
                f"[d][r]libvmaf=model=version={model}:"
                f"n_subsample={subsample}{th}:"
                f"log_fmt=json:log_path={escape_filter_path(log)}",
                "-f", "null", "-",
            ]
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace",
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
            "mean": float(mean) if mean is not None else float("nan"),
            "p5": float(p5) if p5 is not None else float("nan"),
        }

    except (RuntimeError, OSError, ValueError) as e:
        # OSError and ValueError: an exit-0 run whose log is missing or
        # torn is the same failure, not a reason to fail the whole file.
        print(f" {RED}VMAF error: {e}{RESET}")
        try:
            if log.exists():
                log.unlink()
        except OSError:
            pass
        _temp_files.discard(log)
        return {"mean": float("nan"), "p5": float("nan")}


def stored_vmaf(entry, key, size):
    """The score cached under `key` ({'mean','p5'}) when it measured an
    encode of this size, else None.

    The sample and the full encode at one quantizer share an entry, so
    each score records its own encode's size: one shared size, rewritten
    by whichever was measured last, would send the other through a fresh
    measurement, a whole-file VMAF when a new search lands on a verified
    quantizer. An entry without per-score sizes holds one shared `size`,
    which vouches only for the score measured last.
    """
    if not isinstance(entry, dict) or key not in entry:
        return None
    if entry.get(f"{key}_size", entry.get("size")) != size:
        return None
    return {
        "mean": float(entry[key]),
        "p5": float(entry.get(f"{key}_p5", entry[key])),
    }


def vmaf_cached(ref, dist, meta, q, cache, cache_path, *, tag=None,
                threads, log_dir, key_base, q_key, measure=None):
    """Compute VMAF with file-based caching — shared by both pipelines'
    exact-signature wrappers.

    The two cache layouts deliberately differ and are FROZEN:
      av1q       entries[str(cq)]     value keys 'full' / 'sample_full'
      essential  entries[crf_str(q)]  value keys 'vmaf' / 'sample_vmaf'
    Each value key carries '<key>_p5' and '<key>_size' beside it (see
    stored_vmaf).
    Key separation is what keeps essential's SSIMU2-era entries from ever
    being misread as VMAF — the sig never changes by policy, so these key
    names and `q_key` formats must never change either.

    A tagged measurement is a sample probe, and sample pairs are matched
    by frame index (see measure_vmaf); untagged full-file pairs keep
    timestamp pairing.

    `measure` defaults to this module's measure_vmaf; the wrappers inject
    a late-binding closure so their module-level monkeypatch seam stays
    intact for the tests.
    """
    if not dist.exists() or not ref.exists():
        return {"mean": float("nan"), "p5": float("nan")}
    try:
        dist_size = dist.stat().st_size
    except OSError:
        return {"mean": float("nan"), "p5": float("nan")}

    if measure is None:
        measure = measure_vmaf
    if tag:
        meta = {**meta, "vmaf_pair": "index"}
    key = f"{tag}_{key_base}" if tag else key_base
    hit = stored_vmaf(cache["entries"].get(q_key), key, dist_size)
    if hit:
        return hit

    result = measure(ref, dist, meta, 1, threads, log_dir)

    if math.isfinite(result["mean"]) and 0 <= result["mean"] <= 100:
        cache["entries"].setdefault(q_key, {}).update({
            key: result["mean"], f"{key}_p5": result["p5"],
            f"{key}_size": dist_size,
            "t": time.time(),
        })
        atomic_write_json(cache_path, cache)

    return result
