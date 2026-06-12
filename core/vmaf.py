"""VMAF measurement via ffmpeg's libvmaf filter, plus the shared
file-cached wrapper both pipelines build on."""

import json
import math
import subprocess
import time

from .probe import detect_hwaccel, get_fps
from .ui import RED, RESET
from .util import _temp_files, atomic_write_json, make_temp_log, run_cmd

# Codec profiles that must never go through a hardware decoder here.
# ffmpeg's hwaccel capability check only asks the driver about codec +
# chroma + bit depth, never the profile — so a stream using profile
# tools the silicon doesn't implement can be CLAIMED as supported and
# decode to garbage with exit code 0, silently corrupting the scores.
# Concretely: x264 lossless tags High 4:4:4 Predictive even for plain
# 4:2:0 content (so the caps query passes everywhere), and its
# transform-bypass mode mis-decoded on RTX 5090/Blackwell + ffmpeg
# master — collapsing sample VMAF to ~71/P5 12 (issue #4) while older
# GPUs whose caps query fails fell back to software and scored
# correctly. Software decode is bit-exact by definition; for these
# profiles correctness beats decode speed.
_HW_UNSAFE_PROFILES = ("4:4:4", "4:2:2", "444", "422", "rext")


def _sw_decode_only(path):
    """True when this stream's profile is on the hw-unsafe list.
    Unknown/unreadable profiles return False (mainstream profiles are
    the overwhelmingly common case, and a hw decode that errors out
    still falls back to the software attempt)."""
    try:
        r = run_cmd([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=profile",
            "-of", "default=nw=1:nk=1", str(path),
        ])
        prof = (r.stdout or "").strip().lower()
    except RuntimeError:
        return False
    return any(t in prof for t in _HW_UNSAFE_PROFILES)


def measure_vmaf(ref, dist, meta, subsample, threads, cache_dir):
    """Compute VMAF score between reference and distorted video.

    meta["vmaf_pair"] == "index" pairs frames by INDEX instead of by
    timestamp: both chains get identical synthetic timestamps so
    framesync matches frame n with frame n — the same pairing FFVship
    uses. The sample path sets this: its pair is frame-aligned by
    construction (the clean source is CFR and feeds the encoder 1:1),
    while its container timestamps inherit quirks from stream-copied
    cuts of irregular sources — which the timestamp/fps chain turned
    into one-sided dup/drops, collapsing sample VMAF (issue #4: flat
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
            unk = {"", "unknown", "unspecified", "reserved"}
            tags = []
            if meta["ct"] in unk:
                tags.append("color_trc=smpte2084")
            if meta["cp"] in unk:
                tags.append("color_primaries=bt2020")
            if meta["cs"] in unk:
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
    log_esc = log.as_posix().replace("\\", "/").replace("'", "\\'").replace(":", "\\:")

    try:
        hw = detect_hwaccel()
        if hw and (_sw_decode_only(ref) or _sw_decode_only(dist)):
            hw = None
        attempts = [hw, None] if hw else [None]

        for accel in attempts:
            cmd = ["ffmpeg", "-v", "error", "-hide_banner"]
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
                f"log_fmt=json:log_path='{log_esc}'",
                "-f", "null", "-",
            ]
            r = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
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

    except RuntimeError as e:
        print(f" {RED}VMAF error: {e}{RESET}")
        try:
            if log.exists():
                log.unlink()
        except OSError:
            pass
        _temp_files.discard(log)
        return {"mean": float("nan"), "p5": float("nan")}


def vmaf_cached(ref, dist, meta, q, cache, cache_path, *, tag=None,
                threads, log_dir, key_base, q_key, measure=None):
    """Compute VMAF with file-based caching — shared by both pipelines'
    exact-signature wrappers.

    The two cache layouts deliberately differ and are FROZEN:
      av1q       entries[str(cq)]     value keys 'full' / 'sample_full'
      essential  entries[crf_str(q)]  value keys 'vmaf' / 'sample_vmaf'
    Key separation is what keeps essential's SSIMU2-era entries from ever
    being misread as VMAF — the sig never changes by policy, so these key
    names and `q_key` formats must never change either.

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
    key = f"{tag}_{key_base}" if tag else key_base
    entry = cache["entries"].get(q_key)

    if entry and key in entry and entry.get("size") == dist_size:
        return {
            "mean": float(entry[key]),
            "p5": float(entry.get(f"{key}_p5", entry[key])),
        }

    result = measure(ref, dist, meta, 1, threads, log_dir)

    if math.isfinite(result["mean"]) and 0 <= result["mean"] <= 100:
        cache["entries"].setdefault(q_key, {}).update({
            key: result["mean"], f"{key}_p5": result["p5"],
            "size": dist_size,
            "t": time.time(),
        })
        atomic_write_json(cache_path, cache)

    return result
