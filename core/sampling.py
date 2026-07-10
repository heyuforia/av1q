"""Sample selection and extraction for the quality-search stage."""

import hashlib
import json
import math
import os
import time

from .analyze import get_keyframes
from .constants import (
    MINI_SAMPLE_COUNT, MINI_SAMPLE_DURATION, MINI_SAMPLE_MIN_RATIO,
    SAMPLE_COUNT_MAX, SAMPLE_SCALE_K, SAMPLE_SCALE_REF,
)
from .ui import DIM, RED, RESET
from .util import _temp_files, clamp, run_cmd


def sampling_plan(duration, cfg):
    """Per-file sampling plan: (count, sample_duration, mode) or None.

    'standard' — the configured plan, when the source is meaningfully
    longer than its extracted total (1.25×; below that each probe encodes
    nearly the whole file and the final full encode + verify come on top).
    The scene count scales up with duration so long features get enough
    distinct scenes to represent their complexity range (see
    SAMPLE_SCALE_* in constants): flat at the base count up to the
    reference runtime, then +SAMPLE_SCALE_K per doubling, capped at
    SAMPLE_COUNT_MAX. Clip length stays fixed. 'mini' — a scaled-down
    plan for short files that used to fall through to full-file search,
    where every probe is a full encode: a few tiny probes cost a fraction
    of one full encode and seed the search just as well. None —
    ultra-short sources where even mini probes wouldn't amortize the
    sample path's fixed cost; full-file search is cheaper.
    """
    base = cfg["sample_count"]
    sampling_min = max(
        cfg["short_threshold"],
        base * cfg["sample_duration"] * 1.25,
    )
    if duration > sampling_min:
        count = int(clamp(
            round(base + SAMPLE_SCALE_K * math.log2(duration / SAMPLE_SCALE_REF)),
            base, SAMPLE_COUNT_MAX,
        ))
        return count, cfg["sample_duration"], "standard"
    if duration > MINI_SAMPLE_COUNT * MINI_SAMPLE_DURATION * MINI_SAMPLE_MIN_RATIO:
        return MINI_SAMPLE_COUNT, MINI_SAMPLE_DURATION, "mini"
    return None


def select_samples(scenes, complexity, duration, count, keyframes, cfg):
    """Select representative sample segments for quality estimation."""
    if duration < cfg["short_threshold"]:
        return None

    sample_dur = cfg["sample_duration"]

    if not scenes:
        if not keyframes:
            return [
                {"time": duration * (i + 1) / (count + 1), "duration": sample_dur}
                for i in range(count)
            ]

        seg = duration / count
        selected = []
        for i in range(count):
            start, end = i * seg, (i + 1) * seg
            cands = [k for k in keyframes if start <= k < end]
            best = min(cands or keyframes, key=lambda k: abs(k - (start + end) / 2))
            if best not in [s["time"] for s in selected]:
                selected.append({"time": best, "duration": sample_dur})

        if len(selected) >= count // 2:
            return selected
        return [
            {"time": duration * (i + 1) / (count + 1), "duration": sample_dur}
            for i in range(count)
        ]

    comp_map = {int(c["time"] / 5) * 5: c["complexity"] for c in complexity}
    scored = [
        {
            "time": sc["time"],
            "duration": sc["duration"],
            "complexity": comp_map.get(int(sc["time"] / 5) * 5, 50),
        }
        for sc in scenes
        if sc["duration"] >= cfg["min_scene_duration"]
    ]

    if not scored:
        return select_samples([], complexity, duration, count, keyframes, cfg)

    seg = duration / count
    selected = []
    used = set()
    for i in range(count):
        start, end = i * seg, (i + 1) * seg
        cands = (
            [s for s in scored if start <= s["time"] < end and s["time"] not in used]
            or [s for s in scored if s["time"] not in used]
        )
        if cands:
            best = max(cands, key=lambda x: x["complexity"])
            selected.append({
                "time": best["time"],
                "duration": min(best["duration"], sample_dur),
            })
            used.add(best["time"])

    return selected or None


def complexity_bias_margin(complexity, sample_scenes, base_margin, floor_margin):
    """Estimate the sample→full bitrate margin from this file's complexity spread.

    The floor search needs to know how much hotter the sampled scenes
    encode than the whole file — samples are cut from the highest-complexity
    scenes, so they run above the full-file average, and the search must
    clear margin × floor for the video to clear the floor. That bias is
    normally a fixed cold-start guess (base_margin, ~1.20). Here it is
    estimated a priori — before any encode — from the packet-stat
    complexity (analyze_complexity) that already ranked the scenes: the
    ratio of the selected scenes' mean complexity to the whole-file mean.
    Low-variance content (its hottest scenes barely above average) yields a
    margin near 1.0; high-variance content keeps the full margin.

    Bounded to [floor_margin, base_margin]: the estimate can only TIGHTEN
    the conservative default, never widen it past it, so a noisy proxy can't
    push the search below the floor any harder than the fixed margin already
    might — and the two-sided refine loop backstops whatever it misses.
    Returns base_margin when the complexity data is missing or degenerate.

    complexity is analyze_complexity's per-5s-window list; sample_scenes is
    select_samples' output (only `time`/`duration`), so the selected scenes
    are mapped back to their windows the same way select_samples does.
    """
    if not complexity or not sample_scenes:
        return base_margin
    comp_map = {int(c["time"] / 5) * 5: c["complexity"] for c in complexity}
    all_vals = [
        c["complexity"] for c in complexity
        if isinstance(c.get("complexity"), (int, float)) and c["complexity"] > 0
    ]
    sel_vals = []
    for s in sample_scenes:
        v = comp_map.get(int(s["time"] / 5) * 5)
        if isinstance(v, (int, float)) and v > 0:
            sel_vals.append(v)
    if not all_vals or not sel_vals:
        return base_margin
    mean_all = sum(all_vals) / len(all_vals)
    mean_sel = sum(sel_vals) / len(sel_vals)
    if mean_all <= 0:
        return base_margin
    return clamp(mean_sel / mean_all, floor_margin, base_margin)


def extract_samples(source, scenes, keyframes, cfg, file_hash=None):
    """Extract and concatenate sample clips from the source video."""
    if not scenes:
        return None

    sample_dir = cfg["cache_dir"] / "_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    tag = file_hash or f"{os.getpid()}_{int(time.time() * 1000)}"
    # Key the cached concat by the selected scenes too — they change with
    # --samples / sample_duration / scene settings, and a file-hash-only key
    # would silently reuse a concat cut with the old parameters.
    scene_sig = hashlib.sha256(json.dumps(
        [[round(sc["time"], 3), round(sc["duration"], 3)] for sc in scenes]
    ).encode("utf-8")).hexdigest()[:10]
    concat_out = sample_dir / f"samples_{tag}_{scene_sig}.mkv"
    if concat_out.exists() and concat_out.stat().st_size > 0:
        print(f"{'':>11}{DIM}Samples: {concat_out.stat().st_size / 1e6:.1f}MB (cached){RESET}")
        return concat_out

    ts = int(time.time() * 1000)
    clips = []
    if keyframes is None:
        keyframes = get_keyframes(source)

    for i, sc in enumerate(scenes):
        clip = sample_dir / f"sample_{ts}_{i}.mkv"
        _temp_files.add(clip)
        try:
            start = (
                min(keyframes, key=lambda k: abs(k - sc["time"]))
                if keyframes else sc["time"]
            )
            # Map the video stream explicitly: default stream selection
            # would also pick a subtitle stream (mov_text from MP4 fails
            # MKV stream copy outright) and picks the "best" video stream
            # rather than the first — while probe/encode/VMAF all use
            # v:0. Samples are video-only by contract (see -an in the
            # bitrate accounting notes).
            run_cmd([
                "ffmpeg", "-y", "-hide_banner", "-v", "error",
                "-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{sc['duration']:.3f}",
                "-map", "0:v:0",
                "-c", "copy", "-an", "-avoid_negative_ts", "make_zero",
                str(clip),
            ])
            if clip.exists() and clip.stat().st_size > 0:
                clips.append(clip)
        except RuntimeError as e:
            print(f" {RED}Clip error: {e}{RESET}")

    if not clips:
        print(f" {RED}No clips extracted{RESET}")
        return None

    concat_list = sample_dir / f"concat_{ts}.txt"
    _temp_files.add(concat_list)
    concat_list.write_text(
        "\n".join(f"file '{c.as_posix()}'" for c in clips), encoding="utf-8"
    )

    try:
        run_cmd([
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", str(concat_out),
        ])
        for c in clips:
            try:
                c.unlink()
                _temp_files.discard(c)
            except OSError:
                pass
        concat_list.unlink()
        _temp_files.discard(concat_list)

        if concat_out.exists():
            print(f"{'':>11}{DIM}Samples: {concat_out.stat().st_size / 1e6:.1f}MB{RESET}")
            return concat_out
    except RuntimeError as e:
        print(f" {RED}Concat error: {e}{RESET}")
    return None


def clean_sample_source(concat, meta, cfg):
    """Re-encode the stream-copied sample concat into a continuous,
    losslessly-coded CFR file. Returns the clean path, or None on failure.

    The raw concat has timestamp seams at clip boundaries (stream-copy
    cuts), and the two consumers of the sample disagree on them: ffmpeg's
    CFR Y4M feed duplicates frames at the seams while FFVship's FFMS2
    index does not — measured 745 vs 742 frames on a 2-clip concat, which
    misaligns every frame after the first seam and collapses SSIMU2 to
    garbage. One lossless pass (x264 qp0 is mathematically lossless)
    gives both consumers the exact same frame sequence. av1q never needs
    this because VMAF decodes both sides through a single ffmpeg process.
    """
    clean = concat.with_name(concat.stem + "_clean.mkv")
    if clean.exists() and clean.stat().st_size > 0:
        print(f"{'':>11}{DIM}Clean samples: {clean.stat().st_size / 1e6:.1f}MB (cached){RESET}")
        return clean

    tmp = clean.with_suffix(".tmp.mkv")
    _temp_files.add(tmp)
    pix = (
        "yuv420p10le"
        if meta["hdr"] or "10le" in meta["pix_fmt"]
        else "yuv420p"
    )
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-v", "error",
        "-i", str(concat), "-map", "0:v:0",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-qp", "0",
        "-pix_fmt", pix,
    ]
    # FFVship reads colorspace from the file, so HDR tags must survive
    # the lossless re-encode for PQ content to be interpreted correctly.
    if meta.get("cp") and meta.get("ct"):
        cmd += ["-color_primaries", meta["cp"], "-color_trc", meta["ct"]]
        if meta.get("cs"):
            cmd += ["-colorspace", meta["cs"]]
    if meta.get("cr"):
        cmd += ["-color_range", meta["cr"]]
    cmd.append(str(tmp))

    try:
        run_cmd(cmd)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError("empty output")
        if clean.exists():
            clean.unlink()
        tmp.rename(clean)
        _temp_files.discard(tmp)
        print(f"{'':>11}{DIM}Clean samples: {clean.stat().st_size / 1e6:.1f}MB{RESET}")
        return clean
    except (RuntimeError, OSError) as e:
        print(f" {RED}Sample clean-encode error: {e}{RESET}")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        _temp_files.discard(tmp)
        return None
