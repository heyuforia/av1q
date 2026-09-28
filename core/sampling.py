"""Sample selection and extraction for the quality-search stage."""

import bisect
import hashlib
import json
import math
import os
import time

from .analyze import span_complexity, window_of
from .constants import (
    MIN_SCENE_DURATION, MINI_SAMPLE_COUNT, MINI_SAMPLE_DURATION,
    MINI_SAMPLE_MIN_RATIO, SAMPLE_COUNT_MAX, SAMPLE_MIN_RATIO, SAMPLE_SCALE_K,
    SAMPLE_SCALE_REF,
)
from .probe import high_bit_depth
from .tools import ffmpeg_exe
from .ui import BOLD, DIM, MIDDOT, RED, RESET, label
from .util import _temp_files, clamp, run_cmd


MINI_PLAN = (MINI_SAMPLE_COUNT, MINI_SAMPLE_DURATION, "mini")

# Each plan's amortization gate: the source must run longer than this
# many times what the plan's clips cut.
_MIN_RATIO = {"standard": SAMPLE_MIN_RATIO, "mini": MINI_SAMPLE_MIN_RATIO}


def sampling_plan(duration, cfg):
    """The largest sampling plan the runtime allows: (count,
    sample_duration, mode) or None. Judged here on the stretches alone,
    before any scan; choose_samples judges it again on the clips as cut
    and may step it down.

    'standard' — the configured plan, when the source is meaningfully
    longer than its planned total (SAMPLE_MIN_RATIO; below that each
    probe encodes nearly the whole file and the final full encode +
    verify come on top). The scene count scales up with duration so long
    features get enough distinct scenes to represent their complexity
    range (see SAMPLE_SCALE_* in constants): flat at the base count up to
    the reference runtime, then +SAMPLE_SCALE_K per doubling, capped at
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
        base * cfg["sample_duration"] * SAMPLE_MIN_RATIO,
    )
    if duration > sampling_min:
        count = int(clamp(
            round(base + SAMPLE_SCALE_K * math.log2(duration / SAMPLE_SCALE_REF)),
            base, SAMPLE_COUNT_MAX,
        ))
        return count, cfg["sample_duration"], "standard"
    if duration > MINI_SAMPLE_COUNT * MINI_SAMPLE_DURATION * MINI_SAMPLE_MIN_RATIO:
        return MINI_PLAN
    return None


def choose_samples(plan, scenes, complexity, keyframes, duration, cfg):
    """The samples a source is searched on, as (sample_scenes, clips,
    plan, even), or None for full-file search. even is True when the
    samples are evenly spaced rather than complexity-selected.

    plan is sampling_plan's, and it only ever steps down: to the mini
    plan, then to none. Each plan is judged on its clips as cut
    (plan_clips), lead-ins included, against the gate sampling_plan
    applied to the stretches alone. Evenly spaced samples start on
    keyframes and add nothing, so what steps a plan down is scene clips
    between sparse keyframes, or a source with too few keyframes to
    space samples on, whose clips all run from the same few and merge.

    A scene list too thin to fill the plan is discarded and the samples
    re-selected evenly spaced, which is representative by construction.
    A plan that even spacing cannot fill either steps down too: a source
    with one keyframe offers a single clip at its start, and no plan
    rides on a single clip when it asked for more.
    """
    # Only the scenes long enough to hold a sample are candidates; a file
    # whose every scene is shorter samples evenly, and must say so, or
    # its representative samples get the scene margin and cohort.
    candidates = sampleable(scenes, cfg)
    plans = [plan, MINI_PLAN] if plan[2] == "standard" else [plan]
    for count, dur, mode in plans:
        if mode == "mini" and plan[2] == "mini":
            print(
                f"{label('short')}{duration:.0f}s source →"
                f" mini-samples ({count}×{dur:.0f}s)"
            )
        # The plan already decided sampling applies, so disarm
        # select_samples' own short-file bail-out and use the plan's clip
        # length (mini plans cut shorter clips).
        select_cfg = {**cfg, "sample_duration": dur, "short_threshold": 0}
        picked = select_samples(
            candidates, complexity, duration, count, keyframes, select_cfg,
        )
        even = not candidates
        # A degenerate scene list can't fill the plan: select_samples
        # picks each distinct scene at most once, so a source with a lone
        # detected cut yields a single clip, and betting the whole search
        # on it is how one near-static scene misreads a high-bitrate
        # source as floor-bound. Too few scene samples → re-select evenly
        # spaced (mirrors the count//2 guard on select_samples' keyframe
        # path; the max(2, ·) stops mini plans from riding on a single
        # clip, min(count, ·) keeps 1-sample plans valid). The even picks
        # must reach it too, or the plan steps down.
        need = min(count, max(2, count // 2))
        if not even and picked and len(picked) < need:
            print(
                f"{label('fallback')}scenes fill only {len(picked)} of"
                f" {count} samples, switching to evenly-spaced"
            )
            picked = select_samples(
                [], complexity, duration, count, keyframes, select_cfg,
            )
            even = True
        picked = picked or []
        clips = plan_clips(picked, keyframes)
        cut = sum(stop - start for start, stop in clips)
        if len(picked) >= need and duration > cut * _MIN_RATIO[mode]:
            info = (
                "evenly-spaced samples" if even
                else f"samples from {BOLD}{len(scenes)}{RESET} scenes"
            )
            print(f"{label('scenes')}{BOLD}{len(picked)}{RESET} {info}")
            return picked, clips, (count, dur, mode), even
        if len(picked) < need:
            why = (
                f"only {len(picked)} of {count} samples can start on a"
                f" distinct keyframe"
            )
        else:
            why = (
                f"{len(clips)} clip{'s' if len(clips) != 1 else ''} cut from"
                f" keyframes would run {cut:.0f}s of the {duration:.0f}s source"
            )
        then = (
            f"switching to mini-samples ({MINI_PLAN[0]}×{MINI_PLAN[1]:.0f}s)"
            if mode == "standard" else "using full VMAF"
        )
        print(f"{label('fallback')}{why}, {then}")
    return None


def sampleable(scenes, cfg):
    """The scenes long enough to hold a sample (MIN_SCENE_DURATION, as
    cfg["min_scene_duration"]). The rest are never candidates."""
    return [sc for sc in scenes if sc["duration"] >= cfg["min_scene_duration"]]


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

    comp_map = {window_of(c["time"]): c["complexity"] for c in complexity}
    # Each scene is ranked by the picture its sample is meant to hold, its
    # first min(scene, sample_dur) seconds. A span with no window entry (a
    # gap in the stream) ranks at the whole-file mean: it neither beats a
    # measurably hotter scene nor loses to a cooler one. Without packet
    # data every scene ties.
    neutral = sum(comp_map.values()) / len(comp_map) if comp_map else 0.0
    scored = []
    for sc in sampleable(scenes, cfg):
        dur = min(sc["duration"], sample_dur)
        value, _ = span_complexity(comp_map, sc["time"], sc["time"] + dur)
        scored.append({
            "time": sc["time"], "duration": dur,
            "complexity": neutral if value is None else value,
        })

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
            selected.append({"time": best["time"], "duration": best["duration"]})
            used.add(best["time"])

    return selected or None


def complexity_bias(complexity, clips):
    """How much hotter the sample is than the whole file.

    The a-priori measure of complexity-selection bias — available before
    any encode, from the packet-stat complexity (complexity_windows) that
    already ranked the scenes: the ratio of the sample's mean complexity
    to the whole-file mean. 1.0 means the sample turned out
    representative after all (the file's hottest scenes are barely above
    its average); above 1.0 means the sample really is the hard part.

    Both halves of the sample→full prediction read it — the bitrate
    margin (complexity_bias_margin) and the VMAF offset center
    (calibrate.scene_offset_center) — because both errors have the same
    single cause. Returns None when the complexity data is missing or
    degenerate, and callers fall back to their fixed cold-start guess.

    complexity is complexity_windows' per-window list; clips is
    plan_clips' output, the picture actually cut, lead-ins included, so
    the reading is of what the probes encode. Each second of it counts
    once, as each second of the concat weighs once in a probe's bitrate.
    """
    if not complexity or not clips:
        return None
    comp_map = {window_of(c["time"]): c["complexity"] for c in complexity}
    all_vals = [
        c["complexity"] for c in complexity
        if isinstance(c.get("complexity"), (int, float)) and c["complexity"] > 0
    ]
    total = secs = 0.0
    for start, stop in clips:
        value, inside = span_complexity(comp_map, start, stop)
        if value is not None:
            total += value * inside
            secs += inside
    if not all_vals or not secs:
        return None
    return (total / secs) / (sum(all_vals) / len(all_vals))


def complexity_bias_margin(complexity, clips, base_margin, floor_margin):
    """Estimate the sample→full bitrate margin from this file's complexity spread.

    The floor search needs to know how much hotter the sampled scenes
    encode than the whole file — samples are cut from the highest-complexity
    scenes, so they run above the whole-file average, and the search must
    clear margin × floor for the video to clear the floor. That bias is
    normally a fixed cold-start guess (base_margin, ~1.20); here it is the
    file's own measured complexity_bias instead.

    Bounded to [floor_margin, base_margin]: the estimate can only TIGHTEN
    the conservative default, never widen it past it, so a noisy proxy can't
    push the search below the floor any harder than the fixed margin already
    might — and the two-sided refine loop backstops whatever it misses.
    Returns base_margin when the complexity data is missing or degenerate.
    """
    bias = complexity_bias(complexity, clips)
    if bias is None:
        return base_margin
    return clamp(bias, floor_margin, base_margin)


def _clock(t):
    """Compact position for the samples line: '7:05', '1:02:33'."""
    m, s = divmod(int(t), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# A keyframe this close after a scene's start is the scene's first frame.
# The scene scan and the packet list state one frame's time from two
# pipelines: older ffmpeg builds print the scan's at six significant
# digits (whole hundredths past 1000s), and a container start offset is
# taken off in stream ticks by ffmpeg and in floats by read_packets. The
# value covers both, and stays under one frame at any rate up to 100fps,
# so it never takes a neighbouring frame for the cut.
SAME_FRAME_SLACK = 0.01


def clip_span(t, dur, keyframes):
    """The picture a sample clip holds, as (start, stop): the `dur`
    seconds from t that select_samples ranked, reached from a keyframe,
    because a stream-copied clip can only begin on one.

    The clip always ends where the ranked stretch ends. It starts on the
    keyframe at the scene's own first frame when there is one
    (SAME_FRAME_SLACK), else on the keyframe before t, with a lead-in of
    the scene before it, or on the first keyframe after t when that one
    is strictly nearer and still leaves MIN_SCENE_DURATION of the stretch
    (all of it when the stretch is shorter). A clip timed from its
    keyframe instead can hold none of its scene: a 2.2s scene 2.2s past
    a keyframe, with the next one 5.2s on, is cut wholly from the scene
    before it.

    Nothing before a file's first keyframe can be cut, so a stretch that
    starts before it runs its `dur` from that keyframe. Without a keyframe
    list (the packet scan failed) the span is the stretch itself, and the
    cut takes whatever pre-roll its seek lands on, which the amortization
    gate in choose_samples cannot see. That run only: a failed scan is
    never stored, so the next run scans again.
    """
    if not keyframes:
        return t, t + dur
    stop = t + dur
    i = bisect.bisect_right(keyframes, t)
    if i == 0:
        return keyframes[0], keyframes[0] + dur
    before = keyframes[i - 1]
    if i < len(keyframes):
        after = keyframes[i]
        if after - t <= SAME_FRAME_SLACK or (
                after - t < t - before
                and stop - after >= min(dur, MIN_SCENE_DURATION)):
            return after, stop
    return before, stop


def plan_clips(scenes, keyframes):
    """Every selected scene's clip_span, in time order, with clips that
    overlap merged into one: a stretch cut twice would weigh double in
    every probe. The one plan both the cut (extract_samples) and the
    bias reading (complexity_bias) are made from."""
    merged = []
    for start, stop in sorted(
        clip_span(sc["time"], sc["duration"], keyframes) for sc in scenes
    ):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


# How far ahead of the gate the input seek aims (seconds). Landing early
# only demuxes a little more that the gate then drops; the value merely
# has to keep the output -ss clearly positive.
PRE_SEEK = 1.0


def cut_window(start, stop, keyframes):
    """Where ffmpeg seeks, gates and stops to stream-copy the picture in
    [start, stop), start being a keyframe (clip_span): (seek, gate,
    length). seek is the input -ss, gate the output -ss relative to it
    (None for none), and length the -t.

    A stream-copied clip can only begin on a keyframe, and one input -ss
    cannot be trusted to land on the keyframe chosen: each demuxer
    resolves the seek its own way (Matroska takes the last cue at or
    before it, MP4 the last keyframe by presentation time, MPEG-TS the
    last packet by decode time with no keyframe walk-back), and ffmpeg
    itself moves the request 130ms earlier on every container that does
    not declare presentation-time seeking (Matroska and MPEG-TS do not,
    MP4 does) whenever the stream has B-frames. On an MKV that is one
    cue early, so every clip carried a whole GOP of the previous scene.

    So the cut is two steps. The input seek aims halfway back to the
    previous keyframe, which lands at or before that keyframe on every
    container, and an output -ss at that same point drops what was read
    before it. Stream copy discards leading non-keyframes, so the first
    packet kept is exactly the keyframe chosen: the previous keyframe
    sits before the gate and is dropped, the chosen one sits after it by
    half a GOP, more than any reorder delay pulls its decode time back.
    -t counts from the gate, so it is stretched by the same half GOP.
    """
    if not keyframes:
        # No keyframe list (the packet scan failed): seek at the scene
        # time and accept whatever pre-roll the landing brings.
        return math.ceil(start * 1000 + 1e-3) / 1000, None, stop - start
    j = bisect.bisect_left(keyframes, start)
    if j == 0:
        # The first keyframe is the first packet: reading from the start
        # lands on it in every container, and nothing precedes it.
        return 0.0, None, stop
    mid = (keyframes[j - 1] + start) / 2
    seek = max(0.0, math.floor((mid - PRE_SEEK) * 1000) / 1000)
    gate = math.floor((mid - seek) * 1000) / 1000
    return seek, gate, stop - (seek + gate)


def extract_samples(source, scenes, keyframes, cfg, file_hash=None):
    """Cut each selected scene's clip (plan_clips) by stream copy and join
    the clips into one video-only concat under _cache/_samples/. Returns
    the concat, or None when no clip could be cut.

    Only a complete set is kept for the next run. When a clip fails, the
    search still runs on the clips that were cut, but that set is a temp
    under a per-run name, so the next run cuts every clip again instead
    of inheriting the gap.
    """
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

    spans = plan_clips(scenes, keyframes)
    where = (
        f"{len(spans)} clip{'s' if len(spans) != 1 else ''} at"
        f" {' '.join(_clock(start) for start, _ in spans)}"
    )

    if concat_out.exists() and concat_out.stat().st_size > 0:
        print(
            f"{'':>11}{DIM}Samples: {concat_out.stat().st_size / 1e6:.1f}MB"
            f" (cached) {MIDDOT} {where}{RESET}"
        )
        return concat_out

    ts = int(time.time() * 1000)
    attempted, clips = [], []
    for i, (start, stop) in enumerate(spans):
        seek, gate, length = cut_window(start, stop, keyframes)
        clip = sample_dir / f"sample_{ts}_{i}.mkv"
        attempted.append(clip)
        _temp_files.add(clip)
        try:
            # Map the video stream explicitly: default stream selection
            # would also pick a subtitle stream (mov_text from MP4 fails
            # MKV stream copy outright) and picks the "best" video stream
            # rather than the first — while probe/encode/VMAF all use
            # v:0. Samples are video-only by contract: the sample path's
            # kbps math reads their whole byte size as video.
            run_cmd([
                ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
                "-ss", f"{seek:.3f}", "-i", str(source),
                *(["-ss", f"{gate:.3f}"] if gate is not None else []),
                "-t", f"{length:.3f}",
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

    # Bare names, resolved beside the list (as in core.segments): an
    # absolute path would put the cache folder's own name inside the
    # list's quoting, where one apostrophe ends the entry.
    concat_list = sample_dir / f"concat_{ts}.txt"
    _temp_files.add(concat_list)
    concat_list.write_text(
        "\n".join(f"file '{c.name}'" for c in clips), encoding="utf-8"
    )

    complete = len(clips) == len(spans)
    out = concat_out if complete else sample_dir / (
        f"samples_{tag}_{scene_sig}_part{ts}.mkv"
    )
    # Written under a temp name and renamed when whole, so an
    # interrupted concat never sits under the name a later run reuses.
    tmp = out.with_suffix(".tmp.mkv")
    _temp_files.add(tmp)
    try:
        run_cmd([
            ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
            "-f", "concat", "-i", str(concat_list),
            "-c", "copy", str(tmp),
        ])
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError("empty output")
        tmp.replace(out)
    except (RuntimeError, OSError) as e:
        print(f" {RED}Concat error: {e}{RESET}")
        return None
    finally:
        for p in (*attempted, concat_list, tmp):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                continue
            _temp_files.discard(p)

    print(
        f"{'':>11}{DIM}Samples: {out.stat().st_size / 1e6:.1f}MB"
        f" {MIDDOT} {where}{RESET}"
    )
    if not complete:
        _temp_files.add(out)
        print(
            f"{'':>11}{DIM}{len(clips)} of {len(spans)} clips cut; this"
            f" set is not kept, the next run cuts it again{RESET}"
        )
    return out


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
        if meta["hdr"] or high_bit_depth(meta["pix_fmt"])
        else "yuv420p"
    )
    cmd = [
        ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
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
        # A clean pass over a temp concat (a partial clip set, named per
        # run) is itself a temp: no later run can ever reuse it.
        if concat in _temp_files:
            _temp_files.add(clean)
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
