"""Mainline SVT-AV1 engine: av1q's encode path, through ffmpeg's
libsvtav1 wrapper. The picture is encoded first, crop, color and
HDR10 static metadata applied; the shared source-stream mux in
core.segments then adds audio, subtitles, fonts and chapters."""

import collections
import math
import shutil
import subprocess
import sys
import threading
import time

from .. import segments, ssimu2
from ..constants import (
    FALLBACK_MAXRATE, MAXRATE_FACTOR, RESUMABLE_MIN_DURATION, SEGMENT_TIME,
)
from ..crop import crop_token
from ..probe import (
    content_light_str, high_bit_depth, res_tier, svt_mastering_display,
)
from ..tools import ffmpeg_exe, find_ffvship_optional
from ..ui import BOLD, DIM, GREEN, RESET, fmt_time, label
from ..util import _temp_files, fmt_cmd, partial_hash, run_cmd
from .base import Engine, Grid

# SVT-AV1 refuses a max bitrate above 100000 kbps and fails the encode at
# init, so the peak cap is held to it.
SVT_MAX_BITRATE = 100_000_000


def enc_signature(cfg, crop=None):
    """Tag covering everything that changes encoder output for one source
    at a given CQ: preset, film grain, and crop. Used in cache keys and
    sample-encode filenames so stale variants are never reused.
    """
    return f"p{cfg['preset']}g{cfg['film_grain']}{crop_token(crop)}"


def _run_ffmpeg_progress(cmd, duration, prefix, base_time=0.0):
    """Run ffmpeg with -progress pipe:1 and render an inline progress bar
    after `prefix` (the stage label and quantizer).

    Parses key=value blocks on stdout; uses out_time_us against the known
    source duration so the bar stays accurate even when fps/bitrate vary
    (SVT-AV1 lookahead, scene changes). Falls back to silent run_cmd when
    stdout is not a TTY (logs, redirected output).

    base_time is the seek offset of a resumed segmented encode: the bar
    must show progress through the WHOLE source, but whether out_time
    reports absolute output PTS (the -copyts timeline) or zero-based
    time isn't contractual — so the first real report picks: a value far
    below base_time means zero-based, and the offset is added from then
    on.
    """
    if not sys.stdout.isatty():
        run_cmd(cmd)
        return

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    # stderr must be drained concurrently with the stdout progress
    # stream: -v error keeps it normally silent, but a damaged source
    # can flood decode errors, fill the OS pipe buffer, block ffmpeg's
    # stderr write, and deadlock the encode under a frozen bar. The
    # drain thread keeps only the tail, for the failure message.
    stderr_tail = collections.deque(maxlen=80)

    def drain_stderr():
        for err_line in proc.stderr:
            stderr_tail.append(err_line.rstrip("\n"))

    drain = threading.Thread(target=drain_stderr, daemon=True)
    drain.start()
    state = {}
    last_render = 0.0
    active = False
    bar_w = 20
    offset = None

    def render(final=False):
        nonlocal last_render, active, offset
        last_render = time.time()
        try:
            t = int(state.get("out_time_us", "0")) / 1_000_000
        except ValueError:
            t = 0.0
        if base_time and t > 0:
            if offset is None:
                offset = base_time if t < base_time * 0.5 else 0.0
            t += offset
        pct = max(0.0, min(100.0, t / duration * 100)) if duration > 0 else 0.0
        if final:
            pct = 100.0
        speed_val = None
        sp = state.get("speed", "").strip()
        if sp.endswith("x"):
            try:
                speed_val = float(sp[:-1])
            except ValueError:
                pass
        fps_val = None
        try:
            fps_val = float(state.get("fps", "0"))
        except ValueError:
            pass
        # ffmpeg's -progress stream reports a running-average bitrate as
        # e.g. "bitrate=3234.5kbits/s" (or "N/A" before the first frame).
        kbps_val = None
        br = state.get("bitrate", "").strip()
        if br.endswith("kbits/s"):
            try:
                kbps_val = float(br[:-len("kbits/s")])
            except ValueError:
                pass
        filled = int(bar_w * pct / 100)
        bar = (
            f"{DIM}[{RESET}"
            f"{GREEN}{'█' * filled}{RESET}"
            f"{DIM}{'░' * (bar_w - filled)}]{RESET}"
        )
        parts = [f"{BOLD}{pct:5.1f}%{RESET}"]
        if not final and speed_val and speed_val > 0 and duration > 0:
            remaining = max(0, (duration - t) / speed_val)
            parts.append(f"{fmt_time(remaining)} left")
            parts.append(f"{speed_val:.2f}x")
        if not final and fps_val and fps_val > 0:
            parts.append(f"{fps_val:.1f}fps")
        if not final and kbps_val and kbps_val > 0:
            parts.append(f"{kbps_val:.0f}kbps")
        sys.stdout.write(f"\r\033[K{prefix} {bar} {'  '.join(parts)}")
        sys.stdout.flush()
        active = True

    def finish():
        nonlocal active
        if active:
            sys.stdout.write("\n")
            sys.stdout.flush()
            active = False

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, val = line.partition("=")
            state[key] = val
            if key != "progress":
                continue
            if val == "end":
                render(final=True)
                break
            if time.time() - last_render < 0.2:
                continue
            render()
        proc.wait()
        drain.join(timeout=10)
        if proc.returncode != 0:
            tail = "\n".join(stderr_tail)
            raise RuntimeError(
                f"ffmpeg exit {proc.returncode}\n{fmt_cmd(cmd)}\n{tail}"
            )
    except BaseException:
        try:
            proc.terminate()
        except Exception:
            pass
        proc.wait()
        raise
    finally:
        finish()


def encode_av1(source, dest, meta, cq, cfg, show_progress=False,
               resumable=False):
    """Encode `source` to AV1 at `cq` using SVT-AV1 via ffmpeg.

    A sample probe (resumable False) encodes the picture straight to
    dest: its source carries nothing else. A full-file output encode
    (resumable True) writes the picture first, through the segmented
    resume path on long sources (_encode_segmented), then the shared
    source-stream mux adds audio, subtitles, fonts and chapters, the way
    av1q-essential finishes too. So the live kbps on the progress bar is
    the picture alone, like every bitrate this tool decides on.
    """
    # ffmpeg's wrapper reads -crf 0 as "unset" and encodes at the
    # encoder's own default CRF, while the cache would record 0.
    if not 1 <= cq <= 63:
        raise ValueError(f"CQ {cq} is outside libsvtav1's 1-63")

    pix = (
        "yuv420p10le"
        if meta["hdr"] or high_bit_depth(meta["pix_fmt"]) or cfg["force_10bit"]
        else "yuv420p"
    )

    color_args = []
    if meta["cp"] and meta["ct"]:
        color_args += ["-color_primaries", meta["cp"], "-color_trc", meta["ct"]]
        if meta["cs"]:
            color_args += ["-colorspace", meta["cs"]]
    if meta["cr"]:
        color_args += ["-color_range", meta["cr"]]

    # Capped CRF (see MAXRATE_FACTOR). Setting any cap also switches
    # SVT's recode loop on for key and alt-ref frames at every preset.
    bitrate = meta.get("bitrate") or FALLBACK_MAXRATE[res_tier(meta["w"], meta["h"])]
    maxrate = min(int(bitrate * MAXRATE_FACTOR), SVT_MAX_BITRATE)

    fg = cfg["film_grain"]
    # Quantization matrices: off by default in mainline, a ~1-3% rate-
    # distortion win that VMAF credits directly, so the search converts it
    # into smaller files (verified -9.8% at equal CRF on a synthetic A/B).
    # qm-min 2 / chroma-qm-min 4 follow SVT-AV1-Essential's curated
    # defaults (mainline's qm-min 8 barely lets the matrices act).
    #
    # irefresh-type=2 (closed GOP) is already the encoder's default and
    # ffmpeg's (the wrapper sets +cgop), and is pinned here because the
    # segmented resume path cannot work without it: only a closed-GOP
    # refresh is a true key frame, seekable and keyframe-flagged, and the
    # segment muxer cuts only on those. An open-GOP refresh is an
    # intra-only frame that is neither.
    #
    # No scd: mainline SVT-AV1 switches scene change detection off
    # whatever it is asked (its warning hides under -v error) and never
    # inserts key frames at cuts, so the keyint is the only seek
    # granularity there is.
    svt_params = (
        f"tune=0:sharpness=1:film-grain={fg}:film-grain-denoise=0"
        f":enable-tf=0:enable-overlays=1"
        f":enable-qm=1:qm-min=2:chroma-qm-min=4"
        f":irefresh-type=2"
    )
    # HDR10 static metadata (Engine.prepare_meta), stated outright. On
    # its own ffmpeg hands the wrapper only what the first frame into
    # the filter graph carries, and never refreshes it, so a HEVC
    # capture that emits a picture without its SEI first loses it. The
    # wrapper parses these params after that snapshot, so they win.
    # Neither string holds a ':'. The container's copy is stated at the
    # final mux.
    if meta.get("mastering"):
        svt_params += (
            f":mastering-display={svt_mastering_display(meta['mastering'])}"
        )
    if meta.get("cll"):
        svt_params += f":content-light={content_light_str(meta['cll'])}"

    vf_args = []
    if meta.get("crop"):
        vf_args = ["-vf", f"crop={meta['crop']}"]

    # No tiles: ffmpeg's wrapper has no tile option of its own (only
    # tile-columns/tile-rows through -svtav1-params), and tiles cost
    # ~0.6-1.3% compression efficiency for a client decode speed that CPU
    # playback of AV1 at these bitrates doesn't need. No -threads or
    # -bufsize either: the wrapper never hands the thread count to
    # SVT-AV1, which sizes its own pool, and SVT reads the buffer size
    # only in VBR and CBR, never in capped CRF.
    video_args = [
        "-map", "0:v:0", *vf_args,
        "-pix_fmt", pix,
        "-c:v", "libsvtav1",
        "-preset", str(cfg["preset"]),
        "-crf", str(cq),
        # No -g: SVT's own default keyint rounds the frame rate up to
        # whole mini-GOPs and spans five of those seconds (161 frames at
        # 24fps, 321 at 60fps).
        "-svtav1-params", svt_params,
        "-maxrate", str(maxrate),
        "-fps_mode", "passthrough",
        *color_args,
    ]

    if not resumable:
        tmp = dest.with_suffix(".tmp.mkv")
        _temp_files.add(tmp)
        _ffmpeg_encode(["-i", str(source)], video_args, [str(tmp)],
                       meta, cq, show_progress)
        tmp.replace(dest)
        _temp_files.discard(tmp)
        return

    work = None
    if (cfg["resume_encodes"]
            and (meta.get("duration") or 0) >= RESUMABLE_MIN_DURATION):
        video, work = _encode_segmented(source, meta, cq, cfg, pix,
                                        video_args, show_progress)
    else:
        video = dest.with_suffix(".video.tmp.mkv")
        _temp_files.add(video)
        _ffmpeg_encode(["-i", str(source)], video_args, [str(video)],
                       meta, cq, show_progress)

    tmp = dest.with_suffix(".tmp.mkv")
    _temp_files.add(tmp)
    segments.mux_with_source_streams(
        video, source, tmp,
        mastering=meta.get("mastering"), cll=meta.get("cll"),
    )
    tmp.replace(dest)
    _temp_files.discard(tmp)
    if work:
        shutil.rmtree(work, ignore_errors=True)
    else:
        video.unlink(missing_ok=True)
        _temp_files.discard(video)


def _ffmpeg_encode(in_args, video_args, out_args, meta, cq, show_progress,
                   base_time=0.0):
    """One encoder run; out_args ends with the output path. show_progress
    draws the inline bar (see _run_ffmpeg_progress for base_time)."""
    cmd = [
        ffmpeg_exe(), "-y", "-hide_banner", "-v", "error", "-nostats",
        *in_args, *video_args, *out_args,
    ]
    duration = meta.get("duration") or 0.0
    if show_progress and duration > 1.0:
        cmd[-1:-1] = ["-progress", "pipe:1"]
        _run_ffmpeg_progress(
            cmd, duration, f"{label('encode')}CQ {BOLD}{cq}{RESET}",
            base_time=base_time,
        )
    else:
        run_cmd(cmd)


def _encode_segmented(source, meta, cq, cfg, pix, video_args, show_progress):
    """Resumable picture encode: one continuous encoder, segment-muxed.
    Returns (joined video, work dir).

    The segment muxer finalizes each completed ~SEGMENT_TIME segment
    before opening the next, so the bitstream is identical to the
    single-pass encode while every finished segment survives a kill.
    core.segments owns the resume state: it keeps the finished segments
    minus the last one (its first-packet PTS is the only exactly-knowable
    boundary), and the encode re-enters at that PTS.

    Segment files and the manifest deliberately stay OUT of _temp_files:
    surviving Ctrl-C and crashes is their entire purpose. The caller
    removes the work dir once the output is final; the pipeline sweeps a
    file's dirs too.
    """
    file_hash = partial_hash(source)
    enc_tag = enc_signature(cfg, meta.get("crop"))
    sdir = segments.segment_dir(cfg["cache_dir"], file_hash, enc_tag, str(cq))
    manifest, resume_ms = segments.prepare(
        sdir,
        segments.manifest_expected(
            file_hash, enc_tag, str(cq), SEGMENT_TIME, pix
        ),
    )

    if not manifest["complete"]:
        base_time = 0.0
        in_args = ["-i", str(source)]
        ts_args = []
        if resume_ms is not None:
            base_time = resume_ms / 1000.0
            # Accurate seek decodes up to the boundary and starts on the
            # exact frame; -copyts -start_at_zero re-enters the first
            # run's zero-based timeline at that PTS, so the new segments'
            # timestamps continue the kept ones seamlessly. The seek
            # target sits 1ms EARLY: the stored boundary is the MKV
            # muxer's ms-ROUNDED PTS, which can round up past the true
            # frame time on non-ms-exact sources (24000/1001 etc.) and
            # the accurate-seek cutoff would then drop the boundary
            # frame. 1ms of slack can't admit the previous frame (frame
            # durations are >= 8ms) and -copyts means the seek target
            # never shapes output timestamps, only the discard cutoff.
            in_args = ["-ss", segments.ms_ts(max(0, resume_ms - 1)),
                       "-i", str(source)]
            ts_args = ["-copyts", "-start_at_zero"]
            print(
                f"{label('resume')}{BOLD}{len(manifest['segments'])}{RESET}"
                f" finished segment(s) kept"
                f" {DIM}re-encoding from {fmt_time(base_time)}{RESET}"
            )

        _ffmpeg_encode(
            in_args, video_args,
            [
                *ts_args,
                "-f", "segment",
                "-segment_time", str(SEGMENT_TIME),
                "-segment_format", "matroska",
                "-segment_list", str(sdir / segments.SEGMENT_LIST_NAME),
                "-segment_list_type", "csv",
                "-segment_start_number", str(len(manifest["segments"])),
                "-reset_timestamps", "0",
                str(sdir / segments.SEGMENT_PATTERN),
            ],
            meta, cq, show_progress, base_time=base_time,
        )
        segments.finish_run(sdir, manifest)

    joined = sdir / segments.JOINED_NAME
    segments.concat_segments(sdir, manifest["segments"], joined)
    return joined, sdir


class IntGrid(Grid):
    """av1q's integer CQ grid."""

    step = 1

    def quantize(self, v):
        return int(round(v))

    def fmt(self, q):
        return str(q)

    def fmt_delta(self, d):
        return f"{d:+d}"

    def floor(self, v):
        return int(math.floor(v))

    def ceil(self, v):
        return int(math.ceil(v))

    def span(self, lo, hi):
        return list(range(lo, hi + 1))


class SvtAv1FfmpegEngine(Engine):
    sig = "avq1-c1"
    qname = "CQ"
    banner = "av1q"
    banner_extra = ""
    grid = IntGrid()
    vmaf_key_base = "full"
    sample_ext = ".mkv"
    tmp_patterns = ("*_CQ*.tmp.mkv", "sample_enc_*.tmp.mkv")
    ffmpeg_encoders = ("libsvtav1",)
    ffmpeg_filters = ("libvmaf",)
    rec_q_key = "cq"
    rec_bound_keys = ("min_cq", "max_cq")
    rec_extra_keys = ()
    seed_key = "seed_cq"
    seed_prompt_hint = "(Enter = auto)"
    cal_q_key = "at_cq"
    needs_expected_frames = False
    hdr10_passthrough = True

    def cache_root(self, cfg):
        return cfg["cache_dir"]

    def calibration_root(self, cfg):
        return cfg["learned_dir"] / "av1q"

    def q_bounds(self, cfg):
        return cfg["min_cq"], cfg["max_cq"]

    def seed_override(self, cfg):
        return cfg.get("seed_cq")

    def parse_user_q(self, raw):
        return int(raw)

    def signature(self, cfg, crop=None):
        return enc_signature(cfg, crop)

    def setup(self, cfg):
        # Probe (and on first run auto-download) FFVship up front so any
        # download happens before the seed prompt — not mid-search. The
        # result is cached; every later call is instant. av1q itself has
        # no required tool binaries.
        find_ffvship_optional()

    def encode(self, source, dest, meta, q, cfg,
               show_progress=False, expected_frames=0, resumable=False):
        # expected_frames is a Y4M-pipe concern; ffmpeg's own -progress
        # output drives this engine's bar.
        encode_av1(source, dest, meta, q, cfg, show_progress=show_progress,
                   resumable=resumable)

    def ssimu2_info(self, ref, dist, meta, cfg, ref_index=None):
        return ssimu2.measure_ssimu2_display(
            ref, dist, meta, cfg["cache_dir"], ref_index=ref_index,
        )

    def dst_name(self, stem, q, token, ext):
        return f"{stem}_CQ{q}{token}{ext}"

