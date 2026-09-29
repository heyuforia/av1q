"""Mainline SVT-AV1 engine: av1q's encode path, through ffmpeg's
libsvtav1 wrapper. The picture is encoded first, in pieces through
core.chunks, crop, color and HDR10 static metadata applied; the shared
source-stream mux in core.segments then adds audio, subtitles, fonts
and chapters."""

import collections
import math
import subprocess
import threading

from .. import chunks, segments, ssimu2
from ..constants import FALLBACK_MAXRATE, MAXRATE_FACTOR
from ..crop import crop_token
from ..probe import (
    content_light_str, high_bit_depth, res_tier, svt_mastering_display,
)
from ..tools import ffmpeg_exe, find_ffvship_optional
from ..util import _temp_files, fmt_cmd, own_process_group, run_cmd
from .base import Engine, Grid


# SVT-AV1 refuses a max bitrate above 100000 kbps and fails the encode at
# init, so the peak cap is held to it.
SVT_MAX_BITRATE = 100_000_000


def enc_signature(cfg, crop=None):
    """Tag covering the settings that change encoder output for one source
    at a given CQ: preset, film grain, and crop. Used in cache keys and
    sample-encode filenames so stale variants are never reused.

    --no-10bit stays out on purpose. It changes the pixel format of 8-bit
    SDR sources only, but the skip check runs before the probe and sees
    only cfg, so carrying it here would search and encode every finished
    10-bit source again for an identical file. A --no-10bit rerun may
    therefore reuse a 10-bit encode.
    """
    return f"p{cfg['preset']}g{cfg['film_grain']}{crop_token(crop)}"


def _pix(meta, cfg):
    """The pixel format av1q encodes a source at: 10 bits unless
    --no-10bit and the source is 8-bit SDR."""
    return (
        "yuv420p10le"
        if meta["hdr"] or high_bit_depth(meta["pix_fmt"]) or cfg["force_10bit"]
        else "yuv420p"
    )


def _video_args(meta, cq, cfg, trim=None):
    """The encode's output options for the picture at `cq`: map, filters
    (trim first, then crop), pixel format and the encoder's settings.
    trim is a trim filter keeping one piece's span, or None."""
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
    # pieces of a split encode are joined by stream copy: only a
    # closed-GOP refresh is a true key frame, seekable and
    # keyframe-flagged, so each piece stands on its own. An open-GOP
    # refresh is an intra-only frame that is neither.
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
    # capture that emits a picture without its SEI first loses it, and
    # so does a piece that starts between keyframes. The wrapper parses
    # these params after that snapshot, so they win. Neither string
    # holds a ':'. The container's copy is stated at the final mux.
    if meta.get("mastering"):
        svt_params += (
            f":mastering-display={svt_mastering_display(meta['mastering'])}"
        )
    if meta.get("cll"):
        svt_params += f":content-light={content_light_str(meta['cll'])}"

    vf = [trim] if trim else []
    if meta.get("crop"):
        vf.append(f"crop={meta['crop']}")
    vf_args = ["-vf", ",".join(vf)] if vf else []

    # No tiles: ffmpeg's wrapper has no tile option of its own (only
    # tile-columns/tile-rows through -svtav1-params), and tiles cost
    # ~0.6-1.3% compression efficiency for a client decode speed that CPU
    # playback of AV1 at these bitrates doesn't need. No -threads or
    # -bufsize either: the wrapper never hands the thread count to
    # SVT-AV1, which sizes its own pool, and SVT reads the buffer size
    # only in VBR and CBR, never in capped CRF.
    return [
        "-map", "0:v:0", *vf_args,
        "-pix_fmt", _pix(meta, cfg),
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


def encode_av1(source, dest, meta, cq, cfg, show_progress=False,
               resumable=False):
    """Encode `source` to AV1 at `cq` using SVT-AV1 via ffmpeg.

    A sample probe (resumable False) encodes the picture straight to
    dest: its source carries nothing else. A full-file output encode
    (resumable True) goes to core.chunks, which encodes the picture in
    resumable pieces (encode_chunk_av1), then muxes the source's audio,
    subtitles, fonts and chapters beside it, the way av1q-essential
    finishes too. So the live kbps on the progress bar is the picture
    alone, like every bitrate this tool decides on.
    """
    # ffmpeg's wrapper reads -crf 0 as "unset" and encodes at the
    # encoder's own default CRF, while the cache would record 0.
    if not 1 <= cq <= 63:
        raise ValueError(f"CQ {cq} is outside libsvtav1's 1-63")

    if resumable:
        chunks.encode_full(SvtAv1FfmpegEngine(), source, dest, meta, cq, cfg,
                           show_progress=show_progress)
        return

    tmp = dest.with_suffix(".tmp.mkv")
    _temp_files.add(tmp)
    run_cmd([
        ffmpeg_exe(), "-y", "-hide_banner", "-v", "error", "-nostats",
        "-i", str(source), *_video_args(meta, cq, cfg), str(tmp),
    ])
    tmp.replace(dest)
    _temp_files.discard(tmp)


def encode_chunk_av1(source, out, meta, cq, cfg, piece, job):
    """Encode one piece of a full-file picture to `out` (Engine.
    encode_chunk); piece None is the whole picture, today's single pass.

    A piece seeks early (piece.seek lands at or before its first
    keyframe on every container) and a trim filter keeps exactly the
    source frames whose pts fall in its span. -copyts -start_at_zero keeps
    the source's own timeline, with the container start subtracted as
    every other run does, so each piece carries its frames' original
    PTS and the pieces join with no offset. -copyts also switches off
    MPEG-TS discontinuity repair, which is why a timeline that jumps is
    never split (analyze.read_timeline).

    Both trim times sit 1ms before a keyframe's time. The plan holds the
    microsecond times ffprobe printed for the frames' own pts, and ffmpeg
    rounds a trim time to the stream's time base; 1ms under a frame's
    time rounds to that frame's own tick, or to one between it and the
    frame before, at any time base, and frames are at least 8ms apart.
    So a keyframe opens its own piece and closes the one before, whatever
    the rounding.
    """
    in_args = ["-i", str(source)]
    ts_args = []
    trim = None
    base_s = 0.0
    if piece is not None:
        if piece.seek is not None:
            in_args = ["-ss", segments.us_ts(piece.seek), *in_args]
        ts_args = ["-copyts", "-start_at_zero"]
        bounds = []
        if piece.start is not None:
            bounds.append(f"start={segments.us_ts(piece.start - 1000)}")
            base_s = piece.start / 1_000_000
        if piece.end is not None:
            bounds.append(f"end={segments.us_ts(piece.end - 1000)}")
        trim = "trim=" + ":".join(bounds)

    # -nostdin: a piece is trusted on exit 0, and ffmpeg's interactive
    # 'q' would end one early, with a valid short file, at exit 0.
    tmp = out.with_suffix(".tmp.mkv")
    _temp_files.add(tmp)
    _run_ffmpeg_job([
        ffmpeg_exe(), "-y", "-hide_banner", "-v", "error", "-nostats",
        "-nostdin", *in_args, *_video_args(meta, cq, cfg, trim), *ts_args,
        "-progress", "pipe:1", str(tmp),
    ], job, base_s)
    tmp.replace(out)
    _temp_files.discard(tmp)


def _run_ffmpeg_job(cmd, job, base_s):
    """Run one encoder ffmpeg with -progress pipe:1, reporting each
    progress block to job (seconds encoded, fps, bytes written).

    base_s is the piece's start: whether out_time reports the absolute
    output PTS (the -copyts timeline) or zero-based time isn't
    contractual, so the first real report picks: a value far below
    base_s means zero-based, and nothing is subtracted from then on.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        **own_process_group(),
    )
    job.track(proc)
    # stderr must be drained concurrently with the stdout progress
    # stream: -v error keeps it normally silent, but a damaged source
    # can flood decode errors, fill the OS pipe buffer, block ffmpeg's
    # stderr write, and deadlock the encode. The drain thread keeps only
    # the tail, for the failure message.
    stderr_tail = collections.deque(maxlen=80)

    def drain_stderr():
        for err_line in proc.stderr:
            stderr_tail.append(err_line.rstrip("\n"))

    drain = threading.Thread(target=drain_stderr, daemon=True)
    drain.start()
    state = {}
    offset = None
    try:
        for line in proc.stdout:
            key, sep, val = line.strip().partition("=")
            if not sep:
                continue
            state[key] = val
            if key != "progress":
                continue
            try:
                t = int(state.get("out_time_us", "0")) / 1_000_000
            except ValueError:
                t = 0.0
            if t > 0:
                if offset is None:
                    offset = base_s if t >= base_s * 0.5 else 0.0
                t -= offset
            fps = size = None
            try:
                fps = float(state.get("fps", ""))
            except ValueError:
                pass
            try:
                size = int(state.get("total_size", ""))
            except ValueError:
                pass
            job.report(max(0.0, t), fps, size)
            if val == "end":
                break
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
    chunk_ext = ".mkv"
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
               show_progress=False, resumable=False):
        encode_av1(source, dest, meta, q, cfg, show_progress=show_progress,
                   resumable=resumable)

    def chunk_identity(self, meta, cfg):
        # The pixel format, which enc_signature leaves out (--no-10bit):
        # a bit-depth flip between an interruption and the rerun would
        # otherwise join 8-bit and 10-bit pieces into one corrupt file.
        return {"pix": _pix(meta, cfg)}

    def encode_chunk(self, source, out, meta, q, cfg, piece, job):
        encode_chunk_av1(source, out, meta, q, cfg, piece, job)

    def chunk_start_us(self, meta, piece, out):
        # A piece's picture keeps its source PTS, read back from the file:
        # its first frame must be the keyframe it was cut at and its last
        # must come before the next piece's. Within 1ms: the MKV stores
        # ms-rounded times against the plan's microseconds.
        if piece is None:
            ms = segments.probe_start_ms(out)
            return None if ms is None else ms * 1000
        span = segments.probe_pts_span(out)
        if span is None:
            return None
        first, last = span[0] * 1000, span[1] * 1000
        if piece.start is not None and abs(first - piece.start) > 1000:
            return None
        if piece.end is not None and last > piece.end - 1000:
            return None
        return first

    def ssimu2_info(self, ref, dist, meta, cfg, ref_index=None):
        return ssimu2.measure_ssimu2_display(
            ref, dist, meta, cfg["cache_dir"], ref_index=ref_index,
        )

    def dst_name(self, stem, q, token, ext):
        return f"{stem}_CQ{q}{token}{ext}"

