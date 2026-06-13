"""Mainline SVT-AV1 engine: av1q's original encode path, through
ffmpeg's libsvtav1 wrapper. Color metadata, crop, and audio/subtitle
passthrough all ride a single ffmpeg invocation."""

import math
import os
import shlex
import subprocess
import sys
import time

from .. import ssimu2
from ..constants import FALLBACK_MAXRATE
from ..crop import crop_token
from ..probe import res_tier
from ..tools import find_ffvship_optional
from ..ui import BOLD, DIM, GREEN, ORANGE, RESET, fmt_time
from ..util import _temp_files, clamp, run_cmd
from .base import Engine, Grid


def enc_signature(cfg, crop=None):
    """Tag covering everything that changes encoder output for one source
    at a given CQ: preset, film grain, and crop. Used in cache keys and
    sample-encode filenames so stale variants are never reused.
    """
    return f"p{cfg['preset']}g{cfg['film_grain']}{crop_token(crop)}"


def _run_ffmpeg_progress(cmd, duration, label):
    """Run ffmpeg with -progress pipe:1 and render an inline progress bar.

    Parses key=value blocks on stdout; uses out_time_us against the known
    source duration so the bar stays accurate even when fps/bitrate vary
    (SVT-AV1 lookahead, scene changes). Falls back to silent run_cmd when
    stdout is not a TTY (logs, redirected output).
    """
    if not sys.stdout.isatty():
        run_cmd(cmd)
        return

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    state = {}
    last_render = 0.0
    active = False
    bar_w = 20

    def render(final=False):
        nonlocal last_render, active
        last_render = time.time()
        try:
            t = int(state.get("out_time_us", "0")) / 1_000_000
        except ValueError:
            t = 0.0
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
        sys.stdout.write(
            f"\r\033[K{label} {bar} {'  '.join(parts)}"
        )
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
        if proc.returncode != 0:
            stderr_data = proc.stderr.read() if proc.stderr else ""
            tail = "\n".join((stderr_data or "").splitlines()[-80:])
            raise RuntimeError(
                f"ffmpeg exit {proc.returncode}\n"
                f"{subprocess.list2cmdline(cmd) if os.name == 'nt' else ' '.join(map(shlex.quote, cmd))}"
                f"\n{tail}"
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


def encode_av1(source, dest, meta, cq, cfg, show_progress=False):
    """Encode video to AV1 using SVT-AV1 via ffmpeg."""
    pix = (
        "yuv420p10le"
        if meta["hdr"] or "10le" in meta["pix_fmt"] or cfg["force_10bit"]
        else "yuv420p"
    )

    tmp = dest.with_suffix(".tmp.mkv")
    _temp_files.add(tmp)
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass

    color_args = []
    if meta["cp"] and meta["ct"]:
        color_args += ["-color_primaries", meta["cp"], "-color_trc", meta["ct"]]
        if meta["cs"]:
            color_args += ["-colorspace", meta["cs"]]
    if meta["cr"]:
        color_args += ["-color_range", meta["cr"]]

    bitrate = meta.get("bitrate") or FALLBACK_MAXRATE[res_tier(meta["w"], meta["h"])]
    maxrate = min(int(bitrate * cfg["maxrate_factor"]), 100_000_000)
    crf = clamp(cq, 0, 63)

    # No tiles: ffmpeg's libsvtav1 wrapper never exposed a "-tiles" option
    # (only tile-columns/tile-rows via -svtav1-params), SVT-AV1 threads well
    # without them, and tiles cost ~0.6-1.3% compression efficiency — they
    # only pay off for client decode speed, which CPU playback of AV1 at
    # these bitrates doesn't need.
    threads = os.cpu_count() or 1
    fg = cfg["film_grain"]
    # Quantization matrices: off by default in mainline, a ~1-3% rate-
    # distortion win that VMAF credits directly, so the search converts it
    # into smaller files (verified -9.8% at equal CRF on a synthetic A/B).
    # qm-min 2 / chroma-qm-min 4 follow SVT-AV1-Essential's curated
    # defaults (mainline's qm-min 8 barely lets the matrices act).
    svt_params = (
        f"tune=0:sharpness=1:film-grain={fg}:film-grain-denoise=0"
        f":enable-tf=0:enable-overlays=1:scd=1"
        f":enable-qm=1:qm-min=2:chroma-qm-min=4"
    )

    vf_args = []
    if meta.get("crop"):
        vf_args = ["-vf", f"crop={meta['crop']}"]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-v", "error", "-nostats",
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a?", "-map", "0:s?",
        *vf_args,
        "-c:a", "copy", "-c:s", "copy",
        "-pix_fmt", pix,
        "-c:v", "libsvtav1",
        "-preset", str(cfg["preset"]),
        "-crf", str(crf),
        # No -g: SVT's own default keyint is fps-aware ~5s (161 frames at
        # 24fps, 321 at 60fps), mini-gop aligned. The old fixed -g 250
        # meant ~10s seek granularity at 24fps — and since mainline scd=1
        # does NOT insert keyframes at scene cuts (that's a fork-only
        # behavior), the interval is the ONLY seek granularity there is.
        "-svtav1-params", svt_params,
        "-threads", str(threads),
        "-maxrate", str(maxrate),
        "-bufsize", str(maxrate * 2),
        "-fps_mode", "passthrough",
        *color_args, str(tmp),
    ]

    duration = meta.get("duration") or 0.0
    if show_progress and duration > 1.0:
        out_path = cmd.pop()
        cmd += ["-progress", "pipe:1", out_path]
        label = f" {ORANGE}{'encode':<10}{RESET}CQ {BOLD}{cq}{RESET}"
        _run_ffmpeg_progress(cmd, duration, label)
    else:
        run_cmd(cmd)

    if dest.exists():
        dest.unlink()
    tmp.rename(dest)
    _temp_files.discard(tmp)


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
    tmp_patterns = ("*.tmp.mkv",)
    rec_q_key = "cq"
    rec_bound_keys = ("min_cq", "max_cq")
    rec_extra_keys = ()
    seed_key = "seed_cq"
    seed_prompt_hint = "(Enter = auto)"
    cal_q_key = "at_cq"
    needs_expected_frames = False

    def cache_root(self, cfg):
        return cfg["cache_dir"]

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
               show_progress=False, expected_frames=0):
        # expected_frames is a Y4M-pipe concern; ffmpeg's own -progress
        # output drives this engine's bar.
        encode_av1(source, dest, meta, q, cfg, show_progress=show_progress)

    def ssimu2_info(self, ref, dist, meta, cfg, ref_index=None):
        return ssimu2.measure_ssimu2_display(
            ref, dist, meta, cfg["cache_dir"], ref_index=ref_index,
        )

    def full_ref_index(self, cfg, source, file_hash, size):
        # Stem + size keeps the index stable across runs but distinct
        # when the underlying file changes.
        return cfg["cache_dir"] / "_ffindex" / f"{source.stem}_{size}.ffindex"

    def sample_ref_index(self, cfg, sample_src):
        try:
            return (
                cfg["cache_dir"] / "_ffindex"
                / f"{sample_src.stem}_{sample_src.stat().st_size}.ffindex"
            )
        except OSError:
            return None

    def dst_name(self, stem, q, token, ext):
        return f"{stem}_CQ{q}{token}{ext}"

