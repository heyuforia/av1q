"""SVT-AV1-Essential engine: ffmpeg decodes (applying crop) and pipes
10-bit Y4M into the standalone fork binary, then audio/subs/chapters
are remuxed back from the source. Quarter-step CRF grid; color and
HDR10 static metadata re-stated as encoder flags (Y4M carries none);
VFR sources gated out (the Y4M pipe is CFR-only)."""

import math
import os
import re
import subprocess
import sys
import time

from .. import ssimu2
from ..crop import crop_token
from ..probe import get_rfps, is_vfr, probe_hdr_metadata
from ..sampling import clean_sample_source
from ..tools import find_encoder, find_ffvship_optional
from ..ui import BOLD, DIM, GREEN, MIDDOT, ORANGE, RESET, fmt_time
from ..util import _temp_files, make_temp_log, run_cmd
from .base import Engine, Grid


# Quarter-step CRF grid — SVT-AV1-Essential accepts CRF in 0.25 increments
# (values are floored to the grid inside the encoder).
CRF_STEP = 0.25


def qcrf(v):
    """Quantize to the encoder's quarter-step CRF grid."""
    return round(v * 4) / 4


def crf_str(v):
    """Stable string for a quarter-step CRF: '23', '23.5', '23.25'.

    Used for cache keys and filenames, so it must round-trip exactly
    (float(crf_str(v)) == qcrf(v)) and never grow trailing zeros.
    """
    return f"{qcrf(v):.2f}".rstrip("0").rstrip(".")


def crf_range(lo, hi):
    """All quarter-step CRFs from lo to hi inclusive."""
    n = int(round((hi - lo) / CRF_STEP))
    return [qcrf(lo + i * CRF_STEP) for i in range(max(0, n) + 1)]


# Encoder flags av1q-essential sets itself (encode_essential / build_color_args).
# A user --enc-args value repeating one of these would emit it twice on the
# command line, so filter_enc_args drops them (with a warning) and av1q's own
# value wins — users should reach for av1q's matching CLI option instead
# (e.g. --tune, --film-grain). All of these take a following value.
MANAGED_ENC_FLAGS = {
    "-i", "-b", "--preset", "--crf", "--tune",
    "--film-grain", "--film-grain-denoise",
    "--color-primaries", "--transfer-characteristics",
    "--matrix-coefficients", "--color-range",
    "--mastering-display", "--content-light",
    "--progress", "--hide-banner",
}


def filter_enc_args(tokens):
    """Strip any av1q-managed flag (and its value) from a flat token list.

    Returns (kept, dropped) where dropped is the list of managed flag names
    that were removed, so the caller can warn once. Handles both the
    '--flag value' and '--flag=value' spellings; managed flags all take a
    value, so the space-separated form consumes the next token too.
    """
    kept, dropped = [], []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        base = tok.split("=", 1)[0]
        if base in MANAGED_ENC_FLAGS:
            dropped.append(base)
            # Consume an attached value token for the space-separated form.
            # A value never starts with '-' here, so a following flag (e.g.
            # a malformed trailing '--tune') is left for its own iteration.
            if "=" not in tok and i + 1 < n and not tokens[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        kept.append(tok)
        i += 1
    return kept, dropped


def enc_signature_e(cfg, crop=None):
    """Tag covering everything that changes Essential's output for one
    source at a given CRF: preset, film grain, tune, crop, and any extra
    raw encoder flags. The tune knob is new vs av1q's signature —
    Essential exposes it per-encode. The enc-args hash (cfg['enc_args_sig'],
    None when no --enc-args) only widens the tag when extra flags are
    present, so a plain run produces the exact same signature as before.
    """
    xa = cfg.get("enc_args_sig")
    extra = f"x{xa}" if xa else ""
    return f"p{cfg['preset']}g{cfg['film_grain']}t{cfg['tune']}{extra}{crop_token(crop)}"


# ffprobe names -> SvtAv1EncApp names (Appendix A.2). Identical names are
# included for clarity; unknown values are omitted so the encoder keeps
# its 'unspecified' default rather than guessing.
SVT_PRIMARIES = {
    "bt709": "bt709", "bt470m": "bt470m", "bt470bg": "bt470bg",
    "smpte170m": "bt601", "smpte240m": "smpte240", "film": "film",
    "bt2020": "bt2020", "smpte428": "xyz", "smpte431": "smpte431",
    "smpte432": "smpte432", "ebu3213": "ebu3213",
}
SVT_TRANSFER = {
    "bt709": "bt709", "bt470m": "bt470m", "bt470bg": "bt470bg",
    "smpte170m": "bt601", "smpte240m": "smpte240", "linear": "linear",
    "log100": "log100", "log316": "log100-sqrt10",
    "iec61966-2-4": "iec61966", "bt1361e": "bt1361",
    "iec61966-2-1": "srgb", "bt2020-10": "bt2020-10",
    "bt2020-12": "bt2020-12", "smpte2084": "smpte2084",
    "smpte428": "smpte428", "arib-std-b67": "hlg",
}
SVT_MATRIX = {
    "identity": "identity", "gbr": "identity", "bt709": "bt709",
    "fcc": "fcc", "bt470bg": "bt470bg", "smpte170m": "bt601",
    "smpte240m": "smpte240", "ycgco": "ycgco", "bt2020nc": "bt2020-ncl",
    "bt2020c": "bt2020-cl", "smpte2085": "smpte2085",
    "chroma-derived-nc": "chroma-ncl", "chroma-derived-c": "chroma-cl",
    "ictcp": "ictcp",
}
SVT_RANGE = {"tv": "studio", "limited": "studio", "pc": "full", "full": "full"}


def build_color_args(meta):
    """Encoder color flags from probed metadata.

    Y4M carries no color information, so everything av1q passed through
    ffmpeg's -color_* options must be re-stated as encoder flags here —
    including HDR10 static metadata, which is written into the AV1
    bitstream itself (survives any remux). Setting smpte2084 also auto-
    selects Essential's PQ-optimized variance-boost curve, which changes
    encoding — one more reason sample and full encodes must both get
    these flags (enc_signature parity).
    """
    args = []
    cp = SVT_PRIMARIES.get(meta.get("cp", ""))
    ct = SVT_TRANSFER.get(meta.get("ct", ""))
    cs = SVT_MATRIX.get(meta.get("cs", ""))
    cr = SVT_RANGE.get(meta.get("cr", ""))
    if cp:
        args += ["--color-primaries", cp]
    if ct:
        args += ["--transfer-characteristics", ct]
    if cs:
        args += ["--matrix-coefficients", cs]
    if cr:
        args += ["--color-range", cr]
    if meta.get("mastering"):
        args += ["--mastering-display", meta["mastering"]]
    if meta.get("cll"):
        args += ["--content-light", meta["cll"]]
    return args


def _proc_tail(text, n=40):
    return "\n".join((text or "").splitlines()[-n:])


def encode_essential(source, dest, meta, crf, cfg, show_progress=False,
                     expected_frames=0):
    """Encode `source` to AV1 at `crf` via SVT-AV1-Essential.

    ffmpeg decodes (applying crop) and pipes 10-bit Y4M — Essential
    rejects 8-bit input by design — into the encoder, which writes a
    video-only WebM. A `.mkv` dest then gets audio/subs/chapters remuxed
    back from the source; a `.webm` dest (sample probes) is used as-is.

    Hardware decode is deliberately not used on this path: a mid-stream
    hwaccel failure can't be retried without restarting the encoder, and
    CPU decode comfortably outpaces SVT-AV1 at these presets.
    """
    is_full = dest.suffix.lower() != ".webm"
    enc_out = dest.with_suffix(".tmp.webm")
    _temp_files.add(enc_out)
    try:
        if enc_out.exists():
            enc_out.unlink()
    except OSError:
        pass

    ff_cmd = ["ffmpeg", "-y", "-hide_banner", "-v", "error", "-nostats",
              "-i", str(source), "-map", "0:v:0"]
    # Full encodes normalize the feed's timeline to the source's nominal
    # frame cadence: setpts zeroes any start offset / edit-list delay and
    # fps re-times onto a clean CFR grid at r_frame_rate. This is a no-op
    # for well-formed CFR sources, but for irregular ones (e.g. stream-
    # copy concatenations with a per-join timing gap) it is what keeps the
    # full encode frame-aligned with the VMAF reference, which applies the
    # identical setpts+fps normalization on its side. Without it the
    # encoder's own CFR conversion fills the gaps differently than the
    # reference's fps filter and the two drift out of phase, collapsing
    # full VMAF. Samples skip this: their search source is already a clean
    # CFR re-encode (clean_sample_source) and they pair by index.
    vf = []
    if meta.get("crop"):
        vf.append(f"crop={meta['crop']}")
    rfps = meta.get("rfps") if is_full else None
    if rfps:
        vf += ["setpts=PTS-STARTPTS", f"fps={rfps}"]
    if vf:
        ff_cmd += ["-vf", ",".join(vf)]
    if rfps:
        # Pass the fps filter's CFR frames through untouched; without this
        # the yuv4mpegpipe muxer re-runs its own CFR conversion on top,
        # which can diverge from the reference's fps filter at the same
        # rate and reintroduce the drift.
        ff_cmd += ["-fps_mode", "passthrough"]
    ff_cmd += ["-pix_fmt", "yuv420p10le", "-strict", "-1",
               "-f", "yuv4mpegpipe", "-"]

    enc_cmd = [
        str(cfg["encoder_exe"]), "-i", "stdin", "-b", str(enc_out),
        "--preset", str(cfg["preset"]), "--crf", crf_str(crf),
        "--tune", str(cfg["tune"]), "--film-grain", str(cfg["film_grain"]),
        "--film-grain-denoise", "0",
        "--progress", "2", "--hide-banner", "1",
    ]
    enc_cmd += build_color_args(meta)
    # User-supplied raw encoder flags (already filtered of anything av1q
    # manages, so these can't duplicate the flags above). Appended last.
    enc_cmd += list(cfg.get("enc_args") or [])

    ff_log = make_temp_log(cfg["cache_dir"], "y4mfeed", "log")
    bar_w = 20
    last_render = 0.0
    rendered = False
    tail = []  # last stderr lines for error reporting
    # Real-time speed multiplier = encode fps / source fps, matching the
    # "1.4x" field ffmpeg's own -progress output gives av1q's bar. The
    # encoder reports encode fps but no speed, so derive source fps from
    # the known frame count over the source duration.
    source_fps = (
        expected_frames / meta["duration"]
        if expected_frames and meta.get("duration", 0) > 0 else 0
    )

    def render(frames, total, fps_val, kbps, final=False):
        nonlocal last_render, rendered
        last_render = time.time()
        total_known = total or expected_frames
        pct = (
            max(0.0, min(100.0, frames / total_known * 100))
            if total_known else 0.0
        )
        if final:
            pct = 100.0
        filled = int(bar_w * pct / 100)
        bar = (
            f"{DIM}[{RESET}{GREEN}{'█' * filled}{RESET}"
            f"{DIM}{'░' * (bar_w - filled)}]{RESET}"
        )
        parts = [f"{BOLD}{pct:5.1f}%{RESET}"]
        if not final and fps_val and fps_val > 0 and total_known:
            remaining = max(0, (total_known - frames) / fps_val)
            parts.append(f"{fmt_time(remaining)} left")
            if source_fps > 0:
                parts.append(f"{fps_val / source_fps:.2f}x")
            parts.append(f"{fps_val:.1f}fps")
        if not final and kbps:
            parts.append(f"{kbps:.0f}kbps")
        label = f" {ORANGE}{'encode':<10}{RESET}CRF {BOLD}{crf_str(crf)}{RESET}"
        sys.stdout.write(f"\r\033[K{label} {bar} {'  '.join(parts)}")
        sys.stdout.flush()
        rendered = True

    show = show_progress and sys.stdout.isatty()
    prog_re = re.compile(
        rb"Encoding:\s+(\d+)(?:/(\d+))?\s+Frames\s+@\s+([\d.]+)\s+fps"
        rb"\s+\|\s+([\d.]+)\s+kb/s"
    )

    ffp = enc = None
    try:
        with open(ff_log, "wb") as ferr:
            ffp = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=ferr)
            enc = subprocess.Popen(
                enc_cmd, stdin=ffp.stdout,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            ffp.stdout.close()  # let EPIPE reach ffmpeg if the encoder dies

            # Render the 0% bar immediately: the encoder emits no progress
            # until its lookahead fills, and decoding 4K ProRes into the
            # pipe can take tens of seconds — a blank console reads as a
            # hang.
            if show:
                render(0, 0, 0, 0)

            # Drain encoder stderr continuously (it floods \r progress
            # lines); render at most ~5 bars/sec.
            buf = b""
            while True:
                chunk = enc.stderr.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    m = re.search(rb"[\r\n]", buf)
                    if not m:
                        break
                    line, buf = buf[:m.start()], buf[m.end():]
                    if not line.strip():
                        continue
                    pm = prog_re.search(line)
                    if pm:
                        if show and time.time() - last_render >= 0.2:
                            render(
                                int(pm.group(1)),
                                int(pm.group(2)) if pm.group(2) else 0,
                                float(pm.group(3)), float(pm.group(4)),
                            )
                    else:
                        tail.append(line.decode("utf-8", "replace"))
                        if len(tail) > 60:
                            tail.pop(0)
            enc.wait()
            ffp.wait()
            # The encoder's last progress line lands mid-GOP and its final
            # summary lines don't match prog_re, so a successful encode
            # would otherwise leave the bar frozen short of 100%.
            if rendered and enc.returncode == 0 and ffp.returncode == 0:
                render(0, 0, 0, 0, final=True)
    except BaseException:
        for p in (enc, ffp):
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
                p.wait()
        raise
    finally:
        if rendered:
            sys.stdout.write("\n")
            sys.stdout.flush()

    ff_err = ""
    try:
        ff_err = ff_log.read_text(encoding="utf-8", errors="ignore")
        ff_log.unlink()
    except OSError:
        pass
    _temp_files.discard(ff_log)

    # The encoder treats pipe EOF as a normal end ("Failed to read y4m
    # frame delimeter" on stderr is cosmetic) and can exit 0 on a feed
    # that died early — so a decode failure is checked independently.
    if enc.returncode != 0:
        raise RuntimeError(
            f"SvtAv1EncApp exit {enc.returncode}\n"
            f"{_proc_tail(chr(10).join(tail))}\n{_proc_tail(ff_err, 10)}"
        )
    if ffp.returncode != 0:
        raise RuntimeError(
            f"ffmpeg (y4m feed) exit {ffp.returncode}\n{_proc_tail(ff_err)}"
        )
    if not enc_out.exists() or enc_out.stat().st_size == 0:
        raise RuntimeError("Encoder produced no output")

    if not is_full:
        if dest.exists():
            dest.unlink()
        enc_out.rename(dest)
        _temp_files.discard(enc_out)
        return

    # Remux: AV1 video from the encoder + audio/subs/chapters/attachments
    # from the source. Subtitle copy can fail for codecs MKV won't take
    # as-is (e.g. mov_text from MP4) — retried as SRT, then dropped.
    tmp_mkv = dest.with_suffix(".tmp.mkv")
    _temp_files.add(tmp_mkv)

    def mux_cmd(maps, codecs):
        return [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-i", str(enc_out), "-i", str(source),
            *maps, "-map_chapters", "1", "-map_metadata", "1",
            *codecs, str(tmp_mkv),
        ]

    with_subs = ["-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", "-map", "1:t?"]
    no_subs = ["-map", "0:v:0", "-map", "1:a?", "-map", "1:t?"]
    attempts = [
        (with_subs, ["-c", "copy"], None),
        # MKV rejects some subtitle codecs as-is (e.g. mov_text from MP4)
        (with_subs, ["-c", "copy", "-c:s", "srt"], None),
        (no_subs, ["-c", "copy"], "subtitles incompatible with MKV — dropped"),
    ]
    last_err = None
    for maps, codecs, note in attempts:
        try:
            if note:
                print(f" {ORANGE}{'mux':<10}{RESET}{DIM}{note}{RESET}")
            run_cmd(mux_cmd(maps, codecs))
            last_err = None
            break
        except RuntimeError as e:
            last_err = e
    if last_err is not None:
        raise RuntimeError(f"Remux failed: {last_err}")

    try:
        enc_out.unlink()
    except OSError:
        pass
    _temp_files.discard(enc_out)
    if dest.exists():
        dest.unlink()
    tmp_mkv.rename(dest)
    _temp_files.discard(tmp_mkv)


class QuarterGrid(Grid):
    """SVT-AV1-Essential's quarter-step CRF grid."""

    step = CRF_STEP

    def quantize(self, v):
        return qcrf(v)

    def fmt(self, q):
        return crf_str(q)

    def fmt_delta(self, d):
        return f"{d:+.2f}"

    def floor(self, v):
        return math.floor(v * 4) / 4

    def ceil(self, v):
        return math.ceil(v * 4) / 4

    def span(self, lo, hi):
        return crf_range(lo, hi)


class EssentialEngine(Engine):
    sig = "avqe-c1"
    qname = "CRF"
    banner = "av1q-essential"
    banner_extra = f" {DIM}SVT-AV1-Essential {MIDDOT} VMAF{RESET}"
    grid = QuarterGrid()
    vmaf_key_base = "vmaf"
    sample_ext = ".webm"
    tmp_patterns = ("*.tmp.mkv", "*.tmp.webm")
    rec_q_key = "crf"
    rec_bound_keys = ("min_crf", "max_crf")
    # enc_args_sig is None for a plain run, so a recommended block written
    # before this field existed (missing -> .get() None) still matches a
    # plain rerun; a value present means --enc-args changed the output, so
    # the search is correctly re-run instead of resumed at the old CRF.
    rec_extra_keys = ("tune", "enc_args_sig")
    seed_key = "seed_crf"
    seed_prompt_hint = "(0.25 steps, Enter = auto)"
    cal_q_key = "at_crf"
    needs_expected_frames = True

    def cache_root(self, cfg):
        return cfg["e_cache_dir"]

    def q_bounds(self, cfg):
        return cfg["min_crf"], cfg["max_crf"]

    def seed_override(self, cfg):
        return cfg.get("seed_crf")

    def parse_user_q(self, raw):
        return qcrf(float(raw))

    def signature(self, cfg, crop=None):
        return enc_signature_e(cfg, crop)

    def setup(self, cfg):
        cfg["encoder_exe"] = find_encoder()  # raises FileNotFoundError
        # Optional: powers the SSIMULACRA2 info column only.
        cfg["ffvship_exe"] = find_ffvship_optional()
        cfg["vmaf_threads"] = os.cpu_count() or 4

    def make_dirs(self, cfg):
        cfg["e_cache_dir"].mkdir(parents=True, exist_ok=True)
        (cfg["e_cache_dir"] / "_ffindex").mkdir(parents=True, exist_ok=True)

    def gate(self, source, meta):
        # Y4M is CFR-only and FFVship pairs frames by index, so a
        # genuinely VFR source can't go through this pipeline without
        # silent desync. av1q's ffmpeg path handles VFR fine.
        if is_vfr(source, meta):
            return "VFR source — not pipeable as Y4M, use av1q.py for this file"
        return None

    def prepare_meta(self, source, meta, cfg):
        # HDR10 static metadata rides the encoder flags (Y4M carries
        # none); fetched once per file, used by every encode of it.
        meta["mastering"] = meta["cll"] = None
        if meta["hdr"]:
            meta["mastering"], meta["cll"] = probe_hdr_metadata(source)
        # Nominal frame cadence for the full-encode feed's CFR
        # normalization (see encode_essential). Fetched once per file.
        meta["rfps"] = get_rfps(source)
        return bool(meta["mastering"] or meta["cll"])

    def prep_sample(self, concat, meta, cfg):
        # The raw concat is only an intermediate here — the search runs
        # against the lossless clean re-encode (see clean_sample_source).
        return clean_sample_source(concat, meta, cfg)

    def encode(self, source, dest, meta, q, cfg,
               show_progress=False, expected_frames=0):
        encode_essential(
            source, dest, meta, q, cfg,
            show_progress=show_progress, expected_frames=expected_frames,
        )

    def ssimu2_info(self, ref, dist, meta, cfg, ref_index=None):
        return ssimu2.ssimu2_info(ref, dist, meta, cfg, ref_index=ref_index)

    def full_ref_index(self, cfg, source, file_hash, size):
        return cfg["e_cache_dir"] / "_ffindex" / f"{file_hash}.ffindex"

    def sample_ref_index(self, cfg, sample_src):
        # The sample concat gets its own persistent index (it's a cached
        # file reused across probes within this search).
        return cfg["e_cache_dir"] / "_ffindex" / f"{sample_src.stem}.ffindex"

    def dst_name(self, stem, q, token, ext):
        return f"{stem}_CRF{crf_str(q)}{token}{ext}"

