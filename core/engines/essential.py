"""SVT-AV1-Essential engine: ffmpeg decodes (applying crop) and pipes
10-bit Y4M into the standalone fork binary, a full picture in pieces
through core.chunks, then audio/subs/chapters are remuxed back from the
source. Quarter-step CRF grid; color and
HDR10 static metadata re-stated as encoder flags (Y4M carries none);
VFR sources gated out (the Y4M pipe is CFR-only)."""

import hashlib
import math
import os
import re
import subprocess

from .. import chunks, segments, ssimu2
from ..crop import crop_token
from ..probe import (
    content_light_str, is_vfr, picture_timing, svt_mastering_display,
)
from ..sampling import clean_sample_source
from ..tools import ffmpeg_exe, find_encoder, find_ffvship_optional
from ..ui import DIM, MIDDOT, RESET
from ..util import _temp_files, ascii_dir, make_temp_log, own_process_group
from .base import Engine, Grid


# Quarter-step CRF grid — SVT-AV1-Essential accepts CRF in 0.25 increments
# (values are floored to the grid inside the encoder).
CRF_STEP = 0.25

# Cold-start d(log kbps)/d(CRF). Essential's rate curve is roughly half
# as steep as the generic ln2/6 (2x per ~11 CRF, not per 6): four field
# files measured 0.059, 0.063, 0.064 and 0.065, and an independent
# earlier trace read ~0.07. Sitting just above the measured mean is the
# safe side of the error — too steep only under-jumps and spends another
# cheap sample probe, while too shallow can over-jump a full-file probe
# past the floor. Only the cold start: the per-file and cohort decays
# supersede it as soon as either exists.
ESSENTIAL_BITRATE_DECAY = 0.065


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
    "--progress", "--hide-banner", "--webm",
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
        args += ["--mastering-display",
                 svt_mastering_display(meta["mastering"])]
    if meta.get("cll"):
        args += ["--content-light", content_light_str(meta["cll"])]
    return args


def _proc_tail(text, n=40):
    return "\n".join((text or "").splitlines()[-n:])


def _rate(meta):
    """The feed's frame rate (EssentialEngine.prepare_meta) as (num, den),
    or None when ffmpeg has no guess for the stream."""
    rate = meta.get("picture_rate")
    if not rate:
        return None
    num, den = (int(v) for v in rate.split("/"))
    return num, den


def _slot(meta, t_us):
    """The feed's first output frame of a piece whose source span starts
    at t_us (µs on read_packets' timeline), 0 for the first piece.

    The fps filter places each input frame on the slot its pts rounds to
    and shows every slot the latest frame placed at or before it
    (vf_fps.c), so what a slot shows depends only on the frames up to it.
    The keyframe at t_us rounds to floor(x) or floor(x) + 1, x being its
    exact position in slots; float error is far under half a slot.
    Opening the piece at floor(x) + 1 therefore never needs a frame
    before its keyframe, which a piece that seeks there may not decode,
    and every slot it shows is the one the continuous feed shows. The
    keyframe's own slot, when it is floor(x), goes to the piece before,
    which decodes on past its end to fill it.
    """
    if t_us is None:
        return 0
    num, den = _rate(meta)
    x = (t_us / 1_000_000 - meta["picture_start"]) * num / den
    return math.floor(x) + 1


def _feed_cmd(source, meta, piece, full):
    """ffmpeg's Y4M feed of `source`: the whole search source for a
    sample probe (full False), else the whole picture (piece None) or
    one piece of it.

    A full feed normalizes the timeline: setpts zeroes the picture's
    start offset / edit-list delay and fps re-times onto a clean CFR grid
    at ffmpeg's own frame rate for the stream (probe.picture_timing), the
    rate its CFR outputs run at. This is a no-op for well-formed CFR
    sources, but for irregular ones (e.g. stream-copy concatenations with
    a per-join timing gap) it is what keeps the full encode frame-aligned
    with the VMAF reference, which applies the identical setpts+fps
    normalization on its side at the encode's own rate. Without it the
    encoder's own CFR conversion fills the gaps differently than the
    reference's fps filter and the two drift out of phase, collapsing
    full VMAF. r_frame_rate is not that rate: on an interlaced H.264
    source it is the field rate, and the feed would double every frame.
    Samples skip this: their search source is already a clean CFR
    re-encode (clean_sample_source) and they pair by index.

    A piece keeps that exact grid. It subtracts the picture's first pts
    as a constant (STARTPTS would be the piece's own first frame, and
    each piece would round onto a grid of its own), decodes from its
    seek under -copyts -start_at_zero, the timeline the constant was read
    on, and keeps the output slots from _slot(start) up to, not
    including, _slot(end): a trim on the fps filter's own 1/rate ticks
    at both ends, exact. The pieces then hold every slot of the
    continuous feed once, and a piece with fewer frames than its slots
    (a keyframe that did not decode) is short, never shifted.

    -nostdin: a piece is trusted on exit 0, and ffmpeg's interactive
    'q' would end the feed early at exit 0.
    """
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-v", "error", "-nostats",
           "-nostdin"]
    if piece is not None and piece.seek is not None:
        cmd += ["-ss", segments.us_ts(piece.seek)]
    cmd += ["-i", str(source), "-map", "0:v:0"]
    vf = []
    if meta.get("crop"):
        vf.append(f"crop={meta['crop']}")
    rate = meta["picture_rate"] if full else None
    if rate and piece is None:
        vf += ["setpts=PTS-STARTPTS", f"fps={rate}"]
    elif rate:
        vf += [f"setpts=PTS-({meta['picture_start_pts']})", f"fps={rate}"]
        bounds = []
        if _slot(meta, piece.start):
            bounds.append(f"start_pts={_slot(meta, piece.start)}")
        if piece.end is not None:
            bounds.append(f"end_pts={_slot(meta, piece.end)}")
        if bounds:
            vf.append("trim=" + ":".join(bounds))
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if piece is not None:
        cmd += ["-copyts", "-start_at_zero"]
    if rate:
        # Pass the fps filter's CFR frames through untouched; without this
        # the yuv4mpegpipe muxer re-runs its own CFR conversion on top,
        # which can diverge from the reference's fps filter at the same
        # rate and reintroduce the drift.
        cmd += ["-fps_mode", "passthrough"]
    return cmd + ["-pix_fmt", "yuv420p10le", "-strict", "-1",
                  "-f", "yuv4mpegpipe", "-"]


def _encode_y4m(ff_cmd, meta, crf, cfg, enc_out, job=None, frame_s=0.0):
    """Pipe ff_cmd's Y4M into SvtAv1EncApp at `crf`, which writes the
    picture to enc_out, a path in the cache root's ASCII spelling.
    Raises when either process fails or nothing was written.

    job, when given, is a core.chunks Job: both processes go to it, and
    each progress line is reported as seconds encoded (frames times
    frame_s), frames per second and bytes.

    Hardware decode is deliberately not used on this path: a mid-stream
    hwaccel failure can't be retried without restarting the encoder, and
    CPU decode comfortably outpaces SVT-AV1 at these presets.
    """
    try:
        if enc_out.exists():
            enc_out.unlink()
    except OSError:
        pass

    # --webm 0 pins IVF: release builds are compiled without WebM output,
    # a build with it defaults to WebM, and the two must not differ.
    enc_cmd = [
        str(cfg["encoder_exe"]), "-i", "stdin", "-b", str(enc_out),
        "--webm", "0",
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
    tail = []  # last stderr lines for error reporting
    # --progress 2's line. Under 1 frame per second the encoder states
    # frames per MINUTE ("fpm"), which a 4K encode at a slow preset
    # reaches. NO_COLOR below keeps ANSI colors out of the line (Windows
    # builds never color it).
    prog_re = re.compile(
        rb"Encoding:\s+(\d+)(?:/(\d+))?\s+Frames\s+@\s+([\d.]+)\s+fp([sm])"
        rb"\s+\|\s+([\d.]+)\s+kb/s"
    )

    ffp = enc = None
    try:
        with open(ff_log, "wb") as ferr:
            ffp = subprocess.Popen(ff_cmd, stdout=subprocess.PIPE, stderr=ferr,
                                   **own_process_group())
            if job:
                job.track(ffp)
            enc = subprocess.Popen(
                enc_cmd, stdin=ffp.stdout,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                env={**os.environ, "NO_COLOR": "1"}, **own_process_group(),
            )
            if job:
                job.track(enc)
            ffp.stdout.close()  # let EPIPE reach ffmpeg if the encoder dies

            # Drain encoder stderr continuously (it floods \r progress
            # lines).
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
                        if job:
                            per = 60 if pm.group(4) == b"m" else 1
                            done = int(pm.group(1)) * frame_s
                            job.report(
                                done, float(pm.group(3)) / per,
                                int(float(pm.group(5)) * 125 * done),
                            )
                    else:
                        tail.append(line.decode("utf-8", "replace"))
                        if len(tail) > 60:
                            tail.pop(0)
            enc.wait()
            ffp.wait()
    except BaseException:
        for p in (enc, ffp):
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass
                p.wait()
        raise

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


def _scratch(cfg, dest):
    """The encoder's output file for dest. The standalone encoder reads
    its output path as ANSI on Windows, and a full-encode dest carries
    the source's (possibly non-Latin) stem, so every encode writes a
    hash-named scratch file in the ASCII spelling of the cache root
    (EssentialEngine.setup), renamed to dest by Python, which takes
    Unicode paths."""
    tag = hashlib.sha256(str(dest).encode("utf-8")).hexdigest()[:16]
    enc_out = cfg["e_scratch"] / f"_enc_{tag}.tmp.ivf"
    _temp_files.add(enc_out)
    return enc_out


def encode_essential(source, dest, meta, crf, cfg, show_progress=False,
                     full=False):
    """Encode `source` to AV1 at `crf` via SVT-AV1-Essential.

    ffmpeg decodes (applying crop) and pipes 10-bit Y4M — Essential
    rejects 8-bit input by design — into the encoder, which writes a
    video-only IVF. A sample probe (full False) is that picture, renamed
    to dest. A full-file output (full True) goes to core.chunks, which
    encodes the picture in resumable pieces (encode_chunk_essential) and
    muxes audio/subs/chapters back from the source.
    """
    if full:
        chunks.encode_full(EssentialEngine(), source, dest, meta, crf, cfg,
                           show_progress=show_progress)
        return
    enc_out = _scratch(cfg, dest)
    _encode_y4m(_feed_cmd(source, meta, None, False), meta, crf, cfg, enc_out)
    enc_out.replace(dest)
    _temp_files.discard(enc_out)


def encode_chunk_essential(source, out, meta, crf, cfg, piece, job):
    """Encode one piece of a full-file picture to `out` (Engine.
    encode_chunk); piece None is the whole picture in one feed. The IVF
    counts its frames from 0 whatever the piece, so chunk_start_us places
    it on the joined timeline, and the joined picture's start against the
    audio comes from the source (picture_start) at the mux."""
    rate = _rate(meta) or _rate({"picture_rate": meta.get("fps")})
    frame_s = rate[1] / rate[0] if rate else 0.0
    enc_out = _scratch(cfg, out)
    _encode_y4m(_feed_cmd(source, meta, piece, True), meta, crf, cfg,
                enc_out, job, frame_s)
    enc_out.replace(out)
    _temp_files.discard(enc_out)


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
    sample_ext = ".ivf"  # the encoder's own output, pinned by --webm 0
    # The clean-sample temp is written by prep_sample's lossless x264
    # pass beside the shared concats, hence its pattern here.
    tmp_patterns = (
        "*_CRF*.tmp.mkv", "_enc_*.tmp.ivf", "samples_*_clean.tmp.mkv",
    )
    ffmpeg_encoders = ("libx264",)  # clean_sample_source's lossless pass
    ffmpeg_filters = ("libvmaf",)
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
    chunk_ext = ".ivf"  # the encoder's own output, pinned by --webm 0
    default_decay = ESSENTIAL_BITRATE_DECAY

    def cache_root(self, cfg):
        return cfg["e_cache_dir"]

    def calibration_root(self, cfg):
        return cfg["learned_dir"] / "essential"

    def q_bounds(self, cfg):
        return cfg["min_crf"], cfg["max_crf"]

    def seed_override(self, cfg):
        return cfg.get("seed_crf")

    def parse_user_q(self, raw):
        v = float(raw)
        if not math.isfinite(v):  # "inf" parses; qcrf can't round it
            raise ValueError(raw)
        return qcrf(v)

    def signature(self, cfg, crop=None):
        return enc_signature_e(cfg, crop)

    def setup(self, cfg):
        cfg["encoder_exe"] = find_encoder()  # raises FileNotFoundError
        # The encoder writes every encode into the cache root, and reads
        # that path as ANSI on Windows: a folder with no ASCII spelling
        # would fail every file after its scan and samples were paid for.
        self.make_dirs(cfg)
        cfg["e_scratch"] = ascii_dir(cfg["e_cache_dir"])
        if cfg["e_scratch"] is None:
            raise OSError(
                f"SvtAv1EncApp cannot write under {cfg['e_cache_dir']}: it"
                f" reads file paths in the Windows code page, and that"
                f" folder has no short ASCII name. Move the av1q folder to"
                f" a path with only ASCII characters."
            )
        # Optional, powers the SSIMULACRA2 info column only. Probed (and
        # on first run downloaded) here so that happens before the seed
        # prompt, not mid-search; the result is memoized.
        find_ffvship_optional()

    def launch_notes(self, cfg):
        # The file name carries the build's version. Without this line
        # the build that runs is invisible, and two builds under tools/
        # resolve by path order.
        return (f"encoder: {os.path.basename(cfg['encoder_exe'])}",)

    def make_dirs(self, cfg):
        cfg["e_cache_dir"].mkdir(parents=True, exist_ok=True)

    def gate(self, source, meta):
        # Y4M is CFR-only and FFVship pairs frames by index, so a
        # genuinely VFR source can't go through this pipeline without
        # silent desync. av1q's ffmpeg path handles VFR fine.
        if is_vfr(source, meta):
            return "VFR source — not pipeable as Y4M, use av1q.py for this file"
        return None

    def prepare_meta(self, source, meta, cfg):
        # Read once per file, used by every full encode of it: the Y4M
        # pipe carries no timestamps, so the feed's rate and the
        # picture's start against the audio (restored at the mux) come
        # from ffmpeg's own decode of the source. Raises when the source
        # decodes no frame, which no encode of it could survive either.
        timing = picture_timing(source)
        meta["picture_start"] = timing["start"]
        meta["picture_start_pts"] = timing["start_pts"]
        meta["picture_rate"] = timing["rate"]
        # The HDR10 static metadata build_color_args restates. A failed
        # read raises too: the pipe carries no copy of its own.
        return super().prepare_meta(source, meta, cfg)

    def prep_sample(self, concat, meta, cfg):
        # The raw concat is only an intermediate here — the search runs
        # against the lossless clean re-encode (see clean_sample_source).
        return clean_sample_source(concat, meta, cfg)

    def encode(self, source, dest, meta, q, cfg,
               show_progress=False, resumable=False):
        encode_essential(source, dest, meta, q, cfg,
                         show_progress=show_progress, full=resumable)

    def chunk_identity(self, meta, cfg):
        # The pieces are cut on the feed's own frame grid, which the rate
        # and the picture's first pts define (_feed_cmd). Without both
        # there is no grid to cut on, and the picture stays in one piece.
        if (not meta.get("picture_rate")
                or meta.get("picture_start_pts") is None):
            return None
        return {"rate": meta["picture_rate"],
                "start_pts": meta["picture_start_pts"]}

    def encode_chunk(self, source, out, meta, q, cfg, piece, job):
        encode_chunk_essential(source, out, meta, q, cfg, piece, job)

    def chunk_start_us(self, meta, piece, out):
        # The IVF counts from 0 and carries no timestamp to read back,
        # so a piece is checked by its frame count: exactly the slots it
        # was cut for (the last piece runs to the end, uncounted). The
        # feed's trim closes at the next piece's first slot, so a
        # keyframe that did not decode, or a feed or encoder that
        # stopped early, leaves the piece short. Its start is its first
        # slot, placed by the rate.
        if piece is None or piece.start is None:
            start = 0
        else:
            num, den = _rate(meta)
            start = round(_slot(meta, piece.start) * den * 1_000_000 / num)
        if piece is not None and piece.end is not None:
            want = _slot(meta, piece.end) - _slot(meta, piece.start)
            if segments.probe_packet_count(out) != want:
                return None
        return start

    def mux_start_ms(self, meta):
        # The encoder writes its picture from 0, so its start on the
        # source's timeline comes from the source itself.
        return round(meta["picture_start"] * 1000)

    def ssimu2_info(self, ref, dist, meta, cfg, ref_index=None):
        return ssimu2.ssimu2_info(ref, dist, meta, cfg, ref_index=ref_index)

    def dst_name(self, stem, q, token, ext):
        return f"{stem}_CRF{crf_str(q)}{token}{ext}"

