"""The work dirs of full-file encodes (manifest, probes, concat) and the
source-stream mux every full encode of both engines ends with.

core.chunks encodes a full picture as pieces, each written by its own
encoder process into one work dir per (source, settings, quantizer).
A piece is renamed into place only when its encoder exited cleanly and
the manifest then records it, so a killed encode keeps every finished
piece and loses only the ones in flight. The pieces are stream-copy
concatenated, and audio/subs are muxed from the source at the end.

The timestamp contract of the concat: every piece's record holds where
its picture starts on the joined timeline, and the concat list declares
each piece's exact duration (next piece's start minus this one's), which
zeroes the concat demuxer's timestamp delta so every piece keeps its own
timestamps. A split encode starts a new encoder keyframe at each seam,
so its bitstream is not the one a single pass would write; the frames
and their timestamps are. The mux then keeps the video's start offset
against the source's other streams (read from the video file, or
supplied by an engine whose encoder output restarts at 0).
"""

import json
import shutil

from .probe import content_light_str, ffmpeg_mastering_display
from .tools import ffmpeg_exe, ffmpeg_options, ffprobe_exe
from .ui import DIM, RESET, label
from .util import atomic_write_json, run_cmd

# ffmpeg's input options that set a stream's HDR10 static metadata,
# first released in ffmpeg 9.0.
HDR10_MUX_OPTIONS = ("mastering_display", "content_light")

MANIFEST_NAME = "manifest.json"
CONCAT_LIST_NAME = "concat.txt"
JOINED_NAME = "joined.mkv"


def segment_root(cache_root):
    return cache_root / "_segments"


def segment_dir(cache_root, file_hash, enc_tag, q_key):
    """Work directory for one (source, settings, quantizer) encode.

    Per-quantizer dirs keep refine re-encodes independently resumable;
    the manifest inside re-checks the full identity, so the short hash
    prefix only needs to avoid collisions, not carry meaning.
    """
    return segment_root(cache_root) / f"{file_hash[:8]}_{enc_tag}_{q_key}"


def manifest_expected(file_hash, enc_tag, q_key, plan, extra):
    """The identity a work dir must match to be resumed. Any mismatch
    (source changed, settings changed, different quantizer, other
    pieces) means the pieces were produced by a different encode and
    must be discarded.

    plan is core.chunks' piece list in its JSON form (whole numbers
    only, so the identity read back from disk compares equal), [] for a
    picture encoded in one piece. extra holds what the engine's pieces
    also depend on beside its signature (Engine.chunk_identity).

    The identity holds what the settings choose, never the encoder build
    or how the code spells the encode's params. So a dir left before an
    encoder upgrade or a params change joins its pieces to new ones: a
    valid bitstream with one seam (params added later, such as the HDR10
    metadata, then ride only the pieces encoded from then on).
    --overwrite keeps the dir too, on purpose: discarding it would cost
    an interrupted --overwrite run its work on the natural rerun with
    the same command line."""
    return {
        "source_hash": file_hash,
        "enc_tag": enc_tag,
        "q": q_key,
        "plan": plan,
        **extra,
    }


def load_manifest(seg_dir):
    """Parse the manifest, or None when missing/torn (an unreadable
    manifest means the dir can't be trusted and gets rebuilt)."""
    try:
        data = json.loads(
            (seg_dir / MANIFEST_NAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def manifest_matches(manifest, expected):
    return bool(manifest) and all(
        manifest.get(k) == v for k, v in expected.items()
    )


def write_manifest(seg_dir, manifest):
    atomic_write_json(seg_dir / MANIFEST_NAME, manifest)


def _on_disk(seg_dir, s):
    """A manifest record whose piece is still present and non-empty
    (anything malformed, e.g. a hand-edited manifest, is not)."""
    if not (isinstance(s, dict) and isinstance(s.get("name"), str)
            and isinstance(s.get("start_us"), int)):
        return False
    try:
        p = seg_dir / s["name"]
        return p.is_file() and p.stat().st_size > 0
    except OSError:
        return False


def prepare(seg_dir, expected):
    """Bring a work dir to a resumable state for one encode identity and
    return its manifest.

    manifest["done"] maps a piece's index (as a string, JSON's only key
    type) to its record {"name", "start_us"}. A record is kept only while
    its file is still on disk: a manifest that says "done" is not
    evidence. Every other file in the dir (a piece killed in flight, an
    interrupted join) is deleted, and a dir made by another identity is
    discarded whole.
    """
    manifest = load_manifest(seg_dir)
    if not manifest_matches(manifest, expected):
        shutil.rmtree(seg_dir, ignore_errors=True)
        manifest = {**expected, "done": {}}
    seg_dir.mkdir(parents=True, exist_ok=True)

    done = manifest.get("done")
    kept = {
        k: {"name": s["name"], "start_us": s["start_us"]}
        for k, s in (done.items() if isinstance(done, dict) else ())
        if _on_disk(seg_dir, s)
    }
    manifest["done"] = kept
    keep = {s["name"] for s in kept.values()} | {MANIFEST_NAME}
    for p in seg_dir.iterdir():
        if p.name not in keep:
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
            except OSError:
                pass
    write_manifest(seg_dir, manifest)
    return manifest


def probe_start_ms(path):
    """First video packet PTS in ms (MKV's native timescale), or None.

    This is the ms-exact concat and mux anchor: the container is trusted
    for timeline math, never a planned time.
    """
    try:
        r = run_cmd([
            ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts", "-of", "csv=p=0",
            "-read_intervals", "%+#1", str(path),
        ])
    except RuntimeError:
        return None
    for line in (r.stdout or "").splitlines():
        tok = line.strip().rstrip(",")
        if tok:
            try:
                return int(tok)
            except ValueError:
                return None
    return None


def probe_pts_span(path):
    """(first, last) video packet PTS of an MKV piece in ms, or None when
    it has no readable packet: where its picture starts and ends."""
    try:
        r = run_cmd([
            ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts", "-of", "csv=p=0", str(path),
        ])
    except RuntimeError:
        return None
    pts = []
    for line in (r.stdout or "").splitlines():
        try:
            pts.append(int(line.strip().rstrip(",")))
        except ValueError:
            continue  # N/A, or a blank line
    return (min(pts), max(pts)) if pts else None


def probe_packet_count(path):
    """Video packets in a piece, demux only, or None when it cannot be
    read. On an AV1 piece each packet is one shown frame: a temporal
    unit, which the encoder writes as one IVF frame and one MKV block."""
    try:
        r = run_cmd([
            ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "default=nw=1:nk=1", str(path),
        ])
        return int((r.stdout or "").strip())
    except (RuntimeError, ValueError):
        return None


def probe_duration_s(path):
    """Container duration in seconds (a finished piece carries it)."""
    try:
        r = run_cmd([
            ffprobe_exe(), "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ])
        return float((r.stdout or "").strip().rstrip(","))
    except (RuntimeError, ValueError):
        return None


def ms_ts(ms):
    """ms -> an exact 'S.mmm' seconds string for -ss / duration fields."""
    return f"{ms // 1000}.{ms % 1000:03d}"


def us_ts(us):
    """µs -> an exact 'S.uuuuuu' seconds string, the finest step ffmpeg
    parses a time into."""
    return f"{us // 1_000_000}.{us % 1_000_000:06d}"


def build_concat_list(segments, last_duration_s):
    """ffconcat text with exact per-piece durations.

    duration(i) = start(i+1) - start(i): declaring the exact slice length
    zeroes the concat demuxer's per-file timestamp delta, so the original
    PTS pass through unchanged. The last piece's duration only affects
    total-duration metadata; the container's own value is fine there.
    """
    lines = ["ffconcat version 1.0"]
    for i, s in enumerate(segments):
        lines.append(f"file '{s['name']}'")
        if i + 1 < len(segments):
            lines.append(
                f"duration {us_ts(segments[i + 1]['start_us'] - s['start_us'])}"
            )
        elif last_duration_s:
            lines.append(f"duration {last_duration_s:.6f}")
    return "\n".join(lines) + "\n"


def concat_segments(seg_dir, segments, out_path,
                    probe_duration=probe_duration_s):
    """Stream-copy concat all pieces, in order, into one video-only file.

    The concat demuxer re-bases every file by (cumulative start − the
    file's own start), which with exact durations is a uniform shift of
    −first_start across the whole timeline: zero for the normal case of
    video starting at PTS 0, but a video-led start offset would land the
    joined video earlier than a single-pass encode places it (a constant
    A/V offset once audio is remuxed). -output_ts_offset adds exactly
    that first start back, restoring the original PTS in every case.

    The joined file is a second full copy of the video beside its pieces
    for a short time, kept on purpose: disk space is not the constraint,
    and the work dir is deleted once the output is final.
    """
    if not segments:
        raise RuntimeError("No pieces to concatenate")
    last_dur = probe_duration(seg_dir / segments[-1]["name"])
    lst = seg_dir / CONCAT_LIST_NAME
    lst.write_text(build_concat_list(segments, last_dur), encoding="utf-8")
    try:
        if out_path.exists():
            out_path.unlink()
    except OSError:
        pass
    run_cmd([
        ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-map", "0:v:0", "-c", "copy",
        "-output_ts_offset", us_ts(segments[0]["start_us"]),
        str(out_path),
    ])
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Piece concat produced no output")


def mux_states_hdr10():
    """True when the resolved ffmpeg can state HDR10 static metadata for
    the output container (mux_with_source_streams)."""
    return set(HDR10_MUX_OPTIONS) <= ffmpeg_options()


def mux_with_source_streams(video, source, dest_tmp, probe=None,
                            start_ms=None, mastering=None, cll=None):
    """Mux encoded video with audio, subs, attachments (subtitle fonts),
    chapters and metadata from `source` into `dest_tmp`. Subtitle copy can
    fail for codecs MKV won't take as-is (e.g. mov_text from MP4) — retried
    as SRT, then dropped.

    mastering and cll are the source's HDR10 static metadata in
    probe_hdr_metadata's form, stated for the container's own copy
    (Matroska's MasteringMetadata, MaxCLL and MaxFALL) when the build
    has the options. A stream copy takes that copy from its input
    stream, which never has it from essential's IVF and has only
    ffmpeg's first-frame snapshot from av1q's encode; the options
    replace it before the copy is made. The bitstream's copy, the one a
    player decodes, is the encoder's.

    ffmpeg shifts every input so its own first timestamp reads 0 (unless
    -copyts, which also switches off MPEG-TS discontinuity repair). An
    encode that keeps the source's timeline starts where the source's
    picture starts, which on a file whose audio leads is later than the
    source's first timestamp; rebased to 0 alone, the picture would play
    that much early against the audio. -itsoffset by the video's own
    start cancels its shift. Discontinuity repair is per ffmpeg run: the
    picture is repaired in its encode run (a timeline with a jump is
    never split, see analyze.read_timeline) and the audio here, so a
    timestamp jump the two streams do not share equally can land them a
    little apart.

    start_ms is that start for a video file that does not carry it (a
    pipe-fed encoder writes its picture from 0); None reads it from the
    video file.
    """
    if start_ms is None:
        start_ms = (probe or probe_start_ms)(video)
    if start_ms is None:
        raise RuntimeError(f"Remux failed: {video.name} is unreadable")
    offset = ["-itsoffset", ms_ts(start_ms)] if start_ms > 0 else []
    hdr = []
    if (mastering or cll) and mux_states_hdr10():
        if mastering:
            hdr += ["-mastering_display:v:0",
                    ffmpeg_mastering_display(mastering)]
        if cll:
            hdr += ["-content_light:v:0", content_light_str(cll)]

    def mux_cmd(maps, codecs):
        return [
            ffmpeg_exe(), "-y", "-hide_banner", "-v", "error",
            *hdr, *offset, "-i", str(video), "-i", str(source),
            *maps, "-map_chapters", "1", "-map_metadata", "1",
            *codecs, str(dest_tmp),
        ]

    with_subs = [
        "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", "-map", "1:t?",
    ]
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
                print(f"{label('mux')}{DIM}{note}{RESET}")
            run_cmd(mux_cmd(maps, codecs))
            last_err = None
            break
        except RuntimeError as e:
            last_err = e
    if last_err is not None:
        raise RuntimeError(f"Remux failed: {last_err}")


def cleanup_file_segments(cache_root, file_hash):
    """Remove every segment work dir for one source file (all quantizers).
    Called once its final output exists — the resume state has served its
    purpose."""
    root = segment_root(cache_root)
    if not root.is_dir():
        return
    for d in root.glob(f"{file_hash[:8]}_*"):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def sweep_orphan_segments(cache_root):
    """Startup hygiene: drop segment dirs whose manifest is missing or
    torn — without a trustworthy manifest nothing in them can be resumed.
    Dirs with a valid manifest are kept indefinitely: they are the resume
    state, not junk."""
    root = segment_root(cache_root)
    if not root.is_dir():
        return
    for d in root.iterdir():
        if d.is_dir() and load_manifest(d) is None:
            shutil.rmtree(d, ignore_errors=True)
