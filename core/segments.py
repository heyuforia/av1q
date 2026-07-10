"""Resumable segmented full-file encodes: manifest, validation, concat,
and the shared source-stream remux.

One continuous encoder writes keyframe-aligned segment files through
ffmpeg's segment muxer, which finalizes each completed segment (header,
trailer, cues) before opening the next — so a killed encode keeps every
finished segment and only the in-flight one is lost. Resume restarts the
encoder at a segment boundary, the segments are stream-copy concatenated,
and audio/subs are remuxed from the source at the end.

The timestamp contract that makes the concat bit-compatible with a
single-pass encode: segments are written with -reset_timestamps 0 (the
output timeline's PTS pass straight through into the segment files), a
resumed encode re-enters that same timeline via -ss/-copyts/-start_at_zero,
and the concat list declares each segment's exact duration (next segment's
first PTS minus this one's), which zeroes the concat demuxer's timestamp
delta so the original PTS survive unchanged. Full-file VMAF pairs frames
by timestamp — this is what keeps it aligned.
"""

import json
import shutil

from .ui import DIM, ORANGE, RESET
from .util import atomic_write_json, run_cmd

MANIFEST_NAME = "manifest.json"
SEGMENT_LIST_NAME = "segments.csv"
SEGMENT_PATTERN = "seg_%05d.mkv"
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


def manifest_expected(file_hash, enc_tag, q_key, segment_time):
    """The identity a segment dir must match to be resumed. Any mismatch
    (source changed, settings changed, different quantizer) means the
    segments were produced by a different encode and must be discarded."""
    return {
        "source_hash": file_hash,
        "enc_tag": enc_tag,
        "q": q_key,
        "segment_time": segment_time,
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


def parse_segment_list(csv_path):
    """Segment basenames from ffmpeg's -segment_list CSV, in file order.

    Rows are `filename,start_time,end_time`, written as each segment is
    finalized — so the list is the authority on which segments completed.
    A kill can tear the last line mid-write; malformed rows are skipped.
    Only the basename is trusted (the filename column echoes whatever
    pattern path ffmpeg was given).
    """
    try:
        text = csv_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    names = []
    for line in text.splitlines():
        parts = line.rsplit(",", 2)
        if len(parts) != 3:
            continue
        name = parts[0].replace("\\", "/").rsplit("/", 1)[-1].strip()
        try:
            float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if name:
            names.append(name)
    return names


def _probe_start_ms(path):
    """First video packet PTS in ms (MKV's native timescale), or None.

    This is the ms-exact resume/concat anchor — CSV float seconds are
    not trusted for timeline math, the container is.
    """
    try:
        r = run_cmd([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
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


def _probe_duration_s(path):
    """Container duration in seconds (finalized segments carry it)."""
    try:
        r = run_cmd([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", str(path),
        ])
        return float((r.stdout or "").strip().rstrip(","))
    except (RuntimeError, ValueError):
        return None


def validate_new_segments(seg_dir, known, probe=_probe_start_ms):
    """Validate this run's CSV segments against what's on disk.

    Returns the ordered list of {"name", "start_ms"} entries for segments
    that verifiably completed: listed in the CSV, present, non-empty,
    probeable, and monotonically after the previous one. Validation stops
    at the first failure — segments after a gap can't be stitched. Never
    trust the list alone (a finished-looking manifest with a missing file
    is how resume ships a truncated video).
    """
    known_names = {s["name"] for s in known}
    last_ms = known[-1]["start_ms"] if known else -1
    out = []
    for name in parse_segment_list(seg_dir / SEGMENT_LIST_NAME):
        if name in known_names:
            continue
        p = seg_dir / name
        try:
            if not p.is_file() or p.stat().st_size == 0:
                break
        except OSError:
            break
        start = probe(p)
        if start is None or start <= last_ms:
            break
        out.append({"name": name, "start_ms": start})
        last_ms = start
    return out


def resume_state(seg_dir, manifest, probe=_probe_start_ms):
    """Reconcile an interrupted dir: (kept_segments, resume_ms).

    Merges the manifest's previously validated segments with whatever the
    interrupted run's CSV finalized, then drops the LAST segment: the
    resume seek target must be the exact PTS of the first frame not yet
    encoded, and a segment's first-packet PTS is the only boundary that
    is knowable exactly (a last-packet-plus-duration guess can drop or
    duplicate a frame at the seam on irregular sources). Re-encoding one
    segment is the price of an exact seam. Every file in the dir that
    isn't kept (the dropped segment, the in-flight truncated one) is
    deleted. resume_ms is None for a fresh start.
    """
    kept = []
    for s in manifest.get("segments") or []:
        p = seg_dir / s.get("name", "")
        try:
            if not (isinstance(s.get("start_ms"), int) and p.is_file()
                    and p.stat().st_size > 0):
                break
        except OSError:
            break
        kept.append({"name": s["name"], "start_ms": s["start_ms"]})
    kept += validate_new_segments(seg_dir, kept, probe=probe)

    resume_ms = None
    if kept:
        dropped = kept.pop()
        resume_ms = dropped["start_ms"]
    if not kept:
        resume_ms = None

    keep_names = {s["name"] for s in kept}
    for p in seg_dir.glob("seg_*.mkv"):
        if p.name not in keep_names:
            try:
                p.unlink()
            except OSError:
                pass
    return kept, resume_ms


def ms_ts(ms):
    """ms -> an exact 'S.mmm' seconds string for -ss / duration fields."""
    return f"{ms // 1000}.{ms % 1000:03d}"


def build_concat_list(segments, last_duration_s):
    """ffconcat text with exact per-segment durations.

    duration(i) = start(i+1) - start(i): declaring the exact slice length
    zeroes the concat demuxer's per-file timestamp delta, so the original
    PTS pass through unchanged. The last segment's duration only affects
    total-duration metadata; the container's own value is fine there.
    """
    lines = ["ffconcat version 1.0"]
    for i, s in enumerate(segments):
        lines.append(f"file '{s['name']}'")
        if i + 1 < len(segments):
            lines.append(
                f"duration {ms_ts(segments[i + 1]['start_ms'] - s['start_ms'])}"
            )
        elif last_duration_s:
            lines.append(f"duration {last_duration_s:.6f}")
    return "\n".join(lines) + "\n"


def concat_segments(seg_dir, segments, out_path,
                    probe_duration=_probe_duration_s):
    """Stream-copy concat all segments into one video-only file."""
    if not segments:
        raise RuntimeError("No segments to concatenate")
    last_dur = probe_duration(seg_dir / segments[-1]["name"])
    lst = seg_dir / CONCAT_LIST_NAME
    lst.write_text(build_concat_list(segments, last_dur), encoding="utf-8")
    try:
        if out_path.exists():
            out_path.unlink()
    except OSError:
        pass
    run_cmd([
        "ffmpeg", "-y", "-hide_banner", "-v", "error",
        "-f", "concat", "-safe", "0", "-i", str(lst),
        "-map", "0:v:0", "-c", "copy", str(out_path),
    ])
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("Segment concat produced no output")


def mux_with_source_streams(video, source, dest_tmp, attachments=False):
    """Mux encoded video with audio/subs/chapters/metadata from `source`
    into `dest_tmp`. Subtitle copy can fail for codecs MKV won't take
    as-is (e.g. mov_text from MP4) — retried as SRT, then dropped.
    """
    def mux_cmd(maps, codecs):
        return [
            "ffmpeg", "-y", "-hide_banner", "-v", "error",
            "-i", str(video), "-i", str(source),
            *maps, "-map_chapters", "1", "-map_metadata", "1",
            *codecs, str(dest_tmp),
        ]

    extra = ["-map", "1:t?"] if attachments else []
    with_subs = ["-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", *extra]
    no_subs = ["-map", "0:v:0", "-map", "1:a?", *extra]
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
