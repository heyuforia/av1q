"""Chunked full-file encodes, shared by every engine: the plan, the pool
of encoder processes, the one progress bar they share, and the join.

A full encode is split at source keyframes into pieces of about
CHUNK_TIME seconds (plan_chunks). Up to cfg["workers"] encoder processes
run at once, each on one piece (Engine.encode_chunk), inside a work dir
core.segments keeps, so a killed encode loses only the pieces in flight.
The finished pieces are joined with their exact timestamps, and the
source's audio, subtitles, fonts and chapters are muxed back. A source
that is not split is encoded as a single piece through the same route.

The seam rule. A piece is bounded by presentation time, never by a frame
or packet count: one packet is not always one frame (field-coded
interlaced H.264 codes a frame as two), and a packet that decodes to no
frame would slide every later seam. Each piece seeks early enough to
land at or before its first keyframe on every container
(sampling.keyframe_seek), decodes from there, and the engine keeps
exactly its own span, source frames for av1q and output frames of the
constant-rate feed for av1q-essential. The spans meet with no gap and
no overlap, so the joined picture holds every frame once, where a
single pass puts it.
"""

import bisect
import os
import shutil
import sys
import threading
import time

from . import segments
from .analyze import read_timeline
from .constants import CHUNK_TIME, RESUMABLE_MIN_DURATION, WORKER_THREADS
from .sampling import keyframe_seek
from .ui import BOLD, DIM, GREEN, RESET, fmt_time, label
from .util import _temp_files, partial_hash

US = 1_000_000


def default_workers():
    """Encoders to run at once when --workers is not given: one per
    WORKER_THREADS logical threads, never fewer than one."""
    return max(1, (os.cpu_count() or 1) // WORKER_THREADS)


class Piece:
    """One span of the picture, in µs on read_packets' timeline (the
    container start subtracted). start is None for the first piece, which
    reads from the file's start; end is None for the last, which runs to
    its end. seek is the input -ss that lands at or before the keyframe
    at start, None when reading from the start reaches it."""

    def __init__(self, index, start, end, seek):
        self.index, self.start, self.end, self.seek = index, start, end, seek

    def seconds(self, duration):
        """The span's length, the last piece's up to the file's duration."""
        end = duration if self.end is None else self.end / US
        return max(0.0, end - (self.start or 0) / US)


def plan_chunks(source, meta, cfg):
    """The pieces a full encode of this source is split into, or None
    for a single piece.

    Sources of RESUMABLE_MIN_DURATION or longer are split so a kill
    loses only the pieces in flight; shorter ones only when more than one
    encoder runs, or a split would buy nothing a single pass lacks.
    --no-resume keeps every file in one piece. Each cut is the first
    source keyframe at least CHUNK_TIME past the previous one, and a
    last piece shorter than half of that joins the one before it.

    The keyframes come from one packet scan per file, kept in meta for
    the file's later full encodes. A timeline that jumps, or has no
    keyframe to cut at, stays in one piece and says why.
    """
    duration = meta.get("duration") or 0.0
    if not cfg["resume_encodes"] or duration < CHUNK_TIME * 1.5:
        return None
    if duration < RESUMABLE_MIN_DURATION and cfg["workers"] < 2:
        return None
    if "timeline" not in meta:
        meta["timeline"] = read_timeline(
            source, duration, meta.get("start_time", 0.0))
    if meta["timeline"] is None:
        return None  # the packet scan printed why
    keyframes, jump = meta["timeline"]
    if jump is not None:
        print(
            f"{label('chunks')}{DIM}timestamps jump at {fmt_time(jump)},"
            f" encoding in one piece{RESET}"
        )
        return None

    cuts = []
    target = CHUNK_TIME
    while True:
        j = bisect.bisect_left(keyframes, target)
        if j == len(keyframes) or duration - keyframes[j] < CHUNK_TIME / 2:
            break
        cuts.append(keyframes[j])
        target = keyframes[j] + CHUNK_TIME
    if not cuts:
        print(
            f"{label('chunks')}{DIM}no keyframe to cut at,"
            f" encoding in one piece{RESET}"
        )
        return None

    pieces = [Piece(0, None, round(cuts[0] * US), None)]
    for i, cut in enumerate(cuts, 1):
        landing = keyframe_seek(cut, keyframes)
        pieces.append(Piece(
            i, round(cut * US),
            round(cuts[i] * US) if i < len(cuts) else None,
            round(landing[0] * US) if landing else None,
        ))
    return pieces


def plan_identity(pieces):
    """A plan in the manifest's JSON form: whole numbers and nulls only,
    so the identity read back from disk compares equal."""
    return [[p.start, p.end, p.seek] for p in pieces] if pieces else []


class Job:
    """One piece in flight: what its encoder has reported so far, and the
    processes an abort must stop.

    The engine registers every process it starts (track) and reports as
    it goes (report); the main thread reads the reports to draw the bar
    and stops the processes on Ctrl-C or on another piece's failure."""

    def __init__(self, span_s):
        self.span_s = span_s  # the piece's length, the most it can report
        self.done_s = 0.0   # seconds of this piece's picture encoded
        self.fps = 0.0
        self.size = 0       # bytes written so far
        self._procs = []
        self._stopped = False
        self._lock = threading.Lock()

    def track(self, proc):
        with self._lock:
            self._procs.append(proc)
            stopped = self._stopped
        if stopped:
            _terminate(proc)

    def report(self, done_s, fps=None, size=None):
        self.done_s = max(self.done_s, done_s)
        if fps is not None:
            self.fps = fps
        if size is not None:
            self.size = size

    def stop(self):
        with self._lock:
            self._stopped = True
            procs = list(self._procs)
        for p in procs:
            _terminate(p)


def _terminate(proc):
    try:
        proc.terminate()
    except Exception:
        pass


def encode_full(engine, source, dest, meta, q, cfg, show_progress=False):
    """Encode the whole picture of `source` at q and mux it with the
    source's other streams into dest.

    The work dir is keyed by source, settings and quantizer under the
    engine's own cache root, and resumed when its manifest matches: the
    pieces it holds are kept, the missing ones encoded. It is deleted
    once dest is final. Pieces and manifest deliberately stay OUT of
    _temp_files: surviving Ctrl-C and crashes is their entire purpose.
    """
    grid = engine.grid
    q_key = grid.fmt(q)
    file_hash = partial_hash(source)
    enc_tag = engine.signature(cfg, meta.get("crop"))
    extra = engine.chunk_identity(meta, cfg)
    pieces = plan_chunks(source, meta, cfg) if extra is not None else None
    split = pieces is not None
    if not split:
        pieces = [Piece(0, None, None, None)]
    duration = meta.get("duration") or 0.0

    sdir = segments.segment_dir(
        engine.cache_root(cfg), file_hash, enc_tag, q_key)
    manifest = segments.prepare(sdir, segments.manifest_expected(
        file_hash, enc_tag, q_key, plan_identity(pieces if split else None),
        extra or {},
    ))
    pending = [p for p in pieces if str(p.index) not in manifest["done"]]
    kept = len(pieces) - len(pending)
    if kept and split:
        print(
            f"{label('resume')}{BOLD}{kept}{RESET} of {len(pieces)}"
            f" pieces kept"
        )
    elif kept:
        print(f"{label('resume')}encoded picture kept")
    elif split:
        workers = min(cfg["workers"], len(pieces))
        print(
            f"{label('chunks')}{BOLD}{len(pieces)}{RESET} pieces"
            f"{DIM}, {workers} encoder{'s' if workers > 1 else ''}"
            f" at once{RESET}"
        )

    if pending:
        _run_pool(
            engine, source, sdir, manifest, pieces, pending, split,
            meta, q, cfg, duration,
            prefix=f"{label('encode')}{engine.qname} {BOLD}{q_key}{RESET}",
            show=show_progress and sys.stdout.isatty(),
        )

    records = [manifest["done"][str(p.index)] for p in pieces]
    if len(records) == 1:
        picture = sdir / records[0]["name"]
    else:
        picture = sdir / segments.JOINED_NAME
        segments.concat_segments(sdir, records, picture)

    tmp = dest.with_suffix(".tmp.mkv")
    _temp_files.add(tmp)
    segments.mux_with_source_streams(
        picture, source, tmp, start_ms=engine.mux_start_ms(meta),
        mastering=meta.get("mastering"), cll=meta.get("cll"),
    )
    tmp.replace(dest)
    _temp_files.discard(tmp)
    shutil.rmtree(sdir, ignore_errors=True)


def _run_pool(engine, source, sdir, manifest, pieces, pending, split,
              meta, q, cfg, duration, prefix, show):
    """Encode the pending pieces, up to cfg["workers"] at once, longest
    first so the run does not end on one long piece alone. Each finished
    piece is checked (Engine.chunk_start_us) and recorded in the manifest
    before the next starts on that worker.

    A failed piece stops every other one and raises its error; Ctrl-C
    stops them all too. Worker threads never see Ctrl-C (Python delivers
    it to the main thread only), so the main thread polls them and stops
    their processes itself: a join would wait for every piece in flight.
    """
    lock = threading.Lock()
    queue = sorted(pending, key=lambda p: (-p.seconds(duration), p.index))
    active = []
    errors = []
    failed = threading.Event()
    n = len(pieces)

    # What the bar counts as encoded: every recorded piece's span, plus
    # the running pieces' reports. Recorded before this run started, a
    # piece counts toward the total but not toward the speed.
    finished = {"s": 0.0, "size": 0}
    for p in pieces:
        rec = manifest["done"].get(str(p.index))
        if rec:
            finished["s"] += p.seconds(duration)
            try:
                finished["size"] += (sdir / rec["name"]).stat().st_size
            except OSError:
                pass
    resumed_s = finished["s"]
    total_s = duration or sum(p.seconds(duration) for p in pieces)

    def work():
        while not failed.is_set():
            with lock:
                if not queue:
                    return
                piece = queue.pop(0)
                job = Job(piece.seconds(duration))
                active.append(job)
            out = sdir / f"chunk_{piece.index:05d}{engine.chunk_ext}"
            try:
                engine.encode_chunk(
                    source, out, meta, q, cfg, piece if split else None, job)
                start_us = engine.chunk_start_us(
                    meta, piece if split else None, out)
                if start_us is None:
                    raise RuntimeError(
                        f"Piece {piece.index + 1} of {n} does not hold the"
                        f" span it was cut for. The next run encodes it"
                        f" again; if it never does, --no-resume encodes"
                        f" the file in one piece"
                    )
                with lock:
                    manifest["done"][str(piece.index)] = {
                        "name": out.name, "start_us": start_us,
                    }
                    segments.write_manifest(sdir, manifest)
                    finished["s"] += piece.seconds(duration)
                    finished["size"] += out.stat().st_size
            except BaseException as e:
                with lock:
                    errors.append(e)
                failed.set()
                return
            finally:
                with lock:
                    active.remove(job)

    workers = [
        threading.Thread(target=work, daemon=True)
        for _ in range(max(1, min(cfg["workers"], len(queue))))
    ]
    bar = _Bar(prefix, total_s, resumed_s) if show else None
    try:
        for t in workers:
            t.start()
        while any(t.is_alive() for t in workers):
            if failed.is_set():
                with lock:
                    for job in active:
                        job.stop()
            if bar:
                with lock:
                    done_s = finished["s"] + sum(
                        min(j.done_s, j.span_s) for j in active)
                    size = finished["size"] + sum(j.size for j in active)
                    fps = sum(j.fps for j in active)
                bar.draw(done_s, size, fps)
            time.sleep(0.2)
    except BaseException:
        failed.set()
        with lock:
            for job in active:
                job.stop()
        for t in workers:
            # A Ctrl-C between two starts leaves a thread never started,
            # which join refuses.
            if t.is_alive():
                t.join(timeout=10)
        raise
    finally:
        if bar:
            bar.close(final=not errors and not failed.is_set())
    if errors:
        raise errors[0]


class _Bar:
    """The one progress line every running encoder feeds: the share of
    the picture encoded, the time left at this run's pace, that pace
    against real time, the encoders' frames per second together, and the
    picture's bitrate so far."""

    WIDTH = 20

    def __init__(self, prefix, total_s, resumed_s):
        self.prefix, self.total_s, self.resumed_s = prefix, total_s, resumed_s
        self.t0 = time.time()
        self.drawn = False
        self._write(self.resumed_s, 0, 0.0)

    def draw(self, done_s, size, fps):
        self._write(done_s, size, fps)

    def _write(self, done_s, size, fps, final=False):
        pct = (
            100.0 if final
            else max(0.0, min(100.0, done_s / self.total_s * 100))
            if self.total_s > 0 else 0.0
        )
        filled = int(self.WIDTH * pct / 100)
        bar = (
            f"{DIM}[{RESET}{GREEN}{'█' * filled}{RESET}"
            f"{DIM}{'░' * (self.WIDTH - filled)}]{RESET}"
        )
        parts = [f"{BOLD}{pct:5.1f}%{RESET}"]
        if not final:
            elapsed = time.time() - self.t0
            speed = (done_s - self.resumed_s) / elapsed if elapsed > 0 else 0
            if speed > 0:
                left = max(0.0, self.total_s - done_s) / speed
                parts.append(f"{fmt_time(left)} left")
                parts.append(f"{speed:.2f}x")
            if fps > 0:
                parts.append(f"{fps:.1f}fps")
            if size > 0 and done_s > 0:
                parts.append(f"{size * 8 / 1000 / done_s:.0f}kbps")
        sys.stdout.write(f"\r\033[K{self.prefix} {bar} {'  '.join(parts)}")
        sys.stdout.flush()
        self.drawn = True

    def close(self, final):
        if final:
            self._write(self.total_s, 0, 0.0, final=True)
        if self.drawn:
            sys.stdout.write("\n")
            sys.stdout.flush()
