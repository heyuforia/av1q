"""Process, filesystem, and hashing utilities shared by every stage."""

import contextlib
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .constants import SCAN_TIMEOUT_MAX, SCAN_TIMEOUT_MIN

_temp_files = set()


@contextlib.contextmanager
def suppress_win_error_dialog():
    """Stop a child process that fails to start — e.g. FFVship's
    0xc0000142 DLL-init crash — from popping a modal Windows error box
    that blocks the batch until it's clicked away.

    A child inherits its parent's process error mode at spawn time, so
    setting SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX across the
    spawn suppresses both the loader "unable to start correctly" box and
    the crash box. The child still exits non-zero, which callers already
    treat as failure. The prior mode is restored afterward, so nothing
    else is affected. No-op off Windows.
    """
    if os.name != "nt":
        yield
        return
    import ctypes

    SEM_FAILCRITICALERRORS = 0x0001
    SEM_NOGPFAULTERRORBOX = 0x0002
    k32 = ctypes.windll.kernel32
    prev = k32.SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX)
    try:
        yield
    finally:
        k32.SetErrorMode(prev)


def fmt_cmd(cmd):
    """A command line spelled the way this OS's shell would take it, for
    failure messages."""
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return " ".join(map(shlex.quote, cmd))


def run_cmd(cmd):
    """Run a command and return the result. Raises on failure.

    Output is decoded as UTF-8 (ffmpeg/ffprobe always emit UTF-8): the
    default locale codec is cp1252 on Windows and decodes strictly, so a
    non-ASCII filename echoed in an error message would raise
    UnicodeDecodeError instead of surfacing the real failure.
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        tail = "\n".join((p.stderr or "").splitlines()[-80:])
        raise RuntimeError(f"exit {p.returncode}\n{fmt_cmd(cmd)}\n{tail}")
    return p


def cleanup_temp():
    for path in list(_temp_files):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
        _temp_files.discard(path)


def run_launcher(main):
    """Process wrapper behind every launcher's __main__ block.

    Ctrl-C is a clean stop (temps swept, exit 0); any other escape is a
    bug or a broken environment, so the traceback is printed and the
    exit code says failure. Either way the "Press Enter" pause still
    runs: the scripts are double-clicked as often as typed, and a window
    that closes on the traceback loses the one thing worth reading.
    """
    try:
        code = main() or 0
    except KeyboardInterrupt:
        cleanup_temp()
        code = 0
    except Exception:
        cleanup_temp()
        traceback.print_exc()
        code = 1
    try:
        input("\nPress Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        pass
    sys.exit(code)


def atomic_write_json(path, obj, indent=None):
    """Write JSON to `path` atomically: serialize to a sibling .tmp file,
    flush it to disk, then replace. Neither a crash mid-write nor a power
    cut right after the rename can leave a torn or empty cache/sidecar:
    without the fsync the rename can land before the data does.

    A write stopped by an error or Ctrl-C deletes its .tmp. One left by a
    hard kill carries the target's own name, so the next write of that
    target replaces it.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, indent=indent))
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def make_temp_log(cache_dir, prefix, ext):
    """Create and register a unique temp-log path under `cache_dir`.

    Names combine pid and a microsecond timestamp so they stay unique even
    in tight loops (cropdetect runs one per window). The file is added to
    `_temp_files` so cleanup_temp() removes it if the caller doesn't.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    log = cache_dir / f"{prefix}_{os.getpid()}_{int(time.time() * 1_000_000)}.{ext}"
    _temp_files.add(log)
    return log


def escape_filter_path(path):
    """Escape a Path for use as a log-file option inside an ffmpeg filter
    graph (scdet/cropdetect metadata, libvmaf's log).

    A filter graph is parsed twice, and each pass has its own special
    characters: the option value's separator ':' (which collides with a
    Windows drive letter) plus the quote and escape characters, then the
    graph's own '[],;' plus the same two. Each level is escaped in turn
    with a backslash, so a drive colon becomes '\\\\:' and an apostrophe
    or comma in a folder name survives instead of ending the option.
    """
    s = path.as_posix()
    for special in ("\\':", "\\'[],;"):
        s = "".join(f"\\{c}" if c in special else c for c in s)
    return s


def scan_budget(duration):
    """Timeout in seconds for one whole-file pass over a source of this
    runtime (see SCAN_TIMEOUT_* in constants); the floor when the
    duration is unknown."""
    if duration and duration > 0:
        return int(clamp(duration, SCAN_TIMEOUT_MIN, SCAN_TIMEOUT_MAX))
    return SCAN_TIMEOUT_MIN


def _short_path_win(s):
    """Windows 8.3 short-path alias for `s` (an existing file or folder),
    or None on failure. When the volume has no 8.3 name for the path
    (creation disabled, or the path predates it) GetShortPathNameW returns
    the long path unchanged, which the caller rejects as non-ASCII.
    """
    import ctypes
    from ctypes import wintypes

    fn = ctypes.windll.kernel32.GetShortPathNameW
    fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    fn.restype = wintypes.DWORD
    buf = ctypes.create_unicode_buffer(512)
    n = fn(s, buf, len(buf))
    if n > len(buf):  # buffer too small; n is the required size incl. null
        buf = ctypes.create_unicode_buffer(n)
        n = fn(s, buf, len(buf))
    return buf.value if n else None


def ascii_dir(path):
    """An all-ASCII spelling of an existing folder, for output paths
    handed to a Windows program that reads its arguments as ANSI
    (SvtAv1EncApp, FFVship): the folder itself, or its 8.3 alias. None
    when neither is ASCII (8.3 names are off on that volume). File names
    written inside it are ours and already ASCII.
    """
    s = str(path)
    if s.isascii() or os.name != "nt":
        return path
    short = _short_path_win(s)
    return Path(short) if short and short.isascii() else None


def ascii_path(path, scratch_dir):
    """An all-ASCII spelling of `path` for a Windows program that reads its
    file arguments as ANSI (FFVship's bundled FFMS2), plus any temp link to
    clean up afterward.

    A non-ASCII path is '?'-mangled on the ANSI command line and can't be
    opened, so callers pass the returned spelling in place of the raw path.
    Returns (usable_path, link):
      * (path, None)   already ASCII, or non-Windows — nothing to do.
      * (short, None)  the volume's 8.3 alias — stable, no new file.
      * (link, link)   an ASCII-named hardlink in scratch_dir; the caller
                       unlinks `link` when done.
      * (None, None)   no ASCII spelling possible — drop the optional work.
    """
    s = str(path)
    if s.isascii() or os.name != "nt":
        return path, None
    short = _short_path_win(s)
    if short and short.isascii():
        return Path(short), None
    # Content-addressed hardlink: same inode as the file, ASCII name.
    # Same-volume only; a cross-volume link raises OSError and the optional
    # work is dropped. Recreated every call (never reused) so a link left
    # by a hard-killed prior run can't feed a stale inode.
    try:
        scratch_dir.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix if path.suffix.isascii() else ""
        tag = hashlib.sha256(s.encode()).hexdigest()[:16]
        link = scratch_dir / f"_ascii_{tag}{suffix}"
        if link.exists():
            link.unlink()
        os.link(s, link)
        return link, link
    except OSError:
        return None, None


def partial_hash(filepath, block=1 << 16):
    """Fast file identity hash: size + first/last 64KB."""
    h = hashlib.sha256()
    st = filepath.stat()
    h.update(st.st_size.to_bytes(8, "little"))
    with open(filepath, "rb") as f:
        h.update(f.read(block))
        if st.st_size > block * 2:
            f.seek(-block, 2)
            h.update(f.read(block))
    return h.hexdigest()


def clamp(val, lo, hi):
    return max(lo, min(hi, val))
