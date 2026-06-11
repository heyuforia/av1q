"""Process, filesystem, and hashing utilities shared by every stage."""

import hashlib
import json
import os
import shlex
import subprocess
import time

_temp_files = set()


def run_cmd(cmd):
    """Run a command and return the result. Raises on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        tail = "\n".join((p.stderr or "").splitlines()[-80:])
        raise RuntimeError(
            f"exit {p.returncode}\n"
            f"{subprocess.list2cmdline(cmd) if os.name == 'nt' else ' '.join(map(shlex.quote, cmd))}"
            f"\n{tail}"
        )
    return p


def cleanup_temp():
    for path in list(_temp_files):
        try:
            if path.exists():
                path.unlink()
        except OSError:
            pass
        _temp_files.discard(path)


def atomic_write_json(path, obj, indent=None):
    """Write JSON to `path` atomically: serialize to a sibling .tmp file,
    then replace. A crash mid-write can't leave a torn cache or sidecar.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent), encoding="utf-8")
    tmp.replace(path)


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
    graph (scdet/cropdetect metadata). Filter-graph colons collide with
    Windows drive-letter colons, so they're double-escaped to survive both
    the filtergraph and option parsing stages.
    """
    return path.as_posix().replace(":", "\\\\:")


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
