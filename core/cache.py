"""Per-file result cache: sig-gated JSON keyed by partial hash, and the
match rule for its `recommended` block (the skip-existing and resume
contract)."""

import json

from .ui import DIM, RESET, label


def load_cache(cache_dir, file_hash, sig):
    """This file's cache dict and its path.

    A missing, unreadable, malformed, or foreign-sig cache yields a fresh
    dict (the next write replaces it on disk); nothing here raises,
    because the launch's seed-redo scan loads every file's cache before
    the first encode and one torn JSON must not abort the batch. A
    corrupt file is announced — it costs that file a re-search, and the
    reason should be visible; a foreign sig is refused silently, that
    being the normal shape of a stale cache.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{file_hash}.json"
    fresh = {"sig": sig, "entries": {}}
    if not cp.exists():
        return fresh, cp
    try:
        with open(cp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict):
        print(f"{label('cache')}{DIM}{cp.name} unreadable, starting fresh{RESET}")
        return fresh, cp
    if data.get("sig") != sig:
        return fresh, cp
    if not isinstance(data.get("entries"), dict):
        data["entries"] = {}
    return data, cp


def recommended_matches(rec, engine, cfg, crop, target):
    """True when a cache's `recommended` block was written by a search
    under the current settings — every setting that changes the search's
    answer (the engine's bounds, preset, film grain, its extra keys), the
    crop, and the VMAF target. A block from any other run is refused
    rather than resumed from or skipped on.

    `target` None skips the target check: before a file is probed its
    automatic per-resolution target is unknown, so at that point only an
    explicit --vmaf can be compared.
    """
    if not isinstance(rec, dict):
        return False
    keys = (*engine.rec_bound_keys, "preset", "film_grain",
            *engine.rec_extra_keys)
    return (
        all(rec.get(k) == cfg[k] for k in keys)
        and rec.get("crop") == crop
        and (target is None or rec.get("target") == target)
    )
