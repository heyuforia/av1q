"""Per-file result cache: sig-gated JSON keyed by partial hash."""

import json


def load_cache(cache_dir, file_hash, sig):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_dir / f"{file_hash}.json"
    if cp.exists():
        with open(cp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("sig") == sig:
            return data, cp
    return {"sig": sig, "entries": {}}, cp
