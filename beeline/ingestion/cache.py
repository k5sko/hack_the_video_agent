"""Content-addressed disk cache for every external API call.

Hard requirement: a re-run must rebuild graph_payload.json entirely from cache
with ZERO API calls. This is what makes the demo survive venue wifi, so the
cache is the source of truth, not an optimization.

Every external call goes through `cached()`. In OFFLINE mode a cache miss
raises CacheMiss instead of hitting the network, which is how `--from-cache`
proves it made no API calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

from .config import CACHE_DIR

_LOCK = threading.Lock()

# Counters let the runner assert "zero API calls" rather than just hope.
STATS = {"hits": 0, "misses": 0, "writes": 0}

_OFFLINE = False


class CacheMiss(RuntimeError):
    """Raised when OFFLINE mode needs a value that was never cached."""


def set_offline(offline: bool) -> None:
    global _OFFLINE
    _OFFLINE = offline


def is_offline() -> bool:
    return _OFFLINE


def api_calls_made() -> int:
    return STATS["misses"]


def _hash_key(namespace: str, payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(f"{namespace}\x00{blob}".encode("utf-8")).hexdigest()[:32]
    return digest


def cache_path(namespace: str, payload: Any) -> Path:
    ns_dir = CACHE_DIR / namespace
    return ns_dir / f"{_hash_key(namespace, payload)}.json"


def peek(namespace: str, payload: Any) -> Any | None:
    p = cache_path(namespace, payload)
    if p.exists():
        try:
            return json.loads(p.read_text())["value"]
        except Exception:
            return None
    return None


def cached(namespace: str, key_payload: Any, producer: Callable[[], Any]) -> Any:
    """Return cached value for (namespace, key_payload) or call `producer` once.

    `key_payload` must fully describe the request so the hash is stable across
    runs -- never include timestamps, task ids, or anything nondeterministic.
    """
    path = cache_path(namespace, key_payload)
    if path.exists():
        try:
            record = json.loads(path.read_text())
            with _LOCK:
                STATS["hits"] += 1
            return record["value"]
        except Exception:
            pass  # corrupt entry -> fall through and regenerate

    if _OFFLINE:
        raise CacheMiss(
            f"OFFLINE cache miss for {namespace}: {json.dumps(key_payload, default=str)[:200]}"
        )

    with _LOCK:
        STATS["misses"] += 1

    value = producer()

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"key": key_payload, "value": value}, default=str, indent=None))
    os.replace(tmp, path)
    with _LOCK:
        STATS["writes"] += 1
    return value
