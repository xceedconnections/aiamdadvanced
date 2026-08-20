"""Per-VICIdial-server calls-per-second (CPS) admit control.

When max_cps is set on a dialer, only that many analyze/admit requests
are accepted each second. Excess calls return HTTP 429 so the AGI
sets OPENAMD_STATUS=UNAVAILABLE and the dialplan falls back to stock
VICIdial AMD extension 8369 (same path as AIAMD offline).

Uses Redis when available; falls back to in-process memory.
"""

from __future__ import annotations

import threading
import time
from typing import Tuple

from app.config import get_settings

_mem_lock = threading.Lock()
# key -> (window_start_epoch_second, count)
_mem: dict[str, tuple[int, int]] = {}


def _redis():
    settings = get_settings()
    try:
        import redis

        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
            decode_responses=True,
        )
        r.ping()
        return r
    except Exception:
        return None


def normalize_max_cps(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(10000, n))


def try_admit(server_id: int, max_cps: int) -> Tuple[bool, int, int]:
    """Try to admit one call for this server in the current 1-second bucket.

    Returns (allowed, limit, count_in_bucket_after_attempt).
    max_cps <= 0 means unlimited (always allow, count 0).
    """
    limit = normalize_max_cps(max_cps)
    if limit <= 0:
        return True, 0, 0

    sid = int(server_id)
    bucket = int(time.time())
    key = f"openamd:cps:{sid}:{bucket}"

    r = _redis()
    if r is not None:
        try:
            count = int(r.incr(key))
            if count == 1:
                r.expire(key, 3)
            if count > limit:
                return False, limit, count
            return True, limit, count
        except Exception:
            pass

    with _mem_lock:
        prev = _mem.get(str(sid))
        if not prev or prev[0] != bucket:
            _mem[str(sid)] = (bucket, 1)
            return True, limit, 1
        count = prev[1] + 1
        _mem[str(sid)] = (bucket, count)
        if count > limit:
            return False, limit, count
        return True, limit, count


def current_cps(server_id: int) -> int:
    """Admits already counted in the current 1-second bucket."""
    sid = int(server_id)
    bucket = int(time.time())
    key = f"openamd:cps:{sid}:{bucket}"
    r = _redis()
    if r is not None:
        try:
            return int(r.get(key) or 0)
        except Exception:
            pass
    with _mem_lock:
        prev = _mem.get(str(sid))
        if not prev or prev[0] != bucket:
            return 0
        return int(prev[1])
