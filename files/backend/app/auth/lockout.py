"""Login brute-force lockout: 5 failed attempts → block 15 minutes.

Uses Redis when available; falls back to in-process memory.
Keyed by username + client IP.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

from app.config import get_settings

MAX_FAILURES = 5
LOCKOUT_SECONDS = 15 * 60  # 15 minutes
WINDOW_SECONDS = 15 * 60

_mem_lock = threading.Lock()
_mem: dict[str, dict] = {}


def _key(username: str, ip: str) -> str:
    u = (username or "").strip().lower() or "-"
    i = (ip or "").strip() or "-"
    return f"openamd:loginfail:{u}:{i}"


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


def is_locked(username: str, ip: str) -> Tuple[bool, int]:
    """Return (locked, seconds_remaining)."""
    k = _key(username, ip)
    r = _redis()
    if r is not None:
        try:
            ttl = r.ttl(f"{k}:lock")
            if ttl and ttl > 0:
                return True, int(ttl)
            return False, 0
        except Exception:
            pass

    now = time.time()
    with _mem_lock:
        entry = _mem.get(k)
        if not entry:
            return False, 0
        locked_until = float(entry.get("locked_until", 0))
        if locked_until > now:
            return True, int(locked_until - now)
        return False, 0


def record_failure(username: str, ip: str) -> Tuple[int, bool]:
    """Record a failed login. Returns (failure_count, just_locked)."""
    k = _key(username, ip)
    r = _redis()
    if r is not None:
        try:
            lock_key = f"{k}:lock"
            if r.ttl(lock_key) > 0:
                return MAX_FAILURES, False
            count = int(r.incr(f"{k}:count"))
            if count == 1:
                r.expire(f"{k}:count", WINDOW_SECONDS)
            if count >= MAX_FAILURES:
                r.setex(lock_key, LOCKOUT_SECONDS, "1")
                r.delete(f"{k}:count")
                return count, True
            return count, False
        except Exception:
            pass

    now = time.time()
    with _mem_lock:
        entry = _mem.get(k) or {"count": 0, "first": now, "locked_until": 0}
        if entry.get("locked_until", 0) > now:
            return MAX_FAILURES, False
        if now - float(entry.get("first", now)) > WINDOW_SECONDS:
            entry = {"count": 0, "first": now, "locked_until": 0}
        entry["count"] = int(entry.get("count", 0)) + 1
        just_locked = False
        if entry["count"] >= MAX_FAILURES:
            entry["locked_until"] = now + LOCKOUT_SECONDS
            entry["count"] = 0
            just_locked = True
        _mem[k] = entry
        return int(entry["count"] if not just_locked else MAX_FAILURES), just_locked


def clear_failures(username: str, ip: str) -> None:
    k = _key(username, ip)
    r = _redis()
    if r is not None:
        try:
            r.delete(f"{k}:count", f"{k}:lock")
        except Exception:
            pass
    with _mem_lock:
        _mem.pop(k, None)


def client_ip_from_headers(x_forwarded_for: Optional[str], fallback: str) -> str:
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return (fallback or "").strip()
