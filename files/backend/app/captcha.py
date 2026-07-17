"""Math captcha for portal login (HMAC-signed, no server-side session store)."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Tuple

from app.config import get_settings

_MIN = 1
_MAX = 12
_TTL_SECONDS = 300


def _secret() -> bytes:
    settings = get_settings()
    return (settings.JWT_SECRET + ":captcha").encode("utf-8")


def create_math_captcha() -> Tuple[str, str]:
    """Return (captcha_id, question) for a simple a + b challenge."""
    a = secrets.randbelow(_MAX - _MIN + 1) + _MIN
    b = secrets.randbelow(_MAX - _MIN + 1) + _MIN
    exp = int(time.time()) + _TTL_SECONDS
    payload = f"{a}:{b}:{exp}"
    sig = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:20]
    captcha_id = f"{payload}:{sig}"
    question = f"{a} + {b}"
    return captcha_id, question


def verify_math_captcha(captcha_id: str, answer: str) -> bool:
    if not captcha_id or answer is None:
        return False
    parts = str(captcha_id).strip().split(":")
    if len(parts) != 4:
        return False
    a_s, b_s, exp_s, sig = parts
    try:
        a = int(a_s)
        b = int(b_s)
        exp = int(exp_s)
    except ValueError:
        return False
    if a < _MIN or a > _MAX or b < _MIN or b > _MAX:
        return False
    if int(time.time()) > exp:
        return False
    payload = f"{a}:{b}:{exp}"
    expected = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:20]
    if not hmac.compare_digest(expected, sig):
        return False
    try:
        given = int(str(answer).strip())
    except ValueError:
        return False
    return given == (a + b)
