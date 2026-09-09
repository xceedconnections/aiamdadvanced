"""SCAMMER blacklist / speech-to-text match settings (JSON next to amd_settings)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import get_settings

DEFAULT_BLACKLIST = [
    "bank",
    "chase",
    "wells fargo",
    "credit card",
    "debit card",
    "card number",
    "cvv",
    "card",
    "spectrum",
    "direct tv",
    "directv",
    "social security",
    "ssn",
    "routing number",
    "account number",
    "gift card",
    "wire transfer",
    "western union",
    "otp",
    "one time pass",
]

DEFAULTS: dict[str, Any] = {
    "blacklist_words": list(DEFAULT_BLACKLIST),
    "min_seconds_for_scan": 5,
    "mark_status": "SPAM",
}


def settings_path() -> Path:
    settings = get_settings()
    root = Path(settings.RECORDINGS_DIR).resolve().parent
    return root / "scam_settings.json"


def _normalize_words(raw: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(raw, str):
        parts = re.split(r"[\n,;]+", raw)
    elif isinstance(raw, list):
        parts = raw
    else:
        parts = []
    for p in parts:
        w = str(p or "").strip().lower()
        if len(w) < 2 or w in seen:
            continue
        seen.add(w)
        out.append(w[:80])
    return out[:500]


def load_scam_settings() -> dict[str, Any]:
    path = settings_path()
    data = {
        "blacklist_words": list(DEFAULT_BLACKLIST),
        "min_seconds_for_scan": 5,
        "mark_status": "SPAM",
    }
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if "blacklist_words" in raw:
                    words = _normalize_words(raw.get("blacklist_words"))
                    if words:
                        data["blacklist_words"] = words
                if "min_seconds_for_scan" in raw:
                    try:
                        data["min_seconds_for_scan"] = max(1, min(120, int(raw["min_seconds_for_scan"])))
                    except (TypeError, ValueError):
                        pass
                st = str(raw.get("mark_status") or "SPAM").strip().upper()
                if st in ("SPAM", "SCAM"):
                    data["mark_status"] = st
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    data["path"] = str(path)
    return data


def save_scam_settings(
    *,
    blacklist_words: list[str] | str | None = None,
    min_seconds_for_scan: int | None = None,
    mark_status: str | None = None,
) -> dict[str, Any]:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = load_scam_settings()
    words = prev["blacklist_words"]
    if blacklist_words is not None:
        words = _normalize_words(blacklist_words)
        if not words:
            words = list(DEFAULT_BLACKLIST)
    secs = int(prev["min_seconds_for_scan"])
    if min_seconds_for_scan is not None:
        secs = max(1, min(120, int(min_seconds_for_scan)))
    status = str(prev.get("mark_status") or "SPAM").upper()
    if mark_status is not None:
        st = str(mark_status).strip().upper()
        if st in ("SPAM", "SCAM"):
            status = st
    payload = {
        "blacklist_words": words,
        "min_seconds_for_scan": secs,
        "mark_status": status,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out = dict(payload)
    out["path"] = str(path)
    return out
