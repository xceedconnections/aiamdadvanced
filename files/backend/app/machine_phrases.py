"""Editable MACHINE / voicemail phrase list for Whisper transcript matching.

Stored as JSON next to amd_settings. Custom phrases are checked in addition to
built-in voicemail regexes in whisper_amd.classify_transcript.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import get_settings

# Human-readable examples of built-in cues (always active; not stored).
BUILTIN_MACHINE_PHRASES = [
    "leave a message",
    "voicemail",
    "not available",
    "after the tone",
    "record your message",
    "record your name",
    "mailbox",
    "you have reached",
    "the person you're calling",
    "please leave a message",
    "call back later",
    "mailbox is full",
    "press 1",
    "for english",
]

DEFAULT_CUSTOM: list[str] = []

_cache_mtime: float | None = None
_cache_patterns: list[tuple[str, re.Pattern[str]]] = []
_cache_words: list[str] = []


def settings_path() -> Path:
    settings = get_settings()
    root = Path(settings.RECORDINGS_DIR).resolve().parent
    return root / "machine_phrases.json"


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
        w = re.sub(r"\s+", " ", w)
        if len(w) < 2 or w in seen:
            continue
        seen.add(w)
        out.append(w[:120])
    return out[:300]


def _phrase_to_pattern(phrase: str) -> re.Pattern[str]:
    """Word-boundary match; spaces flexible; apostrophes optional."""
    parts = [re.escape(p) for p in phrase.split() if p]
    if not parts:
        return re.compile(r"(?!)")
    body = r"\s+".join(parts)
    body = body.replace(r"\'", r"'?")
    return re.compile(rf"\b{body}\b", re.I)


def load_machine_phrases() -> dict[str, Any]:
    path = settings_path()
    words = list(DEFAULT_CUSTOM)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and "phrases" in raw:
                words = _normalize_words(raw.get("phrases"))
            elif isinstance(raw, list):
                words = _normalize_words(raw)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return {
        "phrases": words,
        "builtin_phrases": list(BUILTIN_MACHINE_PHRASES),
        "path": str(path),
        "count": len(words),
    }


def save_machine_phrases(phrases: list[str] | str | None = None) -> dict[str, Any]:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    words = _normalize_words(phrases if phrases is not None else [])
    payload = {"phrases": words}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _invalidate_cache()
    out = load_machine_phrases()
    return out


def _invalidate_cache() -> None:
    global _cache_mtime, _cache_patterns, _cache_words
    _cache_mtime = None
    _cache_patterns = []
    _cache_words = []


def _ensure_patterns() -> list[tuple[str, re.Pattern[str]]]:
    global _cache_mtime, _cache_patterns, _cache_words
    path = settings_path()
    mtime = path.stat().st_mtime if path.exists() else -1.0
    if _cache_mtime == mtime and _cache_patterns is not None:
        return _cache_patterns
    cfg = load_machine_phrases()
    words = list(cfg.get("phrases") or [])
    patterns = [(w, _phrase_to_pattern(w)) for w in words]
    _cache_mtime = mtime
    _cache_patterns = patterns
    _cache_words = words
    return patterns


def match_custom_machine_phrase(text: str) -> str | None:
    """If transcript matches a custom phrase, return that phrase; else None."""
    t = (text or "").strip()
    if not t:
        return None
    for label, pat in _ensure_patterns():
        if pat.search(t):
            return label
    return None
