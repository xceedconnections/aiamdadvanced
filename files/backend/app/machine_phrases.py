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
    "message system",
    "messaging system",
    "voice message",
    "mailbox is full",
    "full and there is not",
    "not been set up",
    "set up yet",
    "please try your",
    "try your call again",
    "not available",
    "after the tone",
    "record your message",
    "record your name",
    "mailbox",
    "you have reached",
    "the person you're calling",
    "please leave a message",
    "call back later",
    "press 1",
    "for english",
]

DEFAULT_CUSTOM: list[str] = []

_cache_mtime: float | None = None
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


def _normalize_for_match(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("'", "'").replace("'", "'")
    t = re.sub(r"\bit'?s\b", "it is", t)
    t = re.sub(r"\byou'?re\b", "you are", t)
    t = re.sub(r"\byou'?ve\b", "you have", t)
    t = re.sub(r"\bcan'?t\b", "cannot", t)
    t = re.sub(r"[^a-z0-9'\s]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _match_phrase_flexible(text: str, phrase: str) -> bool:
    """Exact phrase, or any contiguous 3+ word core from the phrase (handles Whisper wording drift)."""
    t = _normalize_for_match(text)
    p = _normalize_for_match(phrase)
    if not t or not p:
        return False
    if _phrase_to_pattern(p).search(t):
        return True
    words = p.split()
    if len(words) < 3:
        return False
    # Long custom lines often start differently than Tiny/Base transcripts
    # ("your call has been forwarded…" vs "It's been forwarded…").
    min_core = 3 if len(words) <= 5 else 4
    for length in range(len(words), min_core - 1, -1):
        for i in range(0, len(words) - length + 1):
            core = " ".join(words[i : i + length])
            if _phrase_to_pattern(core).search(t):
                return True
    return False


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
    global _cache_mtime, _cache_words
    _cache_mtime = None
    _cache_words = []


def _ensure_words() -> list[str]:
    global _cache_mtime, _cache_words
    path = settings_path()
    mtime = path.stat().st_mtime if path.exists() else -1.0
    if _cache_mtime == mtime and _cache_words is not None:
        return _cache_words
    cfg = load_machine_phrases()
    words = list(cfg.get("phrases") or [])
    _cache_mtime = mtime
    _cache_words = words
    return words


def match_custom_machine_phrase(text: str) -> str | None:
    """If transcript matches a custom phrase (or a 3–4+ word core of it), return that phrase."""
    t = (text or "").strip()
    if not t:
        return None
    for label in _ensure_words():
        if _match_phrase_flexible(t, label):
            return label
    return None
