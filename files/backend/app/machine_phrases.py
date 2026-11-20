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
    # AI / virtual assistant / call-recording IVR
    "i'm an ai",
    "i am an ai",
    "ai assistant",
    "virtual assistant",
    "automated assistant",
    "assistant recording",
    "this call is being recorded",
    "your call is being recorded",
    "call is being recorded",
]

DEFAULT_CUSTOM: list[str] = []

_cache_mtime: float | None = None
_cache_words: list[str] = []
_cache_index: dict[str, str] = {}


def settings_path() -> Path:
    settings = get_settings()
    root = Path(settings.RECORDINGS_DIR).resolve().parent
    return root / "machine_phrases.json"


# Words that carry no voicemail meaning on their own. A fragment of a custom
# phrase made only of these ("is not", "you have", "at the", "your call",
# "the person you") shows up in ordinary live speech, so it must not count as a
# hit for the whole phrase.
_GENERIC_WORDS = frozenset(
    """
    a an the this that these those it its is are am was were be been being
    i me my we us our you your he she they them their his her
    to of at on in for from with by as if or and but so than then
    not no do does did have has had can cannot could will would shall should may
    when where who what which how there here now
    call calls called calling number name phone person people line time
    please take taking get getting hear trying try able currently another
    """.split()
)


def _has_distinctive_word(words: list[str]) -> bool:
    return any(w not in _GENERIC_WORDS and len(w) > 1 for w in words)


def _normalize_words(raw: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    if isinstance(raw, str):
        # One phrase per line. A comma belongs to the phrase: splitting
        # "leave your name, number, and message" on commas used to create the
        # stand-alone phrase "number", which matched nearly every call.
        parts = re.split(r"[\r\n]+", raw)
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
    """Match phrase anywhere in the transcript (start, middle, or end).

    Also accepts a contiguous 2+ word core from a longer custom line so Whisper
    wording drift still hits (e.g. full forwarded… line vs \"been forwarded\").
    A core only counts when it holds a word that means something: \"been
    forwarded\" and \"the tone\" do, \"is not\" and \"you have\" do not.
    """
    keys = set(_phrase_keys(phrase))
    if not keys:
        return False
    return any(gram in keys for gram in _text_ngrams(_match_words(text), _MAX_KEY_WORDS))


_MAX_KEY_WORDS = 24


def _match_words(text: str) -> list[str]:
    """Lower-case words with contractions expanded and apostrophes dropped."""
    return _normalize_for_match(text).replace("'", "").split()


def _phrase_keys(phrase: str) -> list[str]:
    """Every word sequence that counts as a hit for `phrase`: itself, then cores."""
    words = _match_words(phrase)[:_MAX_KEY_WORDS]
    if not words:
        return []
    # A one- or two-word entry made only of filler ("number", "your call") is
    # almost always a fragment left by an old comma split, never a real cue.
    if len(words) <= 2 and not _has_distinctive_word(words):
        return []
    keys = [" ".join(words)]
    # Long custom lines often start differently than Whisper transcripts
    # ("your call has been forwarded…" vs "It's been forwarded…").
    # Slide across the WHOLE phrase (middle cores included), not only the start.
    min_core = 2 if len(words) <= 4 else 3
    for length in range(len(words) - 1, min_core - 1, -1):
        for i in range(0, len(words) - length + 1):
            core_words = words[i : i + length]
            core = " ".join(core_words)
            if len(core) >= 4 and _has_distinctive_word(core_words):
                keys.append(core)
    return keys


def _text_ngrams(words: list[str], max_len: int):
    """Word n-grams of the transcript, longest first."""
    for length in range(min(max_len, len(words)), 0, -1):
        for i in range(0, len(words) - length + 1):
            yield " ".join(words[i : i + length])


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
    global _cache_mtime, _cache_words, _cache_index
    _cache_mtime = None
    _cache_words = []
    _cache_index = {}


def _ensure_words() -> list[str]:
    global _cache_mtime, _cache_words, _cache_index
    path = settings_path()
    mtime = path.stat().st_mtime if path.exists() else -1.0
    if _cache_mtime == mtime and _cache_words is not None:
        return _cache_words
    cfg = load_machine_phrases()
    words = list(cfg.get("phrases") or [])
    # key -> first phrase that produces it. Whole phrases are indexed before any
    # core so the label shown on the portal is the phrase that really matched.
    index: dict[str, str] = {}
    all_keys = [(label, _phrase_keys(label)) for label in words]
    for label, keys in all_keys:
        if keys:
            index.setdefault(keys[0], label)
    for label, keys in all_keys:
        for key in keys[1:]:
            index.setdefault(key, label)
    _cache_mtime = mtime
    _cache_words = words
    _cache_index = index
    return words


def match_custom_machine_phrase(text: str) -> str | None:
    """If transcript contains a custom phrase (anywhere) or a meaningful 2+ word core of it, return that phrase."""
    t = (text or "").strip()
    if not t:
        return None
    _ensure_words()
    if not _cache_index:
        return None
    for gram in _text_ngrams(_match_words(t), _MAX_KEY_WORDS):
        label = _cache_index.get(gram)
        if label:
            return label
    return None
