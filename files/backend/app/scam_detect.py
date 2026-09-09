"""Speech-to-text + blacklist phrase detection for SCAMMER full-call recordings."""
from __future__ import annotations

import re
from typing import Any

from app.scam_settings import load_scam_settings


def _word_to_pattern(word: str) -> re.Pattern[str] | None:
    w = (word or "").strip().lower()
    if len(w) < 2:
        return None
    parts = [re.escape(p) for p in re.split(r"\s+", w) if p]
    if not parts:
        return None
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.I)


def build_blacklist_patterns(words: list[str] | None = None) -> list[tuple[str, re.Pattern[str]]]:
    cfg_words = words if words is not None else load_scam_settings().get("blacklist_words") or []
    out: list[tuple[str, re.Pattern[str]]] = []
    for w in cfg_words:
        pat = _word_to_pattern(str(w))
        if pat is not None:
            out.append((str(w).strip().lower(), pat))
    return out


def scan_transcript(text: str, *, words: list[str] | None = None) -> dict[str, Any]:
    """Match blacklist words in transcript. Hits → SPAM (red) by default."""
    cfg = load_scam_settings()
    mark = str(cfg.get("mark_status") or "SPAM").upper()
    if mark not in ("SPAM", "SCAM"):
        mark = "SPAM"
    raw = (text or "").strip()
    if not raw:
        return {
            "status": "CLEAN",
            "confidence": 0.4,
            "match_terms": [],
            "note": "empty_transcript",
        }
    hits: list[str] = []
    for label, pat in build_blacklist_patterns(words if words is not None else cfg.get("blacklist_words")):
        if pat.search(raw) and label not in hits:
            hits.append(label)
    if hits:
        return {
            "status": mark,
            "confidence": min(0.99, 0.55 + 0.06 * len(hits)),
            "match_terms": hits,
            "note": "blacklist_hit",
        }
    return {
        "status": "CLEAN",
        "confidence": 0.6,
        "match_terms": [],
        "note": "no_blacklist_hit",
    }


def transcribe_and_scan(wav_bytes: bytes, *, min_seconds_for_scan: float | None = None) -> dict[str, Any]:
    """Full-recording Whisper speech→text then blacklist scan (not AMD short clip)."""
    import io

    import numpy as np
    import soundfile as sf

    cfg = load_scam_settings()
    if min_seconds_for_scan is None:
        try:
            min_seconds_for_scan = float(cfg.get("min_seconds_for_scan") or 5)
        except (TypeError, ValueError):
            min_seconds_for_scan = 5.0
    min_seconds_for_scan = max(1.0, float(min_seconds_for_scan))

    try:
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    except Exception as exc:
        return {
            "status": "ERROR",
            "confidence": 0.0,
            "match_terms": [],
            "transcript": "",
            "audio_seconds": 0.0,
            "note": f"audio_read_failed:{exc}",
        }

    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    duration = float(len(audio) / float(sr or 1))
    transcript = ""
    whisper_ok = False
    whisper_err = ""
    used_sec = 0.0

    if duration >= float(min_seconds_for_scan):
        try:
            from app.ai.whisper_amd import transcribe_full_call

            # Transcribe the entire recording (cap 10 min), not the AMD 2.2s greeting window
            max_sec = min(max(duration, 1.0), 600.0)
            w = transcribe_full_call(audio, int(sr), max_seconds=max_sec)
            whisper_ok = bool(w.get("whisper_ok"))
            transcript = str(w.get("transcript") or "")
            whisper_err = str(w.get("error") or "")
            try:
                used_sec = float(w.get("audio_seconds_used") or 0)
            except (TypeError, ValueError):
                used_sec = 0.0
        except Exception as exc:
            whisper_err = str(exc)

    if duration < float(min_seconds_for_scan):
        return {
            "status": "CLEAN",
            "confidence": 0.5,
            "match_terms": [],
            "transcript": transcript,
            "audio_seconds": duration,
            "note": "below_min_duration",
            "whisper_ok": whisper_ok,
            "whisper_error": whisper_err,
            "audio_seconds_used": used_sec,
        }

    if not transcript:
        return {
            "status": "PENDING",
            "confidence": 0.3,
            "match_terms": [],
            "transcript": "",
            "audio_seconds": duration,
            "note": "no_transcript",
            "whisper_ok": whisper_ok,
            "whisper_error": whisper_err,
            "audio_seconds_used": used_sec,
        }

    scanned = scan_transcript(transcript)
    scanned["transcript"] = transcript
    scanned["audio_seconds"] = duration
    scanned["whisper_ok"] = whisper_ok
    scanned["whisper_error"] = whisper_err
    scanned["audio_seconds_used"] = used_sec
    return scanned
