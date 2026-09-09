"""Heuristic SCAM / card / bank phrase detection on full-call transcripts."""
from __future__ import annotations

import re
from typing import Any

# Phrases that often indicate social-engineering / card / bank harvesting on calls.
_TERM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("credit_card", re.compile(r"\b(credit\s*card|debit\s*card|card\s*number|cvv|cvc|expiry|expiration)\b", re.I)),
    ("card_digits", re.compile(r"\b(sixteen|sixteen digit|16\s*digit|last\s*four|security\s*code)\b", re.I)),
    ("bank_name", re.compile(
        r"\b(bank\s+of\s+america|wells\s*fargo|chase|citibank|citi\s*bank|capital\s*one|"
        r"td\s*bank|pnc|us\s*bank|bank\s*of\s*montreal|rbc|scotiabank|hsbc|"
        r"your\s+bank|banking\s+details|account\s+number|routing\s+number|sort\s+code)\b",
        re.I,
    )),
    ("otp_ssn", re.compile(r"\b(one[\s-]*time\s*pass|otp|social\s*security|ssn|mother'?s?\s*maiden)\b", re.I)),
    ("verify_account", re.compile(
        r"\b(verify\s+(your\s+)?(account|identity|card)|confirm\s+(your\s+)?(card|account|ssn)|"
        r"update\s+(your\s+)?(billing|payment|card))\b",
        re.I,
    )),
    ("gift_wire", re.compile(r"\b(gift\s*card|itunes\s*card|wire\s*transfer|western\s*union|bitcoin|crypto\s*wallet)\b", re.I)),
]


def scan_transcript(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {
            "status": "CLEAN",
            "confidence": 0.4,
            "match_terms": [],
            "note": "empty_transcript",
        }
    hits: list[str] = []
    for label, pat in _TERM_PATTERNS:
        if pat.search(raw):
            hits.append(label)
    if hits:
        return {
            "status": "SCAM",
            "confidence": min(0.99, 0.55 + 0.08 * len(hits)),
            "match_terms": hits,
            "note": "keyword_hit",
        }
    return {
        "status": "CLEAN",
        "confidence": 0.6,
        "match_terms": [],
        "note": "no_keyword_hit",
    }


def transcribe_and_scan(wav_bytes: bytes, *, min_seconds_for_scan: float = 120.0) -> dict[str, Any]:
    """Transcribe full-ish call audio and apply scam heuristics.

    Short uploads (< min_seconds) are stored as PENDING/CLEAN without forcing SCAM.
    """
    import io

    import numpy as np
    import soundfile as sf

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

    # Only run Whisper when the call is long enough (portal policy: >= 2 min)
    if duration >= float(min_seconds_for_scan):
        try:
            from app.ai.whisper_amd import transcribe_for_display

            # Cap to keep CPU bounded (first ~8 minutes is usually enough for pitch)
            max_sec = min(duration, 480.0)
            w = transcribe_for_display(audio, int(sr), max_seconds=max_sec)
            whisper_ok = bool(w.get("whisper_ok"))
            transcript = str(w.get("transcript") or "")
            whisper_err = str(w.get("error") or "")
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
        }

    scanned = scan_transcript(transcript)
    scanned["transcript"] = transcript
    scanned["audio_seconds"] = duration
    scanned["whisper_ok"] = whisper_ok
    scanned["whisper_error"] = whisper_err
    return scanned
