"""Optional Faster-Whisper Tiny refine — used on uncertain / HUMAN-bound calls.

Lazy-imported. If faster-whisper is not installed, refine is skipped (XGB wins).
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, Optional, Tuple

_lock = threading.Lock()
_model = None
_model_error = ""

# Phrase cues (English NA + common IVR). Keep small — tiny model transcripts are noisy.
_MACHINE_RE = re.compile(
    r"\b("
    r"leave\s+(a\s+)?message|voicemail|voice\s*mail|not\s+available|"
    r"can'?t\s+take\s+your\s+call|unable\s+to\s+take|after\s+the\s+(tone|beep)|"
    r"record\s+your\s+message|please\s+leave|mailbox|the\s+person\s+you\s+(have\s+)?called|"
    r"no\s+one\s+is\s+available|forwarded\s+to\s+an?\s+automated"
    r")\b",
    re.I,
)
_IVR_RE = re.compile(
    r"\b("
    r"press\s+\d|for\s+(english|spanish)|menu|your\s+call\s+is\s+important|"
    r"please\s+hold|enter\s+your|dial\s+\d|options?\s+are|to\s+speak\s+to|"
    r"extension|account\s+number|pin\s+number|pound|hash\s+key|"
    r"using\s+your\s+keypad|touch\s*tone"
    r")\b",
    re.I,
)
_HUMAN_RE = re.compile(
    r"\b("
    r"hello+|hullo|hallo|halo|allo|yellow|"
    r"hi|hey|yeah|yes|yep|yup|yo|speaking|this\s+is|"
    r"who(?:'s|\s+is)\s+this|good\s+(morning|afternoon|evening)|"
    r"how\s+are\s+you|can\s+i\s+help|what'?s\s+up|go\s+ahead|"
    r"i'?m\s+here|who(?:'s|\s+is)\s+calling|pardon|sorry"
    r")\b",
    re.I,
)

_DIGIT_WORDS = {
    "zero",
    "oh",
    "o",
    "nought",
    "naught",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "niner",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "double",
    "triple",
    "and",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _is_digit_token(tok: str) -> bool:
    if not tok:
        return False
    if tok.isdigit():
        return True
    return tok in _DIGIT_WORDS


def is_number_readout(text: str) -> bool:
    """True when Whisper heard an IVR / CLI / account number being spoken.

    Examples: 'Four, four, seven.'  '4, 4, 7, 4.'  'zero one two'
    """
    toks = [t for t in _tokens(text) if t not in ("and",)]
    if len(toks) < 2:
        # Compact numeric blob: "447" / "0123"
        compact = re.sub(r"[^0-9]", "", text or "")
        return len(compact) >= 3
    digit_n = sum(1 for t in toks if _is_digit_token(t))
    if digit_n >= 2 and digit_n >= max(2, int(round(len(toks) * 0.6))):
        return True
    compact = re.sub(r"[^0-9]", "", text or "")
    return len(compact) >= 3 and digit_n >= 2


def classify_transcript(
    text: str,
    xgb_probs: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[str], float, str]:
    """Return (status_or_None, confidence, cue). None = no strong cue."""
    probs = xgb_probs or {}
    t = (text or "").strip()
    if not t:
        return None, 0.0, "none"

    if _IVR_RE.search(t) or is_number_readout(t):
        return "IVR", max(0.92, float(probs.get("IVR", 0.5))), "ivr_digits"
    if _MACHINE_RE.search(t):
        return "MACHINE", max(0.92, float(probs.get("MACHINE", 0.5))), "voicemail"
    if _HUMAN_RE.search(t) and len(t.split()) <= 10 and not is_number_readout(t):
        return "HUMAN", max(0.90, float(probs.get("HUMAN", 0.5))), "human_short"
    return None, 0.0, "none"


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _get_model():
    global _model, _model_error
    with _lock:
        if _model is not None:
            return _model
        if _model_error:
            return None
        try:
            from faster_whisper import WhisperModel

            # CPU, int8 — tiny footprint; only loaded if ML+whisper path runs
            _model = WhisperModel("tiny", device="cpu", compute_type="int8")
            return _model
        except Exception as exc:
            _model_error = str(exc)
            return None


def _transcribe_clip(audio, sr: int, max_seconds: float) -> Tuple[str, Any, Dict[str, Any]]:
    model = _get_model()
    if model is None:
        return "", None, {"whisper_ok": False, "error": _model_error or "not_installed"}

    import numpy as np

    if audio is None or len(audio) == 0:
        return "", None, {"whisper_ok": False, "error": "empty_audio"}

    clip = np.asarray(audio, dtype=np.float32)
    n = int(min(len(clip), max(1, int(sr * max_seconds))))
    clip = clip[:n]
    if sr != 16000 and len(clip) > 1:
        duration = len(clip) / float(sr)
        new_len = max(1, int(duration * 16000))
        x_old = np.linspace(0, 1, num=len(clip), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        clip = np.interp(x_new, x_old, clip).astype(np.float32)

    try:
        segments, info = model.transcribe(
            clip,
            language="en",
            beam_size=1,
            vad_filter=False,
            without_timestamps=True,
        )
        text_parts = []
        for seg in segments:
            if seg.text:
                text_parts.append(seg.text.strip())
        text = " ".join(text_parts).strip()
    except Exception as exc:
        return "", None, {"whisper_ok": False, "error": str(exc)}

    return text, info, {"whisper_ok": True}


def refine_with_whisper(
    audio,  # np.ndarray float32 mono
    sr: int,
    *,
    xgb_probs: Dict[str, float],
    max_seconds: float = 4.0,
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """Return (status_or_None, confidence, details). None status = keep XGB decision."""
    text, info, base = _transcribe_clip(audio, sr, max_seconds)
    if not base.get("whisper_ok"):
        return None, 0.0, base

    details: Dict[str, Any] = {
        "whisper_ok": True,
        "transcript": text[:240],
        "language": getattr(info, "language", "") or "",
    }
    if not text:
        return None, 0.0, {**details, "cue": "none"}

    status, conf, cue = classify_transcript(text, xgb_probs)
    details["cue"] = cue
    if status:
        return status, conf, details
    return None, 0.0, details


def transcribe_audio(
    audio,  # np.ndarray float32 mono
    sr: int,
    *,
    max_seconds: float = 5.0,
) -> Dict[str, Any]:
    """WAV→text only (no AMD status refine). Used for Training Logs display."""
    text, info, base = _transcribe_clip(audio, sr, max_seconds)
    if not base.get("whisper_ok"):
        return {
            "whisper_ok": False,
            "transcript": "",
            "error": base.get("error") or "whisper_failed",
        }

    cue = ""
    if text:
        _status, _conf, cue = classify_transcript(text, {})

    return {
        "whisper_ok": True,
        "transcript": text[:500],
        "cue": cue or "none",
        "language": getattr(info, "language", "") or "",
    }
