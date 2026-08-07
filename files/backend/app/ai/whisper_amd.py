"""Optional Faster-Whisper Tiny refine — used ONLY on low-confidence XGB calls.

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
    r"please\s+hold|enter\s+your|dial\s+\d|options?\s+are|to\s+speak\s+to"
    r")\b",
    re.I,
)
_HUMAN_RE = re.compile(
    r"\b(hello|hi|hey|yeah|yes|speaking|this\s+is|who(?:'s|\s+is)\s+this)\b",
    re.I,
)


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


def refine_with_whisper(
    audio,  # np.ndarray float32 mono
    sr: int,
    *,
    xgb_probs: Dict[str, float],
    max_seconds: float = 4.0,
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """Return (status_or_None, confidence, details). None status = keep XGB decision."""
    model = _get_model()
    if model is None:
        return None, 0.0, {"whisper_ok": False, "error": _model_error or "not_installed"}

    import numpy as np

    if audio is None or len(audio) == 0:
        return None, 0.0, {"whisper_ok": False, "error": "empty_audio"}

    # faster-whisper expects 16 kHz float32
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
        return None, 0.0, {"whisper_ok": False, "error": str(exc)}

    details: Dict[str, Any] = {
        "whisper_ok": True,
        "transcript": text[:240],
        "language": getattr(info, "language", "") or "",
    }
    if not text:
        return None, 0.0, details

    if _IVR_RE.search(text):
        return "IVR", max(0.9, float(xgb_probs.get("IVR", 0.5))), {**details, "cue": "ivr"}
    if _MACHINE_RE.search(text):
        return "MACHINE", max(0.92, float(xgb_probs.get("MACHINE", 0.5))), {**details, "cue": "voicemail"}
    if _HUMAN_RE.search(text) and len(text.split()) <= 6:
        return "HUMAN", max(0.88, float(xgb_probs.get("HUMAN", 0.5))), {**details, "cue": "human_short"}

    # No strong cue — leave XGB decision
    return None, 0.0, {**details, "cue": "none"}
