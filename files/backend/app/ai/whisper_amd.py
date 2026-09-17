"""Optional Faster-Whisper Tiny refine — used on uncertain / HUMAN-bound calls.

Lazy-imported. If faster-whisper is not installed, refine is skipped (XGB wins).

Whisper Tiny invents YouTube outros on short telephony "hello" clips
("and I'll see you next time"). We strip those hallucinations before
display / AMD cues.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, Optional, Tuple

_lock = threading.Lock()
_model = None
_model_name_loaded = ""
_model_error = ""

# Free local STT via faster-whisper. base.en is much more accurate than tiny
# on telephony VM scripts; still CPU-friendly with int8.
_ALLOWED_WHISPER_MODELS = ("tiny", "tiny.en", "base", "base.en", "small", "small.en")
_DEFAULT_WHISPER_MODEL = "base.en"

# Phrase cues (English NA + common IVR). Tiny often mishears "person" as "passion".
_MACHINE_RE = re.compile(
    r"\b("
    r"leave\s+(a\s+)?message|voicemails?|voice\s*mails?|not\s+available|unavailable|"
    r"can'?t\s+take\s+your\s+call|unable\s+to\s+take|after\s+the\s+(tone|beep|sound)|"
    r"at\s+the\s+(tone|beep)|record\s+your\s+(message|name)|please\s+leave|mailbox|"
    r"if\s+you\s+record|record\s+your\s+name\s+and|"
    r"please\s+record\s+your\s+(name|message)|"
    # Google Voice / carrier: "Message system. One." / messaging system
    r"message\s+system|messaging\s+system|voice\s+messaging|voice\s+message|"
    r"answering\s+(machine|service)|automated\s+(attendant|message)|"
    r"this\s+is\s+(the\s+)?(voicemail|voice\s*mail|mailbox|messaging)|"
    r"you\s+have\s+reached\s+(the\s+)?(voicemail|mailbox|message)|"
    # Classic carrier VM: "The person you're calling…" (Tiny: passion/party/portion)
    r"the\s+(person|passion|party|portion|passenger|persons?)\s+"
    r"you(?:'re|\s+are|\s+have)?\s*(calling|called|call)|"
    r"you(?:'re|\s+are)\s+calling|"
    # "You have reached. Mail." / "you've reached the voicemail of…"
    r"you\s+have\s+reached|you'?ve\s+reached|you\s+reached|"
    r"reached\s+(the\s+)?(mailbox|voicemail|voice\s*mail|mail)|"
    r"please\s+leave\s+(a\s+)?message|"
    r"no\s+one\s+is\s+available|forwarded\s+to\s+an?\s+automati(?:c|ed)|"
    r"been\s+forwarded\s+to|it'?s\s+been\s+forwarded|"
    r"automatic\s+voice\s+message|automated\s+voice\s+message|"
    r"your\s+call\s+has\s+been\s+forwarded|try\s+again\s+later|"
    r"call\s+back\s+later|mailbox\s+is\s+full|mail\s*box\s+(is\s+)?full|"
    r"is\s+full\s+and|full\s+and\s+there\s+is\s+not|there\s+is\s+no\s+(more\s+)?"
    r"(room|space|storage)|"
    r"is\s+not\s+available|"
    r"please\s+record|leave\s+your\s+(name|message)|after\s+the\s+beep|"
    r"we(?:'ll|\s+will)\s+(get\s+back|return\s+your\s+call|call\s+you\s+back)|"
    r"subscriber|not\s+in\s+service|has\s+a\s+message|"
    # Carrier VM setup / blocked box (often truncated by AMD window)
    r"not\s+been\s+set\s+up|have\s+not\s+been\s+set\s+up|been\s+set\s+up\s+yet|"
    r"set\s+up\s+yet|please\s+try\s+your(\s+call)?|"
    r"try\s+your\s+call\s+again|call\s+again\s+later"
    r")\b",
    re.I,
)
_IVR_RE = re.compile(
    r"\b("
    r"press\s+\d|for\s+(english|spanish)|menu|your\s+call\s+is\s+important|"
    r"please\s+hold|enter\s+your|dial\s+\d|options?\s+are|to\s+speak\s+to|"
    r"extension|account\s+number|pin\s+number|pound|hash\s+key|"
    r"using\s+your\s+keypad|touch\s*tone|"
    r"if\s+you\s+have\s+not|not\s+been\s+set\s+up|set\s+up\s+yet|"
    r"please\s+try\s+your|try\s+your\s+call|enter\s+your\s+(password|passcode|pin)|"
    r"mailbox\s+is\s+full|is\s+full\s+and"
    r")\b",
    re.I,
)
_HUMAN_RE = re.compile(
    r"\b("
    r"hello+|hullo|hallo|halo|allo|yellow|"
    r"hi|hey|yeah|yes|yep|yup|yo|speaking|this\s+is|"
    # Live one-word answers — Whisper often hears "Hello" as "No."
    r"no+|nope|nah|what|huh|ok|okay|sure|alright|right|"
    r"who(?:'s|\s+is)\s+this|good\s+(morning|afternoon|evening)|"
    r"how\s+are\s+you|can\s+i\s+help|what'?s\s+up|go\s+ahead|"
    r"i'?m\s+here|who(?:'s|\s+is)\s+calling|pardon|sorry"
    r")\b",
    re.I,
)

# Whisper Tiny / Base common hallucinations on silence or short clips
_HALLUCINATION_RE = re.compile(
    r"("
    r"i'?ll\s+see\s+you\s+(next\s+time|later|soon)|"
    r"see\s+you\s+(next\s+time|later|soon)|"
    r"here\s+i\s+go|here\s+we\s+go|there\s+(you|we)\s+go|"
    r"here\s+we\s+are|that'?s\s+it\.?|"
    r"thanks?\s+for\s+(watching|listening|tuning\s+in)|"
    r"thank\s+you\s+for\s+(watching|listening)|"
    r"please\s+subscribe|subscribe\s+(and\s+)?(like|comment)|"
    r"like\s+and\s+subscribe|"
    r"don'?t\s+forget\s+to\s+subscribe|"
    r"in\s+the\s+next\s+video|"
    r"thanks?\s+for\s+watching\.?|"
    r"字幕|字幕by|amara\.org|"
    r"^\s*(music|applause|silence|blank)\s*$|"
    r"^\s*\.+\s*$|"
    r"www\.|http|"
    r"foreign\s*$"
    r")",
    re.I,
)

_DIGIT_WORDS = {
    "zero",
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
}

# Tiny often writes elderly / muffled "hello" as "Oh." — never treat alone as digit 0
_OH_AMBIGUOUS = {"oh", "o"}

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

# Bias toward phone greetings + digits + classic / Google Voice VM
_AMD_INITIAL_PROMPT = (
    "Hello. Hello. Hi. Yeah. Yes. Zero. One. Two. Three. Four. Five. "
    "Message system. One. Voicemail. Mailbox is full and there is not. "
    "If you have not been set up yet, please try your call again later. "
    "The person you are calling is not available. "
    "Please leave a message after the beep. Press one for English."
)


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

    Examples: '0'  'zero'  'Four, four, seven.'  '4, 4, 7, 4.'  'zero one two'

    Bare 'Oh.' / 'O' is NOT a digit — Tiny often mishears elderly 'hello' that way.
    """
    raw = (text or "").strip()
    if not raw:
        return False
    compact = re.sub(r"[^0-9]", "", raw)
    toks = [t for t in _tokens(raw) if t not in ("and",)]
    # Lone "oh" / "o" → not a number (hello mishear)
    if len(toks) == 1 and toks[0] in _OH_AMBIGUOUS and not compact:
        return False
    # Bare digit(s): "0", "00", "447"
    if compact and len(toks) <= 1 and len(compact) >= 1:
        return True
    if len(toks) == 1 and _is_digit_token(toks[0]):
        return True
    if len(toks) < 2:
        return len(compact) >= 3
    # Count "oh" as digit only inside a multi-digit readout ("oh four seven")
    digit_n = 0
    for tok in toks:
        if tok in _OH_AMBIGUOUS or _is_digit_token(tok):
            digit_n += 1
    if digit_n >= 2 and digit_n >= max(2, int(round(len(toks) * 0.6))):
        return True
    return len(compact) >= 3 and digit_n >= 2


def is_hello_mishear(text: str, *, audio_seconds: Optional[float] = None) -> bool:
    """Tiny often reduces muffled/elderly 'hello' to 'Oh.' / 'O' / 'Ah'.

    Do NOT trust this on very short clips — Tiny also turns the start of
    'Your call has been forwarded…' / VM into a lone 'Oh.'.
    """
    t = (text or "").strip().lower().rstrip(".")
    if t not in ("oh", "o", "ah", "uh", "mm", "hmm", "huh", "ay", "ey", "go", "guh"):
        return False
    # Sub-1s clips are usually truncated VM / early-media, not a full hello
    if audio_seconds is not None and float(audio_seconds) < 1.05:
        return False
    return True


def is_whisper_hallucination(text: str) -> bool:
    """True for known Tiny hallucinations (YouTube outros, empty junk)."""
    t = (text or "").strip()
    if not t:
        return False
    if _HALLUCINATION_RE.search(t):
        return True
    words = t.split()
    # Short invented filler that is not a phone greeting / digit / VM phrase
    if 2 <= len(words) <= 5:
        if not _HUMAN_RE.search(t) and not _MACHINE_RE.search(t) and not is_number_readout(t):
            if re.search(
                r"\b(go|going|goes|watching|subscribe|video|channel|episode|music)\b",
                t,
                re.I,
            ):
                return True
    if len(words) >= 6 and not _HUMAN_RE.search(t) and not _MACHINE_RE.search(t) and not is_number_readout(t):
        if re.search(r"\b(watching|subscribe|video|channel|episode)\b", t, re.I):
            return True
    return False


def clean_transcript(text: str) -> Tuple[str, bool]:
    """Return (cleaned_text, was_hallucination). Empty string if hallucination."""
    t = (text or "").strip()
    if not t:
        return "", False
    if is_whisper_hallucination(t):
        return "", True
    # Drop leading filler "and" alone before a hallucination fragment
    t2 = re.sub(r"^\s*and\s+", "", t, flags=re.I).strip()
    if t2 and is_whisper_hallucination(t2):
        return "", True
    return t, False


def classify_transcript(
    text: str,
    xgb_probs: Optional[Dict[str, float]] = None,
    *,
    audio_seconds: Optional[float] = None,
) -> Tuple[Optional[str], float, str]:
    """Return (status_or_None, confidence, cue). None = no strong cue.

    Spoken digits / IVR prompts / voicemail phrases → MACHINE (VICIdial AA).
    Bare 'Oh.' (elderly hello mishear) → HUMAN — only when clip is long enough.
    """
    probs = xgb_probs or {}
    t, hall = clean_transcript(text)
    if hall:
        return None, 0.0, "hallucination"
    if not t:
        return None, 0.0, "none"

    # Elderly / muffled hello often becomes "Oh." — send to agent (not on tiny clips)
    if is_hello_mishear(t, audio_seconds=audio_seconds):
        return "HUMAN", max(0.88, float(probs.get("HUMAN", 0.5))), "human_short"

    # Portal custom MACHINE/VM phrases (Settings → Machine Phrases)
    try:
        from app.machine_phrases import match_custom_machine_phrase

        custom_hit = match_custom_machine_phrase(t)
        if custom_hit:
            return (
                "MACHINE",
                max(0.95, float(probs.get("MACHINE", 0.5))),
                f"custom_phrase:{custom_hit}",
            )
    except Exception:
        pass

    # Lone live answer (Hello / No / Yes / …). Whisper often mishears hello as "No."
    # Require ~1s+ clip so ultra-short truncated VM openings stay MACHINE.
    if re.fullmatch(
        r"\s*(hello+|hi|hey|yeah|yes|yep|yup|yo|no+|nope|nah|what|huh|"
        r"ok|okay|sure|alright|right|oh|ah|uh|mm|hmm|go+|guh)\.?\s*",
        t,
        re.I,
    ):
        if audio_seconds is None or float(audio_seconds) >= 0.95:
            return "HUMAN", max(0.90, float(probs.get("HUMAN", 0.5))), "human_short"

    # Voicemail phrases first (incl. Tiny mishears like "passion you're calling")
    if _MACHINE_RE.search(t):
        return "MACHINE", max(0.94, float(probs.get("MACHINE", 0.5))), "voicemail"
    if _IVR_RE.search(t) or is_number_readout(t):
        return "MACHINE", max(0.93, float(probs.get("MACHINE", 0.5))), "ivr_digits"
    # Tiny fragments: "You have reached. Mail." / "reached mail" without full phrase match
    if re.search(r"\breached\b", t, re.I) and re.search(r"\bmail(box)?\b", t, re.I):
        return "MACHINE", max(0.93, float(probs.get("MACHINE", 0.5))), "voicemail"
    if re.search(r"\byou\s+have\s+reached\b", t, re.I):
        return "MACHINE", max(0.93, float(probs.get("MACHINE", 0.5))), "voicemail"
    # "Hi. If you record your name…" — greeting word + VM instruction
    if re.search(r"\brecord\s+your\s+name\b", t, re.I):
        return "MACHINE", max(0.94, float(probs.get("MACHINE", 0.5))), "voicemail"
    if re.search(r"\bif\s+you\s+record\b", t, re.I):
        return "MACHINE", max(0.93, float(probs.get("MACHINE", 0.5))), "voicemail"
    # Standalone "Mail." / "Mailbox." on a short clip after silence → VM
    if re.fullmatch(r"\s*mails?(box)?\.?\s*", t, re.I):
        return "MACHINE", max(0.9, float(probs.get("MACHINE", 0.5))), "voicemail"
    # Standalone "Voicemail." / "Voice mail."
    if re.fullmatch(r"\s*voice\s*mails?\.?\s*", t, re.I):
        return "MACHINE", max(0.94, float(probs.get("MACHINE", 0.5))), "voicemail"
    # "Message system. One." / "Message. System." + optional digit (Google Voice etc.)
    if re.search(r"\bmessage\s+system\b", t, re.I) or re.search(
        r"\bmessaging\s+system\b", t, re.I
    ):
        return "MACHINE", max(0.95, float(probs.get("MACHINE", 0.5))), "voicemail"
    if re.search(r"\b(message|mail|mailbox|voicemail)\b", t, re.I) and re.search(
        r"\b(one|two|three|four|five|six|seven|eight|nine|zero|[0-9])\b", t, re.I
    ):
        return "MACHINE", max(0.93, float(probs.get("MACHINE", 0.5))), "voicemail"
    # Truncated "mailbox is full and there is not [enough space]…"
    if re.search(r"\bfull\b", t, re.I) and re.search(
        r"\b(there\s+is\s+not|no\s+(more\s+)?(room|space)|and\s+there\s+is)\b", t, re.I
    ):
        return "MACHINE", max(0.94, float(probs.get("MACHINE", 0.5))), "voicemail"
    if re.fullmatch(r"\s*full\.?\s*", t, re.I):
        return "MACHINE", max(0.9, float(probs.get("MACHINE", 0.5))), "voicemail"
    # "If you have not been set up yet, please try your…" (carrier VM/IVR setup)
    if re.search(r"\bset\s+up\b", t, re.I) and re.search(
        r"\b(please\s+try|not\s+been|if\s+you\s+have\s+not)\b", t, re.I
    ):
        return "MACHINE", max(0.94, float(probs.get("MACHINE", 0.5))), "ivr_digits"
    if re.search(r"\bplease\s+try\s+your\b", t, re.I) and not _HUMAN_RE.search(t):
        return "MACHINE", max(0.92, float(probs.get("MACHINE", 0.5))), "ivr_digits"
    # Live hello only when the whole clip is a short greeting — not "Hi" + VM script
    if _HUMAN_RE.search(t) and len(t.split()) <= 10 and not is_number_readout(t):
        if re.search(
            r"\b(record|message|mailbox|voicemail|forwarded|reached|beep|tone|"
            r"unavailable|not\s+available|call\s+back|system|full|set\s+up)\b",
            t,
            re.I,
        ):
            return "MACHINE", max(0.92, float(probs.get("MACHINE", 0.5))), "voicemail"
        return "HUMAN", max(0.90, float(probs.get("HUMAN", 0.5))), "human_short"
    # Multi-word scripted line with no live greeting → MACHINE/IVR (AMD often truncates)
    if len(t.split()) >= 4 and not _HUMAN_RE.search(t):
        if re.search(
            r"\b(full|mailbox|voicemail|set\s+up|try\s+your|press|enter\s+your|"
            r"forwarded|reached|leave\s+(a\s+)?message|not\s+available)\b",
            t,
            re.I,
        ):
            return "MACHINE", max(0.9, float(probs.get("MACHINE", 0.5))), "voicemail"
    # Scripted line with "calling" and no human greeting → treat as AM
    if re.search(r"\b(you(?:'re|\s+are)\s+calling|you\s+have\s+called)\b", t, re.I):
        if not _HUMAN_RE.search(t):
            return "MACHINE", max(0.9, float(probs.get("MACHINE", 0.5))), "voicemail"
    return None, 0.0, "none"


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_whisper_model_name() -> str:
    try:
        from app.amd_settings import load_amd_settings

        name = str(load_amd_settings().get("ml_whisper_model") or "").strip()
    except Exception:
        name = ""
    if name not in _ALLOWED_WHISPER_MODELS:
        name = _DEFAULT_WHISPER_MODEL
    return name


def reset_whisper_model() -> None:
    """Drop cached model so next call reloads (after Settings change)."""
    global _model, _model_name_loaded, _model_error
    with _lock:
        _model = None
        _model_name_loaded = ""
        _model_error = ""


def _get_model():
    global _model, _model_name_loaded, _model_error
    want = _resolve_whisper_model_name()
    with _lock:
        if _model is not None and _model_name_loaded == want:
            return _model
        if _model is not None and _model_name_loaded != want:
            _model = None
            _model_name_loaded = ""
            _model_error = ""
        if _model_error and _model_name_loaded == want:
            return None
        try:
            from faster_whisper import WhisperModel

            # CPU int8 — free, local, no API. base.en ≈ better accuracy than tiny.
            beam_hint = want  # noqa: F841 — name kept for logs
            _model = WhisperModel(want, device="cpu", compute_type="int8")
            _model_name_loaded = want
            _model_error = ""
            return _model
        except Exception as exc:
            _model_error = str(exc)
            _model_name_loaded = want
            return None


def _transcribe_beam_size() -> int:
    name = _resolve_whisper_model_name()
    if name.startswith("small"):
        return 3
    if name.startswith("base"):
        return 2
    return 1


def _trim_to_speech(clip, sr: int = 16000):
    """Keep the loudest speech region; drop long leading/trailing silence."""
    import numpy as np

    if clip is None or len(clip) < int(sr * 0.25):
        return clip
    frame = max(1, int(sr * 0.02))
    n = len(clip) // frame
    if n < 3:
        return clip
    frames = clip[: n * frame].reshape(n, frame)
    energy = np.mean(frames ** 2, axis=1)
    thr = float(np.percentile(energy, 55) * 1.8 + 1e-8)
    active = energy > thr
    if not np.any(active):
        # Fallback: keep top-energy 0.8s around peak
        peak_i = int(np.argmax(energy))
        half = max(5, int(0.4 * sr / frame))
        a = max(0, peak_i - half)
        b = min(n, peak_i + half)
        return clip[a * frame : b * frame]
    idx = np.where(active)[0]
    pad = max(2, int(0.12 * sr / frame))
    a = max(0, int(idx[0]) - pad)
    b = min(n, int(idx[-1]) + pad + 1)
    out = clip[a * frame : b * frame]
    # Cap speech window — AMD greetings are short; long silence invites Tiny hallucinations
    max_len = int(sr * 2.2)
    if len(out) > max_len:
        # Keep densest energy block
        out = out[:max_len]
    return out


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
        sr = 16000

    clip = _trim_to_speech(clip, sr=16000)
    if clip is None or len(clip) < 400:
        return "", None, {"whisper_ok": True, "transcript": "", "cue": "too_short"}

    # Soft peak normalize for quiet hellos
    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if peak > 1e-4:
        clip = (clip / peak) * 0.9

    try:
        beam = _transcribe_beam_size()
        segments, info = model.transcribe(
            clip,
            language="en",
            beam_size=beam,
            best_of=beam,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=250,
                speech_pad_ms=120,
            ),
            without_timestamps=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.2,
            no_speech_threshold=0.55,
            temperature=0.0,
            initial_prompt=_AMD_INITIAL_PROMPT,
        )
        text_parts = []
        for seg in segments:
            if seg.text:
                # Drop very low-confidence segments when available
                avg_logprob = getattr(seg, "avg_logprob", None)
                no_speech = getattr(seg, "no_speech_prob", None)
                if no_speech is not None and float(no_speech) > 0.7:
                    continue
                if avg_logprob is not None and float(avg_logprob) < -1.15:
                    continue
                text_parts.append(seg.text.strip())
        text = " ".join(text_parts).strip()
    except Exception as exc:
        return "", None, {"whisper_ok": False, "error": str(exc)}

    text, hall = clean_transcript(text)
    meta = {"whisper_ok": True}
    if hall:
        meta["hallucination_cleared"] = True
    return text, info, meta


def refine_with_whisper(
    audio,  # np.ndarray float32 mono
    sr: int,
    *,
    xgb_probs: Dict[str, float],
    max_seconds: float = 4.0,
) -> Tuple[Optional[str], float, Dict[str, Any]]:
    """Return (status_or_None, confidence, details). None status = keep XGB decision."""
    import numpy as np

    clip = np.asarray(audio, dtype=np.float32)
    audio_seconds = float(len(clip) / float(sr or 1)) if sr and len(clip) else 0.0
    text, info, base = _transcribe_clip(audio, sr, max_seconds)
    if not base.get("whisper_ok"):
        return None, 0.0, base

    details: Dict[str, Any] = {
        "whisper_ok": True,
        "transcript": text[:240],
        "language": getattr(info, "language", "") or "",
        "audio_seconds": round(audio_seconds, 3),
    }
    if base.get("hallucination_cleared"):
        details["hallucination_cleared"] = True
        details["cue"] = "hallucination"
        return None, 0.0, details
    if not text:
        return None, 0.0, {**details, "cue": "none"}

    status, conf, cue = classify_transcript(
        text, xgb_probs, audio_seconds=audio_seconds
    )
    # Lone "Oh." on a sub-1s clip: do not promote to HUMAN (often truncated VM)
    if cue == "human_short" and is_hello_mishear(text) and audio_seconds < 1.05:
        details["cue"] = "oh_too_short"
        details["oh_suppressed"] = True
        return None, 0.0, details
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

    cue = "none"
    if base.get("hallucination_cleared"):
        cue = "hallucination"
    elif text:
        _status, _conf, cue = classify_transcript(text, {})

    return {
        "whisper_ok": True,
        "transcript": text[:500],
        "cue": cue or "none",
        "language": getattr(info, "language", "") or "",
        "hallucination_cleared": bool(base.get("hallucination_cleared")),
    }


_SCAM_INITIAL_PROMPT = (
    "Phone conversation between a call center agent and a customer. "
    "Bank, credit card, chase, spectrum, account number, verify, yes, yeah, hello."
)


def _to_16k_mono(audio, sr: int):
    import numpy as np

    clip = np.asarray(audio, dtype=np.float32)
    if getattr(clip, "ndim", 1) > 1:
        clip = np.mean(clip, axis=1)
    if sr != 16000 and len(clip) > 1:
        duration = len(clip) / float(sr or 1)
        new_len = max(1, int(duration * 16000))
        x_old = np.linspace(0, 1, num=len(clip), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        clip = np.interp(x_new, x_old, clip).astype(np.float32)
    return clip.astype(np.float32)


def transcribe_full_call(
    audio,
    sr: int,
    *,
    max_seconds: float = 480.0,
) -> Dict[str, Any]:
    """Full agent-leg speech→text for SCAMMERS (NOT the short AMD 2.2s greeting trim).

    Walks the whole recording in overlapping chunks so blacklist matching sees
    the complete call, not only the opening \"Hello\".
    """
    model = _get_model()
    if model is None:
        return {
            "whisper_ok": False,
            "transcript": "",
            "error": _model_error or "not_installed",
        }

    import numpy as np

    if audio is None or len(audio) == 0:
        return {"whisper_ok": False, "transcript": "", "error": "empty_audio"}

    max_seconds = max(5.0, min(600.0, float(max_seconds or 480.0)))
    clip = _to_16k_mono(audio, int(sr or 16000))
    n_max = int(min(len(clip), max(1, int(16000 * max_seconds))))
    clip = clip[:n_max]
    if len(clip) < 400:
        return {"whisper_ok": True, "transcript": "", "error": "", "cue": "too_short"}

    peak = float(np.max(np.abs(clip))) if len(clip) else 0.0
    if peak > 1e-4:
        clip = (clip / peak) * 0.9

    # Chunk ~30s with 1s overlap so Tiny stays accurate on longer calls
    chunk = int(16000 * 30)
    hop = int(16000 * 29)
    parts: list[str] = []
    try:
        start = 0
        while start < len(clip):
            end = min(len(clip), start + chunk)
            piece = clip[start:end]
            if len(piece) < 400:
                break
            segments, _info = model.transcribe(
                piece,
                language="en",
                beam_size=1,
                best_of=1,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=400,
                    speech_pad_ms=200,
                ),
                without_timestamps=True,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6,
                temperature=0.0,
                initial_prompt=_SCAM_INITIAL_PROMPT,
            )
            for seg in segments:
                t = (seg.text or "").strip()
                if not t:
                    continue
                no_speech = getattr(seg, "no_speech_prob", None)
                if no_speech is not None and float(no_speech) > 0.85:
                    continue
                parts.append(t)
            if end >= len(clip):
                break
            start += hop
    except Exception as exc:
        return {"whisper_ok": False, "transcript": "", "error": str(exc)}

    text = " ".join(parts).strip()
    text, hall = clean_transcript(text)
    # Keep more text for blacklist matching (full call)
    return {
        "whisper_ok": True,
        "transcript": text[:12000],
        "error": "",
        "hallucination_cleared": bool(hall),
        "audio_seconds_used": float(len(clip) / 16000.0),
    }


# Alias used by older scam_detect imports
def transcribe_for_display(audio, sr: int, *, max_seconds: float = 480.0) -> Dict[str, Any]:
    return transcribe_full_call(audio, sr, max_seconds=max_seconds)
