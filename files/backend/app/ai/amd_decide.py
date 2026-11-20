"""Multi-stage AMD final decision (OpenAMD 6.x policy).

Stages:
  1) Acoustic / Silero VAD features (already computed)
  2) faster-whisper transcript (Classic + Advanced)
  3) Phrase + timing score

Decision philosophy (outbound call center):
  - Only hang as MACHINE with strong evidence (beep, ringback, VM/IVR phrases,
    a greeting still playing when the capture ends, stock AMD MACHINE that the
    transcript does not contradict, or a high machine score).
  - HUMAN needs speech: a clip where Silero finds nothing spoken is BLANK.
  - Speech that fits none of the above → HUMAN (false VM discards customers;
    false HUMAN is recoverable).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _machine_cue(cue: str) -> bool:
    c = str(cue or "")
    return c in ("voicemail", "ivr", "ivr_digits", "digits") or c.startswith("custom_phrase")


# A talker still going after this much unbroken speech when the capture ends is
# reading a greeting. Live answers ("Hello?", "This is Meg.") stop and wait:
# in the ~650 production recordings this was set from, no live answer ran into
# the cut for more than 0.6 s.
TAIL_SPEECH_MACHINE_MS = 1000.0
# …unless Whisper heard a greeting this short (someone repeating "Hello? Hello?").
TAIL_SPEECH_HUMAN_MAX_WORDS = 5


def _silero_spans(silero: Optional[Dict[str, Any]]) -> Optional[list]:
    """Silero speech islands, or None when Silero did not run."""
    if not isinstance(silero, dict) or not silero.get("ok"):
        return None
    if "spans" not in silero:
        return None
    return list(silero.get("spans") or [])


def _lone_word_then_silence(
    transcript: str,
    silero: Optional[Dict[str, Any]],
    clip_seconds: float,
) -> bool:
    """One short word with silence before and after it: a complete utterance.

    Somebody said a single word and is waiting. A number being read out never
    looks like this — it is a run of digits, or a fragment touching the start
    or the end of the clip.
    """
    if len((transcript or "").split()) != 1:
        return False
    spans = _silero_spans(silero)
    if not spans or len(spans) != 1:
        return False
    start, end = spans[0]
    if end - start > 0.7:
        return False
    return start >= 0.3 and clip_seconds - end >= 0.4


def _run_whisper(
    audio,
    sr: int,
    *,
    existing: Optional[Dict[str, Any]] = None,
    silero: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return whisper detail dict with optional status/conf/cue/transcript."""
    if isinstance(existing, dict) and (
        existing.get("transcript") or existing.get("cue") or existing.get("whisper_ok")
    ):
        return dict(existing)

    out: Dict[str, Any] = {"whisper_ok": False, "cue": "none", "transcript": ""}
    try:
        from app.amd_settings import load_amd_settings
        from app.ai.whisper_amd import refine_with_whisper, whisper_available

        cfg = load_amd_settings()
        if not bool(cfg.get("ml_whisper_enabled", True)):
            out["error"] = "whisper_disabled"
            out["cue"] = "disabled"
            return out
        if not whisper_available():
            out["error"] = "whisper_unavailable"
            return out
        status, conf, details = refine_with_whisper(
            audio,
            sr,
            xgb_probs={},
            max_seconds=5.0,
            speech_spans=_silero_spans(silero),
        )
        out = {**details, "whisper_ok": bool(details.get("whisper_ok", True))}
        if status:
            out["status"] = status
            out["confidence"] = float(conf)
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out


def score_amd(
    *,
    feats: Dict[str, float],
    silero: Dict[str, Any],
    whisper: Dict[str, Any],
    looks_short_human: bool,
    looks_digit: bool,
    script_clip: bool,
    front_script: bool,
    dense_greeting: bool,
    xgb_probs: Optional[Dict[str, float]] = None,
) -> Tuple[int, str]:
    """Return (machine_score, note). Higher → more MACHINE."""
    score = 0
    parts: list[str] = []

    cue = str(whisper.get("cue") or "")
    w_status = str(whisper.get("status") or "")
    transcript = str(whisper.get("transcript") or "").strip()

    if _machine_cue(cue) or w_status == "MACHINE":
        score += 80
        parts.append(f"whisper_vm:{cue or 'machine'}")
    elif cue == "human_short" or w_status == "HUMAN":
        score -= 65
        parts.append("whisper_human")
    elif transcript and len(transcript.split()) >= 6:
        score += 25
        parts.append("long_transcript")

    # Asterisk stock AMD (when AGI posts AMDSTATUS/AMDCAUSE)
    a_h = float(feats.get("asterisk_amd_human", 0.0) or 0.0)
    a_m = float(feats.get("asterisk_amd_machine", 0.0) or 0.0)
    if a_m >= 0.5:
        score += 40
        parts.append("asterisk_machine")
    if a_h >= 0.5:
        score -= 40
        parts.append("asterisk_human")

    # XGBoost class probs (trained on your labeled calls when available)
    if xgb_probs:
        p_m = float(xgb_probs.get("MACHINE", 0.0) or 0.0)
        p_i = float(xgb_probs.get("IVR", 0.0) or 0.0)
        p_h = float(xgb_probs.get("HUMAN", 0.0) or 0.0)
        if p_m + p_i >= 0.85:
            score += 55
            parts.append(f"xgb_machine:{p_m + p_i:.2f}")
        elif p_h >= 0.85:
            score -= 50
            parts.append(f"xgb_human:{p_h:.2f}")
        elif p_m + p_i >= 0.60:
            score += 25
            parts.append(f"xgb_lean_m:{p_m + p_i:.2f}")
        elif p_h >= 0.60:
            score -= 20
            parts.append(f"xgb_lean_h:{p_h:.2f}")

    duration = float(feats.get("duration", 0.0))
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))
    s_long = float(silero.get("longest_speech_ms", 0.0) or 0.0)
    s_segs = int(silero.get("num_segments", 0) or 0)

    if looks_short_human:
        score -= 55
        parts.append("short_human")
    if looks_digit and not looks_short_human:
        score += 45
        parts.append("spoken_digit")
    if script_clip:
        score += 35
        parts.append("truncated_script")
    if front_script and not looks_short_human:
        score += 30
        parts.append("front_script")
    if dense_greeting:
        score += 20
        parts.append("dense_greeting")
    if duration >= 2.5 and speech_ratio >= 0.35:
        score += 15
        parts.append("long_speech")
    if s_long >= 1200 and s_segs >= 2:
        score += 15
        parts.append("silero_long")
    if longest_burst >= 1100 and speech_ratio >= 0.38:
        score += 15
        parts.append("long_burst")

    peak = float(feats.get("peak", 0.0))
    if peak >= 0.20 and looks_short_human:
        score -= 20
        parts.append("loud_hello")

    note = "+".join(parts) if parts else "neutral"
    return int(score), note


def _xgb_probs(feats: Dict[str, Any], silero: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        from app.ai.xgb_model import model_path, predict_proba

        if not model_path().exists():
            return None
        return predict_proba(feats, silero)
    except Exception:
        return None


def finalize_amd_decision(
    *,
    status: str,
    confidence: float,
    feats: Dict[str, float],
    silero: Dict[str, Any],
    audio,
    sr: int,
    details: Dict[str, Any],
    looks_short_human_fn,
    looks_digit_fn,
    is_script_clip_fn,
    is_front_script_fn,
    is_blank_fn,
    is_voip_fn,
    dispose_blank_fn,
) -> Tuple[str, float]:
    """Apply multi-stage policy; mutates details. Returns (status, confidence)."""
    st = str(status or "MACHINE").upper()
    conf = float(confidence)

    # Confident leave-message tone only. Weak beep_conf is often a speech formant
    # in a silence+hello MixMonitor window — never hard-lock those at 95%.
    strong_beep = (
        float(feats.get("beep", 0.0)) >= 0.5
        and float(feats.get("beep_conf", 0.0) or 0.0) >= 0.70
    )

    # Hard acoustic locks (never send to agent)
    strong_fax = (
        float(feats.get("fax", 0.0)) >= 0.5
        and float(feats.get("fax_conf", 0.0) or 0.0) >= 0.80
    )
    if strong_fax:
        # 8399 stock-human path: still capture Whisper for portal badge
        if float(feats.get("asterisk_amd_human", 0.0) or 0.0) >= 0.5:
            try:
                ml_block0 = details.get("ml") if isinstance(details.get("ml"), dict) else {}
                existing0 = (
                    ml_block0.get("whisper")
                    if isinstance(ml_block0.get("whisper"), dict)
                    else None
                )
                w0 = _run_whisper(audio, sr, existing=existing0, silero=silero)
                details["whisper_stage"] = w0
                details["stock_second_opinion_whisper"] = True
                if w0.get("transcript") or w0.get("cue") or w0.get("whisper_ok"):
                    details["whisper_used"] = True
                    if not ml_block0:
                        details["ml"] = {
                            "whisper": w0,
                            "whisper_used": True,
                            "ml_note": "stock_second_opinion_whisper",
                        }
            except Exception:
                pass
        details["amd_stage"] = "hard_fax"
        details["fuse_note"] = "fax_modem_tone"
        details["decision_reason"] = "fax_modem_tone"
        details["fax_detected"] = True
        return "FAX", max(conf, float(feats.get("fax_conf", 0.93) or 0.93), 0.93)
    if strong_beep:
        details["amd_stage"] = "hard_beep"
        details["fuse_note"] = "beep"
        return "MACHINE", max(conf, 0.95)
    if float(feats.get("ringback", 0.0)) >= 0.5:
        details["amd_stage"] = "hard_ringback"
        details["fuse_note"] = "ringback_tone"
        return "MACHINE", max(conf, float(feats.get("ringback_conf", 0.93)), 0.93)
    if float(feats.get("sit", 0.0)) >= 0.5:
        details["amd_stage"] = "hard_sit"
        details["fuse_note"] = "sit"
        return "SIT", max(conf, 0.9)

    if is_blank_fn(feats, silero) or is_voip_fn(feats, silero):
        if st == "HUMAN" and looks_short_human_fn(feats, silero) and not is_blank_fn(
            feats, silero
        ):
            pass
        else:
            b_status, b_conf, b_note = dispose_blank_fn(feats=feats)
            details["amd_stage"] = "blank_dispose"
            details["fuse_note"] = b_note
            details["decision_reason"] = b_note
            details["blank_safety_override"] = True
            return b_status, max(conf, b_conf)

    looks_human = bool(looks_short_human_fn(feats, silero))
    looks_digit = bool(looks_digit_fn(feats, silero))
    script_clip = bool(is_script_clip_fn(feats))
    front_script = bool(is_front_script_fn(feats))
    dense_greeting = (
        float(feats.get("duration", 0.0)) >= 1.8
        and (
            float(feats.get("speech_ratio", 0.0)) >= 0.35
            or float(silero.get("longest_speech_ms", 0.0) or 0) >= 1200
            or float(feats.get("abs_speech_ms", 0.0)) >= 600
        )
    )
    stock_human = float(feats.get("asterisk_amd_human", 0.0) or 0.0) >= 0.5
    stock_machine = float(feats.get("asterisk_amd_machine", 0.0) or 0.0) >= 0.5

    # Stage 2: Whisper — always on 8399 stock HUMAN|PERSON|NOTSURE second opinion
    # (even when acoustics already lean HUMAN). Catches AI/IVR/"fax" wording.
    ml_block = details.get("ml") if isinstance(details.get("ml"), dict) else {}
    existing_w = ml_block.get("whisper") if isinstance(ml_block.get("whisper"), dict) else None
    # Force a fresh Whisper pass when stock AMD trusted human and prior stage
    # skipped STT (confident heuristic HUMAN / non_human_accept without text).
    force_stock_whisper = bool(stock_human and not stock_machine)
    if force_stock_whisper and isinstance(existing_w, dict):
        has_text = bool(str(existing_w.get("transcript") or "").strip())
        if not has_text and not existing_w.get("whisper_ok"):
            existing_w = None
    whisper = _run_whisper(audio, sr, existing=existing_w, silero=silero)
    details["whisper_stage"] = whisper
    if force_stock_whisper:
        details["stock_second_opinion_whisper"] = True
    if whisper.get("transcript") or whisper.get("cue") or whisper.get("whisper_ok"):
        details["whisper_used"] = True
        if not ml_block:
            details["ml"] = {
                "whisper": whisper,
                "whisper_used": True,
                "ml_note": (
                    "stock_second_opinion_whisper"
                    if force_stock_whisper
                    else "classic_whisper_stage"
                ),
            }
        else:
            ml_block["whisper"] = whisper
            ml_block["whisper_used"] = True

    cue = str(whisper.get("cue") or "")
    w_status = str(whisper.get("status") or "")
    w_conf = float(whisper.get("confidence") or 0.0)
    transcript = str(whisper.get("transcript") or "").strip()

    # Re-classify transcript so AI/IVR/VM wording overrides a stale human_short
    # cue or stock-HUMAN rescue when Whisper text is clearly automated.
    if transcript and (
        not _machine_cue(cue) and w_status != "MACHINE"
    ):
        try:
            from app.ai.whisper_amd import classify_transcript

            re_st, re_conf, re_cue = classify_transcript(transcript)
            if re_st == "MACHINE" or _machine_cue(re_cue):
                cue = re_cue or "voicemail"
                w_status = "MACHINE"
                w_conf = max(w_conf, float(re_conf or 0.0), 0.93)
                whisper["status"] = "MACHINE"
                whisper["confidence"] = w_conf
                whisper["cue"] = cue
                details["whisper_machine_override"] = True
                details["whisper_machine_override_cue"] = cue
        except Exception:
            pass

    # "Five." / "There." heard for a lone "Hi" or "Hello?" on a call stock AMD
    # did not call a machine. One isolated word is not a number readout.
    if (
        cue == "ivr_digits"
        and not stock_machine
        and _lone_word_then_silence(
            transcript, silero, float(feats.get("duration", 0.0) or 0.0)
        )
    ):
        cue = "lone_word"
        w_status = ""
        w_conf = 0.0
        whisper["cue"] = cue
        whisper.pop("status", None)
        whisper.pop("confidence", None)
        details["lone_word_not_digits"] = True

    feats["whisper_vm_hit"] = 1.0 if (_machine_cue(cue) or w_status == "MACHINE") else 0.0
    feats["whisper_human_hit"] = 1.0 if (cue == "human_short" or w_status == "HUMAN") else 0.0
    feats["whisper_word_count"] = float(len(transcript.split())) if transcript else 0.0

    # Stage 3: XGBoost (always when model present — Classic or Advanced)
    xgb_probs = _xgb_probs(feats, silero)
    if xgb_probs:
        details["xgb_probs"] = {k: round(float(v), 4) for k, v in xgb_probs.items()}
        details["class_probs"] = details["xgb_probs"]

    # Clear VM/IVR/AI phrases still hang up (never overridden by stock HUMAN
    # or short-hello acoustic rescue).
    if _machine_cue(cue) or w_status == "MACHINE":
        details["amd_stage"] = "whisper_machine"
        details["fuse_note"] = cue or "voicemail"
        details["short_human_safety_override"] = False
        details["internal_status"] = "MACHINE"
        return "MACHINE", max(conf, w_conf, 0.93)

    whisper_human = cue == "human_short" or w_status == "HUMAN"
    word_count = len(transcript.split()) if transcript else 0

    # Still talking when the capture ended, after a second or more of
    # unbroken speech: a greeting being read out ("Hello, this is Kelly
    # Blackburn, a Coldwell Banker…"), hold music, or a carrier announcement in
    # a language Whisper could not transcribe. Not someone who said hello and
    # is waiting for an answer.
    tail_ms = float(silero.get("tail_speech_ms", 0.0) or 0.0) if silero.get("ok") else 0.0
    details["tail_speech_ms"] = round(tail_ms, 1)
    if tail_ms >= TAIL_SPEECH_MACHINE_MS and not (
        whisper_human and word_count <= TAIL_SPEECH_HUMAN_MAX_WORDS
    ):
        details["amd_stage"] = "speech_into_cut"
        details["fuse_note"] = "greeting_still_playing"
        details["short_human_safety_override"] = False
        details["internal_status"] = "MACHINE"
        return "MACHINE", max(conf, 0.90)

    if whisper_human:
        details["amd_stage"] = "whisper_human"
        details["fuse_note"] = "human_short"
        details["short_human_safety_override"] = True
        details["internal_status"] = "HUMAN"
        return "HUMAN", max(conf, w_conf, 0.92)

    # Nothing in the clip is speech: Silero found no spoken island, so Whisper
    # was given nothing and returned nothing. Clicks, line noise and tones the
    # hard detectors did not name end here. Loud is not the same as spoken, and
    # an agent must not be connected to a clip nobody spoke in.
    spans = _silero_spans(silero)
    if spans is not None and not spans and not transcript:
        b_status, b_conf, _ = dispose_blank_fn(feats=feats)
        details["amd_stage"] = "no_speech"
        if b_status == "BLANK":
            details["fuse_note"] = "blank_no_speech"
            details["decision_reason"] = "blank_no_speech"
            details["blank_safety_override"] = True
            details["short_human_safety_override"] = False
        else:
            details["fuse_note"] = "blank_to_agent"
            details["decision_reason"] = "blank_to_agent"
        return b_status, max(conf, b_conf)

    # Stock Asterisk AMD already called this call MACHINE and the second look
    # found no live greeting to overrule it. (Clips recorded after stock AMD
    # finished often hold only the tail of the greeting: "…system.")
    if (
        stock_machine
        and not stock_human
        and bool(whisper.get("whisper_ok"))
    ):
        details["amd_stage"] = "stock_machine_unrefuted"
        details["fuse_note"] = "stock_amd_machine"
        details["short_human_safety_override"] = False
        details["internal_status"] = "MACHINE"
        return "MACHINE", max(conf, 0.90)

    # Stock Asterisk AMD said HUMAN + sparse live hello acoustics → agent.
    # Does not apply when stock said MACHINE / dense greeting / digit prompt
    # / Whisper already showed automated wording above.
    if (
        stock_human
        and not stock_machine
        and looks_human
        and not looks_digit
        and not dense_greeting
        and not script_clip
    ):
        details["amd_stage"] = "stock_human_short"
        details["fuse_note"] = "stock_amd_human_short"
        details["short_human_safety_override"] = True
        details["internal_status"] = "HUMAN"
        return "HUMAN", max(conf, 0.92)

    score, score_note = score_amd(
        feats=feats,
        silero=silero,
        whisper=whisper,
        looks_short_human=looks_human,
        looks_digit=looks_digit,
        script_clip=script_clip,
        front_script=front_script,
        dense_greeting=dense_greeting,
        xgb_probs=xgb_probs,
    )
    details["amd_score"] = score
    details["amd_score_note"] = score_note

    if score >= 70:
        details["amd_stage"] = "score_machine"
        details["fuse_note"] = f"score_{score}:{score_note}"
        details["internal_status"] = "MACHINE"
        return "MACHINE", max(conf, 0.90)

    # VM / IVR phrases already returned MACHINE above.
    # Any other speech (hello, or Whisper unsure) goes to an agent.
    # Only true silence stays BLANK, and that is handled before this stage.
    if looks_human or float(feats.get("peak", 0.0)) >= 0.02:
        details["amd_stage"] = "speech_to_agent"
        details["fuse_note"] = "short_human_greeting" if looks_human else "uncertain_to_agent"
        details["short_human_safety_override"] = True
        details["internal_status"] = "HUMAN" if looks_human else "UNCERTAIN"
        return "HUMAN", max(conf, 0.90)

    details["amd_stage"] = "uncertain_to_agent"
    details["fuse_note"] = "uncertain_to_agent"
    details["short_human_safety_override"] = True
    details["internal_status"] = "UNCERTAIN"
    return "HUMAN", max(conf, 0.90)
