"""Optional ML AMD pipeline (Silero features → XGBoost → Whisper).

Whisper runs when:
  - XGBoost would send HUMAN to an agent (any confidence) — catches IVR
    number-readouts like "four four seven" that XGB scores as HUMAN, or
  - XGBoost says MACHINE/IVR without a beep / long greeting — catches a
    live "hello" that XGB scored as a weak MACHINE (e.g. 67%).

Only imported when Settings → ML pipeline is enabled. Disabled = zero overhead
beyond the existing hybrid engine.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.amd_settings import load_amd_settings


def is_ml_pipeline_enabled() -> bool:
    return bool(load_amd_settings().get("ml_pipeline_enabled", False))


def run_ml_pipeline(
    *,
    audio: np.ndarray,
    sr: int,
    feats: Dict[str, Any],
    silero: Dict[str, Any],
    hybrid_status: str,
    hybrid_confidence: float,
    cfg: Optional[Dict[str, Any]] = None,
    call_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Returns (status, confidence, ml_details).

    confidence is the winning class probability from XGBoost (or Whisper refine).

    Policy (agent-protect):
      - BLANK → accept immediately
      - MACHINE / IVR with beep or long greeting → accept
      - MACHINE / IVR that looks like a short "hello" → Whisper (or keep HUMAN)
      - HUMAN + conf >= threshold → pass to agent
      - HUMAN + conf < threshold → Faster-Whisper Tiny refine
    """
    cfg = cfg if cfg is not None else load_amd_settings()
    high_thr = float(cfg.get("ml_xgb_high_confidence", 0.85))
    whisper_on = bool(cfg.get("ml_whisper_enabled", True))
    save_low = bool(cfg.get("ml_save_low_confidence", True))
    low_thr = float(cfg.get("ml_low_confidence_threshold", 0.85))
    call_meta = call_meta or {}

    from app.ai.engine import (
        _blank_disposition,
        _is_blank,
        _is_voip_noise_burst,
        _looks_like_short_human,
        _looks_like_spoken_digit,
    )
    from app.amd_settings import is_blank_as_machine_enabled

    if is_blank_as_machine_enabled() and _is_blank(feats, silero):
        b_status, b_conf, _ = _blank_disposition()
        details: Dict[str, Any] = {
            "ml_pipeline": True,
            "hybrid_status": hybrid_status,
            "hybrid_confidence": round(float(hybrid_confidence), 4),
            "whisper_policy": "whisper_before_agent",
            "ml_note": "blank_audio",
        }
        return b_status, b_conf, details

    if is_blank_as_machine_enabled() and _is_voip_noise_burst(feats, silero):
        b_status, b_conf, _ = _blank_disposition()
        details = {
            "ml_pipeline": True,
            "hybrid_status": hybrid_status,
            "hybrid_confidence": round(float(hybrid_confidence), 4),
            "whisper_policy": "whisper_before_agent",
            "ml_note": "voip_front_noise_burst",
            "voip_noise_burst": True,
        }
        return b_status, b_conf, details

    if float(feats.get("ringback", 0.0) or 0.0) >= 0.5:
        details = {
            "ml_pipeline": True,
            "hybrid_status": hybrid_status,
            "hybrid_confidence": round(float(hybrid_confidence), 4),
            "whisper_policy": "whisper_before_agent",
            "ml_note": "ringback_tone",
            "ringback_detected": True,
        }
        return "MACHINE", max(0.93, float(feats.get("ringback_conf") or 0.93)), details

    details: Dict[str, Any] = {
        "ml_pipeline": True,
        "hybrid_status": hybrid_status,
        "hybrid_confidence": round(float(hybrid_confidence), 4),
        "whisper_policy": "whisper_before_agent",
    }

    try:
        from app.ai.xgb_model import decide_from_proba, ensure_model, predict_proba
    except Exception as exc:
        details["ml_error"] = f"xgboost_unavailable: {exc}"
        details["ml_fallback"] = "hybrid"
        return hybrid_status, float(hybrid_confidence), details

    try:
        ensure_model()
        probs = predict_proba(feats, silero)
    except Exception as exc:
        details["ml_error"] = f"xgb_predict_failed: {exc}"
        details["ml_fallback"] = "hybrid"
        return hybrid_status, float(hybrid_confidence), details

    status, conf = decide_from_proba(probs)
    details["class_probs"] = {k: round(float(v), 4) for k, v in probs.items()}
    details["xgb_status"] = status
    details["xgb_confidence"] = round(float(conf), 4)
    details["whisper_used"] = False
    looks_human = _looks_like_short_human(feats, silero)
    looks_digit = _looks_like_spoken_digit(feats, silero)
    no_beep = float(feats.get("beep", 0.0) or 0.0) < 0.5
    strong_machine = (
        not no_beep
        or (
            float(feats.get("duration", 0.0)) >= 2.1
            and float(feats.get("speech_ratio", 0.0)) >= 0.42
            and float(silero.get("longest_speech_ms", 0.0) or 0.0) >= 1400
        )
    )

    # Keep clear SIT from hybrid when XGB is unsure HUMAN
    if status == "HUMAN" and hybrid_status == "SIT" and hybrid_confidence >= 0.8:
        status, conf = "SIT", float(hybrid_confidence)
        details["ml_note"] = "keep_hybrid_sit"
        return status, float(conf), details

    def _apply_whisper(note: str) -> Optional[Tuple[str, float]]:
        nonlocal status, conf
        if not whisper_on:
            return None
        try:
            from app.ai.whisper_amd import (
                is_number_readout,
                refine_with_whisper,
                whisper_available,
            )

            if not whisper_available():
                details["ml_note"] = note + "_whisper_unavailable"
                return None
            w_status, w_conf, w_det = refine_with_whisper(
                audio, sr, xgb_probs=probs, max_seconds=5.0
            )
            details["whisper"] = w_det
            details["whisper_used"] = True
            transcript = str((w_det or {}).get("transcript") or "")
            cue = str((w_det or {}).get("cue") or "")
            if w_status:
                details["ml_note"] = note + f"_whisper_{w_status.lower()}"
                return w_status, float(w_conf)
            if transcript and is_number_readout(transcript):
                details["ml_note"] = note + "_whisper_digits"
                details["whisper"]["cue"] = "ivr_digits"
                return "MACHINE", max(0.93, float(probs.get("MACHINE", 0.5)))
            if cue == "hallucination" and looks_digit:
                details["ml_note"] = note + "_hallucination_digit_machine"
                return "MACHINE", 0.9
        except Exception as exc:
            details["whisper_error"] = str(exc)
            details["ml_note"] = note + "_whisper_error"
        return None

    # Spoken digit / IVR syllable before any HUMAN path
    if looks_digit and status == "HUMAN":
        details["ml_note"] = "acoustic_digit_check"
        rescued = _apply_whisper("acoustic_digit_check")
        if rescued:
            return rescued[0], rescued[1], details
        details["ml_note"] = "acoustic_digit_machine"
        return "MACHINE", 0.9, details

    # MACHINE/IVR: Whisper unless it is a strong beep / long greeting.
    # 67% "hello" must not hang up — only high-confidence dense AM skips Whisper.
    if status in ("MACHINE", "IVR") and no_beep and not strong_machine:
        details["ml_note"] = "xgb_machine_whisper_check"
        rescued = _apply_whisper("xgb_machine_whisper_check")
        if rescued:
            return rescued[0], rescued[1], details
        if looks_digit:
            details["ml_note"] = (details.get("ml_note") or "") + "+digit_machine"
            return "MACHINE", 0.9, details
        # No Whisper cue: prefer HUMAN when the clip is a sparse pickup or XGB is unsure
        if looks_human or hybrid_status == "HUMAN" or conf < high_thr:
            details["ml_note"] = (details.get("ml_note") or "") + "+uncertain_machine_to_human"
            return "HUMAN", max(float(hybrid_confidence), 0.82), details

    # Answering machine / IVR / SIT / BLANK with strong evidence → accept
    if status != "HUMAN":
        details["ml_note"] = "non_human_accept"
        return status, float(conf), details

    # HUMAN (any confidence) → Whisper before sending to an agent
    if is_blank_as_machine_enabled() and _is_blank(feats, silero):
        b_status, b_conf, _ = _blank_disposition()
        details["ml_note"] = "blank_override_high_human"
        return b_status, b_conf, details

    # Always run Whisper before agent when enabled — short "hello-looking" clips
    # are often carrier/Google Voice VM ("Message system. One.", "Voicemail").
    # Skipping Whisper here was sending scripted VM to agents.
    w_hit = _apply_whisper("human_to_agent")
    if w_hit:
        status, conf = w_hit
        if status != "HUMAN":
            details["ml_note"] = (details.get("ml_note") or "") + "+block_non_human"
            return status, float(conf), details
        # Whisper said HUMAN — still block if transcript is clearly VM wording
        w_text = ""
        if isinstance(details.get("whisper"), dict):
            w_text = str(details["whisper"].get("transcript") or "")
        if w_text:
            try:
                from app.ai.whisper_amd import classify_transcript

                w_status2, w_conf2, w_cue2 = classify_transcript(
                    w_text,
                    xgb_probs=probs,
                    audio_seconds=float(feats.get("duration") or 0),
                )
                if w_status2 == "MACHINE" or str(w_cue2).startswith("custom_phrase"):
                    details["ml_note"] = (details.get("ml_note") or "") + "+vm_override_human_cue"
                    if isinstance(details.get("whisper"), dict):
                        details["whisper"]["cue"] = w_cue2 or "voicemail"
                    return "MACHINE", max(0.93, float(w_conf2 or 0.93)), details
            except Exception:
                pass
    else:
        # Whisper invented junk ("Here I go") or empty — don't send digits to agents
        w_cue = ""
        w_text = ""
        if isinstance(details.get("whisper"), dict):
            w_cue = str(details["whisper"].get("cue") or "")
            w_text = str(details["whisper"].get("transcript") or "")
        if looks_digit or (w_cue == "hallucination" and not looks_human):
            details["ml_note"] = (details.get("ml_note") or "") + "+digit_or_hallucination_machine"
            return "MACHINE", 0.9, details
        # Transcript present but classify returned no cue — re-check hard VM words
        if w_text:
            try:
                from app.ai.whisper_amd import classify_transcript

                w_status2, w_conf2, w_cue2 = classify_transcript(
                    w_text, xgb_probs=probs, audio_seconds=float(feats.get("duration") or 0)
                )
                if w_status2 == "MACHINE" or str(w_cue2).startswith("custom_phrase"):
                    details["ml_note"] = (details.get("ml_note") or "") + "+transcript_vm_block"
                    details["whisper"]["cue"] = w_cue2 or "voicemail"
                    return "MACHINE", max(0.93, float(w_conf2 or 0.93)), details
            except Exception:
                pass
        if whisper_on and not details.get("whisper_used"):
            details["ml_note"] = "human_whisper_unavailable"
        elif not whisper_on:
            details["ml_note"] = "human_whisper_disabled"

    # Still HUMAN below threshold after Whisper (or Whisper off) → same as classic gate
    if status == "HUMAN" and conf < high_thr:
        action = str(cfg.get("below_threshold_action", "MACHINE")).strip().upper()
        if action not in ("MACHINE", "IVR", "SIT", "ERROR"):
            action = "MACHINE"
        details["ml_gate_downgraded"] = True
        details["ml_gate_action"] = action
        details["ml_note"] = (details.get("ml_note") or "") + "+below_threshold_gate"
        _maybe_save_sample(
            save_low,
            audio=audio,
            sr=sr,
            feats=feats,
            silero=silero,
            status="HUMAN",
            confidence=conf,
            probs=probs,
            details=details,
            call_meta=call_meta,
        )
        return action, float(conf), details

    # Whisper raised confidence enough, or flipped to non-HUMAN
    if is_blank_as_machine_enabled() and _is_blank(feats, silero) and status == "HUMAN":
        b_status, b_conf, _ = _blank_disposition()
        details["ml_note"] = (details.get("ml_note") or "") + "+blank_safety"
        return b_status, b_conf, details

    _maybe_save_sample(
        save_low and conf < low_thr,
        audio=audio,
        sr=sr,
        feats=feats,
        silero=silero,
        status=status,
        confidence=conf,
        probs=probs,
        details=details,
        call_meta=call_meta,
    )
    return status, float(conf), details


def _maybe_save_sample(
    should: bool,
    *,
    audio: np.ndarray,
    sr: int,
    feats: Dict[str, Any],
    silero: Dict[str, Any],
    status: str,
    confidence: float,
    probs: Dict[str, float],
    details: Dict[str, Any],
    call_meta: Optional[Dict[str, Any]] = None,
) -> None:
    if not should:
        return
    try:
        from app.ml_data import save_low_confidence_sample

        sample_id = save_low_confidence_sample(
            audio=audio,
            sr=sr,
            feats=feats,
            silero=silero,
            predicted_status=status,
            confidence=confidence,
            class_probs=probs,
            extra=details,
            call_meta=call_meta,
        )
        details["ml_sample_id"] = sample_id
    except Exception as exc:
        details["ml_sample_error"] = str(exc)
