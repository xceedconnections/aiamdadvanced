"""Optional ML AMD pipeline (Silero features → XGBoost → Whisper).

Whisper runs when:
  - XGBoost would send HUMAN to an agent below the Settings threshold, or
  - XGBoost says MACHINE/IVR on a short live pickup ("hello") so we do not
    hang up a person. Beep / long greetings still skip Whisper.

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

    from app.ai.engine import _blank_disposition, _is_blank, _looks_like_short_human
    from app.amd_settings import is_blank_as_machine_enabled

    if is_blank_as_machine_enabled() and _is_blank(feats, silero):
        b_status, b_conf, _ = _blank_disposition()
        details: Dict[str, Any] = {
            "ml_pipeline": True,
            "hybrid_status": hybrid_status,
            "hybrid_confidence": round(float(hybrid_confidence), 4),
            "whisper_policy": "human_low_confidence_only",
            "ml_note": "blank_audio",
        }
        return b_status, b_conf, details

    details: Dict[str, Any] = {
        "ml_pipeline": True,
        "hybrid_status": hybrid_status,
        "hybrid_confidence": round(float(hybrid_confidence), 4),
        "whisper_policy": "human_low_confidence_only",
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
    no_beep = float(feats.get("beep", 0.0) or 0.0) < 0.5

    # Keep clear SIT from hybrid when XGB is unsure HUMAN
    if status == "HUMAN" and hybrid_status == "SIT" and hybrid_confidence >= 0.8:
        status, conf = "SIT", float(hybrid_confidence)
        details["ml_note"] = "keep_hybrid_sit"
        return status, float(conf), details

    # XGB MACHINE/IVR on a short live pickup — do not hang up without Whisper
    if (
        status in ("MACHINE", "IVR")
        and no_beep
        and (looks_human or hybrid_status == "HUMAN")
    ):
        details["ml_note"] = "xgb_machine_possible_human"
        if whisper_on:
            try:
                from app.ai.whisper_amd import refine_with_whisper, whisper_available

                if whisper_available():
                    w_status, w_conf, w_det = refine_with_whisper(
                        audio, sr, xgb_probs=probs, max_seconds=4.0
                    )
                    details["whisper"] = w_det
                    details["whisper_used"] = True
                    if w_status == "HUMAN":
                        details["ml_note"] = "whisper_rescue_human"
                        return w_status, float(w_conf), details
                    if w_status in ("MACHINE", "IVR", "SIT"):
                        details["ml_note"] = "whisper_confirm_non_human"
                        return w_status, float(w_conf), details
                    # Transcript empty / no cue — prefer live pickup
                    if looks_human or hybrid_status == "HUMAN":
                        details["ml_note"] = "whisper_no_cue_keep_human"
                        return "HUMAN", max(float(hybrid_confidence), 0.84), details
            except Exception as exc:
                details["whisper_error"] = str(exc)
                details["ml_note"] = "whisper_rescue_error"
        if looks_human or hybrid_status == "HUMAN":
            details["ml_note"] = (details.get("ml_note") or "") + "+keep_short_human"
            return "HUMAN", max(float(hybrid_confidence), 0.84), details

    # Answering machine / IVR / SIT / BLANK with strong evidence → accept
    if status != "HUMAN":
        details["ml_note"] = "non_human_accept"
        return status, float(conf), details

    # HUMAN + high confidence → pass to agent (no Whisper) unless blank
    if conf >= high_thr:
        if is_blank_as_machine_enabled() and _is_blank(feats, silero):
            b_status, b_conf, _ = _blank_disposition()
            details["ml_note"] = "blank_override_high_human"
            return b_status, b_conf, details
        details["ml_note"] = "human_high_confidence"
        return status, float(conf), details

    # HUMAN + below threshold → optional Faster-Whisper Tiny (protect agents)
    if whisper_on:
        try:
            from app.ai.whisper_amd import refine_with_whisper, whisper_available

            if whisper_available():
                w_status, w_conf, w_det = refine_with_whisper(
                    audio, sr, xgb_probs=probs, max_seconds=4.0
                )
                details["whisper"] = w_det
                details["whisper_used"] = True
                if w_status:
                    status, conf = w_status, float(w_conf)
                    details["ml_note"] = "whisper_refine_uncertain_human"
                else:
                    details["ml_note"] = "human_low_conf_whisper_no_cue"
            else:
                details["ml_note"] = "human_low_conf_whisper_unavailable"
                details["whisper"] = {
                    "whisper_ok": False,
                    "error": "faster-whisper not installed",
                }
        except Exception as exc:
            details["ml_note"] = "human_low_conf_whisper_error"
            details["whisper_error"] = str(exc)
    else:
        details["ml_note"] = "human_low_conf_whisper_disabled"

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
