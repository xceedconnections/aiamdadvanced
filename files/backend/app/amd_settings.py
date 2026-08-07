"""AMD decision settings (confidence gate + blank-call + optional ML pipeline).

Stored as JSON next to recordings so it survives restarts and is shared by
every VICIdial server that talks to this AI AMD server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings

DEFAULTS = {
    "enabled": True,
    "min_human_confidence_percent": 70,
    "below_threshold_action": "MACHINE",
    # When True, blank / near-silent audio is disposed as MACHINE (not to agents)
    "blank_as_machine": True,
    # Optional ML pipeline (XGBoost + Whisper on low conf). OFF = legacy hybrid only.
    "ml_pipeline_enabled": False,
    "ml_whisper_enabled": True,
    "ml_xgb_high_confidence": 0.85,
    "ml_low_confidence_threshold": 0.85,
    "ml_save_low_confidence": True,
}

_ALLOWED_ACTIONS = {"MACHINE", "IVR", "SIT", "ERROR"}


def settings_path() -> Path:
    settings = get_settings()
    root = Path(settings.RECORDINGS_DIR).resolve().parent
    return root / "amd_settings.json"


def load_amd_settings() -> dict[str, Any]:
    path = settings_path()
    data = dict(DEFAULTS)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if "enabled" in raw:
                    data["enabled"] = bool(raw["enabled"])
                if "min_human_confidence_percent" in raw:
                    pct = int(raw["min_human_confidence_percent"])
                    data["min_human_confidence_percent"] = max(0, min(100, pct))
                action = str(raw.get("below_threshold_action", "")).strip().upper()
                if action in _ALLOWED_ACTIONS:
                    data["below_threshold_action"] = action
                if "blank_as_machine" in raw:
                    data["blank_as_machine"] = bool(raw["blank_as_machine"])
                if "ml_pipeline_enabled" in raw:
                    data["ml_pipeline_enabled"] = bool(raw["ml_pipeline_enabled"])
                if "ml_whisper_enabled" in raw:
                    data["ml_whisper_enabled"] = bool(raw["ml_whisper_enabled"])
                if "ml_save_low_confidence" in raw:
                    data["ml_save_low_confidence"] = bool(raw["ml_save_low_confidence"])
                for key in ("ml_xgb_high_confidence", "ml_low_confidence_threshold"):
                    if key in raw:
                        try:
                            data[key] = max(0.5, min(0.99, float(raw[key])))
                        except (TypeError, ValueError):
                            pass
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    data["path"] = str(path)
    return data


def save_amd_settings(
    *,
    enabled: bool,
    min_human_confidence_percent: int,
    below_threshold_action: str = "MACHINE",
    blank_as_machine: bool = True,
    ml_pipeline_enabled: bool = False,
    ml_whisper_enabled: bool = True,
    ml_xgb_high_confidence: float = 0.85,
    ml_low_confidence_threshold: float = 0.85,
    ml_save_low_confidence: bool = True,
) -> dict[str, Any]:
    pct = int(min_human_confidence_percent)
    if pct < 0 or pct > 100:
        raise ValueError("min_human_confidence_percent must be between 0 and 100")
    action = str(below_threshold_action or "MACHINE").strip().upper()
    if action not in _ALLOWED_ACTIONS:
        raise ValueError("below_threshold_action must be one of MACHINE, IVR, SIT, ERROR")

    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve unknown keys from prior file
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(prev, dict):
                prev = {}
        except (OSError, json.JSONDecodeError):
            prev = {}

    payload = {
        **prev,
        "enabled": bool(enabled),
        "min_human_confidence_percent": pct,
        "below_threshold_action": action,
        "blank_as_machine": bool(blank_as_machine),
        "ml_pipeline_enabled": bool(ml_pipeline_enabled),
        "ml_whisper_enabled": bool(ml_whisper_enabled),
        "ml_xgb_high_confidence": max(0.5, min(0.99, float(ml_xgb_high_confidence))),
        "ml_low_confidence_threshold": max(0.5, min(0.99, float(ml_low_confidence_threshold))),
        "ml_save_low_confidence": bool(ml_save_low_confidence),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out = dict(payload)
    out["path"] = str(path)
    return out


def is_blank_as_machine_enabled() -> bool:
    """Whether blank/near-silent audio should be classified as MACHINE."""
    return bool(load_amd_settings().get("blank_as_machine", True))


def is_ml_pipeline_enabled() -> bool:
    return bool(load_amd_settings().get("ml_pipeline_enabled", False))


def apply_confidence_gate(
    status: str,
    confidence: float,
    *,
    server_override: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """
    Return (final_status, downgraded). Only HUMAN results are gated.

    If server_override is provided and confidence_gate_enabled is True,
    use that server's min % / action. Otherwise use global amd_settings.json.
    """
    if status != "HUMAN":
        return status, False

    use_server = bool(server_override and server_override.get("confidence_gate_enabled"))
    if use_server:
        enabled = True
        min_pct = int(server_override.get("min_human_confidence_percent", 70))
        action = str(server_override.get("below_threshold_action") or "MACHINE").strip().upper()
        if action not in _ALLOWED_ACTIONS:
            action = "MACHINE"
        source = "server"
    else:
        cfg = load_amd_settings()
        if not cfg.get("enabled", True):
            return status, False
        enabled = True
        min_pct = int(cfg.get("min_human_confidence_percent", 70))
        action = str(cfg.get("below_threshold_action", "MACHINE"))
        source = "global"

    if not enabled:
        return status, False

    conf_pct = float(confidence) * 100.0
    if conf_pct >= min_pct:
        return status, False

    _ = source
    return action, True


def gate_config_for_server(server: Any | None = None) -> dict[str, Any]:
    """Resolved gate config actually used for a call (for logging / UI)."""
    global_cfg = load_amd_settings()
    if server is not None and bool(getattr(server, "confidence_gate_enabled", False)):
        action = str(getattr(server, "below_threshold_action", None) or "MACHINE").strip().upper()
        if action not in _ALLOWED_ACTIONS:
            action = "MACHINE"
        return {
            "source": "server",
            "enabled": True,
            "min_human_confidence_percent": int(
                getattr(server, "min_human_confidence_percent", 70) or 70
            ),
            "below_threshold_action": action,
        }
    return {
        "source": "global",
        "enabled": bool(global_cfg.get("enabled", True)),
        "min_human_confidence_percent": int(global_cfg.get("min_human_confidence_percent", 70)),
        "below_threshold_action": str(global_cfg.get("below_threshold_action", "MACHINE")),
    }
