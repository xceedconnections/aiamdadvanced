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
    # Portal display only — does not change OS/server clock
    "display_timezone": "UTC",
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
                tz = str(raw.get("display_timezone") or "").strip()
                if tz:
                    data["display_timezone"] = tz[:64]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    data["path"] = str(path)
    return data


def save_display_timezone(timezone: str) -> dict[str, Any]:
    """Persist portal display timezone only (does not change OS clock)."""
    tz = str(timezone or "UTC").strip() or "UTC"
    if len(tz) > 64:
        raise ValueError("timezone too long")
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    prev: dict[str, Any] = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(prev, dict):
                prev = {}
        except (OSError, json.JSONDecodeError):
            prev = {}
    prev["display_timezone"] = tz
    path.write_text(json.dumps(prev, indent=2) + "\n", encoding="utf-8")
    return {"display_timezone": tz, "path": str(path)}


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
    """Whether blank/near-silent audio should be classified as BLANK (not HUMAN)."""
    return bool(load_amd_settings().get("blank_as_machine", True))


def map_status_for_vicidial(status: str) -> str:
    """Map internal disposition to what the VICIdial AGI dialplan expects.

    BLANK is stored in the portal but sent to VICIdial as MACHINE (AA / hangup).
    """
    s = (status or "").strip().upper()
    if s == "BLANK":
        return "MACHINE"
    return s


def is_ml_pipeline_enabled() -> bool:
    return bool(load_amd_settings().get("ml_pipeline_enabled", False))


def _pct_to_frac(pct: Any, default: float = 0.85) -> float:
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return default
    if v > 1.0:
        v = v / 100.0
    return max(0.5, min(0.99, v))


def server_amd_mode(server: Any | None) -> str:
    """Return global | classic | ml for a VICIdial server row."""
    if server is None:
        return "global"
    mode = str(getattr(server, "amd_mode", "") or "").strip().lower()
    if mode in ("global", "classic", "ml"):
        return mode
    # Legacy flags
    if bool(getattr(server, "ml_pipeline_override_enabled", False)) and bool(
        getattr(server, "ml_pipeline_enabled", False)
    ):
        return "ml"
    if bool(getattr(server, "confidence_gate_enabled", False)):
        return "classic"
    return "global"


def resolve_effective_amd_settings(server: Any | None = None) -> dict[str, Any]:
    """
    Merge global Settings with optional per-VICIdial-server mode.

    Server amd_mode:
      global  → use Settings menu (Classic or Advanced ML)
      classic → custom classic Minimum HUMAN % for this server
      ml      → custom Advanced ML for this server
    blank_as_machine is always from global Settings.
    """
    g = load_amd_settings()
    blank = bool(g.get("blank_as_machine", True))
    out: dict[str, Any] = {
        "enabled": bool(g.get("enabled", True)),
        "min_human_confidence_percent": int(g.get("min_human_confidence_percent", 70)),
        "below_threshold_action": str(g.get("below_threshold_action", "MACHINE")),
        "blank_as_machine": blank,
        "ml_pipeline_enabled": bool(g.get("ml_pipeline_enabled", False)),
        "ml_whisper_enabled": bool(g.get("ml_whisper_enabled", True)),
        "ml_xgb_high_confidence": float(g.get("ml_xgb_high_confidence", 0.85)),
        "ml_low_confidence_threshold": float(g.get("ml_low_confidence_threshold", 0.85)),
        "ml_save_low_confidence": bool(g.get("ml_save_low_confidence", True)),
        "source": "global",
        "ml_source": "global",
        "gate_source": "global",
        "amd_mode": "global",
    }

    if server is None:
        if out["ml_pipeline_enabled"]:
            out["gate_source"] = "ml"
            out["amd_mode"] = "ml"
        else:
            out["amd_mode"] = "classic"
        return out

    mode = server_amd_mode(server)
    out["amd_mode"] = mode

    if mode == "global":
        if out["ml_pipeline_enabled"]:
            out["gate_source"] = "ml"
            out["min_human_confidence_percent"] = int(
                round(float(out["ml_xgb_high_confidence"]) * 100)
            )
        return out

    if mode == "ml":
        out["ml_pipeline_enabled"] = True
        out["ml_whisper_enabled"] = bool(getattr(server, "ml_whisper_enabled", True))
        out["ml_save_low_confidence"] = bool(getattr(server, "ml_save_low_confidence", True))
        out["ml_xgb_high_confidence"] = _pct_to_frac(
            getattr(server, "ml_min_human_confidence_percent", 85), 0.85
        )
        out["ml_low_confidence_threshold"] = _pct_to_frac(
            getattr(server, "ml_save_threshold_percent", 85), 0.85
        )
        action = str(
            getattr(server, "below_threshold_action", None) or out["below_threshold_action"]
        ).strip().upper()
        if action in _ALLOWED_ACTIONS:
            out["below_threshold_action"] = action
        out["ml_source"] = "server"
        out["gate_source"] = "server_ml"
        out["source"] = "server_ml"
        out["min_human_confidence_percent"] = int(
            round(float(out["ml_xgb_high_confidence"]) * 100)
        )
        return out

    # classic custom
    out["ml_pipeline_enabled"] = False
    out["enabled"] = True
    out["min_human_confidence_percent"] = int(
        getattr(server, "min_human_confidence_percent", 70) or 70
    )
    action = str(
        getattr(server, "below_threshold_action", None) or "MACHINE"
    ).strip().upper()
    if action in _ALLOWED_ACTIONS:
        out["below_threshold_action"] = action
    out["gate_source"] = "server"
    out["source"] = "server_classic"
    out["ml_source"] = "off"
    return out


def apply_confidence_gate(
    status: str,
    confidence: float,
    *,
    server_override: dict[str, Any] | None = None,
    effective: dict[str, Any] | None = None,
) -> tuple[str, bool]:
    """
    Return (final_status, downgraded). Only HUMAN results are gated.

    Prefer `effective` from resolve_effective_amd_settings(server).
    Legacy `server_override` still supported for older callers.
    """
    if status != "HUMAN":
        return status, False

    if effective is not None:
        # ML already applied its HUMAN threshold inside the pipeline
        if effective.get("ml_pipeline_enabled"):
            return status, False
        if not effective.get("enabled", True):
            return status, False
        min_pct = int(effective.get("min_human_confidence_percent", 70))
        action = str(effective.get("below_threshold_action", "MACHINE")).strip().upper()
        if action not in _ALLOWED_ACTIONS:
            action = "MACHINE"
        conf_pct = float(confidence) * 100.0
        if conf_pct >= min_pct:
            return status, False
        return action, True

    use_server = bool(server_override and server_override.get("confidence_gate_enabled"))
    if use_server:
        min_pct = int(server_override.get("min_human_confidence_percent", 70))
        action = str(server_override.get("below_threshold_action") or "MACHINE").strip().upper()
        if action not in _ALLOWED_ACTIONS:
            action = "MACHINE"
    else:
        cfg = load_amd_settings()
        if cfg.get("ml_pipeline_enabled"):
            return status, False
        if not cfg.get("enabled", True):
            return status, False
        min_pct = int(cfg.get("min_human_confidence_percent", 70))
        action = str(cfg.get("below_threshold_action", "MACHINE"))

    conf_pct = float(confidence) * 100.0
    if conf_pct >= min_pct:
        return status, False
    return action, True


def gate_config_for_server(server: Any | None = None) -> dict[str, Any]:
    """Resolved gate / ML config actually used for a call (for logging / UI)."""
    eff = resolve_effective_amd_settings(server)
    return {
        "source": eff.get("source", "global"),
        "amd_mode": eff.get("amd_mode", "global"),
        "ml_source": eff.get("ml_source", "global"),
        "gate_source": eff.get("gate_source", "global"),
        "enabled": bool(eff.get("enabled", True)) or bool(eff.get("ml_pipeline_enabled")),
        "min_human_confidence_percent": int(eff.get("min_human_confidence_percent", 70)),
        "below_threshold_action": str(eff.get("below_threshold_action", "MACHINE")),
        "ml_pipeline_enabled": bool(eff.get("ml_pipeline_enabled", False)),
        "ml_xgb_high_confidence": float(eff.get("ml_xgb_high_confidence", 0.85)),
        "ml_whisper_enabled": bool(eff.get("ml_whisper_enabled", True)),
        "blank_as_machine": bool(eff.get("blank_as_machine", True)),
    }
