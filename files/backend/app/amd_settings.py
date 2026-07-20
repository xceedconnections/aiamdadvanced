"""AMD decision settings (confidence gate + blank-call handling).

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
) -> dict[str, Any]:
    pct = int(min_human_confidence_percent)
    if pct < 0 or pct > 100:
        raise ValueError("min_human_confidence_percent must be between 0 and 100")
    action = str(below_threshold_action or "MACHINE").strip().upper()
    if action not in _ALLOWED_ACTIONS:
        raise ValueError("below_threshold_action must be one of MACHINE, IVR, SIT, ERROR")

    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": bool(enabled),
        "min_human_confidence_percent": pct,
        "below_threshold_action": action,
        "blank_as_machine": bool(blank_as_machine),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out = dict(payload)
    out["path"] = str(path)
    return out


def is_blank_as_machine_enabled() -> bool:
    """Whether blank/near-silent audio should be classified as MACHINE."""
    return bool(load_amd_settings().get("blank_as_machine", True))


def apply_confidence_gate(status: str, confidence: float) -> tuple[str, bool]:
    """Return (final_status, downgraded). Only HUMAN results are gated."""
    cfg = load_amd_settings()
    if not cfg.get("enabled", True):
        return status, False
    if status != "HUMAN":
        return status, False

    min_pct = int(cfg.get("min_human_confidence_percent", 70))
    conf_pct = float(confidence) * 100.0
    if conf_pct >= min_pct:
        return status, False

    return str(cfg.get("below_threshold_action", "MACHINE")), True
