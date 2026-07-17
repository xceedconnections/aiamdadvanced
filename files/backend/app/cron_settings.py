"""Recording retention cron settings (JSON file on disk)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import get_settings

DEFAULTS = {
    "enabled": True,
    "retention_days": 7,
}


def settings_path() -> Path:
    settings = get_settings()
    root = Path(settings.RECORDINGS_DIR).resolve().parent
    return root / "cron_settings.json"


def load_cron_settings() -> dict[str, Any]:
    path = settings_path()
    data = dict(DEFAULTS)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                if "enabled" in raw:
                    data["enabled"] = bool(raw["enabled"])
                if "retention_days" in raw:
                    days = int(raw["retention_days"])
                    if days < 1:
                        days = 1
                    if days > 3650:
                        days = 3650
                    data["retention_days"] = days
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    data["path"] = str(path)
    return data


def save_cron_settings(*, enabled: bool, retention_days: int) -> dict[str, Any]:
    days = int(retention_days)
    if days < 1:
        raise ValueError("retention_days must be >= 1")
    if days > 3650:
        raise ValueError("retention_days must be <= 3650")
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": bool(enabled), "retention_days": days}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    out = dict(payload)
    out["path"] = str(path)
    return out
