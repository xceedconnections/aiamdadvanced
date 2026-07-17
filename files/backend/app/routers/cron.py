from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.audio_cleanup import delete_old_recordings
from app.auth.security import get_current_user
from app.cron_settings import load_cron_settings, save_cron_settings
from app.models.user import User

router = APIRouter(prefix="/api/cron", tags=["cron"])


class CronRetentionUpdate(BaseModel):
    enabled: bool = True
    retention_days: int = Field(7, ge=1, le=3650)


def _require_admin(user: User):
    if user.role not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


@router.get("/recording-retention")
def get_recording_retention(user: User = Depends(get_current_user)):
    _require_admin(user)
    return load_cron_settings()


@router.put("/recording-retention")
def put_recording_retention(
    payload: CronRetentionUpdate,
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    try:
        saved = save_cron_settings(
            enabled=payload.enabled,
            retention_days=payload.retention_days,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    saved["updated_by"] = user.username
    return saved


@router.post("/recording-retention/run")
def run_recording_retention_now(user: User = Depends(get_current_user)):
    """Manually run the same cleanup the daily cron uses."""
    _require_admin(user)
    cfg = load_cron_settings()
    if not cfg.get("enabled", True):
        return {
            "ok": True,
            "skipped": True,
            "message": "Retention cron is disabled — enable it first",
            **cfg,
        }
    days = int(cfg.get("retention_days", 7))
    result = delete_old_recordings(days)
    result["ran_by"] = user.username
    result["retention_days"] = days
    result["enabled"] = cfg.get("enabled", True)
    return result
