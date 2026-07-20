from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.amd_settings import load_amd_settings, save_amd_settings
from app.auth.security import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/settings", tags=["settings"])


class AmdSettingsUpdate(BaseModel):
    enabled: bool = True
    min_human_confidence_percent: int = Field(70, ge=0, le=100)
    below_threshold_action: str = Field("MACHINE", max_length=16)
    blank_as_machine: bool = True


def _require_admin(user: User):
    if user.role not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


@router.get("/amd")
def get_amd_settings(user: User = Depends(get_current_user)):
    _require_admin(user)
    return load_amd_settings()


@router.put("/amd")
def put_amd_settings(
    payload: AmdSettingsUpdate,
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    try:
        saved = save_amd_settings(
            enabled=payload.enabled,
            min_human_confidence_percent=payload.min_human_confidence_percent,
            below_threshold_action=payload.below_threshold_action,
            blank_as_machine=payload.blank_as_machine,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    saved["updated_by"] = user.username
    return saved
