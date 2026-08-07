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
    ml_pipeline_enabled: bool = False
    ml_whisper_enabled: bool = True
    ml_xgb_high_confidence: float = Field(0.85, ge=0.5, le=0.99)
    ml_low_confidence_threshold: float = Field(0.85, ge=0.5, le=0.99)
    ml_save_low_confidence: bool = True


class MlLabelBody(BaseModel):
    label: str = Field(..., max_length=16)


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
            ml_pipeline_enabled=payload.ml_pipeline_enabled,
            ml_whisper_enabled=payload.ml_whisper_enabled,
            ml_xgb_high_confidence=payload.ml_xgb_high_confidence,
            ml_low_confidence_threshold=payload.ml_low_confidence_threshold,
            ml_save_low_confidence=payload.ml_save_low_confidence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Warm bootstrap model only when enabling (lazy, once)
    if payload.ml_pipeline_enabled:
        try:
            from app.ai.xgb_model import ensure_model

            ensure_model()
        except Exception as exc:
            saved["ml_warmup_warning"] = str(exc)

    saved["updated_by"] = user.username
    return saved


@router.get("/ml/stats")
def get_ml_stats(user: User = Depends(get_current_user)):
    _require_admin(user)
    from app.ml_data import ml_stats

    return ml_stats()


@router.get("/ml/samples")
def get_ml_samples(
    unlabeled_only: bool = True,
    limit: int = 50,
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    from app.ml_data import list_samples

    return {"samples": list_samples(unlabeled_only=unlabeled_only, limit=max(1, min(limit, 200)))}


@router.post("/ml/samples/{sample_id}/label")
def post_ml_label(
    sample_id: str,
    body: MlLabelBody,
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    from app.ml_data import label_sample

    try:
        return label_sample(sample_id, body.label, username=user.username)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ml/retrain")
def post_ml_retrain(user: User = Depends(get_current_user)):
    _require_admin(user)
    from app.ml_data import retrain_model

    try:
        return retrain_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrain failed: {exc}") from exc
