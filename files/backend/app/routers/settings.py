from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.amd_settings import load_amd_settings, save_amd_settings
from app.auth.security import get_current_user, get_current_user_bearer_or_query
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


class MlWipeBody(BaseModel):
    reset_model: bool = True
    confirm: str = Field("", max_length=32)


class DisplayTimezoneBody(BaseModel):
    display_timezone: str = Field("UTC", max_length=64)


def _require_admin(user: User):
    role = str(getattr(user, "role", "") or "").strip().lower()
    if role not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


@router.get("/amd")
def get_amd_settings(user: User = Depends(get_current_user)):
    _require_admin(user)
    return load_amd_settings()


@router.get("/display")
def get_display_settings(user: User = Depends(get_current_user)):
    """Portal display prefs (timezone). Does not change OS/server clock."""
    cfg = load_amd_settings()
    return {
        "display_timezone": cfg.get("display_timezone") or "UTC",
    }


@router.put("/display")
def put_display_settings(
    payload: DisplayTimezoneBody,
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    from app.amd_settings import save_display_timezone

    try:
        saved = save_display_timezone(payload.display_timezone)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    saved["updated_by"] = user.username
    return saved


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
    unlabeled_only: bool = False,
    labeled_only: bool = False,
    limit: int = 100,
    phone: str = "",
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    from app.ml_data import list_samples

    return {
        "samples": list_samples(
            unlabeled_only=unlabeled_only,
            labeled_only=labeled_only,
            limit=max(1, min(limit, 500)),
            phone=phone,
        )
    }


@router.get("/ml/samples/{sample_id}")
def get_ml_sample_detail(sample_id: str, user: User = Depends(get_current_user)):
    _require_admin(user)
    from app.ml_data import get_sample

    meta = get_sample(sample_id)
    if not meta:
        raise HTTPException(status_code=404, detail="sample not found")
    out = {k: v for k, v in meta.items() if k != "feature_vector"}
    out["feature_vector_len"] = len(meta.get("feature_vector") or [])
    return out


@router.get("/ml/samples/{sample_id}/audio")
def get_ml_sample_audio(
    sample_id: str,
    user: User = Depends(get_current_user_bearer_or_query),
):
    _require_admin(user)
    from app.ml_data import sample_audio_path

    path = sample_audio_path(sample_id)
    if not path:
        raise HTTPException(status_code=404, detail="sample audio not found")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename=f"{sample_id}.wav",
        headers={"Cache-Control": "no-store"},
    )


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


@router.delete("/ml/samples/{sample_id}")
def delete_ml_sample(sample_id: str, user: User = Depends(get_current_user)):
    _require_admin(user)
    from app.ml_data import delete_sample

    if not delete_sample(sample_id):
        raise HTTPException(status_code=404, detail="sample not found")
    return {"ok": True, "deleted": sample_id}


@router.post("/ml/samples/{sample_id}/transcribe")
def post_ml_transcribe(sample_id: str, user: User = Depends(get_current_user)):
    """Run Whisper WAV→text on one sample log (fills the Training Logs column)."""
    _require_admin(user)
    from app.database import SessionLocal
    from app.ml_data import enrich_sample_phones_from_db, transcribe_sample

    db = SessionLocal()
    try:
        enrich_sample_phones_from_db(db, limit=500)
        return transcribe_sample(sample_id, force=True, db=db)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcribe failed: {exc}") from exc
    finally:
        db.close()


@router.post("/ml/transcribe-missing")
def post_ml_transcribe_missing(user: User = Depends(get_current_user)):
    """Backfill Whisper text for all samples that have WAV but empty transcript."""
    _require_admin(user)
    from app.database import SessionLocal
    from app.ml_data import enrich_sample_phones_from_db, transcribe_missing_samples

    db = SessionLocal()
    try:
        phones = enrich_sample_phones_from_db(db, limit=500)
        result = transcribe_missing_samples(limit=100, force=False, db=db)
        result["phones_updated"] = phones.get("updated", 0)
        return result
    finally:
        db.close()


@router.post("/ml/retrain")
def post_ml_retrain(user: User = Depends(get_current_user)):
    _require_admin(user)
    from app.ml_data import retrain_model

    try:
        return retrain_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"retrain failed: {exc}") from exc


@router.post("/ml/wipe")
def post_ml_wipe(body: MlWipeBody, user: User = Depends(get_current_user)):
    """Delete all ML sample logs and optionally reset XGBoost to a fresh bootstrap model."""
    _require_admin(user)
    if str(body.confirm or "").strip().upper() != "WIPE":
        raise HTTPException(
            status_code=400,
            detail='Type confirm: "WIPE" to delete all ML training logs',
        )
    from app.ml_data import wipe_ml_training

    result = wipe_ml_training(reset_model=bool(body.reset_model))
    result["ok"] = True
    result["wiped_by"] = user.username
    return result


@router.post("/ml/reset-model")
def post_ml_reset_model(user: User = Depends(get_current_user)):
    """Reset XGBoost only (keep sample logs)."""
    _require_admin(user)
    from app.ml_data import reset_xgb_model

    return reset_xgb_model(bootstrap=True)
