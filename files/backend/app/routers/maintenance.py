from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.audio_cleanup import delete_old_recordings
from app.auth.security import get_current_user
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.correction import TrainingCorrection
from app.models.user import User

router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])


class WipeRequest(BaseModel):
    confirm: str = Field(..., description="Type WIPE to confirm", max_length=32)
    older_than_days: Optional[int] = Field(
        default=None,
        ge=0,
        le=3650,
        description="If set, only delete records older than N days. Null = all.",
    )


class AudioCleanupRequest(BaseModel):
    confirm: str = Field(..., description="Type DELETE to confirm", max_length=32)
    older_than_days: Optional[int] = Field(
        default=None,
        ge=0,
        le=3650,
        description="If set, only delete audio older than N days. Null = all.",
    )


def _require_admin(user: User):
    if user.role not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


def _recordings_stats() -> dict:
    try:
        from app.config import get_settings
        from pathlib import Path

        settings = get_settings()
        root = Path(settings.RECORDINGS_DIR)
        total_files = 0
        total_bytes = 0
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    total_files += 1
                    try:
                        total_bytes += path.stat().st_size
                    except OSError:
                        pass
        return {
            "recordings_dir": str(root),
            "audio_files": total_files,
            "audio_bytes": total_bytes,
            "audio_mb": round(total_bytes / (1024 * 1024), 2),
            "audio_gb": round(total_bytes / (1024**3), 3),
        }
    except Exception:
        return {
            "recordings_dir": "",
            "audio_files": 0,
            "audio_bytes": 0,
            "audio_mb": 0,
            "audio_gb": 0,
        }


@router.get("/stats")
def maintenance_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    calls = db.query(CallAnalysis).count()
    corrections = db.query(TrainingCorrection).count()
    oldest = db.query(CallAnalysis.created_at).order_by(CallAnalysis.created_at.asc()).first()
    newest = db.query(CallAnalysis.created_at).order_by(CallAnalysis.created_at.desc()).first()
    audio = _recordings_stats()
    return {
        "database": "postgresql",
        "call_analyses": calls,
        "training_corrections": corrections,
        "oldest_call": oldest[0].isoformat() if oldest and oldest[0] else None,
        "newest_call": newest[0].isoformat() if newest and newest[0] else None,
        **audio,
    }


@router.post("/wipe-logs")
def wipe_logs(
    payload: WipeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    if payload.confirm.strip().upper() != "WIPE":
        raise HTTPException(status_code=400, detail="Type WIPE in confirm field")

    if payload.older_than_days is not None:
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=payload.older_than_days)
        old_ids = [
            row.id
            for row in db.query(CallAnalysis.id).filter(CallAnalysis.created_at < cutoff).all()
        ]
        deleted_corr = 0
        if old_ids:
            deleted_corr = (
                db.query(TrainingCorrection)
                .filter(TrainingCorrection.call_id.in_(old_ids))
                .delete(synchronize_session=False)
            )
        deleted_corr += (
            db.query(TrainingCorrection)
            .filter(TrainingCorrection.created_at < cutoff)
            .delete(synchronize_session=False)
        )
        deleted_calls = (
            db.query(CallAnalysis)
            .filter(CallAnalysis.created_at < cutoff)
            .delete(synchronize_session=False)
        )
    else:
        deleted_corr = db.query(TrainingCorrection).delete(synchronize_session=False)
        deleted_calls = db.query(CallAnalysis).delete(synchronize_session=False)

    db.commit()
    return {
        "ok": True,
        "deleted_call_analyses": deleted_calls,
        "deleted_training_corrections": deleted_corr,
        "older_than_days": payload.older_than_days,
        "wiped_by": user.username,
        "at": datetime.utcnow().isoformat() + "Z",
    }


@router.post("/delete-audio")
def delete_audio(
    payload: AudioCleanupRequest,
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    if payload.confirm.strip().upper() != "DELETE":
        raise HTTPException(status_code=400, detail="Type DELETE in confirm field")

    try:
        result = delete_old_recordings(payload.older_than_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["deleted_by"] = user.username
    return result
