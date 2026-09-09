from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.audio_cleanup import delete_old_recordings, delete_recordings_for_calls, delete_paths, prune_empty_dirs
from app.auth.security import get_current_user
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.correction import TrainingCorrection, TrainingOverride
from app.models.scam import ScamCall
from app.models.user import User
from app.config import get_settings
from pathlib import Path

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
    overrides = (
        db.query(TrainingOverride)
        .filter(TrainingOverride.is_active == True)  # noqa: E712
        .count()
    )
    oldest = db.query(CallAnalysis.created_at).order_by(CallAnalysis.created_at.asc()).first()
    newest = db.query(CallAnalysis.created_at).order_by(CallAnalysis.created_at.desc()).first()
    audio = _recordings_stats()
    return {
        "database": "postgresql",
        "call_analyses": calls,
        "training_corrections": corrections,
        "training_overrides": overrides,
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

    # Load rows first so we can delete matching WAV files (avoid orphaned recordings)
    if payload.older_than_days is not None:
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=payload.older_than_days)
        rows = (
            db.query(CallAnalysis)
            .filter(CallAnalysis.created_at < cutoff)
            .all()
        )
    else:
        rows = db.query(CallAnalysis).all()

    audio_result = delete_recordings_for_calls(rows)
    # Also remove orphaned/dated files matching the same age window (or everything)
    # Includes AMD WAVs and SCAMMERS 2‑min recordings under recordings/scam/
    orphan = delete_old_recordings(payload.older_than_days)

    # Wipe matching SCAMMERS DB rows (+ their files already covered by orphan sweep)
    scam_q = db.query(ScamCall)
    if payload.older_than_days is not None:
        from datetime import timedelta

        scam_cutoff = datetime.utcnow() - timedelta(days=payload.older_than_days)
        scam_rows = scam_q.filter(ScamCall.created_at < scam_cutoff).all()
    else:
        scam_rows = scam_q.all()
    scam_file_bytes = 0
    for srow in scam_rows:
        if srow.audio_path:
            r = delete_paths([srow.audio_path])
            scam_file_bytes += int(r.get("freed_bytes") or 0)
            orphan["deleted_files"] = int(orphan.get("deleted_files") or 0) + int(r.get("deleted_files") or 0)
        db.delete(srow)
    deleted_scam = len(scam_rows)
    prune_empty_dirs(Path(get_settings().RECORDINGS_DIR) / "scam")

    audio_result = {
        "deleted_files": audio_result.get("deleted_files", 0) + orphan.get("deleted_files", 0),
        "failed_files": audio_result.get("failed_files", 0) + orphan.get("failed_files", 0),
        "freed_bytes": audio_result.get("freed_bytes", 0) + orphan.get("freed_bytes", 0) + scam_file_bytes,
        "freed_mb": round(
            (audio_result.get("freed_bytes", 0) + orphan.get("freed_bytes", 0) + scam_file_bytes) / (1024 * 1024),
            2,
        ),
        "freed_gb": round(
            (audio_result.get("freed_bytes", 0) + orphan.get("freed_bytes", 0) + scam_file_bytes) / (1024**3),
            3,
        ),
        "recordings_dir": orphan.get("recordings_dir") or audio_result.get("recordings_dir"),
    }

    old_ids = [row.id for row in rows]
    deleted_overrides = 0

    if payload.older_than_days is not None:
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=payload.older_than_days)
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
        # Full wipe of call logs also clears all taught AMD knowledge
        deleted_corr = db.query(TrainingCorrection).delete(synchronize_session=False)
        deleted_overrides = db.query(TrainingOverride).delete(synchronize_session=False)
        deleted_calls = db.query(CallAnalysis).delete(synchronize_session=False)

    db.commit()
    return {
        "ok": True,
        "deleted_call_analyses": deleted_calls,
        "deleted_training_corrections": deleted_corr,
        "deleted_training_overrides": deleted_overrides,
        "deleted_scam_calls": deleted_scam,
        "deleted_audio_files": audio_result.get("deleted_files", 0),
        "failed_audio_files": audio_result.get("failed_files", 0),
        "freed_mb": audio_result.get("freed_mb", 0),
        "older_than_days": payload.older_than_days,
        "wiped_by": user.username,
        "at": datetime.utcnow().isoformat() + "Z",
    }


@router.post("/delete-audio")
def delete_audio(
    payload: AudioCleanupRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    if payload.confirm.strip().upper() != "DELETE":
        raise HTTPException(status_code=400, detail="Type DELETE in confirm field")

    try:
        # Deletes AMD WAVs and SCAMMERS recordings under recordings/ (incl. scam/)
        result = delete_old_recordings(payload.older_than_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Clear SCAMMERS DB audio blobs/paths for matching age (keep row metadata optional:
    # null blobs so Play disappears; keep CLEAN/SPAM history unless full wipe)
    scam_q = db.query(ScamCall)
    if payload.older_than_days is not None:
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=payload.older_than_days)
        scam_rows = scam_q.filter(ScamCall.created_at < cutoff).all()
    else:
        scam_rows = scam_q.all()
    cleared = 0
    for srow in scam_rows:
        srow.audio_blob = None
        srow.audio_path = ""
        srow.audio_saved = False
        cleared += 1
    db.commit()
    prune_empty_dirs(Path(get_settings().RECORDINGS_DIR) / "scam")

    result["deleted_by"] = user.username
    result["scam_audio_cleared"] = cleared
    result["includes_scam_recordings"] = True
    if result.get("failed_files"):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Deleted {result.get('deleted_files', 0)} file(s) but "
                f"{result['failed_files']} remain (permission/path error). "
                f"Dir: {result.get('recordings_dir')}"
            ),
        )
    return result