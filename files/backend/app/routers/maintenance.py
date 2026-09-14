"""Portal maintenance — wipe detection logs + audio cleanup (AMD + SCAM)."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.audio_cleanup import (
    delete_old_recordings,
    delete_paths,
    delete_recordings_for_calls,
    prune_empty_dirs,
)
from app.auth.security import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.correction import TrainingCorrection, TrainingOverride
from app.models.scam import ScamCall
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
    role = str(getattr(user, "role", "") or "").strip().lower()
    if role not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


def _disk_recordings_stats() -> dict:
    try:
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


def _db_audio_stats(db: Session) -> dict:
    """CDR playable audio is often stored as Postgres blobs, not only disk WAVs."""
    amd_saved = 0
    scam_saved = 0
    amd_blob_bytes = 0
    scam_blob_bytes = 0
    try:
        row = db.execute(
            text(
                "SELECT COUNT(*) FILTER (WHERE audio_blob IS NOT NULL), "
                "COALESCE(SUM(octet_length(audio_blob)), 0) "
                "FROM call_analyses"
            )
        ).one()
        amd_saved = int(row[0] or 0)
        amd_blob_bytes = int(row[1] or 0)
    except Exception:
        try:
            amd_saved = (
                db.query(CallAnalysis)
                .filter(CallAnalysis.audio_saved == True)  # noqa: E712
                .count()
            )
        except Exception:
            amd_saved = 0
    try:
        row = db.execute(
            text(
                "SELECT COUNT(*) FILTER (WHERE audio_blob IS NOT NULL), "
                "COALESCE(SUM(octet_length(audio_blob)), 0) "
                "FROM scam_calls"
            )
        ).one()
        scam_saved = int(row[0] or 0)
        scam_blob_bytes = int(row[1] or 0)
    except Exception:
        try:
            scam_saved = (
                db.query(ScamCall)
                .filter(ScamCall.audio_saved == True)  # noqa: E712
                .count()
            )
        except Exception:
            scam_saved = 0

    blob_bytes = amd_blob_bytes + scam_blob_bytes
    return {
        "calls_with_audio_blob": amd_saved,
        "scam_with_audio": scam_saved,
        "db_audio_bytes": blob_bytes,
        "db_audio_mb": round(blob_bytes / (1024 * 1024), 2),
        "db_audio_gb": round(blob_bytes / (1024**3), 3),
        "amd_blob_bytes": amd_blob_bytes,
        "scam_blob_bytes": scam_blob_bytes,
    }


def _clear_call_audio_rows(db: Session, older_than_days: Optional[int]) -> int:
    q = db.query(CallAnalysis)
    if older_than_days is not None:
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        q = q.filter(CallAnalysis.created_at < cutoff)
    rows = q.all()
    n = 0
    for row in rows:
        changed = False
        if row.audio_blob is not None:
            row.audio_blob = None
            changed = True
        if row.audio_path:
            row.audio_path = ""
            changed = True
        if row.audio_saved:
            row.audio_saved = False
            changed = True
        if changed:
            n += 1
    return n


def _clear_scam_audio_rows(db: Session, older_than_days: Optional[int]) -> int:
    q = db.query(ScamCall)
    if older_than_days is not None:
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        q = q.filter(ScamCall.created_at < cutoff)
    rows = q.all()
    n = 0
    for row in rows:
        if row.audio_path:
            delete_paths([row.audio_path])
        row.audio_blob = None
        row.audio_path = ""
        row.audio_saved = False
        n += 1
    return n


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
    try:
        scam_calls = db.query(ScamCall).count()
    except Exception:
        scam_calls = 0
    oldest = db.query(CallAnalysis.created_at).order_by(CallAnalysis.created_at.asc()).first()
    newest = db.query(CallAnalysis.created_at).order_by(CallAnalysis.created_at.desc()).first()
    disk = _disk_recordings_stats()
    blobs = _db_audio_stats(db)
    return {
        "database": "postgresql",
        "call_analyses": calls,
        "training_corrections": corrections,
        "training_overrides": overrides,
        "scam_calls": scam_calls,
        "oldest_call": oldest[0].isoformat() if oldest and oldest[0] else None,
        "newest_call": newest[0].isoformat() if newest and newest[0] else None,
        **disk,
        **blobs,
        # Convenience total for UI
        "playable_recordings": int(blobs.get("calls_with_audio_blob") or 0)
        + int(blobs.get("scam_with_audio") or 0),
    }


@router.post("/wipe-logs")
def wipe_logs(
    payload: WipeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete detection/CDR rows, training linked to them, SCAMMERS rows, and all audio (disk + DB blobs)."""
    _require_admin(user)
    if payload.confirm.strip().upper() != "WIPE":
        raise HTTPException(status_code=400, detail="Type WIPE in confirm field")

    if payload.older_than_days is not None:
        cutoff = datetime.utcnow() - timedelta(days=payload.older_than_days)
        rows = db.query(CallAnalysis).filter(CallAnalysis.created_at < cutoff).all()
        scam_rows = db.query(ScamCall).filter(ScamCall.created_at < cutoff).all()
    else:
        rows = db.query(CallAnalysis).all()
        scam_rows = db.query(ScamCall).all()

    old_ids = [row.id for row in rows]

    # Disk WAVs for those calls + orphan sweep (includes recordings/scam/)
    audio_result = delete_recordings_for_calls(rows)
    orphan = delete_old_recordings(payload.older_than_days)

    scam_file_bytes = 0
    for srow in scam_rows:
        if srow.audio_path:
            r = delete_paths([srow.audio_path])
            scam_file_bytes += int(r.get("freed_bytes") or 0)
            orphan["deleted_files"] = int(orphan.get("deleted_files") or 0) + int(
                r.get("deleted_files") or 0
            )
        db.delete(srow)
    deleted_scam = len(scam_rows)

    # Clear AMD DB blobs before deleting rows (CDR Play uses these)
    cleared_amd = _clear_call_audio_rows(db, payload.older_than_days)

    deleted_overrides = 0
    deleted_corr = 0
    deleted_calls = 0

    try:
        if payload.older_than_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=payload.older_than_days)
            # Break FK training_overrides.source_call_id → call_analyses.id
            if old_ids:
                deleted_overrides += (
                    db.query(TrainingOverride)
                    .filter(TrainingOverride.source_call_id.in_(old_ids))
                    .delete(synchronize_session=False)
                )
                # Also null any leftovers that weren't deleted
                db.query(TrainingOverride).filter(
                    TrainingOverride.source_call_id.in_(old_ids)
                ).update({TrainingOverride.source_call_id: None}, synchronize_session=False)

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
            deleted_overrides = db.query(TrainingOverride).delete(synchronize_session=False)
            deleted_calls = db.query(CallAnalysis).delete(synchronize_session=False)

        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Wipe failed (database constraint or lock): {exc}",
        ) from exc

    prune_empty_dirs(Path(get_settings().RECORDINGS_DIR))
    prune_empty_dirs(Path(get_settings().RECORDINGS_DIR) / "scam")

    freed = (
        int(audio_result.get("freed_bytes") or 0)
        + int(orphan.get("freed_bytes") or 0)
        + scam_file_bytes
    )
    return {
        "ok": True,
        "deleted_call_analyses": int(deleted_calls or 0),
        "deleted_training_corrections": int(deleted_corr or 0),
        "deleted_training_overrides": int(deleted_overrides or 0),
        "deleted_scam_calls": deleted_scam,
        "cleared_amd_audio_blobs": cleared_amd,
        "deleted_audio_files": int(audio_result.get("deleted_files") or 0)
        + int(orphan.get("deleted_files") or 0),
        "failed_audio_files": int(audio_result.get("failed_files") or 0)
        + int(orphan.get("failed_files") or 0),
        "freed_mb": round(freed / (1024 * 1024), 2),
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
    """Delete AMD + SCAM audio on disk AND clear playable DB blobs (CDR Play)."""
    _require_admin(user)
    if payload.confirm.strip().upper() != "DELETE":
        raise HTTPException(status_code=400, detail="Type DELETE in confirm field")

    try:
        result = delete_old_recordings(payload.older_than_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        cleared_amd = _clear_call_audio_rows(db, payload.older_than_days)
        cleared_scam = _clear_scam_audio_rows(db, payload.older_than_days)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Audio DB clear failed: {exc}",
        ) from exc

    prune_empty_dirs(Path(get_settings().RECORDINGS_DIR))
    prune_empty_dirs(Path(get_settings().RECORDINGS_DIR) / "scam")

    result["deleted_by"] = user.username
    result["cleared_amd_audio_blobs"] = cleared_amd
    result["scam_audio_cleared"] = cleared_scam
    result["includes_scam_recordings"] = True
    result["includes_amd_db_blobs"] = True
    if result.get("failed_files"):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Deleted {result.get('deleted_files', 0)} file(s) and cleared "
                f"{cleared_amd} AMD / {cleared_scam} SCAM DB recordings, but "
                f"{result['failed_files']} disk file(s) remain. "
                f"Dir: {result.get('recordings_dir')}"
            ),
        )
    return result
