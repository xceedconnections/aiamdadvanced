"""Portal maintenance — wipe detection logs + audio cleanup (AMD + SCAM)."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.audio_cleanup import delete_old_recordings, prune_empty_dirs
from app.auth.security import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.correction import TrainingCorrection, TrainingOverride
from app.models.scam import ScamCall
from app.models.user import User

router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])


class WipeRequest(BaseModel):
    confirm: str = Field(..., description="Type WIPE to confirm", max_length=64)
    older_than_days: Optional[int] = Field(
        default=None,
        ge=0,
        le=3650,
        description="If set, only delete records older than N days. Null = all.",
    )


class AudioCleanupRequest(BaseModel):
    confirm: str = Field(..., description="Type DELETE to confirm", max_length=64)
    older_than_days: Optional[int] = Field(
        default=None,
        ge=0,
        le=3650,
        description="If set, only delete audio older than N days. Null = all.",
    )


def _require_admin(user: User):
    role = str(getattr(user, "role", "") or "").strip().lower()
    if role not in ("superadmin", "admin"):
        raise HTTPException(
            status_code=403,
            detail=f"Admin role required (your role: {role or 'none'})",
        )


def _norm_confirm(value: str) -> str:
    return " ".join(str(value or "").strip().upper().split())


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


def _clear_amd_audio_sql(db: Session, older_than_days: Optional[int]) -> int:
    """Bulk-clear AMD blobs without loading them into Python memory."""
    if older_than_days is None:
        result = db.execute(
            text(
                "UPDATE call_analyses SET audio_blob = NULL, audio_path = '', "
                "audio_saved = false "
                "WHERE audio_blob IS NOT NULL OR COALESCE(audio_path, '') <> '' "
                "OR audio_saved = true"
            )
        )
    else:
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        result = db.execute(
            text(
                "UPDATE call_analyses SET audio_blob = NULL, audio_path = '', "
                "audio_saved = false "
                "WHERE created_at < :cutoff AND ("
                "audio_blob IS NOT NULL OR COALESCE(audio_path, '') <> '' "
                "OR audio_saved = true)"
            ),
            {"cutoff": cutoff},
        )
    return int(result.rowcount or 0)


def _clear_scam_audio_sql(db: Session, older_than_days: Optional[int]) -> int:
    """Bulk-clear SCAM blobs without loading them into Python memory."""
    if older_than_days is None:
        result = db.execute(
            text(
                "UPDATE scam_calls SET audio_blob = NULL, audio_path = '', "
                "audio_saved = false "
                "WHERE audio_blob IS NOT NULL OR COALESCE(audio_path, '') <> '' "
                "OR audio_saved = true"
            )
        )
    else:
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        result = db.execute(
            text(
                "UPDATE scam_calls SET audio_blob = NULL, audio_path = '', "
                "audio_saved = false "
                "WHERE created_at < :cutoff AND ("
                "audio_blob IS NOT NULL OR COALESCE(audio_path, '') <> '' "
                "OR audio_saved = true)"
            ),
            {"cutoff": cutoff},
        )
    return int(result.rowcount or 0)


def _wipe_database(db: Session, older_than_days: Optional[int]) -> dict:
    """
    FK-safe wipe using SQL only (never SELECT audio_blob into app memory).
    Order: clear FKs → delete children → delete parents.
    """
    if older_than_days is None:
        cleared_amd = _clear_amd_audio_sql(db, None)
        cleared_scam = _clear_scam_audio_sql(db, None)
        deleted_corr = db.execute(text("DELETE FROM training_corrections")).rowcount or 0
        deleted_ov = db.execute(text("DELETE FROM training_overrides")).rowcount or 0
        deleted_scam = db.execute(text("DELETE FROM scam_calls")).rowcount or 0
        deleted_calls = db.execute(text("DELETE FROM call_analyses")).rowcount or 0
    else:
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        cleared_amd = _clear_amd_audio_sql(db, older_than_days)
        cleared_scam = _clear_scam_audio_sql(db, older_than_days)

        # Break FKs pointing at calls that will be removed
        db.execute(
            text(
                "UPDATE training_overrides SET source_call_id = NULL "
                "WHERE source_call_id IN ("
                "SELECT id FROM call_analyses WHERE created_at < :cutoff)"
            ),
            {"cutoff": cutoff},
        )
        db.execute(
            text(
                "UPDATE training_corrections SET call_id = NULL "
                "WHERE call_id IN ("
                "SELECT id FROM call_analyses WHERE created_at < :cutoff)"
            ),
            {"cutoff": cutoff},
        )

        deleted_corr = (
            db.execute(
                text("DELETE FROM training_corrections WHERE created_at < :cutoff"),
                {"cutoff": cutoff},
            ).rowcount
            or 0
        )
        # Overrides linked only by age of source call already nulled; delete inactive/orphan optional
        deleted_ov = (
            db.execute(
                text(
                    "DELETE FROM training_overrides WHERE updated_at < :cutoff "
                    "OR created_at < :cutoff"
                ),
                {"cutoff": cutoff},
            ).rowcount
            or 0
        )
        deleted_scam = (
            db.execute(
                text("DELETE FROM scam_calls WHERE created_at < :cutoff"),
                {"cutoff": cutoff},
            ).rowcount
            or 0
        )
        deleted_calls = (
            db.execute(
                text("DELETE FROM call_analyses WHERE created_at < :cutoff"),
                {"cutoff": cutoff},
            ).rowcount
            or 0
        )

    return {
        "cleared_amd_audio_blobs": int(cleared_amd),
        "cleared_scam_audio_blobs": int(cleared_scam),
        "deleted_training_corrections": int(deleted_corr),
        "deleted_training_overrides": int(deleted_ov),
        "deleted_scam_calls": int(deleted_scam),
        "deleted_call_analyses": int(deleted_calls),
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
        "playable_recordings": int(blobs.get("calls_with_audio_blob") or 0)
        + int(blobs.get("scam_with_audio") or 0),
    }


@router.post("/wipe-logs")
def wipe_logs(
    payload: WipeRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete detection/CDR rows, training, SCAMMERS, and all audio (disk + DB blobs)."""
    _require_admin(user)
    if _norm_confirm(payload.confirm) != "WIPE":
        raise HTTPException(status_code=400, detail='Type "WIPE" in the confirm field')

    # Disk first (does not touch DB blobs)
    try:
        orphan = delete_old_recordings(payload.older_than_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        db_result = _wipe_database(db, payload.older_than_days)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Wipe failed (database): {exc}",
        ) from exc

    root = Path(get_settings().RECORDINGS_DIR)
    prune_empty_dirs(root)
    prune_empty_dirs(root / "scam")

    return {
        "ok": True,
        **db_result,
        "deleted_audio_files": int(orphan.get("deleted_files") or 0),
        "failed_audio_files": int(orphan.get("failed_files") or 0),
        "freed_mb": float(orphan.get("freed_mb") or 0),
        "older_than_days": payload.older_than_days,
        "wiped_by": user.username,
        "at": datetime.utcnow().isoformat() + "Z",
        "recordings_dir": str(root),
    }


@router.post("/delete-audio")
def delete_audio(
    payload: AudioCleanupRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete AMD + SCAM audio on disk AND clear playable DB blobs (CDR Play)."""
    _require_admin(user)
    if _norm_confirm(payload.confirm) != "DELETE":
        raise HTTPException(status_code=400, detail='Type "DELETE" in the confirm field')

    try:
        result = delete_old_recordings(payload.older_than_days)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        cleared_amd = _clear_amd_audio_sql(db, payload.older_than_days)
        cleared_scam = _clear_scam_audio_sql(db, payload.older_than_days)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Audio DB clear failed: {exc}",
        ) from exc

    root = Path(get_settings().RECORDINGS_DIR)
    prune_empty_dirs(root)
    prune_empty_dirs(root / "scam")

    result["deleted_by"] = user.username
    result["cleared_amd_audio_blobs"] = cleared_amd
    result["scam_audio_cleared"] = cleared_scam
    result["includes_scam_recordings"] = True
    result["includes_amd_db_blobs"] = True
    result["ok"] = True
    return result
