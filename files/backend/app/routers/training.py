"""Training teach / history / backup APIs."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, undefer

from app.auth.security import get_current_user
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.correction import TrainingCorrection, TrainingOverride
from app.models.user import User
from app.schemas import CorrectionCreate
from app.training import (
    ALLOWED_STATUSES,
    deactivate_override,
    export_training_backup,
    import_training_backup,
    normalize_phone,
    normalize_status,
    upsert_override_from_call,
    wipe_all_training,
)

router = APIRouter(prefix="/api/training", tags=["training"])


class TeachByPhone(BaseModel):
    phone_number: str = Field(..., min_length=7, max_length=32)
    taught_status: str
    notes: str = ""


class BulkDeleteHistory(BaseModel):
    ids: list[int] = Field(default_factory=list)


class WipeTrainingRequest(BaseModel):
    confirm: str = Field(..., max_length=32)


def _require_admin(user: User):
    if user.role not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


@router.post("/correct")
def correct_call(
    payload: CorrectionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    call = (
        db.query(CallAnalysis)
        .options(undefer(CallAnalysis.audio_blob))
        .filter(CallAnalysis.id == payload.call_analysis_id)
        .first()
    )
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    corrected = normalize_status(payload.corrected_status)
    if corrected not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid corrected_status")

    try:
        correction, override = upsert_override_from_call(
            db,
            call=call,
            taught_status=corrected,
            username=user.username,
            notes=payload.notes or "",
            ai_status=call.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db.commit()
    ml_sid = getattr(correction, "ml_sample_id", None)
    msg = (
        f"Logged correction for this call → {corrected}. "
        "Next dial is still judged fresh from its recording (not forced by phone)."
    )
    if ml_sid:
        msg += (
            f" Also saved as ML sample `{ml_sid}` for XGBoost retrain "
            "(Settings → Retrain XGBoost when ready)."
        )
    else:
        msg += " (No ML features on this call — sample not added.)"
    return {
        "ok": True,
        "id": correction.id,
        "call_analysis_id": call.id,
        "phone_number": correction.phone_number,
        "ai_status": correction.ai_status,
        "corrected_status": corrected,
        "override_active": False,
        "ml_sample_id": ml_sid,
        "message": msg,
    }


@router.post("/teach-phone")
def teach_phone(
    payload: TeachByPhone,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Log a manual phone correction (audit only — does not force future AMD)."""
    status = normalize_status(payload.taught_status)
    if status not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail="Invalid taught_status")
    phone = normalize_phone(payload.phone_number)
    if len(phone) < 7:
        raise HTTPException(status_code=400, detail="phone_number too short")

    ov = (
        db.query(TrainingOverride)
        .filter(TrainingOverride.phone_number == phone)
        .first()
    )
    prev = ov.taught_status if ov else ""
    if ov:
        ov.taught_status = status
        ov.taught_by = user.username
        ov.notes = payload.notes or ov.notes or ""
        ov.is_active = False
        ov.updated_at = datetime.utcnow()

    corr = TrainingCorrection(
        call_id=None,
        phone_number=phone,
        ai_status=prev or "",
        corrected_status=status,
        previous_taught_status=prev,
        action="teach",
        corrected_by=user.username,
        notes=payload.notes or "Manual phone teach (audit only)",
        is_active=True,
    )
    db.add(corr)
    db.flush()
    if ov is not None:
        ov.last_correction_id = corr.id
    db.commit()
    return {
        "ok": True,
        "phone_number": phone,
        "taught_status": status,
        "correction_id": corr.id,
        "override_id": ov.id if ov else None,
        "message": "Logged for history only. Future AMD always judges from the live recording.",
    }


@router.get("/history")
def training_history(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    phone: Optional[str] = Query(None, max_length=32),
    active_only: bool = True,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(TrainingCorrection)
    if active_only:
        q = q.filter(TrainingCorrection.is_active == True)  # noqa: E712
    if phone:
        key = normalize_phone(phone)
        if key:
            q = q.filter(TrainingCorrection.phone_number == key)
    total = q.count()
    rows = q.order_by(TrainingCorrection.id.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [
            {
                "id": r.id,
                "call_id": r.call_id,
                "phone_number": r.phone_number or "",
                "ai_status": r.ai_status,
                "corrected_status": r.corrected_status,
                "previous_taught_status": r.previous_taught_status or "",
                "action": r.action or "teach",
                "corrected_by": r.corrected_by or "",
                "notes": r.notes or "",
                "is_active": bool(getattr(r, "is_active", True)),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get("/overrides")
def list_overrides(
    limit: int = Query(200, ge=1, le=1000),
    active_only: bool = True,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(TrainingOverride)
    if active_only:
        q = q.filter(TrainingOverride.is_active == True)  # noqa: E712
    rows = q.order_by(TrainingOverride.updated_at.desc()).limit(limit).all()
    return {
        "items": [
            {
                "id": o.id,
                "phone_number": o.phone_number,
                "taught_status": o.taught_status,
                "taught_by": o.taught_by or "",
                "notes": o.notes or "",
                "hit_count": int(o.hit_count or 0),
                "is_active": bool(o.is_active),
                "source_call_id": o.source_call_id,
                "updated_at": o.updated_at.isoformat() if o.updated_at else None,
                "created_at": o.created_at.isoformat() if o.created_at else None,
            }
            for o in rows
        ]
    }


@router.post("/overrides/{override_id}/revert")
def revert_override(
    override_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    ov = db.query(TrainingOverride).filter(TrainingOverride.id == override_id).first()
    if not ov:
        raise HTTPException(status_code=404, detail="Override not found")
    deactivate_override(
        db,
        phone_number=ov.phone_number,
        username=user.username,
        notes="Reverted from Training History",
    )
    db.commit()
    return {"ok": True, "phone_number": ov.phone_number, "reverted": True}


@router.delete("/history/{correction_id}")
def delete_history_item(
    correction_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Soft-remove a history row and deactivate matching override if this was its last teach."""
    _require_admin(user)
    row = db.query(TrainingCorrection).filter(TrainingCorrection.id == correction_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Correction not found")
    phone = row.phone_number or ""
    row.is_active = False
    row.action = "revert"
    row.notes = (row.notes or "") + " [deleted from history]"
    if phone:
        ov = (
            db.query(TrainingOverride)
            .filter(TrainingOverride.phone_number == phone)
            .first()
        )
        if ov and ov.last_correction_id == row.id:
            ov.is_active = False
            ov.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


@router.post("/history/delete")
def bulk_delete_history(
    payload: BulkDeleteHistory,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Soft-delete selected training history rows (checkbox delete)."""
    _require_admin(user)
    ids = [int(i) for i in (payload.ids or []) if int(i) > 0]
    if not ids:
        raise HTTPException(status_code=400, detail="No history ids selected")
    deleted = 0
    for cid in ids[:500]:
        row = db.query(TrainingCorrection).filter(TrainingCorrection.id == cid).first()
        if not row:
            continue
        phone = row.phone_number or ""
        row.is_active = False
        row.action = "revert"
        note = row.notes or ""
        if "[deleted from history]" not in note:
            row.notes = note + " [deleted from history]"
        if phone:
            ov = (
                db.query(TrainingOverride)
                .filter(TrainingOverride.phone_number == phone)
                .first()
            )
            if ov and ov.last_correction_id == row.id:
                ov.is_active = False
                ov.updated_at = datetime.utcnow()
        deleted += 1
    db.commit()
    return {"ok": True, "deleted": deleted}


@router.get("/backup")
def download_backup(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    payload = export_training_backup(db)
    data = json.dumps(payload, indent=2).encode("utf-8")
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return StreamingResponse(
        BytesIO(data),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="openamd_training_backup_{stamp}.json"'
        },
    )


@router.post("/backup/import")
async def upload_backup(
    file: UploadFile = File(...),
    replace: bool = Query(False),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON backup file") from exc
    try:
        result = import_training_backup(
            db, payload, username=user.username, replace=replace
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@router.post("/wipe")
def wipe_training(
    payload: WipeTrainingRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete all training audit history (AMD always judges from audio regardless)."""
    _require_admin(user)
    if payload.confirm.strip().upper() != "WIPE TRAINING":
        raise HTTPException(status_code=400, detail="Type WIPE TRAINING to confirm")
    result = wipe_all_training(db)
    return {
        "ok": True,
        **result,
        "wiped_by": user.username,
        "at": datetime.utcnow().isoformat() + "Z",
        "message": "Training history deleted. AMD continues to judge every call from its recording.",
    }


@router.get("/stats")
def training_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    active_ov = (
        db.query(TrainingOverride)
        .filter(TrainingOverride.is_active == True)  # noqa: E712
        .count()
    )
    total_ov = db.query(TrainingOverride).count()
    total_corr = db.query(TrainingCorrection).count()
    return {
        "active_overrides": active_ov,
        "total_overrides": total_ov,
        "total_corrections": total_corr,
    }
