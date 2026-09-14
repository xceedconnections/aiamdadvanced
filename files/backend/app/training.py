"""Training corrections: audit history + backup/import (never force future AMD)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.models.call import CallAnalysis
from app.models.correction import TrainingCorrection, TrainingOverride

ALLOWED_STATUSES = {"HUMAN", "MACHINE", "IVR", "FAX", "SIT", "BLANK", "ERROR"}


def normalize_phone(raw: str | None) -> str:
    """Digits-only phone key used for training overrides."""
    if not raw:
        return ""
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    if len(digits) > 15:
        digits = digits[-15:]
    # Prefer last 10 for NANP-style matching when longer
    if len(digits) > 10 and digits.startswith("1") and len(digits) == 11:
        return digits[-10:]
    if len(digits) > 10:
        return digits[-10:] if len(digits) >= 10 else digits
    return digits


def normalize_status(raw: str | None) -> str:
    s = (raw or "").strip().upper()
    if s == "CANCELLED":
        s = "SIT"
    return s


def lookup_override(db: Session, *phones: str) -> Optional[TrainingOverride]:
    """Find active override matching any of the candidate phone fields."""
    seen: set[str] = set()
    for raw in phones:
        key = normalize_phone(raw)
        if not key or len(key) < 7 or key in seen:
            continue
        seen.add(key)
        row = (
            db.query(TrainingOverride)
            .filter(
                TrainingOverride.phone_number == key,
                TrainingOverride.is_active == True,  # noqa: E712
            )
            .first()
        )
        if row:
            return row
        # Also try last-10 if key is longer international form
        if len(key) > 10:
            short = key[-10:]
            if short not in seen:
                seen.add(short)
                row = (
                    db.query(TrainingOverride)
                    .filter(
                        TrainingOverride.phone_number == short,
                        TrainingOverride.is_active == True,  # noqa: E712
                    )
                    .first()
                )
                if row:
                    return row
    return None


def apply_training_override(
    db: Session,
    *,
    engine_status: str,
    engine_confidence: float,
    called: str = "",
    ani: str = "",
    caller: str = "",
) -> tuple[str, float, dict[str, Any]]:
    """
    No-op: AMD always judges from the current recording.
    Kept for API compatibility; never forces a phone status.
    """
    return engine_status, engine_confidence, {"training_override": False}


def upsert_override_from_call(
    db: Session,
    *,
    call: CallAnalysis,
    taught_status: str,
    username: str,
    notes: str = "",
    ai_status: str = "",
) -> tuple[TrainingCorrection, Optional[TrainingOverride]]:
    """
    Correct this call's disposition and write an audit row.
    Does NOT force future AMD for this phone — every call is judged fresh.
    """
    status = normalize_status(taught_status)
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid taught status: {taught_status}")

    phone = normalize_phone(call.called_number or call.ani or "")
    if not phone:
        phone = normalize_phone(call.caller_id or "")

    # Preserve original AI label on the call row
    original_ai = (getattr(call, "raw_status", None) or "").strip() or (ai_status or call.status)
    if not (getattr(call, "raw_status", None) or "").strip():
        call.raw_status = original_ai
    call.status = status

    # Keep any legacy phone override inactive (never force future analyzes)
    prev = ""
    override = None
    if phone:
        override = (
            db.query(TrainingOverride)
            .filter(TrainingOverride.phone_number == phone)
            .first()
        )
        if override:
            prev = override.taught_status or ""
            override.taught_status = status
            override.source_call_id = call.id
            override.taught_by = username
            override.notes = notes or override.notes or ""
            override.is_active = False
            override.updated_at = datetime.utcnow()

    correction = TrainingCorrection(
        call_id=call.id,
        phone_number=phone,
        ai_status=original_ai,
        corrected_status=status,
        previous_taught_status=prev,
        action="teach",
        corrected_by=username,
        notes=notes or "",
        is_active=True,
    )
    db.add(correction)
    db.flush()
    if override is not None:
        override.last_correction_id = correction.id

    # Also feed Advanced ML retrain library (best-effort; never breaks teach)
    ml_sample_id = None
    try:
        from app.ml_data import ingest_call_correction_as_ml_sample

        ml_sample_id = ingest_call_correction_as_ml_sample(
            call=call,
            taught_status=status,
            username=username,
            phone_number=phone,
        )
        if ml_sample_id and notes is not None:
            # stash on correction notes for operators (optional short tag)
            pass
    except Exception as exc:
        print(f"OpenAMD WARNING: ML sample ingest from training failed: {exc}")

    # Attach for API callers (not a DB column)
    setattr(correction, "ml_sample_id", ml_sample_id)
    return correction, override


def deactivate_override(
    db: Session,
    *,
    phone_number: str,
    username: str,
    notes: str = "",
) -> Optional[TrainingOverride]:
    key = normalize_phone(phone_number)
    if not key:
        return None
    ov = (
        db.query(TrainingOverride)
        .filter(TrainingOverride.phone_number == key)
        .first()
    )
    if not ov:
        return None
    prev = ov.taught_status
    ov.is_active = False
    ov.updated_at = datetime.utcnow()
    corr = TrainingCorrection(
        call_id=ov.source_call_id,
        phone_number=key,
        ai_status=prev,
        corrected_status=prev,
        previous_taught_status=prev,
        action="revert",
        corrected_by=username,
        notes=notes or "Override deactivated",
        is_active=False,
    )
    db.add(corr)
    return ov


def wipe_all_training(db: Session) -> dict[str, int]:
    """Remove all taught knowledge — server behaves like fresh install for training."""
    from sqlalchemy import text

    # Raw SQL avoids ORM session issues / FK ordering surprises
    db.execute(
        text("UPDATE training_overrides SET last_correction_id = NULL")
    )
    n_corr = db.execute(text("DELETE FROM training_corrections")).rowcount or 0
    n_ov = db.execute(text("DELETE FROM training_overrides")).rowcount or 0
    db.commit()
    return {"deleted_corrections": int(n_corr), "deleted_overrides": int(n_ov)}


def export_training_backup(db: Session) -> dict[str, Any]:
    overrides = db.query(TrainingOverride).order_by(TrainingOverride.phone_number).all()
    corrections = (
        db.query(TrainingCorrection)
        .order_by(TrainingCorrection.id.asc())
        .all()
    )
    return {
        "format": "openamd-training-v1",
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "overrides": [
            {
                "phone_number": o.phone_number,
                "taught_status": o.taught_status,
                "taught_by": o.taught_by or "",
                "notes": o.notes or "",
                "is_active": bool(o.is_active),
                "hit_count": int(o.hit_count or 0),
                "source_call_id": o.source_call_id,
                "created_at": o.created_at.isoformat() if o.created_at else None,
                "updated_at": o.updated_at.isoformat() if o.updated_at else None,
            }
            for o in overrides
        ],
        "corrections": [
            {
                "call_id": c.call_id,
                "phone_number": c.phone_number or "",
                "ai_status": c.ai_status,
                "corrected_status": c.corrected_status,
                "previous_taught_status": c.previous_taught_status or "",
                "action": c.action or "teach",
                "corrected_by": c.corrected_by or "",
                "notes": c.notes or "",
                "is_active": bool(getattr(c, "is_active", True)),
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in corrections
        ],
    }


def import_training_backup(
    db: Session,
    payload: dict[str, Any],
    *,
    username: str,
    replace: bool = False,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid backup JSON")
    fmt = str(payload.get("format") or "")
    if fmt and fmt != "openamd-training-v1":
        raise ValueError(f"Unsupported backup format: {fmt}")

    if replace:
        wipe_all_training(db)

    overrides_in = payload.get("overrides") or []
    corrections_in = payload.get("corrections") or []
    if not isinstance(overrides_in, list):
        raise ValueError("overrides must be a list")

    imported_ov = 0
    updated_ov = 0
    for item in overrides_in:
        if not isinstance(item, dict):
            continue
        phone = normalize_phone(item.get("phone_number"))
        status = normalize_status(item.get("taught_status"))
        if not phone or status not in ALLOWED_STATUSES:
            continue
        row = (
            db.query(TrainingOverride)
            .filter(TrainingOverride.phone_number == phone)
            .first()
        )
        if row:
            row.taught_status = status
            row.taught_by = str(item.get("taught_by") or username)
            row.notes = str(item.get("notes") or row.notes or "")
            row.is_active = False  # never force future AMD
            row.updated_at = datetime.utcnow()
            updated_ov += 1
        else:
            db.add(
                TrainingOverride(
                    phone_number=phone,
                    taught_status=status,
                    taught_by=str(item.get("taught_by") or username),
                    notes=str(item.get("notes") or ""),
                    is_active=False,  # audit/backup only
                    hit_count=int(item.get("hit_count") or 0),
                )
            )
            imported_ov += 1

    imported_corr = 0
    for item in corrections_in:
        if not isinstance(item, dict):
            continue
        status = normalize_status(item.get("corrected_status"))
        if status not in ALLOWED_STATUSES:
            continue
        phone = normalize_phone(item.get("phone_number"))
        db.add(
            TrainingCorrection(
                call_id=item.get("call_id"),
                phone_number=phone,
                ai_status=str(item.get("ai_status") or ""),
                corrected_status=status,
                previous_taught_status=str(item.get("previous_taught_status") or ""),
                action=str(item.get("action") or "import"),
                corrected_by=str(item.get("corrected_by") or username),
                notes=str(item.get("notes") or "Imported from backup"),
                is_active=bool(item.get("is_active", True)),
            )
        )
        imported_corr += 1

    # Audit that an import happened
    db.add(
        TrainingCorrection(
            call_id=None,
            phone_number="",
            ai_status="",
            corrected_status="IMPORT",
            action="import",
            corrected_by=username,
            notes=f"Imported overrides+{imported_ov}/{updated_ov} corr+{imported_corr} replace={replace}",
            is_active=True,
        )
    )
    db.commit()
    return {
        "ok": True,
        "imported_overrides": imported_ov,
        "updated_overrides": updated_ov,
        "imported_corrections": imported_corr,
        "replaced": replace,
    }
