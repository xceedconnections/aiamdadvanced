"""SCAMMER checker — full agent-leg recordings (separate from AMD 8399 analyze)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, undefer

from app.auth.security import get_current_user, get_current_user_bearer_or_query, get_server_from_api_key
from app.config import get_settings
from app.database import get_db
from app.models.scam import ScamCall
from app.models.server import VicidialServer
from app.models.user import User
from app.recordings import to_browser_wav
from app.scam_detect import scan_transcript, transcribe_and_scan
from app.scam_settings import load_scam_settings, save_scam_settings
from app.audio_cleanup import delete_paths, prune_empty_dirs

router = APIRouter(tags=["scam"])
settings = get_settings()


class ScamCallOut(BaseModel):
    id: int
    server_id: int | None = None
    server_name: str = ""
    call_id: str = ""
    campaign: str = ""
    agent_user: str = ""
    caller_id: str = ""
    called_number: str = ""
    status: str = "PENDING"
    confidence: float = 0.0
    audio_seconds: float = 0.0
    transcript: str = ""
    match_terms: str = ""
    has_recording: bool = False
    created_at: datetime
    reviewed_at: datetime | None = None

    class Config:
        from_attributes = True


class ScamStatusUpdate(BaseModel):
    status: str = Field(..., min_length=2, max_length=32)


class ScamBlacklistUpdate(BaseModel):
    blacklist_words: list[str] | str = Field(default_factory=list)
    min_seconds_for_scan: int | None = Field(None, ge=1, le=120)
    mark_status: str | None = Field(None, max_length=16)
    rescan: bool = False


def _delete_scam_row_files(row: ScamCall) -> dict:
    paths: list[str] = []
    if row.audio_path:
        paths.append(row.audio_path)
    return delete_paths(paths)


def _apply_scan_to_row(row: ScamCall, scanned: dict) -> None:
    status = str(scanned.get("status") or "PENDING").upper()
    if status not in ("SCAM", "SPAM", "CLEAN", "PENDING", "ERROR"):
        status = "PENDING"
    row.status = status
    row.confidence = float(scanned.get("confidence") or 0)
    if scanned.get("audio_seconds") is not None:
        row.audio_seconds = float(scanned.get("audio_seconds") or row.audio_seconds or 0)
    row.transcript = str(scanned.get("transcript") or row.transcript or "")
    row.match_terms = ",".join(scanned.get("match_terms") or [])
    row.details_json = json.dumps(
        {
            "note": scanned.get("note"),
            "whisper_ok": scanned.get("whisper_ok"),
            "whisper_error": scanned.get("whisper_error"),
            "match_terms": scanned.get("match_terms") or [],
        }
    )
    row.error_message = str(scanned.get("whisper_error") or "") if status == "ERROR" else ""
    row.reviewed_at = datetime.utcnow() if status in ("SCAM", "SPAM", "CLEAN") else None



def _scam_dir(server_id: int | None) -> Path:
    root = Path(settings.RECORDINGS_DIR) / "scam"
    if server_id:
        root = root / str(server_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_scam_wav(data: bytes, callid: str, server_id: int | None) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (callid or "call"))[:80]
    name = f"{safe}_{uuid.uuid4().hex[:8]}.wav"
    path = _scam_dir(server_id) / name
    path.write_bytes(data)
    return str(path)


def _to_out(row: ScamCall) -> ScamCallOut:
    return ScamCallOut(
        id=row.id,
        server_id=row.server_id,
        server_name=(row.server.name if row.server else "") or "",
        call_id=row.call_id or "",
        campaign=row.campaign or "",
        agent_user=row.agent_user or "",
        caller_id=row.caller_id or "",
        called_number=row.called_number or "",
        status=row.status or "PENDING",
        confidence=float(row.confidence or 0),
        audio_seconds=float(row.audio_seconds or 0),
        transcript=(row.transcript or "")[:2000],
        match_terms=row.match_terms or "",
        has_recording=bool(row.audio_saved or row.audio_path),
        created_at=row.created_at,
        reviewed_at=row.reviewed_at,
    )


# ---------------------------------------------------------------------------
# Dialer API (X-API-Key) — does NOT touch /api/v1/analyze AMD flow
# ---------------------------------------------------------------------------


@router.get("/api/v1/scam/config")
@router.post("/api/v1/scam/config")
def scam_config(server: VicidialServer = Depends(get_server_from_api_key)):
    """Dialer asks whether full-call SCAM recording is enabled for this server."""
    enabled = bool(getattr(server, "scam_protection_enabled", False))
    try:
        record_secs = int(getattr(server, "scam_record_seconds", 120) or 120)
    except (TypeError, ValueError):
        record_secs = 120
    record_secs = max(30, min(600, record_secs))
    cfg = load_scam_settings()
    return {
        "status": "ok",
        "scam_protection_enabled": enabled,
        "server_id": server.id,
        "server_name": server.name,
        "min_seconds_for_scan": int(cfg.get("min_seconds_for_scan") or 5),
        "record_max_seconds": record_secs,
        "scam_record_seconds": record_secs,
        "upload_url": "/api/v1/scam/recording",
    }


@router.post("/api/v1/scam/arm")
def scam_arm(
    callid: str = Form(...),
    campaign: str = Form(""),
    caller: str = Form(""),
    called: str = Form(""),
    agent: str = Form(""),
    db: Session = Depends(get_db),
    server: VicidialServer = Depends(get_server_from_api_key),
):
    """Dialer notifies portal when MixMonitor starts (HUMAN → agent) so SCAMMERS shows PENDING immediately."""
    if not bool(getattr(server, "scam_protection_enabled", False)):
        raise HTTPException(
            status_code=403,
            detail="SCAM protection is OFF for this VICIdial server in the portal",
        )
    cid = (callid or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="callid required")

    row = (
        db.query(ScamCall)
        .filter(ScamCall.server_id == server.id, ScamCall.call_id == cid)
        .order_by(ScamCall.id.desc())
        .first()
    )
    if row and (row.status or "").upper() == "PENDING" and not row.audio_saved:
        if campaign:
            row.campaign = campaign
        if agent:
            row.agent_user = (agent or "").strip()
        if caller:
            row.caller_id = (caller or "").strip()
        if called:
            row.called_number = (called or "").strip()
    else:
        row = ScamCall(
            server_id=server.id,
            call_id=cid,
            campaign=campaign or "",
            agent_user=(agent or "").strip(),
            caller_id=(caller or "").strip(),
            called_number=(called or "").strip(),
            status="PENDING",
            confidence=0.0,
            audio_seconds=0.0,
            audio_path="",
            audio_saved=False,
            transcript="",
            match_terms="",
            details_json=json.dumps({"note": "recording_armed"}),
            error_message="",
        )
        db.add(row)
    server.last_seen = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return {"status": "PENDING", "scam_id": row.id, "callid": cid}


@router.post("/api/v1/scam/finalize")
def scam_finalize(
    callid: str = Form(...),
    status: str = Form("ERROR"),
    note: str = Form(""),
    campaign: str = Form(""),
    caller: str = Form(""),
    called: str = Form(""),
    agent: str = Form(""),
    db: Session = Depends(get_db),
    server: VicidialServer = Depends(get_server_from_api_key),
):
    """Dialer marks a PENDING scam row finished when WAV upload cannot run."""
    if not bool(getattr(server, "scam_protection_enabled", False)):
        raise HTTPException(
            status_code=403,
            detail="SCAM protection is OFF for this VICIdial server in the portal",
        )
    cid = (callid or "").strip()
    if not cid:
        raise HTTPException(status_code=400, detail="callid required")
    st = (status or "ERROR").strip().upper()
    if st not in ("ERROR", "CLEAN", "PENDING", "SCAM", "SPAM"):
        st = "ERROR"

    row = (
        db.query(ScamCall)
        .filter(ScamCall.server_id == server.id, ScamCall.call_id == cid)
        .order_by(ScamCall.id.desc())
        .first()
    )
    details = {"note": note or "finalize"}
    if row and not row.audio_saved:
        row.status = st
        row.error_message = (note or "")[:500]
        row.details_json = json.dumps(details)
        if campaign:
            row.campaign = campaign
        if agent:
            row.agent_user = (agent or "").strip()
        if caller:
            row.caller_id = (caller or "").strip()
        if called:
            row.called_number = (called or "").strip()
        if st in ("SCAM", "SPAM", "CLEAN", "ERROR"):
            row.reviewed_at = datetime.utcnow()
    elif not row:
        row = ScamCall(
            server_id=server.id,
            call_id=cid,
            campaign=campaign or "",
            agent_user=(agent or "").strip(),
            caller_id=(caller or "").strip(),
            called_number=(called or "").strip(),
            status=st,
            confidence=0.0,
            audio_seconds=0.0,
            audio_path="",
            audio_saved=False,
            transcript="",
            match_terms="",
            details_json=json.dumps(details),
            error_message=(note or "")[:500],
            reviewed_at=datetime.utcnow() if st != "PENDING" else None,
        )
        db.add(row)
    server.last_seen = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return {"status": row.status, "scam_id": row.id, "callid": cid}


@router.post("/api/v1/scam/recording")
async def scam_recording_upload(
    callid: str = Form(...),
    campaign: str = Form(""),
    caller: str = Form(""),
    called: str = Form(""),
    agent: str = Form(""),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
    server: VicidialServer = Depends(get_server_from_api_key),
):
    """Receive full agent-leg WAV from scammerchecker dialer (HTTP POST like analyze)."""
    if not bool(getattr(server, "scam_protection_enabled", False)):
        raise HTTPException(
            status_code=403,
            detail="SCAM protection is OFF for this VICIdial server in the portal",
        )

    raw = await audio.read()
    max_bytes = settings.MAX_AUDIO_MB * 1024 * 1024 * 8  # allow longer full calls
    # Cap hard at 200MB
    hard_cap = 200 * 1024 * 1024
    if len(raw) > hard_cap or len(raw) > max_bytes * 4:
        raise HTTPException(status_code=413, detail="Scam recording too large")
    if len(raw) < 1000:
        raise HTTPException(status_code=400, detail="Scam recording too small / empty")

    playable = to_browser_wav(raw)
    path = ""
    try:
        path = _save_scam_wav(playable, callid, server.id)
    except OSError as exc:
        print(f"OpenAMD SCAM WARNING: save failed for {callid}: {exc}")

    scanned = transcribe_and_scan(playable)
    status = str(scanned.get("status") or "PENDING").upper()
    if status not in ("SCAM", "SPAM", "CLEAN", "PENDING", "ERROR"):
        status = "PENDING"

    details = {
        "note": scanned.get("note"),
        "whisper_ok": scanned.get("whisper_ok"),
        "whisper_error": scanned.get("whisper_error"),
        "match_terms": scanned.get("match_terms") or [],
    }

    cid = (callid or "").strip()
    row = (
        db.query(ScamCall)
        .filter(ScamCall.server_id == server.id, ScamCall.call_id == cid)
        .order_by(ScamCall.id.desc())
        .first()
    )
    if row and not row.audio_saved:
        row.campaign = campaign or row.campaign or ""
        row.agent_user = (agent or "").strip() or row.agent_user or ""
        row.caller_id = (caller or "").strip() or row.caller_id or ""
        row.called_number = (called or "").strip() or row.called_number or ""
        row.status = status
        row.confidence = float(scanned.get("confidence") or 0)
        row.audio_seconds = float(scanned.get("audio_seconds") or 0)
        row.audio_path = path
        row.audio_saved = bool(path)
        row.audio_blob = playable
        row.transcript = str(scanned.get("transcript") or "")
        row.match_terms = ",".join(scanned.get("match_terms") or [])
        row.details_json = json.dumps(details)
        row.error_message = str(scanned.get("whisper_error") or "") if status == "ERROR" else ""
        row.reviewed_at = datetime.utcnow() if status in ("SCAM", "SPAM", "CLEAN") else None
    else:
        row = ScamCall(
            server_id=server.id,
            call_id=cid,
            campaign=campaign or "",
            agent_user=(agent or "").strip(),
            caller_id=(caller or "").strip(),
            called_number=(called or "").strip(),
            status=status,
            confidence=float(scanned.get("confidence") or 0),
            audio_seconds=float(scanned.get("audio_seconds") or 0),
            audio_path=path,
            audio_saved=bool(path),
            audio_blob=playable,
            transcript=str(scanned.get("transcript") or ""),
            match_terms=",".join(scanned.get("match_terms") or []),
            details_json=json.dumps(details),
            error_message=str(scanned.get("whisper_error") or "") if status == "ERROR" else "",
            reviewed_at=datetime.utcnow() if status in ("SCAM", "SPAM", "CLEAN") else None,
        )
        db.add(row)
    server.last_seen = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return {
        "status": row.status,
        "scam_id": row.id,
        "callid": callid,
        "confidence": row.confidence,
        "match_terms": row.match_terms,
        "audio_seconds": row.audio_seconds,
    }


# ---------------------------------------------------------------------------
# Portal API (JWT)
# ---------------------------------------------------------------------------


def _require_staff(user: User):
    if user.role not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


@router.get("/api/scam/blacklist")
def get_scam_blacklist(user: User = Depends(get_current_user)):
    _require_staff(user)
    return load_scam_settings()


@router.put("/api/scam/blacklist")
def put_scam_blacklist(
    payload: ScamBlacklistUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_staff(user)
    saved = save_scam_settings(
        blacklist_words=payload.blacklist_words,
        min_seconds_for_scan=payload.min_seconds_for_scan,
        mark_status=payload.mark_status,
    )
    rescanned = 0
    if payload.rescan:
        rows = (
            db.query(ScamCall)
            .filter(ScamCall.transcript.isnot(None), ScamCall.transcript != "")
            .order_by(ScamCall.id.desc())
            .limit(2000)
            .all()
        )
        for row in rows:
            scanned = scan_transcript(row.transcript or "")
            _apply_scan_to_row(row, scanned)
            rescanned += 1
        db.commit()
    return {**saved, "rescanned": rescanned}


@router.get("/api/scammers", response_model=list[ScamCallOut])
def list_scammers(
    status: str = "ALL",
    server_id: int | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_staff(user)
    limit = max(1, min(500, int(limit)))
    q = db.query(ScamCall).order_by(ScamCall.id.desc())
    st = (status or "").strip().upper()
    if st and st != "ALL":
        q = q.filter(ScamCall.status == st)
    if server_id is not None:
        q = q.filter(ScamCall.server_id == int(server_id))
    rows = q.limit(limit).all()
    return [_to_out(r) for r in rows]


@router.patch("/api/scammers/{scam_id}", response_model=ScamCallOut)
def patch_scammer(
    scam_id: int,
    payload: ScamStatusUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_staff(user)
    row = db.query(ScamCall).filter(ScamCall.id == scam_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    st = payload.status.strip().upper()
    if st not in ("SCAM", "SPAM", "CLEAN", "PENDING", "ERROR"):
        raise HTTPException(status_code=400, detail="Invalid status")
    row.status = st
    row.reviewed_at = datetime.utcnow()
    db.commit()
    db.refresh(row)
    return _to_out(row)


@router.delete("/api/scammers/{scam_id}")
def delete_scammer(
    scam_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_staff(user)
    row = (
        db.query(ScamCall)
        .options(undefer(ScamCall.audio_blob))
        .filter(ScamCall.id == scam_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    files = _delete_scam_row_files(row)
    db.delete(row)
    db.commit()
    root = Path(settings.RECORDINGS_DIR) / "scam"
    prune_empty_dirs(root)
    return {"ok": True, "deleted_id": scam_id, **files}


@router.delete("/api/scammers")
def delete_all_scammers(
    server_id: int | None = None,
    confirm: str = "",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Delete all SCAMMERS rows (+ WAV files). confirm=DELETE required."""
    _require_staff(user)
    if (confirm or "").strip().upper() != "DELETE":
        raise HTTPException(status_code=400, detail="Pass confirm=DELETE")
    q = db.query(ScamCall)
    if server_id is not None:
        q = q.filter(ScamCall.server_id == int(server_id))
    rows = q.all()
    deleted_files = 0
    freed = 0
    for row in rows:
        r = _delete_scam_row_files(row)
        deleted_files += int(r.get("deleted_files") or 0)
        freed += int(r.get("freed_bytes") or 0)
        db.delete(row)
    db.commit()
    root = Path(settings.RECORDINGS_DIR) / "scam"
    prune_empty_dirs(root)
    return {
        "ok": True,
        "deleted_rows": len(rows),
        "deleted_files": deleted_files,
        "freed_mb": round(freed / (1024 * 1024), 2),
        "server_id": server_id,
    }


@router.get("/api/scammers/{scam_id}/play")
def play_scammer(
    scam_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user_bearer_or_query),
):
    _require_staff(user)
    row = (
        db.query(ScamCall)
        .options(undefer(ScamCall.audio_blob))
        .filter(ScamCall.id == scam_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    data = bytes(row.audio_blob) if row.audio_blob else b""
    if not data and row.audio_path:
        p = Path(row.audio_path)
        if p.is_file():
            data = p.read_bytes()
    if not data:
        raise HTTPException(status_code=404, detail="Recording missing")
    return Response(content=data, media_type="audio/wav")
