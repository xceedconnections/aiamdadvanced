from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session, undefer

from app.auth.security import get_current_user, get_current_user_bearer_or_query
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.user import User
from app.recordings import (
    ensure_recordings_dir,
    get_call_audio_bytes,
    list_recording_index,
    recording_meta,
    repair_call_recording,
    resolve_recording_path,
)

router = APIRouter(prefix="/api/recordings", tags=["recordings"])


def _get_call(analysis_id: int, db: Session, *, load_blob: bool = False) -> CallAnalysis:
    q = db.query(CallAnalysis)
    if load_blob:
        q = q.options(undefer(CallAnalysis.audio_blob))
    call = q.filter(CallAnalysis.id == analysis_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


def _audio_response(call: CallAnalysis, db: Session, *, as_attachment: bool) -> Response:
    data = get_call_audio_bytes(call)
    if not data:
        # Try disk path heal once
        path = resolve_recording_path(call)
        if path and path.is_file():
            data = path.read_bytes()
            call.audio_path = str(path)
            try:
                db.commit()
            except Exception:
                db.rollback()
    if not data:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Recording not found for id={call.id} call_id={call.call_id}. "
                "Old calls may have no saved audio — place a new test call after upgrading."
            ),
        )

    safe_name = f"{call.call_id}.wav".replace("/", "_").replace("\\", "_")
    headers = {
        "Cache-Control": "private, max-age=60",
        "Accept-Ranges": "bytes",
    }
    if as_attachment:
        headers["Content-Disposition"] = f'attachment; filename="{safe_name}"'
    else:
        headers["Content-Disposition"] = f'inline; filename="{safe_name}"'

    return Response(content=data, media_type="audio/wav", headers=headers)


@router.get("/status")
def recordings_status(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    root = ensure_recordings_dir()
    files = list_recording_index()
    sample = [str(p) for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:8]]
    missing_path = (
        db.query(CallAnalysis)
        .filter((CallAnalysis.audio_path == None) | (CallAnalysis.audio_path == ""))  # noqa: E711
        .count()
    )
    with_blob = db.query(CallAnalysis).filter(CallAnalysis.audio_saved == True).count()  # noqa: E712
    return {
        "recordings_dir": root,
        "wav_files": len(files),
        "calls_with_audio_blob": with_blob,
        "calls_missing_audio_path": missing_path,
        "sample_files": sample,
    }


@router.post("/repair")
def repair_recordings(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    limit: int = 500,
):
    ensure_recordings_dir()
    index = list_recording_index()
    rows = db.query(CallAnalysis).order_by(CallAnalysis.id.desc()).limit(limit).all()
    fixed = 0
    for row in rows:
        if repair_call_recording(row, index=index):
            fixed += 1
    db.commit()
    return {
        "checked": len(rows),
        "linked": fixed,
        "wav_files_on_disk": len(index),
        "recordings_dir": ensure_recordings_dir(),
    }


@router.get("/{analysis_id}/meta")
def recording_info(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    call = _get_call(analysis_id, db)
    meta = recording_meta(call)
    return {
        "analysis_id": call.id,
        "call_id": call.call_id,
        "audio_saved": bool(getattr(call, "audio_saved", False)),
        **meta,
    }


@router.get("/{analysis_id}/play")
def play_recording(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user_bearer_or_query),
):
    call = _get_call(analysis_id, db, load_blob=True)
    return _audio_response(call, db, as_attachment=False)


@router.get("/{analysis_id}/download")
def download_recording(
    analysis_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user_bearer_or_query),
):
    call = _get_call(analysis_id, db, load_blob=True)
    return _audio_response(call, db, as_attachment=True)
