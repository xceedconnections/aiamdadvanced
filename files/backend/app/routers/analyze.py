import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.ai.engine import analyze_audio
from app.amd_settings import apply_confidence_gate, gate_config_for_server
from app.auth.security import get_server_from_api_key
from app.config import get_settings
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.server import VicidialServer
from app.recordings import link_analysis_recording, save_call_audio, to_browser_wav
from app.schemas import AnalyzeResponse

router = APIRouter(prefix="/api/v1", tags=["analyze"])
settings = get_settings()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    callid: str = Form(...),
    campaign: str = Form(""),
    caller: str = Form(""),
    called: str = Form(""),
    ani: str = Form(""),
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
    server: VicidialServer = Depends(get_server_from_api_key),
):
    raw = await audio.read()
    max_bytes = settings.MAX_AUDIO_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="Audio file too large")
    if len(raw) < 100:
        raise HTTPException(status_code=400, detail="Audio file too small / empty")

    result = analyze_audio(
        raw,
        locale_pack_enabled=bool(getattr(server, "locale_pack_enabled", False)),
        locale_pack=str(getattr(server, "locale_pack", None) or "usa"),
    )

    # Per-server gate if enabled on this VICIdial server; else global Settings
    engine_status = result.status
    gate_cfg = gate_config_for_server(server)
    gated_status, downgraded = apply_confidence_gate(
        result.status,
        result.confidence,
        server_override={
            "confidence_gate_enabled": bool(getattr(server, "confidence_gate_enabled", False)),
            "min_human_confidence_percent": int(
                getattr(server, "min_human_confidence_percent", 70) or 70
            ),
            "below_threshold_action": str(
                getattr(server, "below_threshold_action", None) or "MACHINE"
            ),
        },
    )

    called_number = (called or "").strip()
    caller_id = (caller or "").strip()
    ani_value = (ani or called_number or "").strip()

    # Always judge from this recording (engine + confidence gate).
    # Training corrections are audit/history only — they never force future AMD.
    final_status = gated_status
    final_confidence = result.confidence
    raw_status = engine_status

    # Browser-safe PCM16 WAV for portal play + disk archive
    playable = to_browser_wav(raw)

    path = ""
    try:
        path = save_call_audio(playable, callid, server.id)
    except OSError as exc:
        print(f"OpenAMD WARNING: failed to save recording for {callid}: {exc}")
        path = ""

    details = {
        **result.details,
        "gate_downgraded": downgraded,
        "raw_status": raw_status,
        "confidence_gate": gate_cfg,
        "training_override": False,
    }

    row = CallAnalysis(
        server_id=server.id,
        call_id=callid,
        campaign=campaign or "",
        caller_id=caller_id,
        called_number=called_number,
        ani=ani_value,
        status=final_status,
        raw_status=raw_status,
        confidence=final_confidence,
        processing_ms=result.processing_ms,
        audio_seconds=result.audio_seconds,
        audio_path=path,
        audio_saved=True,
        audio_blob=playable,
        features_json=json.dumps(details),
        error_message=result.details.get("error", "") if result.status == "ERROR" else "",
    )
    db.add(row)
    server.last_seen = datetime.utcnow()
    db.commit()
    db.refresh(row)

    if path:
        linked = link_analysis_recording(row.id, path)
        if linked and linked != path:
            row.audio_path = linked
            db.commit()

    return AnalyzeResponse(
        status=final_status,
        confidence=final_confidence,
        processing_ms=result.processing_ms,
        callid=callid,
        analysis_id=row.id,
        details=details,
    )


@router.get("/health")
def health_public():
    return {"status": "ok", "service": "openamd-analyze"}
