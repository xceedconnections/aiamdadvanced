import json
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.ai.engine import analyze_audio
from app.amd_settings import (
    apply_confidence_gate,
    gate_config_for_server,
    map_status_for_vicidial,
    resolve_effective_amd_settings,
)
from app.auth.security import get_server_from_api_key
from app.config import get_settings
from app.cps_limit import try_admit
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.server import VicidialServer
from app.recordings import link_analysis_recording, save_call_audio, to_browser_wav
from app.schemas import AnalyzeResponse

router = APIRouter(prefix="/api/v1", tags=["analyze"])
settings = get_settings()


def _enforce_cps(server: VicidialServer, *, already_admitted: bool = False) -> None:
    """Reject over-limit calls; 8399 AGI trusts its stock-human result on 429."""
    if already_admitted:
        return
    max_cps = int(getattr(server, "max_cps", 0) or 0)
    allowed, limit, count = try_admit(server.id, max_cps)
    if allowed:
        return
    raise HTTPException(
        status_code=429,
        detail=(
            f"CPS limit exceeded for this VICIdial server "
            f"({count}/{limit} calls this second). "
            "On AMD extension 8399, the dialer trusts a prior stock "
            "HUMAN/PERSON/NOTSURE result and passes to the agent."
        ),
        headers={"Retry-After": "1", "X-OpenAMD-CPS-Limit": str(limit)},
    )


@router.get("/admit")
@router.post("/admit")
def admit_call(server: VicidialServer = Depends(get_server_from_api_key)):
    """Fast CPS + online check before Record (AGI ping).

    200 = AIAMD will accept this call.
    429 = over CPS — 8399 trusts its prior stock HUMAN/PERSON/NOTSURE
    result and passes to the agent (same policy as offline/timeout).
    """
    max_cps = int(getattr(server, "max_cps", 0) or 0)
    allowed, limit, count = try_admit(server.id, max_cps)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"CPS limit exceeded ({count}/{limit} per second). "
                "8399 trusts prior stock-human AMD and passes to the agent."
            ),
            headers={"Retry-After": "1", "X-OpenAMD-CPS-Limit": str(limit)},
        )
    return {
        "status": "ok",
        "admitted": True,
        "server_id": server.id,
        "max_cps": limit,
        "cps_count": count,
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    callid: str = Form(...),
    campaign: str = Form(""),
    caller: str = Form(""),
    called: str = Form(""),
    ani: str = Form(""),
    amdstatus: str = Form(""),
    amdcause: str = Form(""),
    audio: UploadFile = File(...),
    x_openamd_admitted: str | None = Header(None, alias="X-OpenAMD-Admitted"),
    db: Session = Depends(get_db),
    server: VicidialServer = Depends(get_server_from_api_key),
):
    # If AGI already passed /admit (ping), do not consume a second CPS slot.
    already = str(x_openamd_admitted or "").strip().lower() in {"1", "true", "yes", "ok"}
    _enforce_cps(server, already_admitted=already)

    raw = await audio.read()
    max_bytes = settings.MAX_AUDIO_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail="Audio file too large")

    called_number = (called or "").strip()
    caller_id = (caller or "").strip()
    ani_value = (ani or called_number or "").strip()

    # Empty / near-empty WAV = no recording to check. Do not trust stock AMD.
    # Checkbox ON  → BLANK, VICIdial MACHINE (AA hangup)
    # Checkbox OFF → HUMAN (quiet/empty pickup still goes to the agent)
    if len(raw) < 100:
        from app.amd_settings import is_blank_as_machine_enabled

        blank_as_aa = is_blank_as_machine_enabled()
        stock_st = (amdstatus or "").strip().upper()

        if blank_as_aa:
            portal_status = "BLANK"
            vicidial_status = "MACHINE"
            fuse_note = "blank_silence_empty_file"
            conf = 0.99
        else:
            portal_status = "HUMAN"
            vicidial_status = "HUMAN"
            fuse_note = "blank_to_agent_empty_file"
            conf = 0.92

        details = {
            "blank_empty_upload": True,
            "audio_bytes": len(raw),
            "raw_status": portal_status,
            "vicidial_status": vicidial_status,
            "blank_as_machine": blank_as_aa,
            "fuse_note": fuse_note,
            "decision_reason": fuse_note,
            "ml_note": fuse_note,
            "amdstatus": stock_st,
            "asterisk_amdstatus": stock_st,
            "amdcause": (amdcause or "").strip(),
            "stock_amd_trust": False,
            "recording_checked": False,
            "called_number": called_number,
            "caller_id": caller_id,
            "ani": ani_value,
            "cps_admitted_prior": already,
            "max_cps": int(getattr(server, "max_cps", 0) or 0),
            "short_human_safety_override": not blank_as_aa,
        }
        row = CallAnalysis(
            server_id=server.id,
            call_id=callid,
            campaign=campaign or "",
            caller_id=caller_id,
            called_number=called_number,
            ani=ani_value,
            status=portal_status,
            raw_status=portal_status,
            confidence=conf,
            processing_ms=0,
            audio_seconds=0.0,
            audio_path="",
            audio_saved=False,
            audio_blob=None,
            features_json=json.dumps(details),
            error_message=f"empty upload ({len(raw)} bytes) → VICIdial {vicidial_status}",
        )
        db.add(row)
        server.last_seen = datetime.utcnow()
        db.commit()
        db.refresh(row)
        return AnalyzeResponse(
            status=vicidial_status,
            confidence=conf,
            processing_ms=0,
            callid=callid,
            analysis_id=row.id,
            details=details,
        )

    call_meta = {
        "callid": callid,
        "called_number": called_number,
        "caller_id": caller_id,
        "ani": ani_value,
        "phone_number": called_number or ani_value,
        "campaign": campaign or "",
        "amdstatus": (amdstatus or "").strip(),
        "amdcause": (amdcause or "").strip(),
        "asterisk_amdstatus": (amdstatus or "").strip(),
        "asterisk_amdcause": (amdcause or "").strip(),
    }

    effective = resolve_effective_amd_settings(server)
    result = analyze_audio(
        raw,
        locale_pack_enabled=False,
        locale_pack="usa",
        amd_settings=effective,
        call_meta=call_meta,
    )

    engine_status = result.status
    # 0.0s captures are not a checked recording. With the blank checkbox on,
    # they stay BLANK / AA even if an earlier stage scored them HUMAN.
    if (
        engine_status == "HUMAN"
        and float(result.audio_seconds or 0) < 0.20
        and bool(effective.get("blank_as_machine", True))
    ):
        engine_status = "BLANK"
        result.status = "BLANK"
        result.confidence = max(float(result.confidence or 0), 0.95)
        if isinstance(result.details, dict):
            result.details["blank_safety_override"] = True
            result.details["fuse_note"] = "blank_silence"
            result.details["decision_reason"] = "blank_silence_too_short"
            result.details["recording_checked"] = False
            result.details["short_human_safety_override"] = False

    gate_cfg = gate_config_for_server(server)
    gated_status, downgraded = apply_confidence_gate(
        result.status,
        result.confidence,
        effective=effective,
        details=result.details if isinstance(result.details, dict) else None,
    )

    final_status = gated_status
    final_confidence = result.confidence
    raw_status = engine_status
    vicidial_status = map_status_for_vicidial(final_status)

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
        "vicidial_status": vicidial_status,
        "confidence_gate": gate_cfg,
        "amd_effective": {
            "source": effective.get("source"),
            "ml_source": effective.get("ml_source"),
            "gate_source": effective.get("gate_source"),
            "ml_pipeline_enabled": effective.get("ml_pipeline_enabled"),
            "min_human_confidence_percent": effective.get("min_human_confidence_percent"),
        },
        "training_override": False,
        "called_number": called_number,
        "caller_id": caller_id,
        "ani": ani_value,
        "cps_admitted_prior": already,
        "max_cps": int(getattr(server, "max_cps", 0) or 0),
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

    ml_block = details.get("ml") if isinstance(details.get("ml"), dict) else {}
    sample_id = ml_block.get("ml_sample_id")
    if sample_id:
        try:
            from app.ml_data import attach_call_to_sample

            attach_call_to_sample(
                sample_id,
                call_analysis_id=row.id,
                vicidial_call_id=callid,
                called_number=called_number,
                caller_id=caller_id,
                ani=ani_value,
                phone_number=called_number or ani_value,
            )
        except Exception:
            pass

    if path:
        linked = link_analysis_recording(row.id, path)
        if linked and linked != path:
            row.audio_path = linked
            db.commit()

    return AnalyzeResponse(
        status=vicidial_status,
        confidence=final_confidence,
        processing_ms=result.processing_ms,
        callid=callid,
        analysis_id=row.id,
        details=details,
    )


@router.get("/health")
def health_public():
    return {"status": "ok", "service": "openamd-analyze"}
