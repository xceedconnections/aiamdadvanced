from datetime import datetime, timedelta
from io import BytesIO, StringIO
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.auth.security import get_current_user
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.correction import TrainingCorrection
from app.models.server import VicidialServer
from app.models.user import User
from app.recordings import (
    list_recording_index,
    recording_meta,
    recording_meta_fast,
    repair_call_recording,
)
from app.schemas import CallOut, CorrectionCreate, CdrPageOut, ServerReport

router = APIRouter(prefix="/api", tags=["reports"])

_ALLOWED_STATUS = {"HUMAN", "MACHINE", "IVR", "FAX", "SIT", "ERROR", "ALL"}
_EXPORT_MAX_ROWS = 20000


def _normalize_status(status: Optional[str]) -> Optional[str]:
    if not status:
        return None
    value = status.strip().upper()
    if value in ("", "ALL", "*"):
        return None
    if value == "CANCELLED":
        return "SIT"
    if value not in _ALLOWED_STATUS:
        raise HTTPException(status_code=400, detail="Invalid status filter")
    return value


def _sanitize_search(q: Optional[str]) -> Optional[str]:
    if q is None:
        return None
    cleaned = "".join(ch for ch in str(q).strip() if ch.isalnum() or ch in " +-_()")
    cleaned = cleaned[:64]
    return cleaned or None


def _to_call_out(r: CallAnalysis, server_map: dict, meta: dict) -> CallOut:
    return CallOut(
        id=r.id,
        server_id=r.server_id,
        call_id=r.call_id,
        campaign=r.campaign or "",
        caller_id=r.caller_id or "",
        called_number=getattr(r, "called_number", None) or "",
        ani=r.ani or "",
        status=r.status,
        raw_status=getattr(r, "raw_status", "") or "",
        confidence=r.confidence,
        processing_ms=r.processing_ms,
        audio_seconds=r.audio_seconds,
        created_at=r.created_at,
        server_name=server_map.get(r.server_id),
        has_recording=meta["has_recording"],
        recording_filename=meta["recording_filename"],
        recording_bytes=meta["recording_bytes"],
        audio_path=meta.get("audio_path") or (r.audio_path or ""),
    )


def _call_out_list(db: Session, rows: list[CallAnalysis], *, repair: bool = True) -> list[CallOut]:
    server_map = {s.id: s.name for s in db.query(VicidialServer).all()}
    if not repair:
        return [_to_call_out(r, server_map, recording_meta_fast(r)) for r in rows]

    index = list_recording_index()
    out = []
    dirty = False
    for r in rows:
        if repair_call_recording(r, index=index):
            dirty = True
        meta = recording_meta(r, index=index)
        out.append(_to_call_out(r, server_map, meta))
    if dirty:
        try:
            db.commit()
        except Exception:
            db.rollback()
    return out


def _filtered_query(
    db: Session,
    *,
    server_id: Optional[int] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
):
    query = db.query(CallAnalysis).order_by(CallAnalysis.id.desc())
    if server_id is not None and int(server_id) > 0:
        query = query.filter(CallAnalysis.server_id == int(server_id))
    status_norm = _normalize_status(status)
    if status_norm:
        query = query.filter(CallAnalysis.status == status_norm)
    search = _sanitize_search(q)
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                CallAnalysis.called_number.ilike(like),
                CallAnalysis.caller_id.ilike(like),
                CallAnalysis.ani.ilike(like),
                CallAnalysis.call_id.ilike(like),
            )
        )
    return query


def _export_headers() -> list[str]:
    return [
        "id",
        "call_id",
        "created_at",
        "server",
        "called_number",
        "caller_id",
        "ani",
        "campaign",
        "status",
        "raw_status",
        "confidence",
        "processing_ms",
        "audio_seconds",
    ]


def _export_row(r: CallAnalysis, server_map: dict) -> list[str]:
    return [
        str(r.id),
        r.call_id or "",
        r.created_at.isoformat(sep=" ", timespec="seconds") if r.created_at else "",
        server_map.get(r.server_id, ""),
        getattr(r, "called_number", None) or "",
        r.caller_id or "",
        r.ani or "",
        r.campaign or "",
        r.status or "",
        getattr(r, "raw_status", "") or "",
        f"{float(r.confidence or 0):.4f}",
        str(r.processing_ms or 0),
        f"{float(r.audio_seconds or 0):.3f}",
    ]


def _build_csv(rows: list[CallAnalysis], server_map: dict) -> bytes:
    import csv

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(_export_headers())
    for r in rows:
        writer.writerow(_export_row(r, server_map))
    # UTF-8 BOM so Excel opens numbers/phones correctly
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def _build_excel_xml(rows: list[CallAnalysis], server_map: dict) -> bytes:
    """Excel-compatible SpreadsheetML (.xls) — no extra Python packages."""
    headers = _export_headers()
    parts = [
        '<?xml version="1.0"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"',
        ' xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">',
        '<Worksheet ss:Name="CDR"><Table>',
    ]
    parts.append(
        "<Row>"
        + "".join(f'<Cell><Data ss:Type="String">{xml_escape(h)}</Data></Cell>' for h in headers)
        + "</Row>"
    )
    for r in rows:
        cells = []
        for value in _export_row(r, server_map):
            cells.append(f'<Cell><Data ss:Type="String">{xml_escape(value)}</Data></Cell>')
        parts.append("<Row>" + "".join(cells) + "</Row>")
    parts.append("</Table></Worksheet></Workbook>")
    return "\n".join(parts).encode("utf-8")


@router.get("/live", response_model=list[CallOut])
def live_calls(
    limit: int = Query(50, ge=1, le=200),
    server_id: Optional[int] = None,
    status: Optional[str] = Query(None, max_length=16),
    q: Optional[str] = Query(None, max_length=64),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    rows = _filtered_query(db, server_id=server_id, status=status, q=q).limit(limit).all()
    return _call_out_list(db, rows, repair=False)


@router.get("/cdr", response_model=CdrPageOut)
def cdr_calls(
    page: int = Query(1, ge=1, le=100000),
    page_size: int = Query(50, ge=1, le=200),
    server_id: Optional[int] = None,
    status: Optional[str] = Query(None, max_length=16),
    q: Optional[str] = Query(None, max_length=64, description="Search called number or caller ID"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    base = _filtered_query(db, server_id=server_id, status=status, q=q)
    total = base.count()
    offset = (page - 1) * page_size
    rows = base.offset(offset).limit(page_size).all()
    return CdrPageOut(
        total=total,
        page=page,
        page_size=page_size,
        rows=_call_out_list(db, rows, repair=False),
    )


@router.get("/cdr/export")
def cdr_export(
    format: str = Query("csv", pattern="^(csv|xlsx|xls)$"),
    server_id: Optional[int] = None,
    status: Optional[str] = Query(None, max_length=16),
    q: Optional[str] = Query(None, max_length=64),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Export filtered CDR as CSV or Excel (.xls SpreadsheetML)."""
    base = _filtered_query(db, server_id=server_id, status=status, q=q)
    rows = base.limit(_EXPORT_MAX_ROWS).all()
    server_map = {s.id: s.name for s in db.query(VicidialServer).all()}
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    fmt = (format or "csv").lower()
    if fmt == "csv":
        data = _build_csv(rows, server_map)
        filename = f"openamd_cdr_{stamp}.csv"
        media = "text/csv; charset=utf-8"
    else:
        data = _build_excel_xml(rows, server_map)
        filename = f"openamd_cdr_{stamp}.xls"
        media = "application/vnd.ms-excel"

    return StreamingResponse(
        BytesIO(data),
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Export-Rows": str(len(rows)),
        },
    )


@router.get("/training/calls", response_model=list[CallOut])
def training_calls(
    limit: int = Query(50, ge=1, le=200),
    server_id: Optional[int] = None,
    status: Optional[str] = Query(None, max_length=16),
    q: Optional[str] = Query(None, max_length=64, description="Search called number or caller ID"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Calls for Training page — search by called number / caller ID, then correct."""
    rows = _filtered_query(db, server_id=server_id, status=status, q=q).limit(limit).all()
    return _call_out_list(db, rows, repair=False)


@router.get("/reports/servers", response_model=list[ServerReport])
def server_reports(
    days: int = Query(1, ge=1, le=90),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    since = datetime.utcnow() - timedelta(days=days)
    servers = db.query(VicidialServer).order_by(VicidialServer.name).all()
    reports = []

    for s in servers:
        base = db.query(CallAnalysis).filter(
            CallAnalysis.server_id == s.id,
            CallAnalysis.created_at >= since,
        )
        total = base.count()

        def cnt(status: str) -> int:
            return base.filter(CallAnalysis.status == status).count()

        avg_ms = (
            db.query(func.avg(CallAnalysis.processing_ms))
            .filter(CallAnalysis.server_id == s.id, CallAnalysis.created_at >= since)
            .scalar()
            or 0.0
        )
        avg_conf = (
            db.query(func.avg(CallAnalysis.confidence))
            .filter(CallAnalysis.server_id == s.id, CallAnalysis.created_at >= since)
            .scalar()
            or 0.0
        )

        corrections = (
            db.query(TrainingCorrection)
            .join(CallAnalysis, CallAnalysis.id == TrainingCorrection.call_id)
            .filter(
                CallAnalysis.server_id == s.id,
                TrainingCorrection.created_at >= since,
            )
            .all()
        )
        accuracy = None
        if corrections:
            ok = sum(1 for c in corrections if c.ai_status == c.corrected_status)
            accuracy = round(ok / len(corrections) * 100.0, 2)

        reports.append(
            ServerReport(
                server_id=s.id,
                server_name=s.name,
                total=total,
                human=cnt("HUMAN"),
                machine=cnt("MACHINE"),
                ivr=cnt("IVR"),
                fax=cnt("FAX"),
                sit=cnt("SIT"),
                errors=cnt("ERROR"),
                avg_processing_ms=round(float(avg_ms), 1),
                avg_confidence=round(float(avg_conf), 4),
                accuracy=accuracy,
            )
        )
    return reports


@router.get("/reports/hourly")
def hourly_report(
    server_id: Optional[int] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    since = datetime.utcnow() - timedelta(hours=24)
    q = db.query(
        func.date_trunc("hour", CallAnalysis.created_at).label("hour"),
        CallAnalysis.status,
        func.count(CallAnalysis.id),
    ).filter(CallAnalysis.created_at >= since)

    if server_id:
        q = q.filter(CallAnalysis.server_id == server_id)

    rows = q.group_by("hour", CallAnalysis.status).order_by("hour").all()
    return [
        {"hour": h.isoformat() if h else None, "status": status, "count": count}
        for h, status, count in rows
    ]


@router.post("/training/correct")
def correct_call(
    payload: CorrectionCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    call = db.query(CallAnalysis).filter(CallAnalysis.id == payload.call_analysis_id).first()
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")

    corrected = (payload.corrected_status or "").strip().upper()
    if corrected == "CANCELLED":
        corrected = "SIT"
    if corrected not in {"HUMAN", "MACHINE", "IVR", "FAX", "SIT", "ERROR"}:
        raise HTTPException(status_code=400, detail="Invalid corrected_status")

    ai_status = call.status
    correction = TrainingCorrection(
        call_id=call.id,
        ai_status=ai_status,
        corrected_status=corrected,
        corrected_by=user.username,
        notes=payload.notes or "",
    )
    db.add(correction)

    # Keep original AI label in raw_status; update visible status for CDR / future tuning
    if not (getattr(call, "raw_status", None) or "").strip():
        call.raw_status = ai_status
    call.status = corrected

    db.commit()
    return {
        "ok": True,
        "id": correction.id,
        "call_analysis_id": call.id,
        "ai_status": ai_status,
        "corrected_status": corrected,
    }
