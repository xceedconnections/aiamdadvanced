import json
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
from typing import Optional
from xml.sax.saxutils import escape as xml_escape

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.auth.security import get_current_user, require_not_dialer, scoped_server_id
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
from app.schemas import CallOut, CdrPageOut, ServerReport

router = APIRouter(prefix="/api", tags=["reports"])

_ALLOWED_STATUS = {"HUMAN", "MACHINE", "IVR", "FAX", "SIT", "BLANK", "ERROR", "ALL"}
_EXPORT_MAX_ROWS = 20000


def _utc_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Mark naive DB timestamps as UTC so JSON clients convert TZ correctly."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


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


def _ml_fields_from_features(raw: Optional[str]) -> dict:
    """Extract Whisper / ML display fields from call features_json."""
    out = {
        "whisper_used": False,
        "whisper_transcript": "",
        "whisper_cue": "",
        "ml_note": "",
    }
    if not raw:
        return out
    try:
        details = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except (TypeError, json.JSONDecodeError):
        return out
    if not isinstance(details, dict):
        return out
    ml = details.get("ml") if isinstance(details.get("ml"), dict) else {}
    whisper = ml.get("whisper") if isinstance(ml.get("whisper"), dict) else {}
    if not whisper and isinstance(details.get("whisper"), dict):
        whisper = details["whisper"]
    used = bool(ml.get("whisper_used") or details.get("whisper_used") or whisper.get("transcript"))
    transcript = str(whisper.get("transcript") or details.get("whisper_transcript") or "")[:240]
    cue = str(whisper.get("cue") or "")
    note = str(ml.get("ml_note") or details.get("ml_note") or "")
    out["whisper_used"] = used
    out["whisper_transcript"] = transcript
    out["whisper_cue"] = cue
    out["ml_note"] = note
    return out


def _to_call_out(r: CallAnalysis, server_map: dict, meta: dict) -> CallOut:
    ml = _ml_fields_from_features(getattr(r, "features_json", None))
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
        created_at=_utc_aware(r.created_at),
        server_name=server_map.get(r.server_id),
        has_recording=meta["has_recording"],
        recording_filename=meta["recording_filename"],
        recording_bytes=meta["recording_bytes"],
        audio_path=meta.get("audio_path") or (r.audio_path or ""),
        whisper_used=ml["whisper_used"],
        whisper_transcript=ml["whisper_transcript"],
        whisper_cue=ml["whisper_cue"],
        ml_note=ml["ml_note"],
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
    within_seconds: Optional[int] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
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
    if within_seconds is not None and int(within_seconds) > 0:
        # created_at is stored as naive UTC
        since = datetime.utcnow() - timedelta(seconds=int(within_seconds))
        query = query.filter(CallAnalysis.created_at >= since)
        return query
    if date_from:
        try:
            start = datetime.strptime(date_from.strip()[:10], "%Y-%m-%d")
            query = query.filter(CallAnalysis.created_at >= start)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date_from (YYYY-MM-DD)") from exc
    if date_to:
        try:
            end = datetime.strptime(date_to.strip()[:10], "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(CallAnalysis.created_at < end)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid date_to (YYYY-MM-DD)") from exc
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
        "whisper_used",
        "whisper_transcript",
    ]


def _export_row(r: CallAnalysis, server_map: dict) -> list[str]:
    ml = _ml_fields_from_features(getattr(r, "features_json", None))
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
        "1" if ml["whisper_used"] else "0",
        ml["whisper_transcript"],
    ]


def _build_csv(rows: list[CallAnalysis], server_map: dict) -> bytes:
    import csv

    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(_export_headers())
    for r in rows:
        writer.writerow(_export_row(r, server_map))
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
    within_seconds: Optional[int] = Query(
        None, ge=1, le=86400, description="Only calls from the last N seconds"
    ),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    server_id = scoped_server_id(user, server_id)
    rows = _filtered_query(
        db,
        server_id=server_id,
        status=status,
        q=q,
        within_seconds=within_seconds,
    ).limit(limit).all()
    return _call_out_list(db, rows, repair=False)


@router.get("/cdr", response_model=CdrPageOut)
def cdr_calls(
    page: int = Query(1, ge=1, le=100000),
    page_size: int = Query(50, ge=1, le=200),
    server_id: Optional[int] = None,
    status: Optional[str] = Query(None, max_length=16),
    q: Optional[str] = Query(None, max_length=64, description="Search called number or caller ID"),
    within_seconds: Optional[int] = Query(
        None, ge=1, le=604800, description="Only calls from the last N seconds"
    ),
    date_from: Optional[str] = Query(None, max_length=10, description="YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, max_length=10, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    server_id = scoped_server_id(user, server_id)
    base = _filtered_query(
        db,
        server_id=server_id,
        status=status,
        q=q,
        within_seconds=within_seconds,
        date_from=date_from,
        date_to=date_to,
    )
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
    within_seconds: Optional[int] = Query(None, ge=1, le=604800),
    date_from: Optional[str] = Query(None, max_length=10),
    date_to: Optional[str] = Query(None, max_length=10),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Export filtered CDR as CSV or Excel (.xls SpreadsheetML)."""
    server_id = scoped_server_id(user, server_id)
    base = _filtered_query(
        db,
        server_id=server_id,
        status=status,
        q=q,
        within_seconds=within_seconds,
        date_from=date_from,
        date_to=date_to,
    )
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
    require_not_dialer(user)
    rows = _filtered_query(db, server_id=server_id, status=status, q=q).limit(limit).all()
    return _call_out_list(db, rows, repair=False)


@router.get("/reports/servers", response_model=list[ServerReport])
def server_reports(
    days: int = Query(1, ge=1, le=90),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    require_not_dialer(user)
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
                TrainingCorrection.is_active == True,  # noqa: E712
                TrainingCorrection.action == "teach",
                TrainingCorrection.call_id.isnot(None),
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
                blank=cnt("BLANK"),
                errors=cnt("ERROR"),
                avg_processing_ms=round(float(avg_ms), 1),
                avg_confidence=round(float(avg_conf), 4),
                accuracy=accuracy,
            )
        )
    return reports


def _norm_label(value: Optional[str]) -> str:
    v = str(value or "").strip().upper()
    if v == "CANCELLED":
        return "SIT"
    if v in _ALLOWED_STATUS and v != "ALL":
        return v
    return "ERROR"


def _is_agent_path(label: str) -> bool:
    """Only LIVE HUMAN is sent to agents; everything else is hangup/AA."""
    return _norm_label(label) == "HUMAN"


@router.get("/reports/accuracy")
def accuracy_report(
    days: int = Query(7, ge=1, le=365),
    server_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Labeled AMD evaluation from Training teaches (ground truth).

    predicted = TrainingCorrection.ai_status (engine label at teach time)
    actual    = TrainingCorrection.corrected_status (human teach)
    confidence from CallAnalysis (unchanged by teach)
    """
    require_not_dialer(user)
    since = datetime.utcnow() - timedelta(days=days)

    q = (
        db.query(TrainingCorrection, CallAnalysis)
        .join(CallAnalysis, CallAnalysis.id == TrainingCorrection.call_id)
        .filter(
            TrainingCorrection.created_at >= since,
            TrainingCorrection.is_active == True,  # noqa: E712
            TrainingCorrection.action == "teach",
            TrainingCorrection.call_id.isnot(None),
        )
        .order_by(TrainingCorrection.call_id.asc(), TrainingCorrection.id.desc())
    )
    if server_id is not None:
        q = q.filter(CallAnalysis.server_id == int(server_id))

    # Latest teach per call wins
    seen: set[int] = set()
    rows: list[dict] = []
    for tc, ca in q.all():
        cid = int(tc.call_id)
        if cid in seen:
            continue
        seen.add(cid)
        predicted = _norm_label(tc.ai_status)
        actual = _norm_label(tc.corrected_status)
        conf = float(ca.confidence or 0.0)
        conf = max(0.0, min(1.0, conf))
        rows.append(
            {
                "call_analysis_id": cid,
                "call_id": ca.call_id or "",
                "predicted": predicted,
                "actual": actual,
                "confidence": conf,
                "correct": predicted == actual,
                "server_id": ca.server_id,
                "labeled_at": tc.created_at.isoformat() if tc.created_at else None,
            }
        )

    n = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    accuracy = (n_correct / n) if n else None

    # Agent-routing errors (what matters operationally)
    false_human = [
        r for r in rows if _is_agent_path(r["predicted"]) and not _is_agent_path(r["actual"])
    ]
    false_machine = [
        r for r in rows if (not _is_agent_path(r["predicted"])) and _is_agent_path(r["actual"])
    ]

    # Confusion matrix (predicted × actual)
    labels_order = ["HUMAN", "MACHINE", "IVR", "BLANK", "SIT", "FAX", "ERROR"]
    present = set()
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["predicted"], r["actual"])
        counts[key] = counts.get(key, 0) + 1
        present.add(r["predicted"])
        present.add(r["actual"])
    labels = [x for x in labels_order if x in present]
    matrix = []
    for pred in labels:
        row_cells = []
        for act in labels:
            row_cells.append(counts.get((pred, act), 0))
        matrix.append({"predicted": pred, "counts": row_cells})

    # Confidence calibration (10 buckets)
    buckets = []
    ece_num = 0.0
    for i in range(10):
        lo = i / 10.0
        hi = (i + 1) / 10.0
        if i == 9:
            in_b = [r for r in rows if lo <= r["confidence"] <= hi]
        else:
            in_b = [r for r in rows if lo <= r["confidence"] < hi]
        bn = len(in_b)
        if bn:
            b_acc = sum(1 for r in in_b if r["correct"]) / bn
            b_conf = sum(r["confidence"] for r in in_b) / bn
            ece_num += abs(b_conf - b_acc) * bn
        else:
            b_acc = None
            b_conf = None
        buckets.append(
            {
                "lo": lo,
                "hi": hi,
                "label": f"{int(lo * 100)}–{int(hi * 100)}%",
                "n": bn,
                "accuracy": None if b_acc is None else round(b_acc, 4),
                "avg_confidence": None if b_conf is None else round(b_conf, 4),
                "gap": None
                if b_acc is None or b_conf is None
                else round(b_conf - b_acc, 4),
            }
        )
    ece = round(ece_num / n, 4) if n else None

    server_map = {s.id: s.name for s in db.query(VicidialServer).all()}

    def _examples(items: list[dict], limit: int = 15) -> list[dict]:
        out = []
        for r in items[:limit]:
            out.append(
                {
                    "call_analysis_id": r["call_analysis_id"],
                    "call_id": r["call_id"],
                    "predicted": r["predicted"],
                    "actual": r["actual"],
                    "confidence": round(r["confidence"], 4),
                    "server_name": server_map.get(r["server_id"], "—"),
                    "labeled_at": r["labeled_at"],
                }
            )
        return out

    return {
        "days": days,
        "server_id": server_id,
        "n_labeled": n,
        "n_correct": n_correct,
        "accuracy": None if accuracy is None else round(accuracy, 4),
        "accuracy_pct": None if accuracy is None else round(accuracy * 100.0, 2),
        "false_human": len(false_human),
        "false_machine": len(false_machine),
        "false_human_rate": round(len(false_human) / n, 4) if n else None,
        "false_machine_rate": round(len(false_machine) / n, 4) if n else None,
        "confusion_labels": labels,
        "confusion_matrix": matrix,
        "calibration": buckets,
        "ece": ece,
        "false_human_examples": _examples(false_human),
        "false_machine_examples": _examples(false_machine),
        "note": (
            "Metrics use Training teaches only (call-linked). "
            "False HUMAN = AI said HUMAN but teach was not HUMAN (sent to agents wrongly). "
            "False MACHINE = AI hung up / AA'd a call taught as HUMAN."
        ),
    }


@router.get("/reports/hourly")
def hourly_report(
    server_id: Optional[int] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    require_not_dialer(user)
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
