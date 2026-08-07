from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth.security import generate_api_key, get_current_user, hash_api_key
from app.database import get_db
from app.locale_packs import ALLOWED_LOCALE_PACKS, locale_pack_meta, normalize_locale_pack
from app.models.api_key import ApiKey
from app.models.call import CallAnalysis
from app.models.server import VicidialServer
from app.models.user import User
from app.schemas import (
    ApiKeyCreate,
    ApiKeyOut,
    ServerCreate,
    ServerOut,
    ServerUpdate,
)

router = APIRouter(prefix="/api/servers", tags=["servers"])

_ALLOWED_ACTIONS = {"MACHINE", "IVR", "SIT", "ERROR"}


def _normalize_server_amd_fields(data: dict) -> dict:
    out = dict(data)
    if "locale_pack" in out and out["locale_pack"] is not None:
        pack = normalize_locale_pack(out["locale_pack"])
        if pack not in ALLOWED_LOCALE_PACKS:
            raise HTTPException(
                status_code=400,
                detail=f"locale_pack must be one of: {', '.join(ALLOWED_LOCALE_PACKS)}",
            )
        out["locale_pack"] = pack
    if "below_threshold_action" in out and out["below_threshold_action"] is not None:
        action = str(out["below_threshold_action"]).strip().upper()
        if action not in _ALLOWED_ACTIONS:
            raise HTTPException(
                status_code=400,
                detail="below_threshold_action must be MACHINE, IVR, SIT, or ERROR",
            )
        out["below_threshold_action"] = action
    if "min_human_confidence_percent" in out and out["min_human_confidence_percent"] is not None:
        pct = int(out["min_human_confidence_percent"])
        out["min_human_confidence_percent"] = max(0, min(100, pct))
    for key in ("ml_min_human_confidence_percent", "ml_save_threshold_percent"):
        if key in out and out[key] is not None:
            out[key] = max(50, min(99, int(out[key])))
    return out


def _server_out(db: Session, server: VicidialServer) -> ServerOut:
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    total = (
        db.query(func.count(CallAnalysis.id))
        .filter(CallAnalysis.server_id == server.id)
        .scalar()
        or 0
    )
    today_count = (
        db.query(func.count(CallAnalysis.id))
        .filter(CallAnalysis.server_id == server.id, CallAnalysis.created_at >= today)
        .scalar()
        or 0
    )
    return ServerOut(
        id=server.id,
        name=server.name,
        description=server.description or "",
        timezone=server.timezone or "UTC",
        ip_whitelist=server.ip_whitelist or "",
        is_active=server.is_active,
        confidence_gate_enabled=bool(getattr(server, "confidence_gate_enabled", False)),
        min_human_confidence_percent=int(
            getattr(server, "min_human_confidence_percent", 70) or 70
        ),
        below_threshold_action=str(
            getattr(server, "below_threshold_action", None) or "MACHINE"
        ),
        ml_pipeline_override_enabled=bool(
            getattr(server, "ml_pipeline_override_enabled", False)
        ),
        ml_pipeline_enabled=bool(getattr(server, "ml_pipeline_enabled", False)),
        ml_whisper_enabled=bool(getattr(server, "ml_whisper_enabled", True)),
        ml_save_low_confidence=bool(getattr(server, "ml_save_low_confidence", True)),
        ml_min_human_confidence_percent=int(
            getattr(server, "ml_min_human_confidence_percent", 85) or 85
        ),
        ml_save_threshold_percent=int(
            getattr(server, "ml_save_threshold_percent", 85) or 85
        ),
        locale_pack_enabled=bool(getattr(server, "locale_pack_enabled", False)),
        locale_pack=normalize_locale_pack(getattr(server, "locale_pack", None) or "usa"),
        last_seen=server.last_seen,
        created_at=server.created_at,
        total_calls=total,
        calls_today=today_count,
    )


@router.get("/locale-packs")
def list_locale_packs(user: User = Depends(get_current_user)):
    return {"packs": locale_pack_meta()}


@router.get("", response_model=list[ServerOut])
def list_servers(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    servers = db.query(VicidialServer).order_by(VicidialServer.name).all()
    return [_server_out(db, s) for s in servers]


@router.post("", response_model=ServerOut)
def create_server(
    payload: ServerCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    exists = db.query(VicidialServer).filter(VicidialServer.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=400, detail="Server name already exists")

    fields = _normalize_server_amd_fields(payload.model_dump())
    server = VicidialServer(
        name=fields["name"],
        description=fields.get("description") or "",
        timezone=fields.get("timezone") or "UTC",
        ip_whitelist=fields.get("ip_whitelist") or "",
        confidence_gate_enabled=bool(fields.get("confidence_gate_enabled", False)),
        min_human_confidence_percent=int(fields.get("min_human_confidence_percent", 70)),
        below_threshold_action=str(fields.get("below_threshold_action") or "MACHINE"),
        ml_pipeline_override_enabled=bool(fields.get("ml_pipeline_override_enabled", False)),
        ml_pipeline_enabled=bool(fields.get("ml_pipeline_enabled", False)),
        ml_whisper_enabled=bool(fields.get("ml_whisper_enabled", True)),
        ml_save_low_confidence=bool(fields.get("ml_save_low_confidence", True)),
        ml_min_human_confidence_percent=int(
            fields.get("ml_min_human_confidence_percent", 85)
        ),
        ml_save_threshold_percent=int(fields.get("ml_save_threshold_percent", 85)),
        locale_pack_enabled=bool(fields.get("locale_pack_enabled", False)),
        locale_pack=str(fields.get("locale_pack") or "usa"),
    )
    db.add(server)
    db.commit()
    db.refresh(server)
    return _server_out(db, server)


@router.get("/{server_id}", response_model=ServerOut)
def get_server(
    server_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    server = db.query(VicidialServer).filter(VicidialServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    return _server_out(db, server)


@router.patch("/{server_id}", response_model=ServerOut)
def update_server(
    server_id: int,
    payload: ServerUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    server = db.query(VicidialServer).filter(VicidialServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    data = _normalize_server_amd_fields(payload.model_dump(exclude_unset=True))
    if "name" in data and data["name"]:
        clash = (
            db.query(VicidialServer)
            .filter(VicidialServer.name == data["name"], VicidialServer.id != server_id)
            .first()
        )
        if clash:
            raise HTTPException(status_code=400, detail="Server name already exists")
    for k, v in data.items():
        setattr(server, k, v)
    server.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(server)
    return _server_out(db, server)


@router.delete("/{server_id}")
def delete_server(
    server_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    server = db.query(VicidialServer).filter(VicidialServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    # Soft delete
    server.is_active = False
    db.commit()
    return {"ok": True}


@router.post("/api-keys", response_model=ApiKeyOut)
def create_api_key(
    payload: ApiKeyCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    server = db.query(VicidialServer).filter(VicidialServer.id == payload.server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    raw = generate_api_key()
    record = ApiKey(
        server_id=server.id,
        key_prefix=raw[:10],
        key_hash=hash_api_key(raw),
        name=payload.name,
        notes=payload.notes,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return ApiKeyOut(
        id=record.id,
        server_id=record.server_id,
        key_prefix=record.key_prefix,
        name=record.name,
        is_active=record.is_active,
        created_at=record.created_at,
        last_used=record.last_used,
        notes=record.notes or "",
        api_key=raw,
    )


@router.get("/api-keys/list", response_model=list[ApiKeyOut])
def list_api_keys(
    server_id: int | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(ApiKey)
    if server_id:
        q = q.filter(ApiKey.server_id == server_id)
    keys = q.order_by(ApiKey.id.desc()).all()
    return [
        ApiKeyOut(
            id=k.id,
            server_id=k.server_id,
            key_prefix=k.key_prefix,
            name=k.name,
            is_active=k.is_active,
            created_at=k.created_at,
            last_used=k.last_used,
            notes=k.notes or "",
            api_key=None,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}")
def revoke_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = False
    db.commit()
    return {"ok": True}
