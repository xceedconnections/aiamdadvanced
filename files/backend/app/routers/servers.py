from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth.security import generate_api_key, get_current_user, hash_api_key
from app.database import get_db
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
        last_seen=server.last_seen,
        created_at=server.created_at,
        total_calls=total,
        calls_today=today_count,
    )


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

    server = VicidialServer(
        name=payload.name,
        description=payload.description,
        timezone=payload.timezone,
        ip_whitelist=payload.ip_whitelist,
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

    data = payload.model_dump(exclude_unset=True)
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
