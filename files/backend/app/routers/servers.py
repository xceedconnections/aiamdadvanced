from datetime import datetime
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.amd_settings import server_amd_mode
from app.auth.security import (
    dialer_server_id,
    generate_api_key,
    get_current_user,
    hash_api_key,
    hash_password,
)
from app.database import get_db
from app.models.api_key import ApiKey
from app.models.call import CallAnalysis
from app.models.server import VicidialServer
from app.models.user import User
from app.schemas import (
    ApiKeyCreate,
    ApiKeyOut,
    DialerUserCreate,
    DialerUserOut,
    DialerUserUpdate,
    ServerCreate,
    ServerOut,
    ServerUpdate,
)

router = APIRouter(prefix="/api/servers", tags=["servers"])

_ALLOWED_ACTIONS = {"MACHINE", "IVR", "SIT", "ERROR"}
_ALLOWED_AMD_MODES = {"global", "classic", "ml"}
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{2,64}$")


def _require_admin(user: User):
    if user.role not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


def _apply_amd_mode_flags(fields: dict) -> dict:
    """Sync legacy flags from amd_mode so older resolve paths stay consistent."""
    out = dict(fields)
    mode = str(out.get("amd_mode") or "global").strip().lower()
    if mode not in _ALLOWED_AMD_MODES:
        raise HTTPException(status_code=400, detail="amd_mode must be global, classic, or ml")
    out["amd_mode"] = mode
    # Locale packs removed from product UI
    out["locale_pack_enabled"] = False
    if mode == "global":
        out["confidence_gate_enabled"] = False
        out["ml_pipeline_override_enabled"] = False
        out["ml_pipeline_enabled"] = False
    elif mode == "classic":
        out["confidence_gate_enabled"] = True
        out["ml_pipeline_override_enabled"] = False
        out["ml_pipeline_enabled"] = False
    else:  # ml
        out["confidence_gate_enabled"] = False
        out["ml_pipeline_override_enabled"] = True
        out["ml_pipeline_enabled"] = True
    return out


def _normalize_server_amd_fields(data: dict) -> dict:
    out = dict(data)
    if "amd_mode" in out and out["amd_mode"] is not None:
        out = _apply_amd_mode_flags(out)
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
    if "max_cps" in out and out["max_cps"] is not None:
        from app.cps_limit import normalize_max_cps

        out["max_cps"] = normalize_max_cps(out["max_cps"])
    if "scam_record_seconds" in out and out["scam_record_seconds"] is not None:
        try:
            secs = int(out["scam_record_seconds"])
        except (TypeError, ValueError):
            secs = 120
        out["scam_record_seconds"] = max(30, min(600, secs))
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
    mode = server_amd_mode(server)
    return ServerOut(
        id=server.id,
        name=server.name,
        description=server.description or "",
        timezone=server.timezone or "UTC",
        ip_whitelist=server.ip_whitelist or "",
        is_active=server.is_active,
        amd_mode=mode,
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
        locale_pack_enabled=False,
        locale_pack="usa",
        max_cps=int(getattr(server, "max_cps", 0) or 0),
        scam_protection_enabled=bool(getattr(server, "scam_protection_enabled", False)),
        scam_record_seconds=int(getattr(server, "scam_record_seconds", 120) or 120),
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
    q = db.query(VicidialServer).order_by(VicidialServer.name)
    locked = dialer_server_id(user)
    if locked is not None:
        q = q.filter(VicidialServer.id == locked)
    servers = q.all()
    return [_server_out(db, s) for s in servers]


@router.post("", response_model=ServerOut)
def create_server(
    payload: ServerCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    exists = db.query(VicidialServer).filter(VicidialServer.name == payload.name).first()
    if exists:
        raise HTTPException(status_code=400, detail="Server name already exists")

    fields = _normalize_server_amd_fields(payload.model_dump())
    server = VicidialServer(
        name=fields["name"],
        description=fields.get("description") or "",
        timezone=fields.get("timezone") or "UTC",
        ip_whitelist=fields.get("ip_whitelist") or "",
        amd_mode=str(fields.get("amd_mode") or "global"),
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
        locale_pack_enabled=False,
        locale_pack="usa",
        max_cps=int(fields.get("max_cps", 0) or 0),
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
    locked = dialer_server_id(user)
    if locked is not None and locked != server_id:
        raise HTTPException(status_code=403, detail="Access denied")
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
    _require_admin(user)
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
        if hasattr(server, k):
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
    _require_admin(user)
    server = db.query(VicidialServer).filter(VicidialServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    db.query(User).filter(User.server_id == server_id).delete(synchronize_session=False)
    db.query(ApiKey).filter(ApiKey.server_id == server_id).delete(synchronize_session=False)
    db.query(CallAnalysis).filter(CallAnalysis.server_id == server_id).update(
        {CallAnalysis.server_id: None},
        synchronize_session=False,
    )
    db.delete(server)
    db.commit()
    return {"ok": True, "deleted": True, "id": server_id}


def _api_key_out(record: ApiKey, *, raw_key: str | None = None, server_name: str = "") -> ApiKeyOut:
    stored = (getattr(record, "key_value", None) or "").strip()
    full = (raw_key or stored or "").strip() or None
    return ApiKeyOut(
        id=record.id,
        server_id=record.server_id,
        server_name=server_name
        or (record.server.name if getattr(record, "server", None) is not None else ""),
        key_prefix=record.key_prefix,
        name=record.name or "default",
        is_active=bool(record.is_active),
        created_at=record.created_at,
        last_used=record.last_used,
        notes=record.notes or "",
        api_key=full,
    )


@router.post("/api-keys", response_model=ApiKeyOut)
def create_api_key(
    payload: ApiKeyCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    server = db.query(VicidialServer).filter(VicidialServer.id == payload.server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    raw = generate_api_key()
    record = ApiKey(
        server_id=server.id,
        key_prefix=raw[:12],
        key_hash=hash_api_key(raw),
        key_value=raw,
        name=(payload.name or "default").strip() or "default",
        notes=payload.notes or "",
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return _api_key_out(record, raw_key=raw, server_name=server.name)


@router.get("/api-keys/list", response_model=list[ApiKeyOut])
def list_api_keys(
    server_id: int | None = None,
    active_only: bool = True,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    q = db.query(ApiKey)
    if server_id:
        q = q.filter(ApiKey.server_id == server_id)
    if active_only:
        q = q.filter(ApiKey.is_active == True)  # noqa: E712
    keys = q.order_by(ApiKey.id.desc()).all()
    # Prefetch server names
    server_ids = {k.server_id for k in keys}
    names = {}
    if server_ids:
        for s in db.query(VicidialServer).filter(VicidialServer.id.in_(server_ids)).all():
            names[s.id] = s.name
    return [_api_key_out(k, server_name=names.get(k.server_id, "")) for k in keys]


@router.delete("/api-keys/{key_id}")
def revoke_api_key(
    key_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="API key not found")
    key.is_active = False
    db.commit()
    return {"ok": True, "id": key_id, "deleted": True}


def _dialer_user_out(u: User, server_name: str = "") -> DialerUserOut:
    return DialerUserOut(
        id=u.id,
        username=u.username,
        full_name=u.full_name or "",
        role=u.role,
        server_id=int(u.server_id),
        server_name=server_name,
        is_active=bool(u.is_active),
        created_at=u.created_at,
        last_login=u.last_login,
    )


@router.get("/{server_id}/portal-users", response_model=list[DialerUserOut])
def list_portal_users(
    server_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    server = db.query(VicidialServer).filter(VicidialServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    rows = (
        db.query(User)
        .filter(User.server_id == server_id, User.role == "dialer")
        .order_by(User.username)
        .all()
    )
    return [_dialer_user_out(u, server.name) for u in rows]


@router.post("/{server_id}/portal-users", response_model=DialerUserOut)
def create_portal_user(
    server_id: int,
    payload: DialerUserCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    server = db.query(VicidialServer).filter(VicidialServer.id == server_id).first()
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")

    username = (payload.username or "").strip()
    if not _USERNAME_RE.match(username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 2–64 chars: letters, numbers, . _ @ -",
        )
    if len((payload.password or "").strip()) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    exists = db.query(User).filter(User.username == username).first()
    if exists:
        raise HTTPException(status_code=400, detail="Username already exists")

    email = f"{username}@dialer{server_id}.openamd.local"
    if db.query(User).filter(User.email == email).first():
        email = f"{username}.{server_id}.{int(datetime.utcnow().timestamp())}@dialer.openamd.local"

    row = User(
        username=username,
        email=email,
        hashed_password=hash_password(payload.password.strip()),
        full_name=(payload.full_name or "").strip() or username,
        role="dialer",
        server_id=server_id,
        is_active=True,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _dialer_user_out(row, server.name)


@router.patch("/portal-users/{user_id}", response_model=DialerUserOut)
def update_portal_user(
    user_id: int,
    payload: DialerUserUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    row = db.query(User).filter(User.id == user_id, User.role == "dialer").first()
    if not row:
        raise HTTPException(status_code=404, detail="Portal user not found")
    data = payload.model_dump(exclude_unset=True)
    if "password" in data and data["password"]:
        pwd = str(data["password"]).strip()
        if len(pwd) < 8:
            raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
        row.hashed_password = hash_password(pwd)
    if "full_name" in data and data["full_name"] is not None:
        row.full_name = str(data["full_name"]).strip()
    if "is_active" in data and data["is_active"] is not None:
        row.is_active = bool(data["is_active"])
    db.commit()
    db.refresh(row)
    srv = db.query(VicidialServer).filter(VicidialServer.id == row.server_id).first()
    return _dialer_user_out(row, srv.name if srv else "")


@router.delete("/portal-users/{user_id}")
def delete_portal_user(
    user_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    row = db.query(User).filter(User.id == user_id, User.role == "dialer").first()
    if not row:
        raise HTTPException(status_code=404, detail="Portal user not found")
    db.delete(row)
    db.commit()
    return {"ok": True, "deleted": True, "id": user_id}

