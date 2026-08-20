from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Query, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, APIKeyHeader
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models.user import User
from app.models.api_key import ApiKey
from app.models.server import VicidialServer
import hashlib

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(
        minutes=expires_minutes or settings.JWT_EXPIRE_MINUTES
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    import secrets

    return f"oam_{secrets.token_urlsafe(32)}"


def _user_from_token(token: str, db: Session) -> User:
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(User.username == username, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return _user_from_token(credentials.credentials, db)


def is_dialer_user(user: User) -> bool:
    return (getattr(user, "role", "") or "").lower() == "dialer"


def dialer_server_id(user: User) -> Optional[int]:
    if not is_dialer_user(user):
        return None
    sid = getattr(user, "server_id", None)
    try:
        return int(sid) if sid is not None else None
    except (TypeError, ValueError):
        return None


def require_admin(user: User) -> None:
    if getattr(user, "role", "") not in ("superadmin", "admin"):
        raise HTTPException(status_code=403, detail="Admin role required")


def require_not_dialer(user: User) -> None:
    if is_dialer_user(user):
        raise HTTPException(
            status_code=403,
            detail="Dialer portal users can only view Live Calls and CDR",
        )


def scoped_server_id(user: User, requested: Optional[int] = None) -> Optional[int]:
    """Force dialer users onto their server; admins keep the requested filter."""
    locked = dialer_server_id(user)
    if locked is not None:
        return locked
    if requested is None:
        return None
    try:
        value = int(requested)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def get_current_user_bearer_or_query(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    token: Optional[str] = Query(None, description="JWT for audio/download links"),
    db: Session = Depends(get_db),
) -> User:
    """Allow Authorization header OR ?token= for native <audio src> playback."""
    raw = credentials.credentials if credentials is not None else token
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return _user_from_token(raw, db)


def _normalize_ip(ip: str) -> str:
    value = (ip or "").strip()
    if value.lower().startswith("::ffff:"):
        value = value[7:]
    # Strip surrounding brackets from IPv6 literals
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return value.strip()


def _client_ip(request: Request) -> str:
    """Resolve dialer IP behind Nginx.

    Prefer X-Real-IP ($remote_addr). For X-Forwarded-For, use the *last*
    hop so a client cannot spoof a whitelisted IP as the first entry.
    """
    real = _normalize_ip(request.headers.get("x-real-ip") or "")
    if real and real not in {"127.0.0.1", "::1", "localhost"}:
        return real

    xff = (request.headers.get("x-forwarded-for") or "").strip()
    if xff:
        parts = [_normalize_ip(p) for p in xff.split(",") if p.strip()]
        parts = [p for p in parts if p]
        if parts:
            # Last entry is the TCP peer nginx saw (with proxy_add_x_forwarded_for)
            return parts[-1]

    if real:
        return real

    if request.client and request.client.host:
        return _normalize_ip(request.client.host)
    return ""


def _parse_whitelist(whitelist: str) -> set[str]:
    return {
        _normalize_ip(p)
        for p in (whitelist or "").split(",")
        if _normalize_ip(p)
    }


def _ip_allowed(whitelist: str, client_ip: str) -> bool:
    allowed = _parse_whitelist(whitelist)
    if not allowed:
        return True  # blank = allow any IP (key still required)

    client = _normalize_ip(client_ip)
    if not client or client in {"127.0.0.1", "::1", "localhost"}:
        # Whitelist is set but we only see loopback → do not bypass
        return False
    return client in allowed


def get_server_from_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db),
) -> VicidialServer:
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    key_hash = hash_api_key(api_key)
    record = (
        db.query(ApiKey)
        .filter(ApiKey.key_hash == key_hash, ApiKey.is_active == True)
        .first()
    )
    if not record:
        # Reject before any audio decode / AMD / recording work
        raise HTTPException(status_code=401, detail="Invalid API key")

    server = (
        db.query(VicidialServer)
        .filter(VicidialServer.id == record.server_id, VicidialServer.is_active == True)
        .first()
    )
    if not server:
        raise HTTPException(status_code=401, detail="Server inactive or missing")

    client = _client_ip(request)
    if not _ip_allowed(server.ip_whitelist or "", client):
        raise HTTPException(
            status_code=403,
            detail=(
                f"Client IP '{client or 'unknown'}' is not in this server's IP whitelist. "
                "Only listed dialer IPs may use AI AMD (blank whitelist = allow all)."
            ),
        )

    record.last_used = datetime.utcnow()
    server.last_seen = datetime.utcnow()
    db.commit()
    return server
