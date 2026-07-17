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


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return ""


def _ip_allowed(whitelist: str, client_ip: str) -> bool:
    raw = (whitelist or "").strip()
    if not raw:
        return True  # blank = allow any IP (key still required)
    allowed = {p.strip() for p in raw.split(",") if p.strip()}
    return client_ip in allowed


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
        raise HTTPException(status_code=403, detail="Client IP not allowed for this server")

    record.last_used = datetime.utcnow()
    server.last_seen = datetime.utcnow()
    db.commit()
    return server
