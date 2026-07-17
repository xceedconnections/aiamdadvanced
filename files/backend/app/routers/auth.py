import re

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.captcha import create_math_captcha, verify_math_captcha
from app.database import get_db
from app.models.call import CallAnalysis
from app.models.server import VicidialServer
from app.models.user import User
from app.schemas import (
    ChangePasswordRequest,
    DashboardStats,
    LoginRequest,
    TokenResponse,
    UserOut,
)

router = APIRouter(prefix="/api", tags=["auth"])

_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


@router.get("/captcha")
def captcha_challenge():
    captcha_id, question = create_math_captcha()
    return {"captcha_id": captcha_id, "question": question}


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    username = (payload.username or "").strip()
    if not _USERNAME_RE.match(username):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not verify_math_captcha(payload.captcha_id, payload.captcha_answer):
        raise HTTPException(status_code=400, detail="Incorrect or expired captcha")

    # Parameterized ORM lookup — never concatenate user input into SQL
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="User disabled")

    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token({"sub": user.username, "role": user.role})
    return TokenResponse(
        access_token=token,
        role=user.role,
        username=user.username,
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if payload.new_password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="New passwords do not match")
    if len(payload.new_password.strip()) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    if not verify_password(payload.current_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if payload.current_password == payload.new_password:
        raise HTTPException(status_code=400, detail="New password must be different")

    db_user = db.query(User).filter(User.id == user.id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db_user.hashed_password = hash_password(payload.new_password)
    db.commit()
    return {"ok": True, "message": "Password updated successfully"}


@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    q = db.query(CallAnalysis).filter(CallAnalysis.created_at >= today)
    total = q.count()

    def count_status(status: str) -> int:
        return q.filter(CallAnalysis.status == status).count()

    avg_ms = db.query(func.avg(CallAnalysis.processing_ms)).filter(
        CallAnalysis.created_at >= today
    ).scalar() or 0.0
    avg_conf = db.query(func.avg(CallAnalysis.confidence)).filter(
        CallAnalysis.created_at >= today
    ).scalar() or 0.0

    total_servers = db.query(VicidialServer).count()
    active_servers = db.query(VicidialServer).filter(VicidialServer.is_active == True).count()

    return DashboardStats(
        total_calls_today=total,
        human=count_status("HUMAN"),
        machine=count_status("MACHINE"),
        ivr=count_status("IVR"),
        fax=count_status("FAX"),
        sit=count_status("SIT"),
        errors=count_status("ERROR"),
        avg_processing_ms=round(float(avg_ms), 1),
        avg_confidence=round(float(avg_conf), 4),
        active_servers=active_servers,
        total_servers=total_servers,
    )
