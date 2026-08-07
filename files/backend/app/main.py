import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import Base, SessionLocal, engine
from app.auth.security import hash_password
from app.models.user import User
from app.models.correction import TrainingCorrection, TrainingOverride  # noqa: F401 — register tables
from app.models.call import CallAnalysis  # noqa: F401
from app.models.server import VicidialServer  # noqa: F401
from app.routers import (
    analyze,
    auth,
    cron,
    maintenance,
    recordings,
    reports,
    servers,
    settings as settings_router,
    system,
    training,
)

settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"


def ensure_dirs():
    for d in (settings.RECORDINGS_DIR, settings.MODELS_DIR, settings.LOGS_DIR):
        os.makedirs(d, exist_ok=True)


def ensure_schema():
    """Add columns introduced after first install."""
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(
            text(
                "ALTER TABLE call_analyses "
                "ADD COLUMN IF NOT EXISTS called_number VARCHAR(64) DEFAULT ''"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE call_analyses "
                "ADD COLUMN IF NOT EXISTS audio_saved BOOLEAN DEFAULT FALSE"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE call_analyses "
                "ADD COLUMN IF NOT EXISTS audio_blob BYTEA"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE call_analyses "
                "ADD COLUMN IF NOT EXISTS raw_status VARCHAR(32) DEFAULT ''"
            )
        )
        # Per-server AMD gate + locale packs + ML overrides
        for stmt in (
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS confidence_gate_enabled BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS min_human_confidence_percent INTEGER DEFAULT 70",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS below_threshold_action VARCHAR(32) DEFAULT 'MACHINE'",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS locale_pack_enabled BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS locale_pack VARCHAR(32) DEFAULT 'usa'",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS ml_pipeline_override_enabled BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS ml_pipeline_enabled BOOLEAN DEFAULT FALSE",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS ml_whisper_enabled BOOLEAN DEFAULT TRUE",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS ml_save_low_confidence BOOLEAN DEFAULT TRUE",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS ml_min_human_confidence_percent INTEGER DEFAULT 85",
            "ALTER TABLE vicidial_servers ADD COLUMN IF NOT EXISTS ml_save_threshold_percent INTEGER DEFAULT 85",
            "ALTER TABLE training_corrections ADD COLUMN IF NOT EXISTS phone_number VARCHAR(32) DEFAULT ''",
            "ALTER TABLE training_corrections ADD COLUMN IF NOT EXISTS previous_taught_status VARCHAR(32) DEFAULT ''",
            "ALTER TABLE training_corrections ADD COLUMN IF NOT EXISTS action VARCHAR(32) DEFAULT 'teach'",
            "ALTER TABLE training_corrections ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE training_corrections ALTER COLUMN call_id DROP NOT NULL",
            # Never force future AMD from phone teaches — deactivate any legacy overrides
            "UPDATE training_overrides SET is_active = FALSE WHERE is_active = TRUE",
        ):
            try:
                conn.execute(text(stmt))
            except Exception as col_exc:
                print(f"OpenAMD schema note: {col_exc}")


def init_db():
    Base.metadata.create_all(bind=engine)
    try:
        ensure_schema()
    except Exception as exc:
        print(f"OpenAMD WARNING: ensure_schema failed: {exc}")

    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == settings.ADMIN_USER).first()
        if not admin:
            admin = User(
                username=settings.ADMIN_USER,
                email=settings.ADMIN_EMAIL,
                hashed_password=hash_password(settings.ADMIN_PASS),
                full_name="OpenAMD Administrator",
                role="superadmin",
                is_active=True,
            )
            db.add(admin)
            db.commit()
            print(f"OpenAMD: created admin user '{settings.ADMIN_USER}'")
        else:
            print(f"OpenAMD: admin user '{settings.ADMIN_USER}' already exists")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_dirs()
    try:
        init_db()
    except Exception as exc:
        # Log loudly but still start so /api/health can report DB errors
        print(f"OpenAMD WARNING: init_db failed: {exc}")
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Offline AI Answering Machine Detection platform for VICIdial",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(servers.router)
app.include_router(analyze.router)
app.include_router(reports.router)
app.include_router(training.router)
app.include_router(recordings.router)
app.include_router(system.router)
app.include_router(maintenance.router)
app.include_router(cron.router)
app.include_router(settings_router.router)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/health")
def health():
    db_ok = "ok"
    redis_ok = "unknown"
    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception as exc:
        db_ok = f"error: {exc}"

    try:
        import redis

        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, socket_connect_timeout=1)
        r.ping()
        redis_ok = "ok"
    except Exception:
        redis_ok = "unavailable"

    try:
        from app.ai.engine import ENGINE_INFO as _ENGINE_INFO
    except Exception:
        _ENGINE_INFO = {
            "name": "OpenAMD Hybrid (Heuristic + Silero)",
            "model": "Rule-based acoustic features + Silero VAD ONNX",
            "version": "4.0.0",
            "runtime": "NumPy + SoundFile + ONNX Runtime (Silero)",
        }

    return {
        "status": "ok" if db_ok == "ok" else "degraded",
        "version": settings.APP_VERSION,
        "database": db_ok,
        "redis": redis_ok,
        "amd_engine": _ENGINE_INFO,
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard.php", response_class=HTMLResponse)
@app.get("/livecalls.php", response_class=HTMLResponse)
@app.get("/cdr.php", response_class=HTMLResponse)
@app.get("/vicidialservers.php", response_class=HTMLResponse)
@app.get("/reports.php", response_class=HTMLResponse)
@app.get("/training.php", response_class=HTMLResponse)
@app.get("/training-history.php", response_class=HTMLResponse)
@app.get("/settings.php", response_class=HTMLResponse)
@app.get("/wipe.php", response_class=HTMLResponse)
@app.get("/audio.php", response_class=HTMLResponse)
@app.get("/cronjob.php", response_class=HTMLResponse)
def portal_index():
    index = TEMPLATES_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>OpenAMD</h1><p>Portal templates missing.</p>")


@app.get("/favicon.ico")
def favicon():
    icon = STATIC_DIR / "favicon.ico"
    if icon.exists():
        return FileResponse(icon)
    return HTMLResponse("")
