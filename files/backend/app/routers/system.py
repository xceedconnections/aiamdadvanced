import os
import shutil
import time

import psutil
from fastapi import APIRouter, Depends

from app.ai.engine import ENGINE_INFO
from app.auth.security import get_current_user
from app.config import get_settings
from app.models.user import User

router = APIRouter(prefix="/api/system", tags=["system"])
settings = get_settings()


def _gb(value: int) -> float:
    return round(value / (1024**3), 2)


@router.get("/stats")
def system_stats(user: User = Depends(get_current_user)):
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    load_1, load_5, load_15 = os.getloadavg()
    boot_time = psutil.boot_time()

    cpu_count = psutil.cpu_count(logical=True) or 1
    cpu_percent = round(psutil.cpu_percent(interval=0.15), 1)
    status = "ok"
    if cpu_percent >= 90 or load_1 >= cpu_count * 1.5 or memory.percent >= 90:
        status = "warning"
    if cpu_percent >= 98 or memory.percent >= 97 or (disk.used / disk.total) >= 0.95:
        status = "critical"

    engine = dict(ENGINE_INFO)
    return {
        "status": status,
        "cpu_percent": cpu_percent,
        "cpu_count": cpu_count,
        "load_1": round(load_1, 2),
        "load_5": round(load_5, 2),
        "load_15": round(load_15, 2),
        "ram_total_gb": _gb(memory.total),
        "ram_used_gb": _gb(memory.used),
        "ram_available_gb": _gb(memory.available),
        "ram_percent": round(memory.percent, 1),
        "disk_total_gb": _gb(disk.total),
        "disk_used_gb": _gb(disk.used),
        "disk_available_gb": _gb(disk.free),
        "disk_free_gb": _gb(disk.free),
        "disk_percent": round((disk.used / disk.total) * 100, 1),
        "uptime_seconds": max(0, int(time.time() - boot_time)),
        "application_version": settings.APP_VERSION,
        "amd_engine": engine,
        # Flat fields for older portal JS / debugging
        "amd_engine_name": engine.get("name", "Unknown"),
        "amd_engine_model": engine.get("model", "Unknown"),
        "amd_engine_version": engine.get("version", "Unknown"),
        "amd_engine_runtime": engine.get("runtime", "Unknown"),
    }
