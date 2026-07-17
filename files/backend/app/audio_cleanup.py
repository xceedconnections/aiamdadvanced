"""Shared recording file cleanup used by portal maintenance + cron."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.config import get_settings


def delete_old_recordings(older_than_days: Optional[int]) -> dict:
    """
    Delete WAV/audio files under RECORDINGS_DIR.
    older_than_days=None deletes all files.
    """
    settings = get_settings()
    root = Path(settings.RECORDINGS_DIR)
    if not root.exists():
        return {
            "ok": True,
            "deleted_files": 0,
            "freed_bytes": 0,
            "freed_mb": 0.0,
            "freed_gb": 0.0,
            "older_than_days": older_than_days,
            "recordings_dir": str(root),
            "message": "Recordings directory does not exist",
        }

    cutoff = None
    if older_than_days is not None:
        if older_than_days < 0:
            raise ValueError("older_than_days must be >= 0")
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)

    deleted_files = 0
    freed_bytes = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if cutoff is not None:
                mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
                if mtime >= cutoff:
                    continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            deleted_files += 1
            freed_bytes += size
        except OSError:
            continue

    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath == str(root):
            continue
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass

    return {
        "ok": True,
        "deleted_files": deleted_files,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "freed_gb": round(freed_bytes / (1024**3), 3),
        "older_than_days": older_than_days,
        "recordings_dir": str(root),
        "at": datetime.utcnow().isoformat() + "Z",
    }
