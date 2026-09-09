"""Shared recording file cleanup used by portal maintenance + cron."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, Sequence

from app.config import get_settings

logger = logging.getLogger("openamd.audio_cleanup")

_AUDIO_SUFFIXES = {".wav", ".wave", ".mp3", ".gsm", ".ulaw", ".alaw", ".sln", ".raw"}


def _safe_unlink(path: Path) -> tuple[bool, int]:
    """Delete one file. Returns (ok, bytes_freed)."""
    try:
        if not path.is_file():
            return False, 0
        size = path.stat().st_size
        path.unlink(missing_ok=True)
        return True, size
    except OSError as exc:
        logger.warning("Failed to delete recording %s: %s", path, exc)
        return False, 0


def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for dirpath, _dirnames, _filenames in os.walk(root, topdown=False):
        if dirpath == str(root):
            continue
        try:
            if not os.listdir(dirpath):
                os.rmdir(dirpath)
        except OSError:
            pass


def delete_paths(paths: Iterable[Path | str]) -> dict:
    """Delete a specific set of recording paths (best-effort)."""
    deleted_files = 0
    failed_files = 0
    freed_bytes = 0
    seen: set[str] = set()

    for raw in paths:
        if not raw:
            continue
        path = Path(str(raw))
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        ok, size = _safe_unlink(path)
        if ok:
            deleted_files += 1
            freed_bytes += size
        elif path.exists():
            failed_files += 1

    return {
        "deleted_files": deleted_files,
        "failed_files": failed_files,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "freed_gb": round(freed_bytes / (1024**3), 3),
    }


def delete_recordings_for_calls(calls: Sequence[object]) -> dict:
    """
    Remove on-disk WAV copies for CallAnalysis rows (dated + by_id + by_callid).
    Used when wiping SQL logs so files are not left orphaned.
    """
    from app.recordings import by_analysis_path, by_callid_path, recordings_root

    root = recordings_root()
    targets: list[Path] = []

    for call in calls:
        audio_path = getattr(call, "audio_path", None) or ""
        if audio_path and not str(audio_path).startswith("db:"):
            p = Path(str(audio_path))
            targets.append(p)
            if not p.is_absolute():
                targets.append(root / p)

        call_id = getattr(call, "call_id", None)
        if call_id:
            targets.append(by_callid_path(str(call_id)))

        analysis_id = getattr(call, "id", None)
        if analysis_id:
            targets.append(by_analysis_path(int(analysis_id)))

    result = delete_paths(targets)
    prune_empty_dirs(root)
    result["recordings_dir"] = str(root)
    return result


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
            "failed_files": 0,
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
    failed_files = 0
    freed_bytes = 0
    skipped_non_audio = 0

    # Collect first so directory mutation during walk is safe
    candidates: list[Path] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Delete known audio + any file when wiping everything
            suffix = path.suffix.lower()
            if older_than_days is not None and suffix not in _AUDIO_SUFFIXES:
                # Retention mode: only audio-like files
                if suffix not in {".json", ".txt", ".log", ".tmp"}:
                    skipped_non_audio += 1
                continue
            candidates.append(path)
    except OSError as exc:
        logger.warning("Failed scanning recordings dir %s: %s", root, exc)

    for path in candidates:
        try:
            if cutoff is not None:
                mtime = datetime.utcfromtimestamp(path.stat().st_mtime)
                if mtime >= cutoff:
                    continue
            ok, size = _safe_unlink(path)
            if ok:
                deleted_files += 1
                freed_bytes += size
            elif path.exists():
                failed_files += 1
        except OSError as exc:
            failed_files += 1
            logger.warning("Failed to delete recording %s: %s", path, exc)

    prune_empty_dirs(root)

    message = "ok"
    if failed_files:
        message = f"Deleted {deleted_files} file(s); {failed_files} could not be removed (check permissions)"

    return {
        "ok": failed_files == 0,
        "deleted_files": deleted_files,
        "failed_files": failed_files,
        "freed_bytes": freed_bytes,
        "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        "freed_gb": round(freed_bytes / (1024**3), 3),
        "older_than_days": older_than_days,
        "recordings_dir": str(root),
        "skipped_non_audio": skipped_non_audio,
        "message": message,
        "at": datetime.utcnow().isoformat() + "Z",
    }
