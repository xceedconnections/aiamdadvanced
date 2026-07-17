"""Helpers to resolve and store call recordings under RECORDINGS_DIR."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from app.config import get_settings
from app.models.call import CallAnalysis


def recordings_root() -> Path:
    raw = (get_settings().RECORDINGS_DIR or "/opt/openamd/recordings").strip()
    path = Path(raw).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def ensure_recordings_dir() -> str:
    root = recordings_root()
    os.makedirs(root, exist_ok=True)
    os.makedirs(root / "by_callid", exist_ok=True)
    os.makedirs(root / "by_id", exist_ok=True)
    return str(root)


def safe_callid_token(call_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in (call_id or ""))[:80]


def callid_match_tokens(call_id: str) -> list[str]:
    raw = (call_id or "").strip()
    if not raw:
        return []
    tokens = {
        raw,
        safe_callid_token(raw),
        raw.replace("-", "_"),
        safe_callid_token(raw).replace("-", "_"),
    }
    if "." in raw:
        tokens.add(raw.split(".", 1)[0])
    if "-" in raw:
        tokens.add(raw.split("-", 1)[0])
    return [t for t in tokens if t and len(t) >= 4]


def _path_if_usable(path: Path) -> Optional[Path]:
    try:
        if path.is_file() and path.stat().st_size > 50:
            return path if path.is_absolute() else path.resolve()
    except OSError:
        return None
    return None


def list_recording_index() -> list[Path]:
    root = recordings_root()
    if not root.is_dir():
        return []
    out: list[Path] = []
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".wav", ".wave", ".mp3", ".gsm"):
                out.append(p)
    except OSError:
        pass
    return out


def by_callid_path(call_id: str) -> Path:
    token = safe_callid_token(call_id) or "unknown"
    return recordings_root() / "by_callid" / f"{token}.wav"


def by_analysis_path(analysis_id: int) -> Path:
    return recordings_root() / "by_id" / f"{int(analysis_id)}.wav"


def save_call_audio(raw: bytes, call_id: str, server_id: int) -> str:
    """
    Save browser-playable WAV to dated folder + stable by_callid link.
    Returns absolute path of primary dated file.
    """
    import uuid
    from datetime import datetime

    raw = to_browser_wav(raw)
    root = ensure_recordings_dir()
    day = datetime.utcnow().strftime("%Y%m%d")
    day_dir = os.path.join(root, day)
    os.makedirs(day_dir, exist_ok=True)

    safe = safe_callid_token(call_id) or "unknown"
    fname = f"{server_id}_{safe}_{uuid.uuid4().hex[:8]}.wav"
    primary = os.path.abspath(os.path.join(day_dir, fname))

    with open(primary, "wb") as f:
        f.write(raw)

    flat = by_callid_path(call_id)
    try:
        shutil.copy2(primary, flat)
    except OSError as exc:
        print(f"OpenAMD WARNING: by_callid copy failed: {exc}")

    return primary


def to_browser_wav(raw: bytes) -> bytes:
    """
    Re-encode to PCM16 WAV so Chrome/Firefox can play it.
    Asterisk often sends ulaw/alaw/gsm WAV that browsers reject.
    """
    if not raw or len(raw) < 50:
        return raw
    try:
        import io
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:
            data = np.mean(data, axis=1)
        buf = io.BytesIO()
        sf.write(buf, data, int(sr), format="WAV", subtype="PCM_16")
        out = buf.getvalue()
        return out if len(out) > 50 else raw
    except Exception as exc:
        print(f"OpenAMD WARNING: wav re-encode failed, storing original: {exc}")
        return raw


def link_analysis_recording(analysis_id: int, source_path: str) -> Optional[str]:
    """Copy/link saved WAV to by_id/{analysis_id}.wav for portal play."""
    if not source_path:
        return None
    src = Path(source_path)
    if not src.is_file():
        return None
    ensure_recordings_dir()
    dest = by_analysis_path(analysis_id)
    try:
        shutil.copy2(src, dest)
        return str(dest.resolve())
    except OSError as exc:
        print(f"OpenAMD WARNING: by_id copy failed: {exc}")
        return str(src)


def resolve_recording_path(
    call: CallAnalysis,
    index: Optional[list[Path]] = None,
) -> Optional[Path]:
    """Return the audio path for a call if it exists on disk."""
    root = recordings_root()

    # 1) by analysis id (most reliable for portal)
    if call.id:
        found = _path_if_usable(by_analysis_path(call.id))
        if found:
            return found

    # 2) exact DB path
    if call.audio_path:
        p = Path(str(call.audio_path).strip())
        found = _path_if_usable(p)
        if found:
            return found
        found = _path_if_usable(root / p)
        if found:
            return found
        if p.name:
            candidates = index if index is not None else list_recording_index()
            for c in candidates:
                if c.name == p.name:
                    got = _path_if_usable(c)
                    if got:
                        return got

    # 3) stable by_callid/{callid}.wav
    found = _path_if_usable(by_callid_path(call.call_id))
    if found:
        return found

    # 4) scan filenames containing callid tokens
    tokens = callid_match_tokens(call.call_id)
    if not tokens:
        return None

    candidates = index if index is not None else list_recording_index()
    if not candidates:
        return None

    matches: list[Path] = []
    for path in candidates:
        name = path.name
        for token in tokens:
            if token in name:
                matches.append(path)
                break

    if not matches:
        return None

    if call.server_id is not None:
        prefix = f"{call.server_id}_"
        prefixed = [p for p in matches if p.name.startswith(prefix)]
        if prefixed:
            matches = prefixed

    matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return _path_if_usable(matches[0])


def resolve_recording_path_fast(call: CallAnalysis) -> Optional[Path]:
    """Quick path lookup without scanning the whole recordings tree."""
    if call.id:
        found = _path_if_usable(by_analysis_path(call.id))
        if found:
            return found
    if call.audio_path:
        p = Path(str(call.audio_path).strip())
        found = _path_if_usable(p)
        if found:
            return found
        found = _path_if_usable(recordings_root() / p)
        if found:
            return found
    return _path_if_usable(by_callid_path(call.call_id))


def recording_meta_fast(call: CallAnalysis) -> dict:
    """Lightweight meta for list/CDR pages (no directory-wide scan)."""
    path = resolve_recording_path_fast(call)
    in_db = bool(getattr(call, "audio_saved", False))
    if path:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return {
            "has_recording": True,
            "recording_filename": path.name,
            "recording_bytes": size,
            "audio_path": str(path),
        }
    if in_db:
        return {
            "has_recording": True,
            "recording_filename": f"{call.call_id}.wav",
            "recording_bytes": 0,
            "audio_path": "db:audio_blob",
        }
    return {
        "has_recording": bool(call.audio_path),
        "recording_filename": Path(call.audio_path).name if call.audio_path else None,
        "recording_bytes": 0,
        "audio_path": (call.audio_path or ""),
    }


def recording_meta(call: CallAnalysis, index: Optional[list[Path]] = None) -> dict:
    path = resolve_recording_path(call, index=index)
    in_db = bool(getattr(call, "audio_saved", False))
    if path:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return {
            "has_recording": True,
            "recording_filename": path.name,
            "recording_bytes": size,
            "audio_path": str(path),
        }
    if in_db:
        return {
            "has_recording": True,
            "recording_filename": f"{call.call_id}.wav",
            "recording_bytes": 0,
            "audio_path": "db:audio_blob",
        }
    return {
        "has_recording": False,
        "recording_filename": None,
        "recording_bytes": 0,
        "audio_path": (call.audio_path or ""),
    }


def get_call_audio_bytes(call: CallAnalysis) -> Optional[bytes]:
    """Prefer DB blob (browser PCM), else disk file."""
    blob = getattr(call, "audio_blob", None)
    if blob:
        return bytes(blob)
    path = resolve_recording_path(call)
    if path and path.is_file():
        try:
            data = path.read_bytes()
            return to_browser_wav(data)
        except OSError:
            return None
    return None


def repair_call_recording(call: CallAnalysis, index: Optional[list[Path]] = None) -> bool:
    """If a file is found, ensure by_id + by_callid links and audio_path are set."""
    path = resolve_recording_path(call, index=index)
    if not path:
        # Still "ok" if audio lives only in DB
        return bool(getattr(call, "audio_saved", False))
    ensure_recordings_dir()
    try:
        shutil.copy2(path, by_callid_path(call.call_id))
    except OSError:
        pass
    if call.id:
        try:
            shutil.copy2(path, by_analysis_path(call.id))
        except OSError:
            pass
    call.audio_path = str(path)
    return True
