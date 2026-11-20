"""Web UI portal self-update from GitHub (admin only)."""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth.security import get_current_user
from app.config import get_settings
from app.models.user import User

router = APIRouter(prefix="/api/system/update", tags=["update"])

GITHUB_REPO = "xceedconnections/aiamdadvanced"
GITHUB_BRANCH = "main"
DEFAULT_SOURCE_DIRS = (
    "/root/aiamdadvanced",
    "/opt/openamd/src",
)
REVISION_FILE = Path("/opt/openamd/git_revision.meta")
UPDATE_LOG = Path("/opt/openamd/logs/web_update.log")
UPDATE_LOCK = Path("/opt/openamd/logs/web_update.lock")


class UpdateApplyRequest(BaseModel):
    confirm: str = Field("", max_length=64)


def _require_admin(user: User) -> None:
    role = str(getattr(user, "role", "") or "").strip().lower()
    if role not in ("superadmin", "admin"):
        raise HTTPException(
            status_code=403,
            detail=f"Admin role required (your role: {role or 'none'})",
        )


def _run_git(cwd: Path, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()
    except FileNotFoundError:
        return 127, "", "git not found"
    except subprocess.TimeoutExpired:
        return 124, "", "git timed out"


def _find_source_dir() -> Path | None:
    env = (os.environ.get("OPENAMD_SOURCE_DIR") or "").strip()
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(Path(p) for p in DEFAULT_SOURCE_DIRS)
    # Repo next to this running backend (dev / unusual layouts)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() and (parent / "files" / "backend" / "app" / "main.py").exists():
            candidates.append(parent)
            break
        if (parent / ".git").exists() and (parent / "backend" / "app" / "main.py").exists():
            candidates.append(parent)
            break
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir() and (path / ".git").exists():
            return path
    return None


def _parse_revision_meta(text: str) -> dict[str, str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    full = lines[0] if len(lines) > 0 else ""
    short = lines[1] if len(lines) > 1 else (full[:7] if full else "")
    subject = lines[2] if len(lines) > 2 else ""
    date = lines[3] if len(lines) > 3 else ""
    return {
        "commit": full,
        "short": short,
        "subject": subject,
        "date": date,
    }


def _local_revision(source: Path | None) -> dict[str, Any]:
    if source is not None:
        code, out, err = _run_git(
            source,
            "log",
            "-1",
            "--format=%H%n%h%n%s%n%ci",
        )
        if code == 0 and out:
            meta = _parse_revision_meta(out)
            meta["source"] = "git"
            meta["source_dir"] = str(source)
            return meta
        return {
            "commit": "",
            "short": "",
            "subject": "",
            "date": "",
            "source": "git_error",
            "source_dir": str(source),
            "error": err or "git log failed",
        }
    if REVISION_FILE.exists():
        try:
            meta = _parse_revision_meta(REVISION_FILE.read_text(encoding="utf-8"))
            meta["source"] = "revision_file"
            meta["source_dir"] = ""
            return meta
        except OSError as exc:
            return {
                "commit": "",
                "short": "",
                "subject": "",
                "date": "",
                "source": "revision_file_error",
                "error": str(exc),
            }
    return {
        "commit": "",
        "short": "",
        "subject": "",
        "date": "",
        "source": "unknown",
            "error": "No /root/aiamdadvanced git clone and no /opt/openamd/git_revision.meta",
    }


def _github_latest() -> dict[str, Any]:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits/{GITHUB_BRANCH}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "OpenAMD-Portal-Update",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        commit = data.get("commit") or {}
        author = commit.get("author") or {}
        return {
            "ok": True,
            "commit": str(data.get("sha") or ""),
            "short": str(data.get("sha") or "")[:7],
            "subject": str((commit.get("message") or "").splitlines()[0] if commit.get("message") else ""),
            "date": str(author.get("date") or ""),
            "html_url": str(data.get("html_url") or ""),
            "repo": GITHUB_REPO,
            "branch": GITHUB_BRANCH,
        }
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        return {
            "ok": False,
            "error": f"GitHub HTTP {exc.code}: {body or exc.reason}",
            "repo": GITHUB_REPO,
            "branch": GITHUB_BRANCH,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "repo": GITHUB_REPO,
            "branch": GITHUB_BRANCH,
        }


def _update_busy() -> bool:
    if not UPDATE_LOCK.exists():
        return False
    try:
        age = time.time() - UPDATE_LOCK.stat().st_mtime
        # Stale lock after 15 minutes
        if age > 900:
            return False
        return True
    except OSError:
        return False


def _tail_log(max_lines: int = 40) -> str:
    if not UPDATE_LOG.exists():
        return ""
    try:
        lines = UPDATE_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except OSError:
        return ""


@router.get("/status")
def update_status(user: User = Depends(get_current_user)):
    _require_admin(user)
    settings = get_settings()
    source = _find_source_dir()
    local = _local_revision(source)
    remote = _github_latest()
    local_sha = str(local.get("commit") or "").strip().lower()
    remote_sha = str(remote.get("commit") or "").strip().lower()
    update_available = bool(
        remote.get("ok")
        and local_sha
        and remote_sha
        and local_sha != remote_sha
    )
    return {
        "app_version": settings.APP_VERSION,
        "repo": GITHUB_REPO,
        "branch": GITHUB_BRANCH,
        "source_dir": str(source) if source else "",
        "local": local,
        "remote": remote,
        "update_available": update_available,
        "up_to_date": bool(remote.get("ok") and local_sha and local_sha == remote_sha),
        "busy": _update_busy(),
        "log_tail": _tail_log(),
        "can_update": bool(source is not None and (source / "install4.sh").exists()),
    }


@router.post("/apply")
def update_apply(
    payload: UpdateApplyRequest,
    user: User = Depends(get_current_user),
):
    _require_admin(user)
    if str(payload.confirm or "").strip().upper() != "UPDATE":
        raise HTTPException(
            status_code=400,
            detail='Type UPDATE to confirm (confirm="UPDATE")',
        )
    if _update_busy():
        raise HTTPException(
            status_code=409,
            detail="An update is already running. Wait and refresh status.",
        )
    source = _find_source_dir()
    if source is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No git clone found. Expected /root/aiamdadvanced "
                "(or set OPENAMD_SOURCE_DIR)."
            ),
        )
    install4 = source / "install4.sh"
    if not install4.exists():
        raise HTTPException(
            status_code=400,
            detail=f"install4.sh missing under {source}",
        )

    UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    helper = Path("/opt/openamd/logs/web_update_run.sh")
    script = f"""#!/bin/bash
set -euo pipefail
SRC={json.dumps(str(source))}
LOG={json.dumps(str(UPDATE_LOG))}
LOCK={json.dumps(str(UPDATE_LOCK))}
META={json.dumps(str(REVISION_FILE))}
trap 'rm -f "$LOCK"' EXIT
touch "$LOCK"
exec >>"$LOG" 2>&1
echo ""
echo "=== $(date -Is) web update start by portal ==="
echo "Source: $SRC"
cd "$SRC"
find "$SRC" -type f -name '*.sh' -exec sed -i 's/\\r$//' {{}} + 2>/dev/null || true
git fetch origin
git checkout {GITHUB_BRANCH}
git pull --ff-only origin {GITHUB_BRANCH}
find "$SRC" -type f -name '*.sh' -exec sed -i 's/\\r$//' {{}} + 2>/dev/null || true
bash "$SRC/install4.sh"
git log -1 --format='%H%n%h%n%s%n%ci' > "$META" || true
echo "Restarting openamd…"
systemctl restart openamd || true
echo "=== $(date -Is) web update finished ==="
"""
    helper.write_text(script, encoding="utf-8")
    helper.chmod(0o755)
    UPDATE_LOCK.write_text(str(time.time()), encoding="utf-8")

    # Detach so restart does not kill the apply request mid-flight.
    subprocess.Popen(  # noqa: S603
        ["/bin/bash", str(helper)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "status": "started",
        "message": (
            "Update started. Portal will restart in ~30–90 seconds. "
            "Refresh this page when it comes back."
        ),
        "source_dir": str(source),
        "log_file": str(UPDATE_LOG),
    }
