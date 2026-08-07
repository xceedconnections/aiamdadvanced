#!/bin/bash
# Phase 3 — Python virtualenv and dependencies

set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"
ensure_project_dir

log "[3/7] Creating Python virtualenv..."

if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip wheel setuptools

REQ="${PROJECT_DIR}/backend/requirements.txt"
[[ -f "${REQ}" ]] || die "Missing ${REQ}"

log "Installing Python packages from requirements.txt..."
pip install -r "${REQ}"

# Known fix: passlib + bcrypt>=4.1 crashes login
pip install 'bcrypt==4.0.1' --force-reinstall

# Optional Faster-Whisper for ML pipeline low-confidence refine (best effort).
# Skipped on failure — XGBoost still works without Whisper when ML is enabled.
log "Optional: faster-whisper (ML pipeline Whisper stage)..."
pip install 'faster-whisper>=1.1.0' || echo "WARNING: faster-whisper not installed; ML pipeline will use XGBoost only"

# Hybrid engine uses Silero VAD via onnxruntime (already in requirements).
# Pre-download ONNX weights into /opt/openamd/models (best effort).
log "Downloading Silero VAD ONNX model (best effort)..."
mkdir -p "${APP_ROOT}/models"
python - <<'PY' || true
from pathlib import Path
import urllib.request
dest = Path("/opt/openamd/models/silero_vad.onnx")
urls = [
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://cdn.jsdelivr.net/gh/snakers4/silero-vad@master/src/silero_vad/data/silero_vad.onnx",
]
if dest.exists() and dest.stat().st_size > 50000:
    print(f"Silero model already present: {dest}")
else:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in urls:
        try:
            tmp = dest.with_suffix(".onnx.download")
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 50000:
                raise RuntimeError("file too small")
            tmp.replace(dest)
            print(f"Downloaded Silero VAD -> {dest}")
            break
        except Exception as exc:
            last = exc
    else:
        print(f"WARNING: Silero download failed ({last}); engine will retry at runtime")
PY

log "Phase 3 complete — Python environment ready."
