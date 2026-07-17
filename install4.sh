#!/bin/bash
# Phase 4 — deploy backend + portal and .env

set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"
ensure_project_dir

log "[4/7] Deploying OpenAMD application (API + portal) to ${APP_ROOT}..."

[[ -f "${PROJECT_DIR}/backend/app/main.py" ]] \
  || die "Missing backend at ${PROJECT_DIR}/backend (run install/pack.ps1 first)"

[[ -f "${PROJECT_DIR}/backend/app/templates/index.html" ]] \
  || die "Missing portal template: backend/app/templates/index.html"

[[ -d "${PROJECT_DIR}/backend/app/static" ]] \
  || die "Missing portal static files: backend/app/static"

rsync -a --delete \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'recordings/*' \
  --exclude 'logs/*' \
  --exclude 'openamd*.tar.gz' \
  "${PROJECT_DIR}/backend/" "${BACKEND}/"

mkdir -p "${APP_ROOT}/recordings" "${APP_ROOT}/models" "${APP_ROOT}/logs" "${APP_ROOT}/install"
rsync -a \
  --exclude 'files' \
  --exclude '*.tar.gz' \
  "${INSTALL_DIR}/" "${APP_ROOT}/install/" 2>/dev/null || true

ENV_FILE="${BACKEND}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ -f "${PROJECT_DIR}/backend/.env" ]]; then
    cp "${PROJECT_DIR}/backend/.env" "${ENV_FILE}"
  elif [[ -f "${INSTALL_DIR}/openamd.env" ]]; then
    cp "${INSTALL_DIR}/openamd.env" "${ENV_FILE}"
  else
    cat >"${ENV_FILE}" <<EOF
DB_HOST=127.0.0.1
DB_PORT=5432
DB_NAME=${DB_NAME}
DB_USER=${DB_USER}
DB_PASS=${DB_PASS}

REDIS_HOST=127.0.0.1
REDIS_PORT=6379

API_HOST=0.0.0.0
API_PORT=${API_PORT}

JWT_SECRET=OpenAMD-JWT-Secret-Change-In-Production-2024

ADMIN_USER=${ADMIN_USER}
ADMIN_PASS=${ADMIN_PASS}
ADMIN_EMAIL=${ADMIN_EMAIL}

RECORDINGS_DIR=${APP_ROOT}/recordings
MODELS_DIR=${APP_ROOT}/models
LOGS_DIR=${APP_ROOT}/logs
EOF
  fi
fi

strip_crlf "${ENV_FILE}"

log "Testing Python import..."
export PYTHONPATH="${BACKEND}"
"${VENV}/bin/python" -c "from app.main import app; print('IMPORT OK:', app.title)"

log "Phase 4 complete — backend + portal deployed."
