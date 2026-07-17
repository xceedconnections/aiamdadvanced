#!/bin/bash
# Shared variables and helpers for OpenAMD installation.
# shellcheck disable=SC2034

set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APP_ROOT="/opt/openamd"
VENV="${APP_ROOT}/venv"
BACKEND="${APP_ROOT}/backend"

DB_NAME="openamd"
DB_USER="openamd"
DB_PASS="Openaccount@123"

ADMIN_USER="admin"
ADMIN_PASS="Openaccount@123"
ADMIN_EMAIL="admin@openamd.local"

API_PORT="8000"
NGINX_PORT="80"

export DEBIAN_FRONTEND=noninteractive

log() {
  echo ""
  echo ">>> $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_root() {
  [[ "$(id -u)" -eq 0 ]] || die "Run as root: sudo bash install.sh"
}

require_ubuntu() {
  if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    source /etc/os-release
    if [[ "${ID:-}" != "ubuntu" ]]; then
      echo "WARNING: This installer targets Ubuntu 24.04. Detected: ${PRETTY_NAME:-unknown}"
      read -r -p "Continue anyway? [y/N] " ans
      [[ "${ans,,}" == "y" ]] || exit 1
    fi
  fi
}

strip_crlf() {
  local f
  for f in "$@"; do
    [[ -f "$f" ]] && sed -i 's/\r$//' "$f" || true
  done
}

detect_project_dir() {
  local candidates=()

  candidates+=("$(cd "${INSTALL_DIR}/.." && pwd)")
  candidates+=("${INSTALL_DIR}/files")
  candidates+=("/root/AIAMD")
  candidates+=("/root/openamd")
  candidates+=("/root")

  for dir in "${candidates[@]}"; do
    if [[ -f "${dir}/backend/app/main.py" ]]; then
      PROJECT_DIR="$(cd "${dir}" && pwd)"
      return 0
    fi
  done

  die "Cannot find OpenAMD source (backend/app/main.py).
Upload either:
  1) full AIAMD project to /root/AIAMD, or
  2) self-contained install folder with install/files/backend (API + portal).

Recommended:
  powershell -ExecutionPolicy Bypass -File install/pack.ps1
  # or: bash install/pack.sh
  scp openamd-install.tar.gz root@SERVER:/root/
  tar -xzf /root/openamd-install.tar.gz -C /root/
  bash /root/install/install.sh"
}

ensure_project_dir() {
  if [[ -z "${PROJECT_DIR:-}" ]]; then
    detect_project_dir
  fi
  log "Project source: ${PROJECT_DIR}"
}

server_ip() {
  local ip
  ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  echo "${ip:-YOUR_SERVER_IP}"
}

wait_for_health() {
  local url="http://127.0.0.1:${API_PORT}/api/health"
  local i
  for i in $(seq 1 20); do
    if curl -fsS "${url}" >/tmp/openamd_health.json 2>/dev/null; then
      cat /tmp/openamd_health.json
      echo ""
      return 0
    fi
    echo "  waiting for API (${i}/20)..."
    sleep 2
  done
  die "API did not become healthy. Check: journalctl -u openamd -n 80 --no-pager"
}
