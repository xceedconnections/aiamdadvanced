#!/bin/bash
# Phase 1 — system packages and base directories

set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"

log "[1/7] Updating system and installing packages..."

apt-get update -y
apt-get upgrade -y

apt-get install -y \
  python3 \
  python3-pip \
  python3-venv \
  git \
  ffmpeg \
  sox \
  build-essential \
  nginx \
  redis-server \
  postgresql \
  postgresql-contrib \
  curl \
  wget \
  unzip \
  htop \
  net-tools \
  rsync \
  libpq-dev \
  libsndfile1 \
  libffi-dev \
  libssl-dev

systemctl enable redis-server
systemctl start redis-server

systemctl enable postgresql
systemctl start postgresql

log "Creating ${APP_ROOT} directories..."
mkdir -p "${APP_ROOT}"/{backend,install,models,recordings,logs,database,worker,docs}

log "Phase 1 complete — system packages ready."
