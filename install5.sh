#!/bin/bash
# Phase 5 — systemd service + daily recording cleanup timer

set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"

log "[5/7] Installing systemd service openamd..."

cat >/etc/systemd/system/openamd.service <<EOF
[Unit]
Description=OpenAMD AI AMD API + Portal
After=network.target postgresql.service redis-server.service
Wants=postgresql.service redis-server.service

[Service]
Type=simple
User=root
WorkingDirectory=${BACKEND}
Environment=PYTHONPATH=${BACKEND}
EnvironmentFile=${BACKEND}/.env
ExecStart=${VENV}/bin/uvicorn app.main:app --host 0.0.0.0 --port ${API_PORT} --workers 1 --log-level info
Restart=always
RestartSec=3
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

# Daily auto-delete of old recordings (retention days set in portal Cron Job page)
cat >/etc/systemd/system/openamd-cleanup.service <<EOF
[Unit]
Description=OpenAMD recording retention cleanup
After=network.target

[Service]
Type=oneshot
User=root
WorkingDirectory=${BACKEND}
Environment=PYTHONPATH=${BACKEND}
EnvironmentFile=${BACKEND}/.env
ExecStart=${VENV}/bin/python ${BACKEND}/scripts/cleanup_recordings.py
EOF

cat >/etc/systemd/system/openamd-cleanup.timer <<EOF
[Unit]
Description=Run OpenAMD recording cleanup daily

[Timer]
OnCalendar=*-*-* 02:15:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

# Default cron settings if missing
if [[ ! -f "${APP_ROOT}/cron_settings.json" ]]; then
  cat >"${APP_ROOT}/cron_settings.json" <<EOF
{
  "enabled": true,
  "retention_days": 7
}
EOF
fi

systemctl daemon-reload
systemctl enable openamd
systemctl enable --now openamd-cleanup.timer
systemctl restart openamd

sleep 2
systemctl is-active --quiet openamd || {
  journalctl -u openamd -n 40 --no-pager
  die "openamd service failed to start"
}

log "Phase 5 complete — openamd service running; cleanup timer enabled (daily 02:15)."
