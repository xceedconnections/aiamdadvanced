#!/bin/bash
# Phase 7 — verify installation and create admin user via app startup

set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"

log "[7/7] Verifying installation..."

systemctl restart openamd
sleep 3

log "API health check..."
wait_for_health

log "Service status:"
systemctl status openamd --no-pager -l | head -20 || true

IP="$(server_ip)"

log "Firewall reminder (if ufw enabled):"
echo "  ufw allow ${NGINX_PORT}/tcp"
echo "  ufw allow from VICIBOX_IP to any port ${NGINX_PORT}"

log "Phase 7 complete — OpenAMD is ready at http://${IP}/"
