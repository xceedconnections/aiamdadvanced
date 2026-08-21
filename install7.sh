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

log "Firewall reminder:"
echo "  # Portal (public)"
echo "  ufw allow ${NGINX_PORT}/tcp"
echo "  # AMD API — dialer IPs only (example)"
echo "  ufw allow from VICIBOX_IP to any port ${AMD_PORT} proto tcp"
echo "  ufw reload"
echo ""
echo "  Portal:  http://${IP}/"
echo "  AMD:     http://${IP}:${AMD_PORT}/api/v1/analyze"

log "Phase 7 complete — OpenAMD portal at http://${IP}/  |  AMD API on :${AMD_PORT}"
