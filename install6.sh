#!/bin/bash
# Phase 6 — Nginx:
#   Port 80  = web portal (public)
#   Port 2130 = AMD API for VICIdial dialers only (firewall allowlist)

set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"

AMD_PORT="${AMD_PORT:-2130}"

log "[6/7] Configuring Nginx — portal :${NGINX_PORT}, AMD API :${AMD_PORT}..."

# Ensure AMD port is allowed by nginx (no package change needed)
# Shared rate-limit zones
cat >/etc/nginx/conf.d/openamd_limits.conf <<'EOF'
# OpenAMD — request rate limits
limit_req_zone $binary_remote_addr zone=openamd_login:10m rate=5r/m;
limit_req_zone $binary_remote_addr zone=openamd_api:10m rate=60r/s;
limit_req_zone $binary_remote_addr zone=openamd_general:10m rate=20r/s;
limit_conn_zone $binary_remote_addr zone=openamd_conn:10m;
EOF

cat >/etc/nginx/sites-available/openamd <<EOF
# ---------------------------------------------------------------------------
# Portal — public (port ${NGINX_PORT})
# VICIdial AMD endpoints are NOT served here (use port ${AMD_PORT}).
# ---------------------------------------------------------------------------
server {
    listen ${NGINX_PORT} default_server;
    listen [::]:${NGINX_PORT} default_server;
    server_name _;

    server_tokens off;
    client_max_body_size 10M;
    client_body_timeout 30s;
    client_header_timeout 30s;

    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
    add_header X-XSS-Protection "1; mode=block" always;

    limit_conn openamd_conn 40;

    # Block dialer AMD traffic on the public portal port
    location /api/v1/ {
        default_type application/json;
        return 403 '{"detail":"AMD API is on port ${AMD_PORT}. Point OPENAMD_URL to http://HOST:${AMD_PORT}/api/v1/analyze"}';
    }

    location = /api/login {
        limit_req zone=openamd_login burst=3 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:${API_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 60s;
    }

    location / {
        limit_req zone=openamd_general burst=40 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:${API_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 60s;
    }
}

# ---------------------------------------------------------------------------
# AMD API — VICIdial dialers (port ${AMD_PORT})
# Firewall: allow only dialer IPs to this port.
# ---------------------------------------------------------------------------
server {
    listen ${AMD_PORT} default_server;
    listen [::]:${AMD_PORT} default_server;
    server_name _;

    server_tokens off;
    client_max_body_size 10M;
    client_body_timeout 30s;
    client_header_timeout 30s;

    limit_conn openamd_conn 200;

    # Dialer AMD + admit + public health for AGI failover checks
    # Full-call SCAM uploads (larger body) — same auth as analyze
    location /api/v1/scam/ {
        limit_req zone=openamd_api burst=40 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:${API_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$remote_addr;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 300s;
        client_max_body_size 200M;
    }

    location /api/v1/ {
        limit_req zone=openamd_api burst=120 nodelay;
        limit_req_status 429;
        proxy_pass http://127.0.0.1:${API_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$remote_addr;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 60s;
        client_max_body_size 10M;
    }

    location = /api/health {
        proxy_pass http://127.0.0.1:${API_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$remote_addr;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 10s;
    }

    # No portal UI on the AMD port
    location / {
        default_type application/json;
        return 404 '{"detail":"Portal is on port ${NGINX_PORT}. AMD API is /api/v1/analyze on this port."}';
    }
}
EOF

rm -f /etc/nginx/sites-enabled/default
ln -sfn /etc/nginx/sites-available/openamd /etc/nginx/sites-enabled/openamd

nginx -t
systemctl enable nginx
systemctl restart nginx

# Best-effort open AMD port in ufw if enabled (still restrict by source IP yourself)
if command -v ufw >/dev/null 2>&1 && ufw status 2>/dev/null | grep -qi 'Status: active'; then
  ufw allow "${NGINX_PORT}/tcp" comment 'OpenAMD portal' >/dev/null 2>&1 || true
  echo ""
  echo "NOTE: ufw is active. Allow dialer IPs only to port ${AMD_PORT}, e.g.:"
  echo "  ufw allow from DIALER_PUBLIC_IP to any port ${AMD_PORT} proto tcp comment 'OpenAMD VICIdial'"
  echo "  ufw reload"
fi

log "Phase 6 complete:"
echo "  Portal (public):  http://SERVER_IP:${NGINX_PORT}/"
echo "  AMD API (dialers): http://SERVER_IP:${AMD_PORT}/api/v1/analyze"
echo "  Firewall: open ${NGINX_PORT} publicly; allow ${AMD_PORT} only from VICIdial IPs."
