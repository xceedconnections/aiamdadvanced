#!/bin/bash
# Phase 6 — Nginx reverse proxy (port 80) with basic hardening

set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"

log "[6/7] Configuring Nginx (security headers + rate limits)..."

# Shared rate-limit zones (idempotent overwrite of snippet)
cat >/etc/nginx/conf.d/openamd_limits.conf <<'EOF'
# OpenAMD Advanced — request rate limits
limit_req_zone $binary_remote_addr zone=openamd_login:10m rate=5r/m;
limit_req_zone $binary_remote_addr zone=openamd_api:10m rate=30r/s;
limit_req_zone $binary_remote_addr zone=openamd_general:10m rate=20r/s;
limit_conn_zone $binary_remote_addr zone=openamd_conn:10m;
EOF

cat >/etc/nginx/sites-available/openamd <<EOF
server {
    listen ${NGINX_PORT} default_server;
    listen [::]:${NGINX_PORT} default_server;
    server_name _;

    server_tokens off;
    client_max_body_size 10M;
    client_body_timeout 30s;
    client_header_timeout 30s;

    # Security headers (portal on port ${NGINX_PORT})
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
    add_header X-XSS-Protection "1; mode=block" always;

    limit_conn openamd_conn 40;

    # Login — slow brute force
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

    # Analyze API — allow legitimate dialer bursts, cap floods
    # Use \$remote_addr only for forwarded IP (ignore client-supplied XFF spoofing)
    location /api/v1/ {
        limit_req zone=openamd_api burst=60 nodelay;
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
EOF

rm -f /etc/nginx/sites-enabled/default
ln -sfn /etc/nginx/sites-available/openamd /etc/nginx/sites-enabled/openamd

nginx -t
systemctl enable nginx
systemctl restart nginx

log "Phase 6 complete — Nginx proxy on port ${NGINX_PORT} with rate limits + security headers."
