#!/bin/bash
# Phase 6 — Nginx reverse proxy

set -euo pipefail
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"

log "[6/7] Configuring Nginx..."

cat >/etc/nginx/sites-available/openamd <<EOF
server {
    listen ${NGINX_PORT} default_server;
    listen [::]:${NGINX_PORT} default_server;
    server_name _;

    client_max_body_size 10M;

    location / {
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

log "Phase 6 complete — Nginx proxy on port ${NGINX_PORT}."
