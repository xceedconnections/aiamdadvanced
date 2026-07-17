#!/bin/bash
# Create self-contained install folder tarball for a new AI AMD server.
# Run from project root (Git Bash / WSL / Linux):
#   bash install/pack.sh
#
# Bundles: install/ + current backend (API + portal). Vicibox is separate.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_DIR="${ROOT}/install"
BUNDLE="${INSTALL_DIR}/files"
OUT="${ROOT}/openamd-install.tar.gz"

echo "Refreshing bundled backend + portal in ${BUNDLE}..."
rm -rf "${BUNDLE}"
mkdir -p "${BUNDLE}/backend"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'recordings/*' \
    --exclude 'logs/*' \
    "${ROOT}/backend/" "${BUNDLE}/backend/"
else
  cp -a "${ROOT}/backend/." "${BUNDLE}/backend/"
  find "${BUNDLE}" -type d \( -name 'venv' -o -name '__pycache__' -o -name '.git' \) -prune -exec rm -rf {} + 2>/dev/null || true
  find "${BUNDLE}" -type f -name '*.pyc' -delete 2>/dev/null || true
fi

cp -f "${INSTALL_DIR}/README.md" "${BUNDLE}/README.md" 2>/dev/null || true

[[ -f "${BUNDLE}/backend/app/templates/index.html" ]] \
  || { echo "ERROR: portal template missing after pack"; exit 1; }
[[ -f "${BUNDLE}/backend/app/static/app.js" ]] \
  || { echo "ERROR: portal static missing after pack"; exit 1; }

find "${INSTALL_DIR}" -name '*.sh' -exec sed -i 's/\r$//' {} \; 2>/dev/null || true

echo "Creating ${OUT}..."
tar -czf "$OUT" \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='recordings/*' \
  --exclude='logs/*' \
  --exclude='openamd*.tar.gz' \
  -C "$ROOT" \
  install

echo "Created: $OUT"
echo ""
echo "Upload to new Ubuntu 24.04 server:"
echo "  scp openamd-install.tar.gz root@NEW_SERVER:/root/"
echo ""
echo "On server:"
echo "  tar -xzf /root/openamd-install.tar.gz -C /root/"
echo "  bash /root/install/install.sh"
echo ""
echo "ViciBox dialer (separate):"
echo "  scp -r vicibox root@VICIBOX:/root/"
echo "  bash /root/vicibox/vicibox_install.sh AI_AMD_IP oam_API_KEY"
