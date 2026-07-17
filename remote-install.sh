#!/bin/bash
#
# One-command OpenAMD Advanced (Heuristic + Silero) install from GitHub.
# Run as root on a new Ubuntu 24.04 server:
#
#   curl -fsSL https://raw.githubusercontent.com/xceedconnections/aiamdadvanced/main/remote-install.sh | bash
#
# Or clone then install:
#
#   git clone https://github.com/xceedconnections/aiamdadvanced.git /root/aiamdadvanced
#   bash /root/aiamdadvanced/install.sh
#
set -euo pipefail

REPO_URL="${OPENAMD_REPO_URL:-https://github.com/xceedconnections/aiamdadvanced.git}"
BRANCH="${OPENAMD_BRANCH:-main}"
DEST="${OPENAMD_DEST:-/root/aiamdadvanced}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root."
  exit 1
fi

echo "==========================================="
echo " OpenAMD Advanced — download + install"
echo " Hybrid: OpenAMD Heuristic + Silero VAD"
echo " Repo: ${REPO_URL} (${BRANCH})"
echo " Dest: ${DEST}"
echo "==========================================="

if ! command -v git >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y git
fi

rm -rf "${DEST}"
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${DEST}"

# Support either flat repo (install.sh at root) or install/ subfolder
if [[ -f "${DEST}/install.sh" ]]; then
  INSTALL_SCRIPT="${DEST}/install.sh"
elif [[ -f "${DEST}/install/install.sh" ]]; then
  INSTALL_SCRIPT="${DEST}/install/install.sh"
else
  echo "ERROR: install.sh not found in ${DEST}"
  exit 1
fi

# Normalize CRLF if present
find "${DEST}" -type f -name '*.sh' -exec sed -i 's/\r$//' {} +

bash "${INSTALL_SCRIPT}"
