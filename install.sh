#!/bin/bash
#
# OpenAMD master installer
# Ubuntu 24.04 LTS — AI AMD server + portal
#
# Usage (on a fresh Ubuntu server as root):
#   1) Upload openamd-install.tar.gz to /root
#   2) tar -xzf /root/openamd-install.tar.gz -C /root/
#   3) bash /root/install/install.sh
#
# The install folder includes install/files/backend (API + portal).
# ViciBox dialer install is separate — see ../vicibox/
#

set -euo pipefail

INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=common.sh
source "${INSTALL_DIR}/common.sh"

require_root
require_ubuntu
detect_project_dir

echo "==========================================="
echo " OpenAMD Full Installer"
echo " AI AMD Server + Portal"
echo "==========================================="
echo " Install dir : ${INSTALL_DIR}"
echo " Source      : ${PROJECT_DIR}"
echo " Target      : ${APP_ROOT}"
echo "==========================================="

STEPS=(
  install1.sh
  install2.sh
  install3.sh
  install4.sh
  install5.sh
  install6.sh
  install7.sh
)

for step in "${STEPS[@]}"; do
  script="${INSTALL_DIR}/${step}"
  [[ -f "${script}" ]] || die "Missing ${script}"
  strip_crlf "${script}"
  echo ""
  echo "###########################################"
  echo " Running ${step}"
  echo "###########################################"
  bash "${script}"
done

IP="$(server_ip)"

echo ""
echo "==========================================="
echo " OpenAMD installation COMPLETE"
echo "==========================================="
echo ""
echo " Portal:   http://${IP}/"
echo " API docs: http://${IP}/docs"
echo " Health:   http://${IP}/api/health"
echo ""
echo " Login:"
echo "   Username: ${ADMIN_USER}"
echo "   Password: ${ADMIN_PASS}"
echo ""
echo " VICIdial analyze API:"
echo "   POST http://${IP}/api/v1/analyze"
echo "   Header: X-API-Key: <from portal>"
echo ""
echo " Service:"
echo "   systemctl status openamd"
echo "   journalctl -u openamd -f"
echo ""
echo " Next — connect a ViciBox dialer:"
echo "   1) Portal → VICIdial Servers → add server → generate API key"
echo "   2) Upload the vicibox/ folder to the dialer and run:"
echo "      bash /root/vicibox/vicibox_install.sh ${IP} oam_YOUR_API_KEY"
echo "   See vicibox/README.md"
echo ""
