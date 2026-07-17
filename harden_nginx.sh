#!/bin/bash
# Re-apply Nginx hardening (port 80) on an existing OpenAMD server.
# Usage (as root):
#   bash /root/aiamdadvanced/install6.sh
# or:
#   bash /opt/openamd/install/install6.sh

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
bash "${DIR}/install6.sh"
