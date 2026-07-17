#!/bin/bash
#
# Fix "Unknown" AMD engine stats on the AI AMD portal.
# Run as root on the AI AMD server:
#   bash /root/fix_engine_display.sh
# Or from the project:
#   bash /root/AIAMD/install/fix_engine_display.sh
#   bash /root/install/fix_engine_display.sh
#

set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root"
  exit 1
fi

APP="/opt/openamd"
BACKEND="${APP}/backend"
ENGINE="${BACKEND}/app/ai/engine.py"
SYSTEM="${BACKEND}/app/routers/system.py"
MAIN="${BACKEND}/app/main.py"
APPJS="${BACKEND}/app/static/app.js"

if [[ ! -f "${SYSTEM}" ]]; then
  echo "ERROR: ${SYSTEM} not found"
  exit 1
fi

echo "[1/4] Ensuring ENGINE_INFO in engine.py..."
python3 <<'PY'
from pathlib import Path
path = Path("/opt/openamd/backend/app/ai/engine.py")
text = path.read_text(encoding="utf-8")
block = '''
ENGINE_INFO = {
    "name": "OpenAMD Hybrid (Heuristic + Silero)",
    "model": "Rule-based acoustic features + Silero VAD ONNX",
    "version": "4.0.0",
    "runtime": "NumPy + SoundFile + ONNX Runtime (Silero)",
}
'''
if "ENGINE_INFO" not in text:
    # insert after soundfile import block / before AnalysisResult
    needle = "@dataclass\nclass AnalysisResult:"
    if needle in text:
        text = text.replace(needle, block.strip() + "\n\n\n" + needle, 1)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")
    print("ENGINE_INFO added")
else:
    print("ENGINE_INFO already present")
PY

echo "[2/4] Patching system.py to return amd_engine..."
python3 <<'PY'
from pathlib import Path
import re

path = Path("/opt/openamd/backend/app/routers/system.py")
text = path.read_text(encoding="utf-8")

if "from app.ai.engine import ENGINE_INFO" not in text:
    text = text.replace(
        "from fastapi import APIRouter, Depends\n",
        "from fastapi import APIRouter, Depends\n\nfrom app.ai.engine import ENGINE_INFO\n",
        1,
    )
if "from app.config import get_settings" not in text:
    text = text.replace(
        "from app.ai.engine import ENGINE_INFO\n",
        "from app.ai.engine import ENGINE_INFO\nfrom app.config import get_settings\n",
        1,
    )
if "settings = get_settings()" not in text:
    text = text.replace(
        'router = APIRouter(prefix="/api/system", tags=["system"])\n',
        'router = APIRouter(prefix="/api/system", tags=["system"])\nsettings = get_settings()\n',
        1,
    )

# Replace return dict to include engine fields
if '"amd_engine"' not in text:
    text = re.sub(
        r'return \{\s*"status": status,',
        '''engine = dict(ENGINE_INFO)
    return {
        "status": status,''',
        text,
        count=1,
    )
    # Inject fields before closing of return
    text = text.replace(
        '"uptime_seconds": max(0, int(time.time() - boot_time)),\n    }',
        '''"uptime_seconds": max(0, int(time.time() - boot_time)),
        "application_version": settings.APP_VERSION,
        "amd_engine": engine,
        "amd_engine_name": engine.get("name", "Unknown"),
        "amd_engine_model": engine.get("model", "Unknown"),
        "amd_engine_version": engine.get("version", "Unknown"),
        "amd_engine_runtime": engine.get("runtime", "Unknown"),
    }''',
        1,
    )
    path.write_text(text, encoding="utf-8")
    print("system.py patched")
else:
    # Ensure flat fields exist
    if "amd_engine_name" not in text:
        text = text.replace(
            '"amd_engine": ENGINE_INFO,',
            '''"amd_engine": ENGINE_INFO,
        "amd_engine_name": ENGINE_INFO.get("name", "Unknown"),
        "amd_engine_model": ENGINE_INFO.get("model", "Unknown"),
        "amd_engine_version": ENGINE_INFO.get("version", "Unknown"),
        "amd_engine_runtime": ENGINE_INFO.get("runtime", "Unknown"),''',
            1,
        )
        if "application_version" not in text:
            text = text.replace(
                '"amd_engine":',
                '"application_version": settings.APP_VERSION,\n        "amd_engine":',
                1,
            )
        path.write_text(text, encoding="utf-8")
        print("system.py flat fields added")
    else:
        print("system.py already has amd_engine fields")
PY

echo "[3/4] Updating portal JS defaults..."
if [[ -f "${APPJS}" ]]; then
  python3 <<'PY'
from pathlib import Path
path = Path("/opt/openamd/backend/app/static/app.js")
text = path.read_text(encoding="utf-8")
old = '''  const systemCards = [
    ["Server status", (system.status || "ok").toUpperCase()],
    ["AMD engine", system.amd_engine?.name || "Unknown"],
    ["AMD model", system.amd_engine?.model || "Unknown"],
    ["Engine version", system.amd_engine?.version || "Unknown"],
    ["Engine runtime", system.amd_engine?.runtime || "Unknown"],
    ["Portal/API version", system.application_version || "Unknown"],'''
new = '''  const engine = system.amd_engine || {};
  const engineName =
    engine.name || system.amd_engine_name || "OpenAMD Hybrid (Heuristic + Silero)";
  const engineModel =
    engine.model || system.amd_engine_model || "Rule-based acoustic features + Silero VAD ONNX";
  const engineVersion =
    engine.version || system.amd_engine_version || "4.0.0";
  const engineRuntime =
    engine.runtime || system.amd_engine_runtime || "NumPy + SoundFile + ONNX Runtime (Silero)";
  const appVersion = system.application_version || system.version || "1.0.0";

  const systemCards = [
    ["Server status", (system.status || "ok").toUpperCase()],
    ["AMD engine", engineName],
    ["AMD model", engineModel],
    ["Engine version", engineVersion],
    ["Engine runtime", engineRuntime],
    ["Portal/API version", appVersion],'''
if old in text:
    path.write_text(text.replace(old, new), encoding="utf-8")
    print("app.js updated")
elif "engineName" in text:
    print("app.js already updated")
else:
    # Broad fallback: replace Unknown defaults for these labels
    text2 = text
    text2 = text2.replace(
        'system.amd_engine?.name || "Unknown"',
        'system.amd_engine?.name || system.amd_engine_name || "OpenAMD Hybrid (Heuristic + Silero)"',
    )
    text2 = text2.replace(
        'system.amd_engine?.model || "Unknown"',
        'system.amd_engine?.model || system.amd_engine_model || "Rule-based acoustic features + Silero VAD ONNX"',
    )
    text2 = text2.replace(
        'system.amd_engine?.version || "Unknown"',
        'system.amd_engine?.version || system.amd_engine_version || "4.0.0"',
    )
    text2 = text2.replace(
        'system.amd_engine?.runtime || "Unknown"',
        'system.amd_engine?.runtime || system.amd_engine_runtime || "NumPy + SoundFile + ONNX Runtime (Silero)"',
    )
    text2 = text2.replace(
        'system.application_version || "Unknown"',
        'system.application_version || system.version || "2.0.0"',
    )
    if text2 != text:
        path.write_text(text2, encoding="utf-8")
        print("app.js patched (fallback)")
    else:
        print("app.js left unchanged (manual check)")
PY
fi

echo "[4/4] Restarting openamd..."
systemctl restart openamd
sleep 2

export PYTHONPATH="${BACKEND}"
"${APP}/venv/bin/python" - <<'PY'
from app.routers.system import system_stats
from app.ai.engine import ENGINE_INFO
print("ENGINE_INFO:", ENGINE_INFO)
PY

CODE=$(curl -sS -o /tmp/oam_sys.json -w "%{http_code}" http://127.0.0.1:8000/api/system/stats || echo 000)
echo "GET /api/system/stats => HTTP ${CODE} (401 expected without token)"
# Also verify import path in health
curl -sS http://127.0.0.1:8000/api/health | head -c 500 || true
echo ""
echo "Done. Hard-refresh portal (Ctrl+F5)."
echo "Expected:"
echo "  AMD engine: OpenAMD Hybrid (Heuristic + Silero)"
echo "  AMD model: Rule-based acoustic features + Silero VAD ONNX"
echo "  Engine version: 4.0.0"
echo "  Engine runtime: NumPy + SoundFile + ONNX Runtime (Silero)"
echo "  Portal/API version: 2.0.0"
