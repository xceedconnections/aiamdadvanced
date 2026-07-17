#!/usr/bin/env python3
"""
Daily cron / systemd timer entrypoint.

Reads /opt/openamd/cron_settings.json (or RECORDINGS_DIR parent) and deletes
recordings older than retention_days when enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.audio_cleanup import delete_old_recordings  # noqa: E402
from app.cron_settings import load_cron_settings  # noqa: E402


def main() -> int:
    cfg = load_cron_settings()
    if not cfg.get("enabled", True):
        print("OpenAMD cleanup: disabled — skipping")
        return 0
    days = int(cfg.get("retention_days", 7))
    result = delete_old_recordings(days)
    print(
        f"OpenAMD cleanup: retention={days}d deleted={result['deleted_files']} "
        f"freed_mb={result['freed_mb']} dir={result['recordings_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
