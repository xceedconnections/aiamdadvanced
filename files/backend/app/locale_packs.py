"""Locale / market threshold packs for hybrid AMD.

These adjust acoustic cutoffs only (no extra models / negligible CPU).

- default (locale pack OFF / global): North-America style AM blocking
- usa / canada / multilingual: same NA voicemail detection
- uk: left HUMAN-friendly (na_vm_aggressive=0) — do not change UK behaviour
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Shared NA voicemail detection keys (USA / Canada / multilingual / default)
_NA_VM = {
    "na_vm_aggressive": 1.0,
    "vm_struct_duration_min": 1.85,
    "vm_struct_speech_ratio": 0.36,
    "vm_struct_min_bursts": 3,
    "vm_struct_silence_lo_ms": 300,
    "vm_struct_silence_hi_ms": 1500,
    "vm_struct_total_speech_ms": 1000,
    "vm_dense_duration": 2.2,
    "vm_dense_speech_ratio": 0.45,
    "vm_dense_bursts": 4,
    "vm_dense_min_burst_ms": 350,
    "vm_choppy_duration_min": 1.35,
    "vm_choppy_duration_max": 2.4,
    "vm_choppy_min_bursts": 6,
    "vm_choppy_max_burst_ms": 220,
    "vm_choppy_min_speech_ratio": 0.10,
}

# Core NA acoustic thresholds
_NA_CORE = {
    "machine_duration_min": 1.9,
    "machine_longest_burst_ms": 1600,
    "machine_num_bursts": 3,
    "machine_ivr_silence_ms": 400,
    "machine_ivr_bursts": 4,
    "machine_dense_duration": 1.8,
    "machine_dense_bursts": 4,
    "machine_dense_speech_ratio": 0.3,
    "human_max_bursts": 2,
    "human_max_burst_ms": 1000,
    "machine_long_speech_ms": 1800,
    "machine_long_speech_ratio": 0.45,
    "machine_long_duration": 2.1,
    "silero_long_duration": 1.8,
    "silero_long_speech_ms": 1400,
    "silero_long_speech_ratio": 0.38,
    "silero_long_segments": 2,
    "silero_ivr_segments": 4,
    "silero_ivr_speech_ratio": 0.45,
    "silero_many_segments": 4,
    "silero_many_speech_ratio": 0.3,
    "silero_human_max_segments": 2,
    "silero_human_max_speech_ms": 850,
    "silero_human_max_speech_ratio": 0.35,
    "silero_quiet_mean_prob": 0.32,
    "silero_quiet_duration": 1.8,
    "prefer_human_machine_conf": 0.65,
    "prefer_human_speech_ms": 600,
    "upgrade_machine_speech_ms": 1600,
    "upgrade_machine_speech_ratio": 0.42,
    "upgrade_machine_duration": 2.0,
    "strong_machine_conf": 0.82,
    **_NA_VM,
}


def _na_pack() -> dict[str, float]:
    return deepcopy(_NA_CORE)


LOCALE_PACKS: dict[str, dict[str, float]] = {
    # Global / locale pack disabled — same NA voicemail blocking
    "default": _na_pack(),
    "usa": _na_pack(),
    "canada": _na_pack(),
    "multilingual": _na_pack(),
    # UK: keep HUMAN-friendly; disable aggressive NA VM rules
    "uk": {
        "machine_duration_min": 2.4,
        "machine_longest_burst_ms": 2200,
        "machine_num_bursts": 5,
        "machine_ivr_silence_ms": 550,
        "machine_ivr_bursts": 5,
        "machine_dense_duration": 2.2,
        "machine_dense_bursts": 6,
        "machine_dense_speech_ratio": 0.38,
        "human_max_bursts": 5,
        "human_max_burst_ms": 1400,
        "machine_long_speech_ms": 2400,
        "machine_long_speech_ratio": 0.58,
        "machine_long_duration": 2.6,
        "silero_long_duration": 2.2,
        "silero_long_speech_ms": 2000,
        "silero_long_speech_ratio": 0.48,
        "silero_long_segments": 2,
        "silero_ivr_segments": 4,
        "silero_ivr_speech_ratio": 0.52,
        "silero_many_segments": 5,
        "silero_many_speech_ratio": 0.38,
        "silero_human_max_segments": 4,
        "silero_human_max_speech_ms": 1100,
        "silero_human_max_speech_ratio": 0.58,
        "silero_quiet_mean_prob": 0.38,
        "silero_quiet_duration": 2.1,
        "prefer_human_machine_conf": 0.85,
        "prefer_human_speech_ms": 1200,
        "upgrade_machine_speech_ms": 2400,
        "upgrade_machine_speech_ratio": 0.58,
        "upgrade_machine_duration": 2.6,
        "strong_machine_conf": 0.9,
        "na_vm_aggressive": 0.0,
        "vm_struct_duration_min": 99.0,
        "vm_struct_speech_ratio": 0.99,
        "vm_struct_min_bursts": 99,
        "vm_struct_silence_lo_ms": 9999,
        "vm_struct_silence_hi_ms": 9999,
        "vm_struct_total_speech_ms": 99999,
        "vm_dense_duration": 99.0,
        "vm_dense_speech_ratio": 0.99,
        "vm_dense_bursts": 99,
        "vm_dense_min_burst_ms": 9999,
        "vm_choppy_duration_min": 99.0,
        "vm_choppy_duration_max": 0.0,
        "vm_choppy_min_bursts": 99,
        "vm_choppy_max_burst_ms": 0,
        "vm_choppy_min_speech_ratio": 0.99,
    },
}

ALLOWED_LOCALE_PACKS = ("usa", "canada", "uk", "multilingual")


def normalize_locale_pack(name: str | None) -> str:
    key = str(name or "usa").strip().lower()
    if key in ("us", "usa", "united states", "america"):
        return "usa"
    if key in ("ca", "can", "canada", "canadian"):
        return "canada"
    if key in ("uk", "gb", "britain", "united kingdom"):
        return "uk"
    if key in ("multi", "multilingual", "mixed", "intl", "international"):
        return "multilingual"
    if key in LOCALE_PACKS and key != "default":
        return key
    return "usa"


def resolve_locale_pack(*, enabled: bool, pack: str | None) -> tuple[str, dict[str, float]]:
    """Return (pack_name, thresholds).

    Disabled → default (NA voicemail blocking for global settings).
    """
    if not enabled:
        return "default", dict(LOCALE_PACKS["default"])
    name = normalize_locale_pack(pack)
    return name, dict(LOCALE_PACKS.get(name, LOCALE_PACKS["usa"]))


def locale_pack_meta() -> list[dict[str, Any]]:
    return [
        {
            "id": "usa",
            "label": "USA",
            "description": "US — blocks speech–pause and short choppy AM greetings before the beep",
        },
        {
            "id": "canada",
            "label": "Canada",
            "description": "Canada — same North-America voicemail blocking as USA",
        },
        {
            "id": "uk",
            "label": "UK",
            "description": "UK — shorter cadence; more HUMAN-friendly (unchanged)",
        },
        {
            "id": "multilingual",
            "label": "Multilingual / mixed",
            "description": "Mixed trunks — uses North-America voicemail blocking",
        },
    ]
