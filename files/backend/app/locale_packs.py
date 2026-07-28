"""Locale / market threshold packs for hybrid AMD.

These adjust acoustic cutoffs only (no extra models / negligible CPU).
When locale packs are disabled on a VICIdial server, the engine uses DEFAULT
(identical to the previous hard-coded behaviour).
"""

from __future__ import annotations

from typing import Any

# Keys used by heuristic classify + Silero fuse
LOCALE_PACKS: dict[str, dict[str, float]] = {
    # Exact previous engine behaviour
    "default": {
        "machine_duration_min": 2.2,
        "machine_longest_burst_ms": 2000,
        "machine_num_bursts": 4,
        "machine_ivr_silence_ms": 500,
        "machine_ivr_bursts": 5,
        "machine_dense_duration": 2.0,
        "machine_dense_bursts": 6,
        "machine_dense_speech_ratio": 0.35,
        "human_max_bursts": 4,
        "human_max_burst_ms": 1500,
        "machine_long_speech_ms": 2200,
        "machine_long_speech_ratio": 0.55,
        "machine_long_duration": 2.5,
        "silero_long_duration": 2.0,
        "silero_long_speech_ms": 1800,
        "silero_long_speech_ratio": 0.45,
        "silero_long_segments": 2,
        "silero_ivr_segments": 4,
        "silero_ivr_speech_ratio": 0.5,
        "silero_many_segments": 5,
        "silero_many_speech_ratio": 0.35,
        "silero_human_max_segments": 3,
        "silero_human_max_speech_ms": 1200,
        "silero_human_max_speech_ratio": 0.55,
        "silero_quiet_mean_prob": 0.35,
        "silero_quiet_duration": 2.0,
        "prefer_human_machine_conf": 0.8,
        "prefer_human_speech_ms": 1000,
        "upgrade_machine_speech_ms": 2200,
        "upgrade_machine_speech_ratio": 0.55,
        "upgrade_machine_duration": 2.5,
        "strong_machine_conf": 0.88,
    },
    # US: longer scripted AM greetings — detect MACHINE a bit earlier
    "usa": {
        "machine_duration_min": 2.0,
        "machine_longest_burst_ms": 1800,
        "machine_num_bursts": 4,
        "machine_ivr_silence_ms": 450,
        "machine_ivr_bursts": 5,
        "machine_dense_duration": 1.9,
        "machine_dense_bursts": 5,
        "machine_dense_speech_ratio": 0.32,
        "human_max_bursts": 4,
        "human_max_burst_ms": 1600,
        "machine_long_speech_ms": 2000,
        "machine_long_speech_ratio": 0.5,
        "machine_long_duration": 2.3,
        "silero_long_duration": 1.9,
        "silero_long_speech_ms": 1600,
        "silero_long_speech_ratio": 0.42,
        "silero_long_segments": 2,
        "silero_ivr_segments": 4,
        "silero_ivr_speech_ratio": 0.48,
        "silero_many_segments": 5,
        "silero_many_speech_ratio": 0.32,
        "silero_human_max_segments": 3,
        "silero_human_max_speech_ms": 1300,
        "silero_human_max_speech_ratio": 0.55,
        "silero_quiet_mean_prob": 0.35,
        "silero_quiet_duration": 2.0,
        "prefer_human_machine_conf": 0.78,
        "prefer_human_speech_ms": 900,
        "upgrade_machine_speech_ms": 2000,
        "upgrade_machine_speech_ratio": 0.5,
        "upgrade_machine_duration": 2.3,
        "strong_machine_conf": 0.88,
    },
    # UK: often shorter cadence — slightly more HUMAN-friendly on mid cases
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
    },
    # Mixed trunks: balanced; prefer HUMAN when uncertain
    "multilingual": {
        "machine_duration_min": 2.3,
        "machine_longest_burst_ms": 2100,
        "machine_num_bursts": 4,
        "machine_ivr_silence_ms": 500,
        "machine_ivr_bursts": 5,
        "machine_dense_duration": 2.1,
        "machine_dense_bursts": 6,
        "machine_dense_speech_ratio": 0.36,
        "human_max_bursts": 4,
        "human_max_burst_ms": 1500,
        "machine_long_speech_ms": 2300,
        "machine_long_speech_ratio": 0.55,
        "machine_long_duration": 2.5,
        "silero_long_duration": 2.1,
        "silero_long_speech_ms": 1900,
        "silero_long_speech_ratio": 0.46,
        "silero_long_segments": 2,
        "silero_ivr_segments": 4,
        "silero_ivr_speech_ratio": 0.5,
        "silero_many_segments": 5,
        "silero_many_speech_ratio": 0.36,
        "silero_human_max_segments": 3,
        "silero_human_max_speech_ms": 1200,
        "silero_human_max_speech_ratio": 0.55,
        "silero_quiet_mean_prob": 0.35,
        "silero_quiet_duration": 2.0,
        "prefer_human_machine_conf": 0.82,
        "prefer_human_speech_ms": 1100,
        "upgrade_machine_speech_ms": 2300,
        "upgrade_machine_speech_ratio": 0.55,
        "upgrade_machine_duration": 2.5,
        "strong_machine_conf": 0.88,
    },
}

ALLOWED_LOCALE_PACKS = ("usa", "uk", "multilingual")


def normalize_locale_pack(name: str | None) -> str:
    key = str(name or "usa").strip().lower()
    if key in ("us", "usa", "united states", "america"):
        return "usa"
    if key in ("uk", "gb", "britain", "united kingdom"):
        return "uk"
    if key in ("multi", "multilingual", "mixed", "intl", "international"):
        return "multilingual"
    if key in LOCALE_PACKS:
        return key
    return "usa"


def resolve_locale_pack(*, enabled: bool, pack: str | None) -> tuple[str, dict[str, float]]:
    """Return (pack_name, thresholds). Disabled → default (current behaviour)."""
    if not enabled:
        return "default", dict(LOCALE_PACKS["default"])
    name = normalize_locale_pack(pack)
    return name, dict(LOCALE_PACKS.get(name, LOCALE_PACKS["usa"]))


def locale_pack_meta() -> list[dict[str, Any]]:
    return [
        {
            "id": "usa",
            "label": "USA",
            "description": "US market — longer AM greetings detected earlier as MACHINE",
        },
        {
            "id": "uk",
            "label": "UK",
            "description": "UK market — shorter cadence; more HUMAN-friendly mid cases",
        },
        {
            "id": "multilingual",
            "label": "Multilingual / mixed",
            "description": "Mixed trunks — balanced; prefers HUMAN when uncertain",
        },
    ]
