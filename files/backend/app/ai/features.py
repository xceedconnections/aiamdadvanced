"""Feature vector for the optional XGBoost AMD classifier.

OpenAMD 6.x — acoustic + Silero + Whisper cues + optional Asterisk AMD.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

# Stable order — changing this invalidates saved models (bump MODEL_VERSION)
FEATURE_KEYS: List[str] = [
    "duration",
    "speech_ratio",
    "num_bursts",
    "longest_burst_ms",
    "longest_silence_ms",
    "avg_burst_ms",
    "internal_silence_ms",
    "activity_ratio",
    "rms",
    "peak",
    "beep",
    "beep_conf",
    "sit",
    "sit_conf",
    "silero_speech_ratio",
    "silero_num_segments",
    "silero_longest_speech_ms",
    "silero_mean_prob",
    # v2 — Asterisk stock AMD + Whisper phrase features
    "asterisk_amd_human",
    "asterisk_amd_machine",
    "whisper_vm_hit",
    "whisper_human_hit",
    "whisper_word_count",
]

LABELS: List[str] = ["HUMAN", "MACHINE", "IVR"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}
MODEL_VERSION = 2


def _asterisk_flags(feats: Dict[str, Any]) -> tuple[float, float]:
    st = str(feats.get("asterisk_amdstatus") or feats.get("amdstatus") or "").strip().upper()
    if st in ("HUMAN", "PERSON"):
        return 1.0, 0.0
    if st in ("MACHINE", "AM", "ANSWERMACHINE", "MAXWORDS", "LONGGREETING", "MAXWORDLENGTH"):
        return 0.0, 1.0
    cause = str(feats.get("asterisk_amdcause") or feats.get("amdcause") or "").strip().upper()
    if "HUMAN" in cause:
        return 1.0, 0.0
    if any(
        x in cause
        for x in ("MACHINE", "TOOLONG", "MAXWORD", "LONGGREET", "INITIALSILENCE")
    ):
        return 0.0, 1.0
    return 0.0, 0.0


def vectorize(feats: Dict[str, Any], silero: Dict[str, Any] | None = None) -> np.ndarray:
    """Build a float32 feature row from hybrid engine outputs."""
    silero = silero or {}
    a_h, a_m = _asterisk_flags(feats)
    row = [
        float(feats.get("duration", 0.0) or 0.0),
        float(feats.get("speech_ratio", 0.0) or 0.0),
        float(feats.get("num_bursts", 0.0) or 0.0),
        float(feats.get("longest_burst_ms", 0.0) or 0.0),
        float(feats.get("longest_silence_ms", 0.0) or 0.0),
        float(feats.get("avg_burst_ms", 0.0) or 0.0),
        float(feats.get("internal_silence_ms", 0.0) or 0.0),
        float(feats.get("activity_ratio", feats.get("speech_ratio", 0.0)) or 0.0),
        float(feats.get("rms", 0.0) or 0.0),
        float(feats.get("peak", 0.0) or 0.0),
        float(feats.get("beep", 0.0) or 0.0),
        float(feats.get("beep_conf", 0.0) or 0.0),
        float(feats.get("sit", 0.0) or 0.0),
        float(feats.get("sit_conf", 0.0) or 0.0),
        float(silero.get("speech_ratio", 0.0) or 0.0),
        float(silero.get("num_segments", 0) or 0.0),
        float(silero.get("longest_speech_ms", 0.0) or 0.0),
        float(silero.get("mean_prob", 0.0) or 0.0),
        a_h,
        a_m,
        float(feats.get("whisper_vm_hit", 0.0) or 0.0),
        float(feats.get("whisper_human_hit", 0.0) or 0.0),
        float(feats.get("whisper_word_count", 0.0) or 0.0),
    ]
    return np.asarray(row, dtype=np.float32)


def pad_feature_vector(vec: Sequence[float]) -> np.ndarray:
    """Pad/truncate legacy sample vectors to current FEATURE_KEYS length."""
    arr = np.asarray(list(vec), dtype=np.float32).ravel()
    n = len(FEATURE_KEYS)
    if arr.size == n:
        return arr
    if arr.size > n:
        return arr[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: arr.size] = arr
    return out


def matrix_from_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(FEATURE_KEYS)), dtype=np.float32)
    padded = [pad_feature_vector(r) for r in rows]
    return np.vstack(padded).astype(np.float32)
