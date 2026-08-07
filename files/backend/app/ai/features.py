"""Feature vector for the optional XGBoost AMD classifier.

Uses the same OpenAMD acoustic + Silero features already computed by the
hybrid engine — no extra audio passes when building the vector.
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
]

LABELS: List[str] = ["HUMAN", "MACHINE", "IVR"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}
MODEL_VERSION = 1


def vectorize(feats: Dict[str, Any], silero: Dict[str, Any] | None = None) -> np.ndarray:
    """Build a float32 feature row from hybrid engine outputs."""
    silero = silero or {}
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
    ]
    return np.asarray(row, dtype=np.float32)


def matrix_from_rows(rows: Sequence[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((0, len(FEATURE_KEYS)), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)
