"""Low-confidence AMD samples for agent labeling + XGBoost retrain."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.ai.features import vectorize
from app.config import get_settings


def ml_data_root() -> Path:
    root = Path(get_settings().MODELS_DIR).resolve().parent / "ml_data"
    (root / "samples").mkdir(parents=True, exist_ok=True)
    return root


def samples_dir() -> Path:
    d = ml_data_root() / "samples"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_low_confidence_sample(
    *,
    audio: np.ndarray,
    sr: int,
    feats: Dict[str, Any],
    silero: Dict[str, Any],
    predicted_status: str,
    confidence: float,
    class_probs: Dict[str, float],
    extra: Optional[Dict[str, Any]] = None,
    max_seconds: float = 5.0,
) -> str:
    """Persist first max_seconds of audio + feature JSON. Returns sample id."""
    sid = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    n = int(min(len(audio), max(1, int(sr * max_seconds))))
    clip = np.asarray(audio[:n], dtype=np.float32)

    wav_path = samples_dir() / f"{sid}.wav"
    meta_path = samples_dir() / f"{sid}.json"

    try:
        import soundfile as sf

        sf.write(str(wav_path), clip, sr, subtype="PCM_16")
    except Exception:
        # Fallback: raw float32 dump is useless for agents — skip audio, keep meta
        wav_path = Path("")

    feat_vec = vectorize(feats, silero).tolist()
    meta = {
        "id": sid,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "predicted_status": predicted_status,
        "confidence": round(float(confidence), 4),
        "class_probs": {k: round(float(v), 4) for k, v in class_probs.items()},
        "label": None,
        "labeled_by": None,
        "labeled_at": None,
        "feature_vector": feat_vec,
        "feats": {
            k: feats.get(k)
            for k in (
                "duration",
                "speech_ratio",
                "num_bursts",
                "longest_burst_ms",
                "internal_silence_ms",
                "activity_ratio",
                "beep",
            )
        },
        "silero": {
            "speech_ratio": silero.get("speech_ratio"),
            "num_segments": silero.get("num_segments"),
            "longest_speech_ms": silero.get("longest_speech_ms"),
            "mean_prob": silero.get("mean_prob"),
        },
        "audio_path": str(wav_path) if wav_path else "",
        "extra_note": (extra or {}).get("ml_note", ""),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return sid


def list_samples(*, unlabeled_only: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(samples_dir().glob("*.json"), reverse=True):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if unlabeled_only and meta.get("label"):
            continue
        rows.append(
            {
                "id": meta.get("id"),
                "created_at": meta.get("created_at"),
                "predicted_status": meta.get("predicted_status"),
                "confidence": meta.get("confidence"),
                "class_probs": meta.get("class_probs"),
                "label": meta.get("label"),
                "audio_path": meta.get("audio_path"),
                "extra_note": meta.get("extra_note"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def label_sample(sample_id: str, label: str, *, username: str = "") -> Dict[str, Any]:
    label_u = str(label).strip().upper()
    if label_u in ("VOICEMAIL", "AM", "FAX", "SIT", "ERROR"):
        label_u = "MACHINE"
    if label_u not in ("HUMAN", "MACHINE", "IVR"):
        raise ValueError("label must be HUMAN, MACHINE, or IVR")

    path = samples_dir() / f"{sample_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"sample not found: {sample_id}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["label"] = label_u
    meta["labeled_by"] = username or "admin"
    meta["labeled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_labeled_for_train() -> tuple[list, list]:
    import numpy as np

    xs = []
    ys = []
    for path in samples_dir().glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        lab = meta.get("label")
        vec = meta.get("feature_vector")
        if not lab or not vec:
            continue
        xs.append(np.asarray(vec, dtype=np.float32))
        ys.append(str(lab).upper())
    return xs, ys


def retrain_model() -> Dict[str, Any]:
    from app.ai.xgb_model import retrain_from_labeled

    xs, ys = load_labeled_for_train()
    meta = retrain_from_labeled(xs, ys, mix_bootstrap=True)
    meta["n_labeled_used"] = len(ys)
    return meta


def ml_stats() -> Dict[str, Any]:
    from app.ai.xgb_model import model_status
    from app.ai.whisper_amd import whisper_available

    all_s = list_samples(limit=5000)
    unlabeled = [s for s in all_s if not s.get("label")]
    labeled = [s for s in all_s if s.get("label")]
    return {
        "samples_total": len(all_s),
        "samples_unlabeled": len(unlabeled),
        "samples_labeled": len(labeled),
        "samples_dir": str(samples_dir()),
        "whisper_installed": whisper_available(),
        "model": model_status(),
    }
