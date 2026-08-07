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
                "source": meta.get("source") or "ml_low_conf",
                "call_analysis_id": meta.get("call_analysis_id"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _map_train_label(label: str) -> str:
    label_u = str(label).strip().upper()
    if label_u in ("VOICEMAIL", "AM", "FAX", "SIT", "ERROR", "CANCELLED"):
        return "MACHINE"
    if label_u not in ("HUMAN", "MACHINE", "IVR"):
        return "MACHINE"
    return label_u


def _feats_silero_from_details(details: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    silero_raw = details.get("silero") if isinstance(details.get("silero"), dict) else {}
    silero = {
        "speech_ratio": silero_raw.get("speech_ratio", details.get("speech_ratio", 0.0)),
        "num_segments": silero_raw.get("num_segments", 0),
        "longest_speech_ms": silero_raw.get("longest_speech_ms", 0.0),
        "mean_prob": silero_raw.get("mean_prob", 0.0),
    }
    feats = {
        "duration": details.get("duration", 0.0),
        "speech_ratio": details.get("speech_ratio", 0.0),
        "num_bursts": details.get("num_bursts", 0.0),
        "longest_burst_ms": details.get("longest_burst_ms", 0.0),
        "longest_silence_ms": details.get("longest_silence_ms", 0.0),
        "avg_burst_ms": details.get("avg_burst_ms", 0.0),
        "internal_silence_ms": details.get("internal_silence_ms", 0.0),
        "activity_ratio": details.get("activity_ratio", details.get("speech_ratio", 0.0)),
        "rms": details.get("rms", 0.0),
        "peak": details.get("peak", 0.0),
        "beep": details.get("beep", 0.0),
        "beep_conf": details.get("beep_conf", 0.0),
        "sit": details.get("sit", 0.0),
        "sit_conf": details.get("sit_conf", 0.0),
    }
    return feats, silero


def ingest_call_correction_as_ml_sample(
    *,
    call: Any,
    taught_status: str,
    username: str = "",
    max_seconds: float = 5.0,
) -> Optional[str]:
    """
    Turn a Training Mark-as correction into a labeled XGBoost sample.

    Uses features_json from the call (and optional audio clip). Safe to call
    even when ML pipeline is off — samples wait until Retrain XGBoost.
    Returns sample id or None if features are missing.
    """
    details: Dict[str, Any] = {}
    raw_json = getattr(call, "features_json", None) or "{}"
    try:
        parsed = json.loads(raw_json) if isinstance(raw_json, str) else (raw_json or {})
        if isinstance(parsed, dict):
            details = parsed
    except (TypeError, json.JSONDecodeError):
        details = {}

    feats, silero = _feats_silero_from_details(details)
    # Need at least some acoustic signal in the vector
    if not any(float(feats.get(k) or 0) for k in ("duration", "speech_ratio", "rms", "num_bursts")):
        # Still allow if silero has signal
        if not any(float(silero.get(k) or 0) for k in ("speech_ratio", "mean_prob", "num_segments")):
            return None

    label = _map_train_label(taught_status)
    predicted = str(
        getattr(call, "raw_status", None) or details.get("heuristic_status") or getattr(call, "status", "") or ""
    ).strip().upper() or "UNKNOWN"
    conf = float(getattr(call, "confidence", 0.0) or details.get("confidence") or 0.0)
    class_probs = details.get("class_probs") if isinstance(details.get("class_probs"), dict) else {}

    call_id = int(getattr(call, "id", 0) or 0)
    sid = f"teach-{call_id}" if call_id else f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    meta_path = samples_dir() / f"{sid}.json"
    wav_path = samples_dir() / f"{sid}.wav"

    # Optional audio clip (best-effort)
    audio_saved = ""
    try:
        import soundfile as sf

        audio = None
        sr = 8000
        blob = getattr(call, "audio_blob", None)
        path = (getattr(call, "audio_path", None) or "").strip()
        if blob:
            import io

            audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = np.mean(audio, axis=1)
        elif path and Path(path).is_file():
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if getattr(audio, "ndim", 1) > 1:
                audio = np.mean(audio, axis=1)
        if audio is not None and len(audio) > 0:
            n = int(min(len(audio), max(1, int(sr * max_seconds))))
            clip = np.asarray(audio[:n], dtype=np.float32)
            sf.write(str(wav_path), clip, int(sr) or 8000, subtype="PCM_16")
            audio_saved = str(wav_path)
    except Exception:
        audio_saved = ""

    feat_vec = vectorize(feats, silero).tolist()
    meta = {
        "id": sid,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "training_teach",
        "call_analysis_id": call_id,
        "vicidial_call_id": getattr(call, "call_id", "") or "",
        "predicted_status": predicted,
        "confidence": round(float(conf), 4),
        "class_probs": {k: round(float(v), 4) for k, v in class_probs.items()},
        "label": label,
        "labeled_by": username or "admin",
        "labeled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
        "silero": silero,
        "audio_path": audio_saved,
        "extra_note": "from_training_correction",
    }
    # Preserve first-created timestamp if updating same call teach
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("created_at"):
                meta["created_at"] = old["created_at"]
        except (OSError, json.JSONDecodeError):
            pass
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return sid


def label_sample(sample_id: str, label: str, *, username: str = "") -> Dict[str, Any]:
    label_u = _map_train_label(label)

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
