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
    call_meta: Optional[Dict[str, Any]] = None,
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
        wav_path = Path("")

    extra = extra or {}
    call_meta = call_meta or {}
    whisper = extra.get("whisper") if isinstance(extra.get("whisper"), dict) else {}
    feat_vec = vectorize(feats, silero).tolist()
    meta = {
        "id": sid,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "ml_low_conf",
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
        "extra_note": extra.get("ml_note", ""),
        "whisper_transcript": (whisper.get("transcript") or "")[:500],
        "whisper_cue": whisper.get("cue") or "",
        "whisper_used": bool(extra.get("whisper_used")),
        "called_number": str(call_meta.get("called_number") or call_meta.get("called") or ""),
        "caller_id": str(call_meta.get("caller_id") or call_meta.get("caller") or ""),
        "ani": str(call_meta.get("ani") or ""),
        "vicidial_call_id": str(call_meta.get("callid") or call_meta.get("vicidial_call_id") or ""),
        "call_analysis_id": call_meta.get("call_analysis_id"),
        "phone_number": str(
            call_meta.get("phone_number")
            or call_meta.get("called_number")
            or call_meta.get("called")
            or call_meta.get("ani")
            or ""
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return sid


def attach_call_to_sample(sample_id: str, **fields: Any) -> bool:
    """Update an existing sample with call_analysis_id / phones after DB insert."""
    path = samples_dir() / f"{sample_id}.json"
    if not path.exists():
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for k, v in fields.items():
        if v is None or v == "":
            continue
        meta[k] = v
    if fields.get("called_number") or fields.get("ani"):
        meta["phone_number"] = str(
            fields.get("phone_number")
            or fields.get("called_number")
            or fields.get("ani")
            or meta.get("phone_number")
            or ""
        )
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return True


def list_samples(
    *,
    unlabeled_only: bool = False,
    labeled_only: bool = False,
    limit: int = 100,
    phone: str = "",
) -> List[Dict[str, Any]]:
    phone_q = "".join(ch for ch in str(phone or "") if ch.isdigit())
    rows: List[Dict[str, Any]] = []
    for path in sorted(samples_dir().glob("*.json"), reverse=True):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if unlabeled_only and meta.get("label"):
            continue
        if labeled_only and not meta.get("label"):
            continue
        if phone_q:
            hay = "".join(
                ch
                for ch in str(
                    meta.get("phone_number")
                    or meta.get("called_number")
                    or meta.get("ani")
                    or meta.get("caller_id")
                    or ""
                )
                if ch.isdigit()
            )
            if phone_q not in hay:
                continue
        rows.append(
            {
                "id": meta.get("id"),
                "created_at": meta.get("created_at"),
                "predicted_status": meta.get("predicted_status"),
                "confidence": meta.get("confidence"),
                "class_probs": meta.get("class_probs"),
                "label": meta.get("label"),
                "labeled_by": meta.get("labeled_by"),
                "audio_path": meta.get("audio_path"),
                "has_audio": bool(
                    (meta.get("audio_path") and Path(str(meta.get("audio_path"))).is_file())
                    or (samples_dir() / f"{meta.get('id')}.wav").is_file()
                ),
                "extra_note": meta.get("extra_note"),
                "source": meta.get("source") or "ml_low_conf",
                "call_analysis_id": meta.get("call_analysis_id"),
                "vicidial_call_id": meta.get("vicidial_call_id") or "",
                "phone_number": meta.get("phone_number")
                or meta.get("called_number")
                or meta.get("ani")
                or "",
                "called_number": meta.get("called_number") or "",
                "caller_id": meta.get("caller_id") or "",
                "ani": meta.get("ani") or "",
                "whisper_transcript": meta.get("whisper_transcript") or "",
                "whisper_cue": meta.get("whisper_cue") or "",
                "whisper_used": bool(meta.get("whisper_used")),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def get_sample(sample_id: str) -> Optional[Dict[str, Any]]:
    path = samples_dir() / f"{sample_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def sample_audio_path(sample_id: str) -> Optional[Path]:
    meta = get_sample(sample_id)
    if not meta:
        return None
    p = Path(str(meta.get("audio_path") or ""))
    if p.is_file():
        return p
    fallback = samples_dir() / f"{sample_id}.wav"
    return fallback if fallback.is_file() else None


def delete_sample(sample_id: str) -> bool:
    meta_path = samples_dir() / f"{sample_id}.json"
    wav_path = samples_dir() / f"{sample_id}.wav"
    ok = False
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ap = Path(str(meta.get("audio_path") or ""))
            if ap.is_file() and ap != wav_path:
                ap.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        meta_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        ok = True
    if wav_path.exists():
        wav_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        ok = True
    return ok


def wipe_all_samples() -> Dict[str, int]:
    n_json = 0
    n_wav = 0
    for path in list(samples_dir().glob("*.json")):
        try:
            path.unlink()
            n_json += 1
        except OSError:
            pass
    for path in list(samples_dir().glob("*.wav")):
        try:
            path.unlink()
            n_wav += 1
        except OSError:
            pass
    return {"deleted_json": n_json, "deleted_wav": n_wav}


def reset_xgb_model(*, bootstrap: bool = True) -> Dict[str, Any]:
    """Delete saved XGBoost model; optionally rebuild synthetic bootstrap."""
    from app.ai.xgb_model import ensure_model, meta_path, model_path

    removed = []
    for p in (model_path(), meta_path()):
        if p.exists():
            p.unlink()
            removed.append(str(p))
    # Clear in-memory booster
    try:
        import app.ai.xgb_model as xm

        with xm._lock:
            xm._booster = None
            xm._loaded_path = None
    except Exception:
        pass
    out: Dict[str, Any] = {"removed": removed, "bootstrapped": False}
    if bootstrap:
        ensure_model()
        out["bootstrapped"] = True
    return out


def wipe_ml_training(*, reset_model: bool = True) -> Dict[str, Any]:
    """Delete all ML sample logs and optionally reset XGBoost to fresh bootstrap."""
    samples = wipe_all_samples()
    model = reset_xgb_model(bootstrap=True) if reset_model else {"removed": [], "bootstrapped": False}
    return {"samples": samples, "model": model}


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
    whisper = details.get("whisper") if isinstance(details.get("whisper"), dict) else {}
    ml_block = details.get("ml") if isinstance(details.get("ml"), dict) else {}
    if not whisper and isinstance(ml_block.get("whisper"), dict):
        whisper = ml_block["whisper"]
    phone = (
        str(getattr(call, "called_number", "") or "")
        or str(getattr(call, "ani", "") or "")
        or str(details.get("called_number") or details.get("ani") or "")
    )
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
        "whisper_transcript": (whisper.get("transcript") or details.get("whisper_transcript") or "")[:500],
        "whisper_cue": whisper.get("cue") or "",
        "whisper_used": bool(
            ml_block.get("whisper_used")
            or details.get("whisper_used")
            or whisper.get("transcript")
        ),
        "called_number": str(getattr(call, "called_number", "") or details.get("called_number") or ""),
        "caller_id": str(getattr(call, "caller_id", "") or details.get("caller_id") or ""),
        "ani": str(getattr(call, "ani", "") or details.get("ani") or ""),
        "phone_number": phone,
    }
    # Preserve first-created timestamp if updating same call teach
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("created_at"):
                meta["created_at"] = old["created_at"]
            # Keep prior whisper text if teach overwrite has none
            if not meta.get("whisper_transcript") and old.get("whisper_transcript"):
                meta["whisper_transcript"] = old["whisper_transcript"]
                meta["whisper_cue"] = old.get("whisper_cue") or meta.get("whisper_cue")
                meta["whisper_used"] = old.get("whisper_used") or meta.get("whisper_used")
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
