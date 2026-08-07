"""XGBoost AMD classifier — lazy-loaded, optional.

When the ML pipeline is disabled, this module is never imported by the
hot path (engine imports only inside ml_pipeline when enabled).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.ai.features import FEATURE_KEYS, LABEL_TO_ID, LABELS, MODEL_VERSION, matrix_from_rows, vectorize
from app.config import get_settings

_lock = threading.Lock()
_booster = None  # type: ignore
_loaded_path: Optional[Path] = None


def models_dir() -> Path:
    root = Path(get_settings().MODELS_DIR).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_path() -> Path:
    return models_dir() / "amd_xgb.json"


def meta_path() -> Path:
    return models_dir() / "amd_xgb.meta.json"


def _make_bootstrap_dataset(n_per_class: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic feature rows that mirror OpenAMD heuristic patterns."""
    rng = np.random.default_rng(42)
    xs: List[np.ndarray] = []
    ys: List[int] = []

    def row(**kwargs: float) -> np.ndarray:
        base = {k: 0.0 for k in FEATURE_KEYS}
        base.update(kwargs)
        return vectorize(base, {
            "speech_ratio": base.get("silero_speech_ratio", 0.0),
            "num_segments": base.get("silero_num_segments", 0.0),
            "longest_speech_ms": base.get("silero_longest_speech_ms", 0.0),
            "mean_prob": base.get("silero_mean_prob", 0.0),
        })

    for _ in range(n_per_class):
        # HUMAN — short sparse "hello?"
        xs.append(row(
            duration=float(rng.uniform(0.8, 2.0)),
            speech_ratio=float(rng.uniform(0.05, 0.35)),
            num_bursts=float(rng.integers(1, 3)),
            longest_burst_ms=float(rng.uniform(80, 900)),
            longest_silence_ms=float(rng.uniform(200, 1200)),
            avg_burst_ms=float(rng.uniform(80, 600)),
            activity_ratio=float(rng.uniform(0.1, 0.4)),
            rms=float(rng.uniform(0.05, 0.25)),
            peak=float(rng.uniform(0.3, 1.0)),
            silero_speech_ratio=float(rng.uniform(0.05, 0.35)),
            silero_num_segments=float(rng.integers(1, 3)),
            silero_longest_speech_ms=float(rng.uniform(100, 900)),
            silero_mean_prob=float(rng.uniform(0.2, 0.55)),
        ))
        ys.append(LABEL_TO_ID["HUMAN"])

        # MACHINE — AM greeting / choppy / beep
        beep = float(rng.random() < 0.25)
        xs.append(row(
            duration=float(rng.uniform(1.5, 3.5)),
            speech_ratio=float(rng.uniform(0.35, 0.75)),
            num_bursts=float(rng.integers(3, 10)),
            longest_burst_ms=float(rng.uniform(200, 2200)),
            longest_silence_ms=float(rng.uniform(200, 900)),
            avg_burst_ms=float(rng.uniform(150, 800)),
            internal_silence_ms=float(rng.uniform(300, 1200)),
            activity_ratio=float(rng.uniform(0.4, 0.9)),
            rms=float(rng.uniform(0.08, 0.35)),
            peak=float(rng.uniform(0.4, 1.0)),
            beep=beep,
            beep_conf=float(rng.uniform(0.7, 0.99)) if beep else 0.0,
            silero_speech_ratio=float(rng.uniform(0.35, 0.8)),
            silero_num_segments=float(rng.integers(2, 8)),
            silero_longest_speech_ms=float(rng.uniform(800, 2500)),
            silero_mean_prob=float(rng.uniform(0.4, 0.9)),
        ))
        ys.append(LABEL_TO_ID["MACHINE"])

        # IVR — many scripted segments
        xs.append(row(
            duration=float(rng.uniform(2.2, 3.8)),
            speech_ratio=float(rng.uniform(0.4, 0.85)),
            num_bursts=float(rng.integers(5, 12)),
            longest_burst_ms=float(rng.uniform(400, 1800)),
            longest_silence_ms=float(rng.uniform(300, 800)),
            avg_burst_ms=float(rng.uniform(200, 700)),
            internal_silence_ms=float(rng.uniform(250, 900)),
            activity_ratio=float(rng.uniform(0.45, 0.95)),
            rms=float(rng.uniform(0.1, 0.4)),
            peak=1.0,
            silero_speech_ratio=float(rng.uniform(0.45, 0.9)),
            silero_num_segments=float(rng.integers(4, 12)),
            silero_longest_speech_ms=float(rng.uniform(600, 2000)),
            silero_mean_prob=float(rng.uniform(0.45, 0.95)),
        ))
        ys.append(LABEL_TO_ID["IVR"])

    return matrix_from_rows(xs), np.asarray(ys, dtype=np.int32)


def _train_booster(X: np.ndarray, y: np.ndarray):
    import xgboost as xgb

    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_KEYS)
    params = {
        "objective": "multi:softprob",
        "num_class": len(LABELS),
        "max_depth": 5,
        "eta": 0.2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "nthread": 1,  # keep CPU impact tiny on dialer hosts
    }
    return xgb.train(params, dtrain, num_boost_round=60)


def ensure_model(force_bootstrap: bool = False):
    """Load model from disk, or train a bootstrap model once."""
    global _booster, _loaded_path
    path = model_path()
    with _lock:
        if _booster is not None and _loaded_path == path and not force_bootstrap:
            return _booster
        if path.exists() and not force_bootstrap:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(str(path))
            _booster = booster
            _loaded_path = path
            return _booster

        X, y = _make_bootstrap_dataset()
        booster = _train_booster(X, y)
        path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(path))
        meta_path().write_text(
            json.dumps(
                {
                    "model_version": MODEL_VERSION,
                    "labels": LABELS,
                    "feature_keys": FEATURE_KEYS,
                    "source": "bootstrap_synthetic",
                    "n_samples": int(len(y)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _booster = booster
        _loaded_path = path
        return _booster


def predict_proba(feats: Dict[str, Any], silero: Dict[str, Any] | None = None) -> Dict[str, float]:
    """Return class probabilities, e.g. {'HUMAN': 0.12, 'MACHINE': 0.81, 'IVR': 0.07}."""
    booster = ensure_model()
    import xgboost as xgb

    x = vectorize(feats, silero).reshape(1, -1)
    dmat = xgb.DMatrix(x, feature_names=FEATURE_KEYS)
    probs = booster.predict(dmat)[0]
    out = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    return out


def decide_from_proba(probs: Dict[str, float]) -> Tuple[str, float]:
    status = max(probs, key=probs.get)  # type: ignore[arg-type]
    return status, float(probs[status])


def retrain_from_labeled(
    feature_rows: List[np.ndarray],
    labels: List[str],
    *,
    mix_bootstrap: bool = True,
) -> Dict[str, Any]:
    """Retrain and persist model. Optionally mix bootstrap so rare classes remain."""
    global _booster, _loaded_path

    xs = list(feature_rows)
    ys: List[int] = []
    for lab in labels:
        key = str(lab).strip().upper()
        if key in ("VOICEMAIL", "AM", "FAX", "SIT", "ERROR"):
            key = "MACHINE" if key != "IVR" else "IVR"
        if key == "SIT":
            key = "MACHINE"
        if key not in LABEL_TO_ID:
            key = "MACHINE"
        ys.append(LABEL_TO_ID[key])

    if mix_bootstrap or len(xs) < 30:
        Xb, yb = _make_bootstrap_dataset(n_per_class=200)
        X = np.vstack([matrix_from_rows(xs), Xb]) if xs else Xb
        y = np.concatenate([np.asarray(ys, dtype=np.int32), yb]) if ys else yb
    else:
        X = matrix_from_rows(xs)
        y = np.asarray(ys, dtype=np.int32)

    booster = _train_booster(X, y)
    path = model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    meta = {
        "model_version": MODEL_VERSION,
        "labels": LABELS,
        "feature_keys": FEATURE_KEYS,
        "source": "retrain_labeled",
        "n_labeled": len(feature_rows),
        "n_train_total": int(len(y)),
    }
    meta_path().write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    with _lock:
        _booster = booster
        _loaded_path = path
    return meta


def model_status() -> Dict[str, Any]:
    path = model_path()
    meta: Dict[str, Any] = {}
    if meta_path().exists():
        try:
            meta = json.loads(meta_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    return {
        "model_path": str(path),
        "model_exists": path.exists(),
        "loaded": _booster is not None,
        "meta": meta,
    }
