"""Silero VAD (ONNX) helper for Hybrid OpenAMD.

Pure onnxruntime + NumPy — no torch required at inference time.
Model is loaded from MODELS_DIR or downloaded once on first use.

ONNX I/O matches snakers4/silero-vad OnnxWrapper:
  inputs:  input [B, context+window], state [2, B, 128], sr int64
  outputs: prob, state
  8 kHz window=256, context=32; 16 kHz window=512, context=64
"""

from __future__ import annotations

import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

_MODEL_URLS = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
    "https://cdn.jsdelivr.net/gh/snakers4/silero-vad@master/src/silero_vad/data/silero_vad.onnx",
)

_lock = threading.Lock()
_session: Optional[Any] = None
_model_path: Optional[Path] = None
_load_error: str = ""


def _default_model_path() -> Path:
    try:
        from app.config import get_settings

        return Path(get_settings().MODELS_DIR) / "silero_vad.onnx"
    except Exception:
        return Path("/opt/openamd/models/silero_vad.onnx")


def _download_model(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    tmp = dest.with_suffix(".onnx.download")
    for url in _MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size < 50_000:
                raise RuntimeError(f"downloaded file too small from {url}")
            tmp.replace(dest)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download Silero VAD model: {last_err}")


def ensure_model(path: Optional[Path] = None) -> Path:
    dest = Path(path) if path else _default_model_path()
    if dest.exists() and dest.stat().st_size > 50_000:
        return dest
    _download_model(dest)
    return dest


def _get_session() -> Optional[Any]:
    global _session, _model_path, _load_error
    if _session is not None:
        return _session
    if ort is None:
        _load_error = "onnxruntime not installed"
        return None

    with _lock:
        if _session is not None:
            return _session
        try:
            model = ensure_model()
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            _session = ort.InferenceSession(
                str(model),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            _model_path = model
            _load_error = ""
        except Exception as exc:  # noqa: BLE001
            _load_error = str(exc)
            _session = None
    return _session


def available() -> bool:
    return _get_session() is not None


def status_info() -> Dict[str, Any]:
    sess = _get_session()
    return {
        "silero_available": sess is not None,
        "silero_model_path": str(_model_path) if _model_path else "",
        "silero_error": _load_error,
    }


def _to_8k(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    if sr == 8000:
        return audio.astype(np.float32), 8000
    if sr == 16000:
        return audio.astype(np.float32), 16000
    duration = len(audio) / float(sr) if sr else 0.0
    new_len = max(1, int(duration * 8000))
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32), 8000
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    x_new = np.linspace(0, 1, num=new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32), 8000


def _run_vad(audio: np.ndarray, sr: int = 8000) -> Tuple[np.ndarray, int]:
    """Return per-chunk speech probabilities and window size in samples."""
    session = _get_session()
    if session is None:
        raise RuntimeError(_load_error or "Silero VAD unavailable")

    audio, sr = _to_8k(audio, sr)
    window = 256 if sr == 8000 else 512
    context_size = 32 if sr == 8000 else 64

    if len(audio) < window:
        audio = np.pad(audio, (0, window - len(audio)))
    n = (len(audio) // window) * window
    audio = audio[:n].astype(np.float32)

    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, context_size), dtype=np.float32)
    sr_arr = np.array(sr, dtype=np.int64)

    probs: List[float] = []
    for i in range(0, len(audio), window):
        chunk = audio[i : i + window].reshape(1, -1)
        x = np.concatenate([context, chunk], axis=1)
        ort_outs = session.run(
            None,
            {"input": x, "state": state, "sr": sr_arr},
        )
        out, state = ort_outs[0], ort_outs[1]
        probs.append(float(np.squeeze(out)))
        context = x[:, -context_size:]

    return np.asarray(probs, dtype=np.float32), window


def _speech_spans(
    mask: np.ndarray,
    frame_ms: float,
    *,
    min_frames: int = 2,
    merge_gap_ms: float = 200.0,
) -> List[Tuple[float, float]]:
    """Speech islands as (start_s, end_s), merged across short pauses.

    Islands shorter than `min_frames` (one 32 ms frame is a click, not a
    syllable) are dropped before merging.
    """
    spans: List[Tuple[int, int]] = []
    start = -1
    for i, v in enumerate(mask):
        if bool(v):
            if start < 0:
                start = i
        elif start >= 0:
            if i - start >= min_frames:
                spans.append((start, i))
            start = -1
    if start >= 0 and len(mask) - start >= min_frames:
        spans.append((start, len(mask)))

    max_gap = int(merge_gap_ms / frame_ms) if frame_ms > 0 else 0
    merged: List[Tuple[int, int]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a * frame_ms / 1000.0, b * frame_ms / 1000.0) for a, b in merged]


def _tail_speech_ms(
    spans: List[Tuple[float, float]],
    clip_seconds: float,
    *,
    end_slack_ms: float = 130.0,
) -> float:
    """Length of the speech island that is still running when the clip ends.

    0 when the clip ends in silence. A caller who is still talking at the cut
    after more than a second is reading a greeting, not answering "hello".
    """
    if not spans:
        return 0.0
    a, b = spans[-1]
    if (clip_seconds - b) * 1000.0 > end_slack_ms:
        return 0.0
    return max(0.0, (b - a) * 1000.0)


def _no_speech(error: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "speech_ratio": 0.0,
        "num_segments": 0,
        "longest_speech_ms": 0.0,
        "avg_speech_prob": 0.0,
        "mean_prob": 0.0,
        "max_prob": 0.0,
        "spans": [],
        "total_speech_ms": 0.0,
        "tail_speech_ms": 0.0,
    }


def analyze_speech(
    audio: np.ndarray,
    sr: int = 8000,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Derive AMD-relevant speech stats from Silero VAD probabilities."""
    if len(audio) == 0:
        return _no_speech("empty audio")

    try:
        probs, window = _run_vad(audio, sr=sr)
        effective_sr = 8000 if sr not in (8000, 16000) else sr
    except Exception as exc:  # noqa: BLE001
        return _no_speech(str(exc))

    frame_ms = (window / float(effective_sr)) * 1000.0
    mask = probs >= threshold

    segments = 0
    longest = 0
    cur = 0
    prev = False
    for v in mask:
        if bool(v):
            if not prev:
                segments += 1
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
        prev = bool(v)

    speech_probs = probs[mask] if np.any(mask) else np.array([], dtype=np.float32)
    spans = _speech_spans(mask, frame_ms)
    clip_seconds = len(probs) * frame_ms / 1000.0
    return {
        "max_prob": float(np.max(probs)) if len(probs) else 0.0,
        "spans": [(round(a, 3), round(b, 3)) for a, b in spans],
        "total_speech_ms": float(sum(b - a for a, b in spans) * 1000.0),
        "tail_speech_ms": float(_tail_speech_ms(spans, clip_seconds)),
        "ok": True,
        "error": "",
        "speech_ratio": float(np.mean(mask)) if len(mask) else 0.0,
        "num_segments": int(segments),
        "longest_speech_ms": float(longest * frame_ms),
        "avg_speech_prob": float(np.mean(speech_probs)) if len(speech_probs) else 0.0,
        "mean_prob": float(np.mean(probs)) if len(probs) else 0.0,
        "threshold": float(threshold),
        "frame_ms": float(frame_ms),
        "num_frames": int(len(probs)),
    }
