"""
OpenAMD Advanced AI Engine — Hybrid (Heuristic + Silero VAD)

Combines the Phase-3 outbound heuristic AMD rules with Silero VAD speech
features. Prefer HUMAN when uncertain so live agents get calls.
Blank / near-silent audio is disposed as MACHINE when enabled in portal
Settings (blank_as_machine). Only return MACHINE/IVR/SIT with strong
evidence otherwise.
"""

from __future__ import annotations

import io
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

from app.ai import silero_vad
from app.amd_settings import is_blank_as_machine_enabled
from app.locale_packs import resolve_locale_pack

ENGINE_INFO = {
    "name": "OpenAMD Hybrid (Heuristic + Silero)",
    "model": "Rule-based acoustic features + Silero VAD ONNX",
    "version": "4.1.0",
    "runtime": "NumPy + SoundFile + ONNX Runtime (Silero)",
}


@dataclass
class AnalysisResult:
    status: str
    confidence: float
    processing_ms: int
    audio_seconds: float
    details: Dict[str, Any]


def _load_audio(data: bytes, target_sr: int = 8000) -> Tuple[np.ndarray, int, float]:
    """Return (audio, sample_rate, peak_before_normalize)."""
    if sf is None:
        raise RuntimeError("soundfile is not installed")

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr and len(audio) > 0:
        duration = len(audio) / float(sr)
        new_len = max(1, int(duration * target_sr))
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sr = target_sr

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1e-6:
        audio = audio / peak

    return audio.astype(np.float32), sr, peak


def _is_blank(
    feats: Dict[str, float],
    silero: Optional[Dict[str, Any]] = None,
) -> bool:
    """True when audio is empty, too short, or has no usable speech."""
    duration = float(feats.get("duration", 0.0))
    peak = float(feats.get("peak", 1.0))
    rms = float(feats.get("rms", 0.0))
    speech_ratio = float(feats.get("speech_ratio", 0.0))

    if duration < 0.6:
        return True
    # Peak before normalize — true silence / near-silence files
    if peak < 0.008:
        return True
    if peak < 0.02 and speech_ratio < 0.08:
        return True
    if rms < 0.02 and speech_ratio < 0.03:
        return True

    if silero and silero.get("ok"):
        s_ratio = float(silero.get("speech_ratio", 0.0))
        s_mean = float(silero.get("mean_prob", 0.0))
        s_long = float(silero.get("longest_speech_ms", 0.0))
        # No meaningful speech across the clip
        if s_ratio < 0.05 and s_mean < 0.15 and s_long < 200:
            return True

    return False


def _frame_energy(audio: np.ndarray, sr: int, frame_ms: int = 20) -> np.ndarray:
    frame = max(1, int(sr * frame_ms / 1000))
    if len(audio) < frame:
        return np.array([float(np.mean(audio ** 2))])
    n = len(audio) // frame
    frames = audio[: n * frame].reshape(n, frame)
    return np.mean(frames ** 2, axis=1)


def _burst_stats(mask: np.ndarray, frame_ms: int = 20) -> Dict[str, float]:
    if len(mask) == 0:
        return {
            "speech_ratio": 0.0,
            "num_bursts": 0.0,
            "longest_burst_ms": 0.0,
            "longest_silence_ms": 0.0,
            "avg_burst_ms": 0.0,
        }

    bursts = []
    silences = []
    cur = 0
    state = bool(mask[0])
    for v in mask:
        if bool(v) == state:
            cur += 1
        else:
            (bursts if state else silences).append(cur * frame_ms)
            state = bool(v)
            cur = 1
    (bursts if state else silences).append(cur * frame_ms)

    return {
        "speech_ratio": float(np.mean(mask)),
        "num_bursts": float(len(bursts)),
        "longest_burst_ms": float(max(bursts) if bursts else 0),
        "longest_silence_ms": float(max(silences) if silences else 0),
        "avg_burst_ms": float(np.mean(bursts) if bursts else 0),
    }


def _detect_beep(audio: np.ndarray, sr: int) -> Tuple[bool, float]:
    """Voicemail beep: strong narrow tone near end of greeting."""
    if len(audio) < int(sr * 0.5):
        return False, 0.0

    segment = audio[-sr:]
    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)

    band = (freqs >= 850) & (freqs <= 1100)
    if not np.any(band):
        return False, 0.0

    band_spec = spectrum[band]
    peak = float(np.max(band_spec))
    mean = float(np.mean(spectrum) + 1e-9)
    ratio = peak / mean
    is_beep = ratio > 25.0 and peak > np.percentile(spectrum, 97)
    conf = min(0.99, max(0.0, (ratio - 20.0) / 30.0))
    return bool(is_beep), float(conf)


def _detect_sit(audio: np.ndarray, sr: int) -> Tuple[bool, float]:
    """Real SIT tones are dual-frequency (e.g. 914+1429, 1777+1371 Hz)."""
    if len(audio) < int(sr * 1.0):
        return False, 0.0

    segment = audio[: min(len(audio), sr * 2)]
    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)

    bands = [
        (900, 950),
        (1350, 1450),
        (1750, 1800),
    ]
    peaks = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs <= hi)
        if np.any(m):
            peaks.append(float(np.max(spectrum[m])))
        else:
            peaks.append(0.0)

    overall = float(np.mean(spectrum) + 1e-9)
    strong = [p for p in peaks if p > overall * 18.0]
    if len(strong) >= 2:
        return True, 0.85
    return False, 0.0


def _classify_heuristic(
    feats: Dict[str, float],
    beep: bool,
    sit: bool,
    pack: Dict[str, float],
) -> Tuple[str, float]:
    duration = feats.get("duration", 0.0)
    speech_ratio = feats.get("speech_ratio", 0.0)
    num_bursts = feats.get("num_bursts", 0.0)
    longest_burst = feats.get("longest_burst_ms", 0.0)
    longest_silence = feats.get("longest_silence_ms", 0.0)

    if sit:
        return "SIT", 0.85

    if beep:
        return "MACHINE", max(0.9, feats.get("beep_conf", 0.9))

    # Blank / near-silent — MACHINE when setting enabled (portal Settings)
    if _is_blank(feats):
        if is_blank_as_machine_enabled():
            return "MACHINE", 0.92
        return "HUMAN", 0.55

    if (
        duration >= pack["machine_duration_min"]
        and longest_burst >= pack["machine_longest_burst_ms"]
        and num_bursts >= pack["machine_num_bursts"]
    ):
        if (
            longest_silence > pack["machine_ivr_silence_ms"]
            and num_bursts >= pack["machine_ivr_bursts"]
        ):
            return "IVR", 0.8
        return "MACHINE", 0.88

    if (
        duration >= pack["machine_dense_duration"]
        and num_bursts >= pack["machine_dense_bursts"]
        and speech_ratio > pack["machine_dense_speech_ratio"]
    ):
        return "MACHINE", 0.8

    if num_bursts <= pack["human_max_bursts"] and longest_burst <= pack["human_max_burst_ms"]:
        return "HUMAN", 0.82

    if (
        longest_burst >= pack["machine_long_speech_ms"]
        and speech_ratio > pack["machine_long_speech_ratio"]
        and duration >= pack["machine_long_duration"]
    ):
        return "MACHINE", 0.7

    return "HUMAN", 0.7


def _fuse_with_silero(
    status: str,
    confidence: float,
    feats: Dict[str, float],
    silero: Dict[str, Any],
    pack: Dict[str, float],
) -> Tuple[str, float, str]:
    """Blend heuristic decision with Silero VAD speech structure.

    Returns (status, confidence, fuse_note).
    """
    # Blank / no speech → MACHINE when portal setting is enabled
    if _is_blank(feats, silero) and is_blank_as_machine_enabled():
        return "MACHINE", max(float(confidence), 0.92), "blank_silence"

    if not silero.get("ok"):
        return status, confidence, "heuristic_only"

    duration = float(feats.get("duration", 0.0))
    s_ratio = float(silero.get("speech_ratio", 0.0))
    s_segs = int(silero.get("num_segments", 0))
    s_long = float(silero.get("longest_speech_ms", 0.0))
    s_mean = float(silero.get("mean_prob", 0.0))
    strong_m = float(pack["strong_machine_conf"])

    # Strong acoustic events already decided — Silero only confirms confidence
    if status in ("SIT",):
        return status, confidence, "heuristic_sit"

    if status == "MACHINE" and confidence >= strong_m:
        # Beep / blank / strong machine — keep
        if s_long >= 1500 or s_ratio >= 0.4:
            return status, min(0.99, confidence + 0.05), "agree_machine_strong"
        return status, confidence, "heuristic_machine_strong"

    if status == "IVR" and confidence >= 0.75:
        return status, confidence, "heuristic_ivr"

    # Silero: long continuous speech greeting → machine / IVR
    if (
        duration >= pack["silero_long_duration"]
        and s_long >= pack["silero_long_speech_ms"]
        and s_ratio >= pack["silero_long_speech_ratio"]
        and s_segs >= pack["silero_long_segments"]
    ):
        if s_segs >= pack["silero_ivr_segments"] and s_ratio >= pack["silero_ivr_speech_ratio"]:
            return "IVR", max(confidence, 0.82), "silero_ivr_script"
        return "MACHINE", max(confidence, 0.84), "silero_long_greeting"

    # Silero: many speech islands over 2s → scripted machine
    if (
        duration >= pack["silero_long_duration"]
        and s_segs >= pack["silero_many_segments"]
        and s_ratio >= pack["silero_many_speech_ratio"]
    ):
        return "MACHINE", max(confidence, 0.8), "silero_many_segments"

    # Silero: short sparse speech → human ("hello?", "yeah?")
    # Do not override high-confidence MACHINE (beep / blank / strong AM)
    if (
        s_segs <= pack["silero_human_max_segments"]
        and s_long <= pack["silero_human_max_speech_ms"]
        and s_ratio <= pack["silero_human_max_speech_ratio"]
    ):
        if status == "MACHINE" and confidence < strong_m:
            return "HUMAN", max(0.75, 1.0 - confidence + 0.2), "silero_override_human"
        if status == "HUMAN":
            return "HUMAN", max(confidence, 0.85), "agree_human_short"

    # Both sides lean human
    if (
        status == "HUMAN"
        and s_mean < pack["silero_quiet_mean_prob"]
        and duration < pack["silero_quiet_duration"]
    ):
        return "HUMAN", max(confidence, 0.78), "agree_human_quiet"

    # Mild disagreement: prefer HUMAN (agent connect rate)
    if (
        status == "MACHINE"
        and confidence < pack["prefer_human_machine_conf"]
        and s_long < pack["prefer_human_speech_ms"]
    ):
        return "HUMAN", 0.72, "prefer_human_uncertain"

    if (
        status == "HUMAN"
        and s_long >= pack["upgrade_machine_speech_ms"]
        and s_ratio >= pack["upgrade_machine_speech_ratio"]
        and duration >= pack["upgrade_machine_duration"]
    ):
        return "MACHINE", 0.76, "silero_upgrade_machine"

    return status, confidence, "heuristic_primary"


def analyze_audio(
    data: bytes,
    sample_hint_sr: int = 8000,
    *,
    locale_pack_enabled: bool = False,
    locale_pack: str = "usa",
) -> AnalysisResult:
    t0 = time.perf_counter()

    try:
        audio, sr, peak = _load_audio(data, target_sr=sample_hint_sr)
    except Exception as exc:
        ms = int((time.perf_counter() - t0) * 1000)
        if is_blank_as_machine_enabled():
            return AnalysisResult(
                status="MACHINE",
                confidence=0.9,
                processing_ms=ms,
                audio_seconds=0.0,
                details={
                    "error": str(exc),
                    "fallback": "MACHINE",
                    "fuse_note": "blank_or_unreadable",
                    "engine": "hybrid",
                },
            )
        return AnalysisResult(
            status="HUMAN",
            confidence=0.4,
            processing_ms=ms,
            audio_seconds=0.0,
            details={"error": str(exc), "fallback": "HUMAN", "engine": "hybrid"},
        )

    duration = float(len(audio) / sr) if sr else 0.0
    energy = _frame_energy(audio, sr, frame_ms=20)
    thr = float(np.percentile(energy, 35) * 2.0 + 1e-6)
    mask = energy > thr
    stats = _burst_stats(mask, frame_ms=20)

    beep, beep_conf = _detect_beep(audio, sr)
    sit, sit_conf = _detect_sit(audio, sr)

    feats = {
        "duration": duration,
        "sample_rate": float(sr),
        "energy_threshold": thr,
        "peak": float(peak),
        "rms": float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0,
        "beep": float(beep),
        "beep_conf": beep_conf,
        "sit": float(sit),
        "sit_conf": sit_conf,
        **stats,
    }

    pack_name, pack = resolve_locale_pack(
        enabled=bool(locale_pack_enabled),
        pack=locale_pack,
    )
    h_status, h_confidence = _classify_heuristic(feats, beep, sit, pack)
    silero = silero_vad.analyze_speech(audio, sr=sr, threshold=0.5)
    status, confidence, fuse_note = _fuse_with_silero(
        h_status, h_confidence, feats, silero, pack
    )

    details: Dict[str, Any] = {
        **feats,
        "engine": "hybrid",
        "locale_pack_enabled": bool(locale_pack_enabled),
        "locale_pack": pack_name,
        "blank_as_machine": is_blank_as_machine_enabled(),
        "heuristic_status": h_status,
        "heuristic_confidence": round(float(h_confidence), 4),
        "fuse_note": fuse_note,
        "silero": {
            "ok": bool(silero.get("ok")),
            "speech_ratio": round(float(silero.get("speech_ratio", 0.0)), 4),
            "num_segments": int(silero.get("num_segments", 0)),
            "longest_speech_ms": round(float(silero.get("longest_speech_ms", 0.0)), 1),
            "mean_prob": round(float(silero.get("mean_prob", 0.0)), 4),
            "error": silero.get("error", ""),
        },
        **silero_vad.status_info(),
    }

    ms = int((time.perf_counter() - t0) * 1000)
    return AnalysisResult(
        status=status,
        confidence=round(float(confidence), 4),
        processing_ms=ms,
        audio_seconds=round(duration, 3),
        details=details,
    )


def result_to_dict(result: AnalysisResult) -> Dict[str, Any]:
    return asdict(result)
