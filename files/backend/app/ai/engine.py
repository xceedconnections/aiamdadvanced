"""
OpenAMD Advanced AI Engine — Hybrid (Heuristic + Silero VAD)

Combines the Phase-3 outbound heuristic AMD rules with Silero VAD speech
features. Prefer HUMAN when uncertain so live agents get calls.
Short live greetings ("hello") must not be treated as voicemail.
Blank / near-silent audio is disposed as BLANK (never to agents). VICIdial
receives MACHINE (AA). Only return MACHINE/IVR/SIT with strong evidence
otherwise.
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
    "version": "5.1.2",
    "runtime": "NumPy + SoundFile + ONNX Runtime (Silero); optional XGBoost + Whisper",
}


@dataclass
class AnalysisResult:
    status: str
    confidence: float
    processing_ms: int
    audio_seconds: float
    details: Dict[str, Any]


def _load_audio(
    data: bytes, target_sr: int = 8000
) -> Tuple[np.ndarray, int, float, float, float, float]:
    """Return (audio, sr, peak, raw_rms, abs_speech_ratio, abs_speech_ms).

    Soft-normalizes so a single click/spike cannot crush real speech used by
    VAD and blank detection (common on short AMD windows).
    abs_* metrics are measured on the original level before normalize.
    """
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
    raw_rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

    # Absolute speech (pre-normalize) — independent of clicks / gain
    abs_speech_ratio = 0.0
    abs_speech_ms = 0.0
    if len(audio) > 0 and sr > 0:
        energy_pre = _frame_energy(audio, sr, frame_ms=20)
        if len(energy_pre):
            abs_thr = 0.012
            mask = energy_pre >= abs_thr
            abs_speech_ratio = float(np.mean(mask))
            # longest contiguous run
            longest = 0
            cur = 0
            for v in mask:
                if bool(v):
                    cur += 1
                    longest = max(longest, cur)
                else:
                    cur = 0
            abs_speech_ms = float(longest * 20)

    if peak > 1e-6 and len(audio) >= 80:
        abs_a = np.abs(audio)
        soft = float(np.percentile(abs_a, 99.0))
        if soft > 1e-4 and peak > soft * 2.5:
            audio = np.clip(audio, -soft, soft) / (soft * 1.05)
        else:
            audio = audio / peak
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    elif peak > 1e-6:
        audio = (audio / peak).astype(np.float32)

    return (
        audio.astype(np.float32),
        sr,
        peak,
        raw_rms,
        abs_speech_ratio,
        abs_speech_ms,
    )

def _compute_zcr(audio: np.ndarray) -> float:
    """Zero-crossing rate — high for noise/static, lower for voiced speech."""
    if audio is None or len(audio) < 2:
        return 0.0
    s = np.sign(audio.astype(np.float64))
    s[s == 0] = 1.0
    return float(np.mean(s[:-1] != s[1:]))


def _front_energy_ratio(audio: np.ndarray) -> float:
    """Fraction of energy in the first half of the clip (1.0 = all front-loaded)."""
    if audio is None or len(audio) < 8:
        return 0.5
    mid = len(audio) // 2
    e1 = float(np.sum(audio[:mid] ** 2))
    e2 = float(np.sum(audio[mid:] ** 2))
    return e1 / (e1 + e2 + 1e-12)


def _is_voip_noise_burst(
    feats: Dict[str, float],
    silero: Optional[Dict[str, Any]] = None,
) -> bool:
    """Loud non-speech noise dump with silence on the other side — VoIP artifact.

    Same comfort-noise / early-media burst often appears on every call on a bad
    trunk. Can be front-loaded (noise then silence) OR back-loaded (silence then
    noise). Must not go to agents.
    """
    duration = float(feats.get("duration", 0.0))
    peak = float(feats.get("peak", 0.0))
    front = float(feats.get("front_energy_ratio", 0.5))
    zcr = float(feats.get("zcr", 0.0))
    longest_silence = float(feats.get("longest_silence_ms", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))
    num_bursts = float(feats.get("num_bursts", 0.0))
    beep = float(feats.get("beep", 0.0))
    ring = float(feats.get("ringback", 0.0))
    if beep >= 0.5 or ring >= 0.5:
        return False
    if duration < 0.7 or duration > 4.2:
        return False
    if peak < 0.06:
        return False
    # Energy almost entirely in one half (front dump OR rear dump after lead-in silence)
    one_sided = front >= 0.85 or front <= 0.18
    if not one_sided or longest_silence < 350:
        return False
    # High ZCR = noise/static, not a voiced "hello"
    if zcr >= 0.18:
        return True
    # Extreme one-sided dump + long continuous burst
    if num_bursts <= 2 and longest_burst >= 550 and zcr >= 0.12:
        return True
    # Silero disagrees that the loud dump is speech
    if silero and silero.get("ok"):
        s_mean = float(silero.get("mean_prob", 0.0) or 0.0)
        s_long = float(silero.get("longest_speech_ms", 0.0) or 0.0)
        if longest_silence >= 400 and s_mean < 0.28 and s_long < 900:
            return True
    return False


def _has_audible_voice(feats: Dict[str, float]) -> bool:
    """True when the clip has real voice energy (not digital silence / empty)."""
    peak = float(feats.get("peak", 0.0))
    raw_rms = float(feats.get("raw_rms", feats.get("rms", 0.0)))
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    activity_ratio = float(feats.get("activity_ratio", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))
    abs_ratio = float(feats.get("abs_speech_ratio", 0.0))
    abs_long = float(feats.get("abs_speech_ms", 0.0))

    if abs_long >= 140 and peak >= 0.04:
        return True
    if abs_ratio >= 0.10 and peak >= 0.05:
        return True
    if peak >= 0.08 and longest_burst >= 120 and (
        speech_ratio >= 0.08 or activity_ratio >= 0.12
    ):
        return True
    if peak >= 0.10 and raw_rms >= 0.015 and activity_ratio >= 0.10:
        return True
    if peak >= 0.08 and speech_ratio >= 0.15:
        return True
    if longest_burst >= 200 and peak >= 0.04 and activity_ratio >= 0.15:
        return True
    return False


def _is_line_noise_only(feats: Dict[str, float]) -> bool:
    """True for quiet line/carrier noise with no absolute speech islands.

    Relative percentile activity can look 'busy' on flat noise and must not
    override absolute silence (false HUMAN to agents).
    """
    peak = float(feats.get("peak", 1.0))
    raw_rms = float(feats.get("raw_rms", feats.get("rms", 0.0)))
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    abs_ratio = float(feats.get("abs_speech_ratio", 0.0))
    abs_long = float(feats.get("abs_speech_ms", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))
    if peak >= 0.045:
        return False
    if abs_long >= 100 or abs_ratio >= 0.06:
        return False
    if speech_ratio >= 0.12 and longest_burst >= 200:
        return False
    return raw_rms < 0.014 and peak < 0.045


def _is_blank(
    feats: Dict[str, float],
    silero: Optional[Dict[str, Any]] = None,
) -> bool:
    """True only when audio has no usable voice at all.

    Short AMD windows often contain a quiet 'hello' under 0.6s — those must
    NOT be blank. Silero-alone blank is not enough when energy shows speech.
    """
    duration = float(feats.get("duration", 0.0))
    peak = float(feats.get("peak", 1.0))
    raw_rms = float(feats.get("raw_rms", feats.get("rms", 0.0)))
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    activity_ratio = float(feats.get("activity_ratio", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))
    abs_ratio = float(feats.get("abs_speech_ratio", 0.0))
    abs_long = float(feats.get("abs_speech_ms", 0.0))

    # Empty / digital silence only
    if duration < 0.20:
        return True
    if peak < 0.008:
        return True

    # Quiet carrier / line noise with no absolute speech → blank (not HUMAN)
    if _is_line_noise_only(feats):
        return True

    # VoIP early-media / comfort-noise burst (loud front, silent tail) → blank
    if _is_voip_noise_burst(feats, silero):
        return True

    # Any clear voice → never blank (human hello or short VM fragment)
    if _has_audible_voice(feats):
        return False

    # Short clip with no usable voice energy
    if duration < 0.55:
        return True

    energy_quiet = (
        speech_ratio < 0.10
        and longest_burst < 180
        and raw_rms < 0.025
        and abs_long < 160
        and (
            activity_ratio < 0.20
            or abs_ratio < 0.05
            or peak < 0.04
        )
    )

    if silero and silero.get("ok"):
        s_ratio = float(silero.get("speech_ratio", 0.0))
        s_mean = float(silero.get("mean_prob", 0.0))
        s_long = float(silero.get("longest_speech_ms", 0.0))
        s_segs = int(silero.get("num_segments", 0))
        # Silero hears speech → not blank (unless absolute levels are line-noise)
        if s_long >= 150 and s_mean >= 0.18 and not _is_line_noise_only(feats):
            return False
        if s_segs >= 1 and s_long >= 160 and peak >= 0.05:
            return False
        # Blank only when Silero AND energy agree there is no voice
        if energy_quiet and s_ratio < 0.08 and s_mean < 0.22 and s_long < 280 and s_segs <= 1:
            return True
        if energy_quiet and peak < 0.06 and s_ratio < 0.12 and s_long < 350:
            return True
        return False

    if peak < 0.025 and speech_ratio < 0.12 and activity_ratio < 0.15:
        return True
    if raw_rms < 0.003 and speech_ratio < 0.08:
        return True
    if raw_rms < 0.008 and speech_ratio < 0.05 and (activity_ratio < 0.18 or abs_ratio < 0.05):
        return True
    if peak < 0.04 and speech_ratio < 0.06 and longest_burst < 150 and abs_long < 120:
        return True
    if energy_quiet and peak < 0.05:
        return True

    return False


def _looks_like_spoken_digit(
    feats: Dict[str, float],
    silero: Optional[Dict[str, Any]] = None,
) -> bool:
    """Isolated spoken digit / prompt syllable ('zero', 'one') in a longer AMD window.

    Not a live 'hello'. Short clips (~1s) are usually hello, not keypad digits.
    """
    if float(feats.get("beep", 0.0)) >= 0.5:
        return False
    if float(feats.get("sit", 0.0)) >= 0.5:
        return False
    duration = float(feats.get("duration", 0.0))
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    num_bursts = float(feats.get("num_bursts", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))
    # Digit prompts usually sit in a ~2–3s AMD record with silence around them
    if duration < 1.25 or duration > 3.8:
        return False
    # One compact syllable; speech fills little of a ~3s AMD clip
    if not (num_bursts <= 2 and 100.0 <= longest_burst <= 750.0):
        return False
    if not (0.06 <= speech_ratio <= 0.38):
        return False
    if silero and silero.get("ok"):
        s_long = float(silero.get("longest_speech_ms", 0.0))
        s_segs = int(silero.get("num_segments", 0))
        if s_segs > 2:
            return False
        if s_long > 0 and not (80.0 <= s_long <= 800.0):
            return False
    return True


def _is_truncated_script_clip(feats: Dict[str, float]) -> bool:
    """Ultra-short AMD windows with dense speech are usually truncated VM/IVR.

    Live hello in <0.85s is a single sparse syllable; continuous fill is the
    start of a scripted greeting cut by the AMD timer.
    """
    duration = float(feats.get("duration", 0.0))
    if duration < 0.35 or duration >= 0.90:
        return False
    if float(feats.get("beep", 0.0)) >= 0.5 or float(feats.get("ringback", 0.0)) >= 0.5:
        return False
    peak = float(feats.get("peak", 0.0))
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))
    abs_long = float(feats.get("abs_speech_ms", 0.0))
    num_bursts = float(feats.get("num_bursts", 0.0))
    activity_ratio = float(feats.get("activity_ratio", speech_ratio))
    if peak < 0.05:
        return False
    # Dense fill in a tiny window → script fragment, not "hello"
    if speech_ratio >= 0.42 or activity_ratio >= 0.50:
        return True
    if longest_burst >= 220 and speech_ratio >= 0.28:
        return True
    if abs_long >= 200 and speech_ratio >= 0.25:
        return True
    if num_bursts >= 2 and speech_ratio >= 0.32 and duration < 0.75:
        return True
    return False


def _looks_like_short_human(
    feats: Dict[str, float],
    silero: Optional[Dict[str, Any]] = None,
) -> bool:
    """Live pickup: short 'hello' / 'yeah?' — not a voicemail greeting.

    AMD windows are ~2s and often include leading/trailing silence, so a
    human answer looks sparse. Scripted AM is longer / denser.
    """
    if float(feats.get("beep", 0.0)) >= 0.5:
        return False
    if float(feats.get("sit", 0.0)) >= 0.5:
        return False
    if float(feats.get("ringback", 0.0)) >= 0.5:
        return False
    # Line noise / empty air / VoIP noise burst must never look like a short hello
    if _is_line_noise_only(feats) or _is_blank(feats, silero) or _is_voip_noise_burst(feats, silero):
        return False
    # Truncated dense VM/IVR openings must not look like hello
    if _is_truncated_script_clip(feats):
        return False
    peak = float(feats.get("peak", 0.0))
    abs_long = float(feats.get("abs_speech_ms", 0.0))
    if peak < 0.05 and abs_long < 80:
        return False
    # Spoken digit / IVR prompt syllable is not a live hello
    if _looks_like_spoken_digit(feats, silero):
        return False
    duration = float(feats.get("duration", 0.0))
    # Allow short AMD windows (~0.4–0.6s) that still contain a hello syllable
    if duration < 0.35 or duration > 3.8:
        return False
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    num_bursts = float(feats.get("num_bursts", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))

    # Sub-second windows: only sparse single-syllable hellos count as human
    if duration < 0.90:
        if speech_ratio > 0.38 or longest_burst > 380 or num_bursts >= 3:
            return False
        if abs_long >= 280 and speech_ratio >= 0.30:
            return False

    # Long continuous talk in the AMD window → greeting / IVR, not hello
    if longest_burst >= 1100 and speech_ratio >= 0.38:
        return False
    if speech_ratio >= 0.55 and duration >= 1.8:
        return False
    # Digit/IVR readout is several short words with real speech fill
    # A noisy "hello" can also split into 3 energy blips — do not treat those as IVR
    if num_bursts >= 3 and longest_burst <= 550 and speech_ratio >= 0.28:
        return False

    if silero and silero.get("ok"):
        s_long = float(silero.get("longest_speech_ms", 0.0))
        s_segs = int(silero.get("num_segments", 0))
        s_ratio = float(silero.get("speech_ratio", 0.0))
        if s_long >= 1200 and s_ratio >= 0.32:
            return False
        if s_segs >= 3 and s_ratio >= 0.25:
            return False
        # Ultra-short dense Silero speech → truncated script
        if duration < 0.90 and s_ratio >= 0.40 and s_long >= 200:
            return False
        if s_segs <= 2 and s_long <= 900 and s_long >= 80:
            return True

    if num_bursts <= 3 and longest_burst <= 900 and speech_ratio <= 0.40:
        return True
    return False


def _blank_disposition() -> Tuple[str, float, str]:
    """Internal BLANK disposition (VICIdial maps to MACHINE / AA)."""
    return "BLANK", 0.95, "blank_silence"


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
    """Voicemail beep: narrow tone near end (or mid–end) of greeting.

    US carriers often use ~1000 Hz; some use 900–1400 Hz. AMD clips are short,
    so also scan the last 1.5s in overlapping windows.
    """
    if len(audio) < int(sr * 0.4):
        return False, 0.0

    best_ratio = 0.0
    best_peak = 0.0
    best_p97 = 0.0
    win = max(int(sr * 0.35), 256)
    hop = max(win // 2, 128)
    start = max(0, len(audio) - int(sr * 1.5))
    segment_full = audio[start:]

    for off in range(0, max(1, len(segment_full) - win + 1), hop):
        segment = segment_full[off : off + win]
        if len(segment) < win // 2:
            continue
        window = np.hanning(len(segment))
        spectrum = np.abs(np.fft.rfft(segment * window))
        freqs = np.fft.rfftfreq(len(segment), d=1.0 / sr)
        band = (freqs >= 850) & (freqs <= 1400)
        if not np.any(band):
            continue
        peak = float(np.max(spectrum[band]))
        mean = float(np.mean(spectrum) + 1e-9)
        ratio = peak / mean
        if ratio > best_ratio:
            best_ratio = ratio
            best_peak = peak
            best_p97 = float(np.percentile(spectrum, 97))

    if best_ratio <= 0:
        return False, 0.0

    # Slightly looser than before — many live AMD beeps are short / compressed
    is_beep = best_ratio > 18.0 and best_peak > best_p97 * 0.85
    conf = min(0.99, max(0.0, (best_ratio - 14.0) / 28.0))
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


def _band_peak(spectrum: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0
    return float(np.max(spectrum[m]))


def _detect_ringback(audio: np.ndarray, sr: int) -> Tuple[bool, float]:
    """Audible ringback / ringing (call not answered) — must not go to agents.

    North America: dual-tone 440 Hz + 480 Hz.
    Many other regions: strong ~400–450 Hz tone.
    AMD windows often capture one ring cycle (tone then silence).
    """
    if audio is None or len(audio) < int(sr * 0.45):
        return False, 0.0

    seg = audio[: min(len(audio), int(sr * 1.8))]
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    if rms < 0.015:
        return False, 0.0

    window = np.hanning(len(seg))
    spectrum = np.abs(np.fft.rfft(seg * window))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / sr)
    overall = float(np.mean(spectrum) + 1e-9)

    p440 = _band_peak(spectrum, freqs, 432, 448)
    p480 = _band_peak(spectrum, freqs, 472, 488)
    p425 = _band_peak(spectrum, freqs, 415, 435)
    p400 = _band_peak(spectrum, freqs, 390, 410)
    # Speech/VM energy is broader; ringback is narrowly tonal in 400–490 Hz
    speech_band = _band_peak(spectrum, freqs, 700, 3000)
    low_tone = max(p440, p480, p425, p400)

    # Classic US dual-tone ringback (very distinctive)
    if p440 > overall * 18.0 and p480 > overall * 18.0:
        conf = min(0.99, max(0.9, (min(p440, p480) / overall) / 60.0))
        return True, float(conf)
    if p440 > overall * 14.0 and p480 > overall * 12.0 and low_tone > speech_band * 1.5:
        return True, 0.94
    if p480 > overall * 14.0 and p440 > overall * 12.0 and low_tone > speech_band * 1.5:
        return True, 0.94

    # Single-tone ring (~400/425 Hz) that dominates over speech bands
    if low_tone > overall * 28.0 and low_tone > speech_band * 4.0:
        # Prefer clips that look like tone-then-silence (not continuous speech)
        mid = len(audio) // 2
        e1 = float(np.sum(audio[:mid] ** 2))
        e2 = float(np.sum(audio[mid:] ** 2))
        front = e1 / (e1 + e2 + 1e-12)
        if front >= 0.75:
            return True, 0.9

    return False, 0.0


def _internal_silence_ms(mask: np.ndarray, frame_ms: int = 20) -> float:
    """Longest silence that is not leading or trailing (AM greeting pause)."""
    if len(mask) < 3:
        return 0.0
    silences = []
    cur = 0
    state = bool(mask[0])
    # skip leading silence by starting after first speech
    started = False
    for v in mask:
        speech = bool(v)
        if not started:
            if speech:
                started = True
                state = True
                cur = 1
            continue
        if speech == state:
            cur += 1
        else:
            if not state:
                silences.append(cur * frame_ms)
            state = speech
            cur = 1
    # trailing silence ignored on purpose
    return float(max(silences) if silences else 0.0)


def _detect_voicemail_structure(
    feats: Dict[str, float],
    pack: Dict[str, float],
) -> Tuple[bool, float, str]:
    """North-America / global AM patterns in short AMD windows.

    Covers:
      1) speech → mid pause → more speech (greeting before beep)
      2) dense scripted talk filling a 2s+ clip
      3) choppy / compressed AM (many micro-bursts in ~1.5–2s)

    UK packs set na_vm_aggressive=0 to skip these aggressive rules.
    """
    if float(pack.get("na_vm_aggressive", 1.0)) < 0.5:
        return False, 0.0, ""

    duration = float(feats.get("duration", 0.0))
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    activity_ratio = float(feats.get("activity_ratio", speech_ratio))
    num_bursts = float(feats.get("num_bursts", 0.0))
    longest_burst = float(feats.get("longest_burst_ms", 0.0))
    total_speech_ms = speech_ratio * duration * 1000.0
    internal_sil = float(feats.get("internal_silence_ms", 0.0))

    # --- Short choppy / carrier-compressed AM (no beep, <2s windows) ---
    choppy_min_dur = float(pack.get("vm_choppy_duration_min", 1.35))
    choppy_max_dur = float(pack.get("vm_choppy_duration_max", 2.4))
    choppy_min_bursts = float(pack.get("vm_choppy_min_bursts", 6))
    choppy_max_burst = float(pack.get("vm_choppy_max_burst_ms", 220))
    choppy_min_speech = float(pack.get("vm_choppy_min_speech_ratio", 0.10))
    if (
        choppy_min_dur <= duration <= choppy_max_dur
        and num_bursts >= choppy_min_bursts
        and longest_burst <= choppy_max_burst
        and (speech_ratio >= choppy_min_speech or activity_ratio >= 0.5)
    ):
        return True, 0.84, "na_vm_choppy_short"

    # Continuous low-level energy with many tiny islands (soft AM / network chop)
    if (
        duration >= 1.45
        and duration <= 2.3
        and num_bursts >= 7
        and longest_burst <= 150
        and activity_ratio >= 0.45
    ):
        return True, 0.83, "na_vm_choppy_fragmented"

    min_dur = float(pack.get("vm_struct_duration_min", 1.9))
    min_speech = float(pack.get("vm_struct_speech_ratio", 0.38))
    min_bursts = float(pack.get("vm_struct_min_bursts", 3))
    sil_lo = float(pack.get("vm_struct_silence_lo_ms", 350))
    sil_hi = float(pack.get("vm_struct_silence_hi_ms", 1400))
    min_total_speech = float(pack.get("vm_struct_total_speech_ms", 1100))

    if duration < min_dur or speech_ratio < min_speech:
        return False, 0.0, ""
    if num_bursts < min_bursts:
        return False, 0.0, ""
    if total_speech_ms < min_total_speech:
        return False, 0.0, ""

    # Classic AM: internal pause between phrases
    if sil_lo <= internal_sil <= sil_hi and num_bursts >= min_bursts:
        conf = 0.86
        if duration >= 2.4 and speech_ratio >= 0.45:
            conf = 0.9
        return True, conf, "na_vm_mid_pause"

    # Dense scripted talk filling most of a 2s+ AMD clip (no beep yet)
    if (
        duration >= float(pack.get("vm_dense_duration", 2.4))
        and speech_ratio >= float(pack.get("vm_dense_speech_ratio", 0.48))
        and num_bursts >= float(pack.get("vm_dense_bursts", 4))
        and longest_burst >= float(pack.get("vm_dense_min_burst_ms", 400))
    ):
        return True, 0.84, "na_vm_dense_greeting"

    return False, 0.0, ""


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

    # Audible ringing / ringback — call not answered
    if float(feats.get("ringback", 0.0)) >= 0.5:
        return "MACHINE", max(0.93, float(feats.get("ringback_conf", 0.93)))

    # Blank / near-silent — never pass to agents when blank detection is on
    if is_blank_as_machine_enabled() and (
        _is_blank(feats) or _is_voip_noise_burst(feats)
    ):
        return _blank_disposition()[:2]

    # Truncated dense AMD clip (often <0.7s of VM/IVR opening) → MACHINE
    if _is_truncated_script_clip(feats):
        return "MACHINE", 0.88

    # Short live "hello" before aggressive NA voicemail rules
    if _looks_like_short_human(feats):
        return "HUMAN", 0.86

    vm_hit, vm_conf, _vm_note = _detect_voicemail_structure(feats, pack)
    if vm_hit:
        return "MACHINE", float(vm_conf)

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
    # Blank / no speech → BLANK (VICIdial AA) when portal setting is enabled
    if _is_blank(feats, silero) and is_blank_as_machine_enabled():
        b_status, b_conf, b_note = _blank_disposition()
        return b_status, max(float(confidence), b_conf), b_note

    # Live pickup ("hello") wins over choppy-VM / XGB-style machine guesses
    if _looks_like_short_human(feats, silero) and status != "SIT":
        return "HUMAN", max(float(confidence), 0.86), "short_human_greeting"

    if not silero.get("ok"):
        return status, confidence, "heuristic_only"

    duration = float(feats.get("duration", 0.0))
    speech_ratio = float(feats.get("speech_ratio", 0.0))
    s_ratio = float(silero.get("speech_ratio", 0.0))
    s_segs = int(silero.get("num_segments", 0))
    s_long = float(silero.get("longest_speech_ms", 0.0))
    s_mean = float(silero.get("mean_prob", 0.0))
    strong_m = float(pack["strong_machine_conf"])
    dense_speech = duration >= 2.1 and speech_ratio >= 0.38
    vm_hit, _, vm_note = _detect_voicemail_structure(feats, pack)

    # Strong acoustic events already decided — Silero only confirms confidence
    if status in ("SIT",):
        return status, confidence, "heuristic_sit"

    if status == "MACHINE" and (confidence >= strong_m or vm_hit):
        # Beep / blank / USA VM structure / strong machine — keep
        if s_long >= 1500 or s_ratio >= 0.4:
            return status, min(0.99, confidence + 0.05), "agree_machine_strong"
        return status, confidence, vm_note or "heuristic_machine_strong"

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
    # Do not override blank / high-confidence MACHINE or dense USA greetings
    if (
        s_segs <= pack["silero_human_max_segments"]
        and s_long <= pack["silero_human_max_speech_ms"]
        and s_ratio <= pack["silero_human_max_speech_ratio"]
        and not _is_blank(feats, silero)
    ):
        if status == "MACHINE" and confidence < strong_m and not dense_speech and not vm_hit:
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

    # Mild disagreement: prefer HUMAN only when not blank and not a dense/scripted clip
    if (
        status == "MACHINE"
        and confidence < pack["prefer_human_machine_conf"]
        and s_long < pack["prefer_human_speech_ms"]
        and not dense_speech
        and not vm_hit
        and not _is_blank(feats, silero)
    ):
        return "HUMAN", 0.72, "prefer_human_uncertain"

    if (
        status == "HUMAN"
        and s_long >= pack["upgrade_machine_speech_ms"]
        and s_ratio >= pack["upgrade_machine_speech_ratio"]
        and duration >= pack["upgrade_machine_duration"]
    ):
        return "MACHINE", 0.76, "silero_upgrade_machine"

    # Heuristic said HUMAN but clip looks like USA AM structure
    if status == "HUMAN" and vm_hit:
        return "MACHINE", max(0.84, confidence), vm_note or "na_vm_structure"

    return status, confidence, "heuristic_primary"


def analyze_audio(
    data: bytes,
    sample_hint_sr: int = 8000,
    *,
    locale_pack_enabled: bool = False,
    locale_pack: str = "usa",
    amd_settings: Optional[Dict[str, Any]] = None,
    call_meta: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    t0 = time.perf_counter()

    try:
        audio, sr, peak, raw_rms, abs_speech_ratio, abs_speech_ms = _load_audio(
            data, target_sr=sample_hint_sr
        )
    except Exception as exc:
        ms = int((time.perf_counter() - t0) * 1000)
        if is_blank_as_machine_enabled():
            b_status, b_conf, b_note = _blank_disposition()
            return AnalysisResult(
                status=b_status,
                confidence=b_conf,
                processing_ms=ms,
                audio_seconds=0.0,
                details={
                    "error": str(exc),
                    "fallback": b_status,
                    "fuse_note": b_note,
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
    # Lower threshold activity — soft continuous AM often fails the primary mask
    thr_lo = float(np.percentile(energy, 20) * 1.5 + 1e-6)
    activity_ratio = float(np.mean(energy > thr_lo)) if len(energy) else 0.0
    stats = _burst_stats(mask, frame_ms=20)

    beep, beep_conf = _detect_beep(audio, sr)
    sit, sit_conf = _detect_sit(audio, sr)
    ring, ring_conf = _detect_ringback(audio, sr)
    internal_sil = _internal_silence_ms(mask, frame_ms=20)

    feats = {
        "duration": duration,
        "sample_rate": float(sr),
        "energy_threshold": thr,
        "peak": float(peak),
        "raw_rms": float(raw_rms),
        "rms": float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0,
        "beep": float(beep),
        "beep_conf": beep_conf,
        "sit": float(sit),
        "sit_conf": sit_conf,
        "ringback": float(ring),
        "ringback_conf": float(ring_conf),
        "internal_silence_ms": float(internal_sil),
        "activity_ratio": float(activity_ratio),
        "abs_speech_ratio": float(abs_speech_ratio),
        "abs_speech_ms": float(abs_speech_ms),
        "zcr": float(_compute_zcr(audio)),
        "front_energy_ratio": float(_front_energy_ratio(audio)),
        **stats,
    }

    pack_name, pack = resolve_locale_pack(
        enabled=bool(locale_pack_enabled),
        pack=locale_pack,
    )
    h_status, h_confidence = _classify_heuristic(feats, beep, sit, pack)
    # Slightly lower threshold so quiet telephony hello/VM is not missed
    silero = silero_vad.analyze_speech(audio, sr=sr, threshold=0.4)

    # Ringback / ringing (not answered) — never send to agents
    if float(feats.get("ringback", 0.0)) >= 0.5:
        ms = int((time.perf_counter() - t0) * 1000)
        return AnalysisResult(
            status="MACHINE",
            confidence=max(0.93, float(feats.get("ringback_conf", 0.93))),
            processing_ms=ms,
            audio_seconds=round(duration, 3),
            details={
                **feats,
                "engine": "hybrid",
                "locale_pack_enabled": bool(locale_pack_enabled),
                "locale_pack": pack_name,
                "blank_as_machine": is_blank_as_machine_enabled(),
                "heuristic_status": h_status,
                "heuristic_confidence": round(float(h_confidence), 4),
                "fuse_note": "ringback_tone",
                "ringback_detected": True,
                "ml_pipeline_enabled": False,
                "silero": {
                    "ok": bool(silero.get("ok")),
                    "speech_ratio": round(float(silero.get("speech_ratio", 0.0)), 4),
                    "num_segments": int(silero.get("num_segments", 0)),
                    "longest_speech_ms": round(float(silero.get("longest_speech_ms", 0.0)), 1),
                    "mean_prob": round(float(silero.get("mean_prob", 0.0)), 4),
                },
            },
        )

    # VoIP trunk noise (same burst on every call) → BLANK before fuse/ML can false-HUMAN
    if is_blank_as_machine_enabled() and _is_voip_noise_burst(feats, silero):
        b_status, b_conf, _ = _blank_disposition()
        ms = int((time.perf_counter() - t0) * 1000)
        return AnalysisResult(
            status=b_status,
            confidence=b_conf,
            processing_ms=ms,
            audio_seconds=round(duration, 3),
            details={
                **feats,
                "engine": "hybrid",
                "locale_pack_enabled": bool(locale_pack_enabled),
                "locale_pack": pack_name,
                "blank_as_machine": True,
                "heuristic_status": h_status,
                "heuristic_confidence": round(float(h_confidence), 4),
                "fuse_note": "voip_front_noise_burst",
                "voip_noise_burst": True,
                "ml_pipeline_enabled": False,
                "silero": {
                    "ok": bool(silero.get("ok")),
                    "speech_ratio": round(float(silero.get("speech_ratio", 0.0)), 4),
                    "num_segments": int(silero.get("num_segments", 0)),
                    "longest_speech_ms": round(float(silero.get("longest_speech_ms", 0.0)), 1),
                    "mean_prob": round(float(silero.get("mean_prob", 0.0)), 4),
                },
            },
        )

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
        "ml_pipeline_enabled": False,
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

    # Optional ML stage — only when effective settings enable it
    # (global Settings or per-server override). Off = no XGBoost/Whisper import.
    ml_on = bool((amd_settings or {}).get("ml_pipeline_enabled"))
    if not ml_on and amd_settings is None:
        from app.amd_settings import is_ml_pipeline_enabled

        ml_on = is_ml_pipeline_enabled()

    if ml_on:
        try:
            from app.ai.ml_pipeline import run_ml_pipeline

            ml_status, ml_conf, ml_details = run_ml_pipeline(
                audio=audio,
                sr=sr,
                feats=feats,
                silero=silero,
                hybrid_status=status,
                hybrid_confidence=confidence,
                cfg=amd_settings,
                call_meta=call_meta,
            )
            status, confidence = ml_status, ml_conf
            details["ml_pipeline_enabled"] = True
            details["engine"] = "hybrid+ml"
            details["ml"] = ml_details
            if ml_details.get("class_probs"):
                details["class_probs"] = ml_details["class_probs"]
        except Exception as exc:
            details["ml_pipeline_enabled"] = True
            details["ml_error"] = str(exc)
            # Keep hybrid decision on any ML failure

    # Safety net: XGBoost often scores a short "hello" as MACHINE at 98%+
    # Never undo Whisper IVR / digit-readout / voicemail.
    ml_block = details.get("ml") if isinstance(details.get("ml"), dict) else {}
    w_block = ml_block.get("whisper") if isinstance(ml_block.get("whisper"), dict) else {}
    w_cue = str(w_block.get("cue") or "")
    w_text = str(w_block.get("transcript") or "")
    whisper_non_human = (
        w_cue in ("ivr", "ivr_digits", "voicemail", "digits")
        or w_cue.startswith("custom_phrase")
    )
    if not whisper_non_human and w_text:
        try:
            from app.ai.whisper_amd import classify_transcript, is_number_readout

            whisper_non_human = is_number_readout(w_text)
            if not whisper_non_human:
                w_status, _, w_cue2 = classify_transcript(w_text)
                if w_status == "MACHINE" or w_cue2 in ("voicemail", "ivr_digits", "ivr") or str(
                    w_cue2
                ).startswith("custom_phrase"):
                    whisper_non_human = True
                    details["whisper_machine_from_text"] = True
        except Exception:
            whisper_non_human = False
    # Dense/long greeting acoustics must not be overridden to HUMAN by short-human net
    dense_greeting = (
        float(feats.get("duration", 0.0)) >= 1.8
        and (
            float(feats.get("speech_ratio", 0.0)) >= 0.35
            or float(silero.get("longest_speech_ms", 0.0) or 0) >= 1200
            or float(feats.get("abs_speech_ms", 0.0)) >= 600
        )
    )
    if (
        status in ("MACHINE", "IVR")
        and float(feats.get("beep", 0.0)) < 0.5
        and float(feats.get("ringback", 0.0)) < 0.5
        and _looks_like_short_human(feats, silero)
        and not whisper_non_human
        and not dense_greeting
    ):
        status, confidence = "HUMAN", max(float(confidence), 0.86)
        details["short_human_safety_override"] = True
        details["fuse_note"] = "short_human_greeting"
    elif (
        status in ("MACHINE", "IVR")
        and float(confidence) < 0.80
        and float(feats.get("beep", 0.0)) < 0.5
        and float(feats.get("ringback", 0.0)) < 0.5
        and not whisper_non_human
        and not dense_greeting
        and not _is_blank(feats, silero)
    ):
        # Weak MACHINE (e.g. 67%) is not evidence enough to hang up a live hello
        status, confidence = "HUMAN", max(float(confidence), 0.78)
        details["uncertain_machine_to_human"] = True
        details["fuse_note"] = "uncertain_machine_to_human"

    # Final guard: never send agent a call whose Whisper text is clearly VM/IVR
    if status == "HUMAN" and w_text:
        try:
            from app.ai.whisper_amd import classify_transcript, is_number_readout

            w_status, w_conf, w_cue2 = classify_transcript(w_text)
            if (
                w_status == "MACHINE"
                or is_number_readout(w_text)
                or w_cue2 in ("voicemail", "ivr_digits", "ivr")
                or str(w_cue2).startswith("custom_phrase")
            ):
                status = "MACHINE"
                confidence = max(float(confidence), float(w_conf or 0.93), 0.93)
                details["whisper_vm_final_block"] = True
                details["fuse_note"] = w_cue2 or "voicemail"
        except Exception:
            pass

    # Final guard: audible ringback must never reach agents
    if status == "HUMAN" and float(feats.get("ringback", 0.0)) >= 0.5:
        status = "MACHINE"
        confidence = max(float(confidence), float(feats.get("ringback_conf", 0.93)), 0.93)
        details["ringback_final_block"] = True
        details["fuse_note"] = "ringback_tone"

    # Final guard: ultra-short dense script fragments must never reach agents
    if status == "HUMAN" and _is_truncated_script_clip(feats):
        status = "MACHINE"
        confidence = max(float(confidence), 0.9)
        details["truncated_script_final_block"] = True
        details["fuse_note"] = "truncated_script_clip"

    # Safety net: spoken digit / IVR syllable must not reach agents
    if status == "HUMAN" and _looks_like_spoken_digit(feats, silero) and not (
        w_cue == "human_short"
    ):
        status, confidence = "MACHINE", max(float(confidence), 0.9)
        details["digit_safety_override"] = True
        details["fuse_note"] = "spoken_digit_machine"

    # Safety net: never send blank/silent clips to agents (ML can false-HUMAN)
    if (
        is_blank_as_machine_enabled()
        and _is_blank(feats, silero)
        and status == "HUMAN"
    ):
        b_status, b_conf, b_note = _blank_disposition()
        status, confidence = b_status, max(float(confidence), b_conf)
        details["blank_safety_override"] = True
        details["fuse_note"] = b_note

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
