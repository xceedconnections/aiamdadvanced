"""Regression: continuous fax/modem carriers must never route as HUMAN."""

from __future__ import annotations

import unittest

import numpy as np

from app.ai.engine import (
    _detect_fax_modem,
    _looks_like_short_human,
    _classify_heuristic,
)
from app.locale_packs import resolve_locale_pack


class FaxModemDetectionTests(unittest.TestCase):
    sample_rate = 8000

    def _tone(self, hz: float, seconds: float = 3.0, amp: float = 0.2) -> np.ndarray:
        t = np.arange(int(seconds * self.sample_rate), dtype=np.float64) / self.sample_rate
        # Mild 2nd harmonic mimics real fax/modem carriers on telephony
        audio = amp * (
            np.sin(2 * np.pi * hz * t) + 0.35 * np.sin(2 * np.pi * (2 * hz) * t)
        )
        return audio.astype(np.float32)

    def test_cng_1100_is_fax(self):
        detected, conf = _detect_fax_modem(self._tone(1100.0), self.sample_rate)
        self.assertTrue(detected)
        self.assertGreaterEqual(conf, 0.85)

    def test_ced_2100_is_fax(self):
        detected, conf = _detect_fax_modem(self._tone(2100.0), self.sample_rate)
        self.assertTrue(detected)
        self.assertGreaterEqual(conf, 0.85)

    def test_modemish_1212_is_fax(self):
        """Observed false-HUMAN clip: continuous ~1212 Hz carrier."""
        detected, conf = _detect_fax_modem(self._tone(1212.0, seconds=5.0), self.sample_rate)
        self.assertTrue(detected)
        self.assertGreaterEqual(conf, 0.85)

    def test_short_voiced_hello_is_not_fax(self):
        t = np.arange(int(2.0 * self.sample_rate)) / self.sample_rate
        audio = np.zeros_like(t, dtype=np.float32)
        active = (t >= 0.7) & (t < 1.05)
        ta = t[active]
        env = np.sin(np.pi * (ta - 0.7) / 0.35) ** 2
        audio[active] = (
            0.12
            * env
            * (
                np.sin(2 * np.pi * 180 * ta)
                + 0.55 * np.sin(2 * np.pi * 900 * ta)
                + 0.25 * np.sin(2 * np.pi * 1400 * ta)
            )
        ).astype(np.float32)

        detected, conf = _detect_fax_modem(audio, self.sample_rate)
        self.assertFalse(detected)
        self.assertEqual(conf, 0.0)

    def test_fax_feats_not_short_human_and_heuristic_fax(self):
        audio = self._tone(1212.0, seconds=5.0)
        detected, conf = _detect_fax_modem(audio, self.sample_rate)
        self.assertTrue(detected)
        # Flat-energy fax fools percentile VAD (speech_ratio≈0); short-human
        # must still reject via fax flag / high ZCR guard.
        zcr = float(np.mean(np.sign(audio[:-1]) != np.sign(audio[1:])))
        feats = {
            "duration": 5.0,
            "peak": 0.22,
            "raw_rms": 0.08,
            "rms": 0.35,
            "speech_ratio": 0.0,
            "num_bursts": 0.0,
            "longest_burst_ms": 0.0,
            "longest_silence_ms": 5000.0,
            "avg_burst_ms": 0.0,
            "abs_speech_ms": 0.0,
            "abs_speech_ratio": 0.0,
            "activity_ratio": 0.0,
            "zcr": zcr,
            "beep": 0.0,
            "beep_conf": 0.0,
            "sit": 0.0,
            "ringback": 0.0,
            "fax": 1.0,
            "fax_conf": float(conf),
        }
        self.assertFalse(_looks_like_short_human(feats))
        _, pack = resolve_locale_pack(enabled=False, pack="usa")
        status, hconf = _classify_heuristic(feats, beep=False, sit=False, pack=pack)
        self.assertEqual(status, "FAX")
        self.assertGreaterEqual(hconf, 0.90)


if __name__ == "__main__":
    unittest.main()
