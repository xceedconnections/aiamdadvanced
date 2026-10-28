"""Regression tests for ringback versus sparse voiced speech."""

import unittest

import numpy as np

from app.ai.engine import _detect_ringback


class RingbackDetectionTests(unittest.TestCase):
    sample_rate = 8000

    def test_fixed_440_480_dual_tone_is_ringback(self):
        t = np.arange(int(1.98 * self.sample_rate)) / self.sample_rate
        audio = np.zeros_like(t, dtype=np.float32)
        active = t < 0.8
        audio[active] = 0.12 * (
            np.sin(2 * np.pi * 440 * t[active])
            + np.sin(2 * np.pi * 480 * t[active])
        )

        detected, confidence = _detect_ringback(audio, self.sample_rate)

        self.assertTrue(detected)
        self.assertGreaterEqual(confidence, 0.9)

    def test_broad_voiced_formant_between_ring_tones_is_not_ringback(self):
        """One dominant moving/voiced region must not count as two tones.

        The target false positive has a sparse hello with its strongest
        spectral energy around 454 Hz.  This deterministic signal recreates
        the causal shape: energy reaches both broad 440/480 detector bands,
        while the inter-tone voice component remains dominant.
        """
        t = np.arange(int(1.98 * self.sample_rate)) / self.sample_rate
        audio = np.zeros_like(t, dtype=np.float32)
        active = (t >= 0.65) & (t < 1.01)
        ta = t[active]
        envelope = np.sin(np.pi * (ta - 0.65) / 0.36) ** 2
        audio[active] = 0.08 * envelope * (
            np.sin(2 * np.pi * 454 * ta)
            + 0.45 * np.sin(2 * np.pi * 440 * ta)
            + 0.45 * np.sin(2 * np.pi * 480 * ta)
            + 0.20 * np.sin(2 * np.pi * 908 * ta)
        )

        detected, confidence = _detect_ringback(audio, self.sample_rate)

        self.assertFalse(detected)
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
