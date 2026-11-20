"""Blank / 0-second audio must not become HUMAN while blank-as-AA is on."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.ai.engine import _dispose_blank_or_noise, _is_blank


def _silent(duration: float) -> dict:
    return {
        "duration": duration,
        "peak": 0.001,
        "raw_rms": 0.0,
        "rms": 0.0,
        "speech_ratio": 0.0,
        "activity_ratio": 0.0,
        "longest_burst_ms": 0.0,
        "num_bursts": 0.0,
        "abs_speech_ratio": 0.0,
        "abs_speech_ms": 0.0,
        "asterisk_amdstatus": "HUMAN",
        "amdstatus": "HUMAN",
    }


class BlankNotToAgentTests(unittest.TestCase):
    def test_zero_second_clip_is_blank(self):
        self.assertTrue(_is_blank(_silent(0.0)))
        self.assertTrue(_is_blank(_silent(0.04)))

    def test_stock_human_does_not_override_blank_checkbox(self):
        feats = _silent(0.0)
        with patch("app.ai.engine.is_blank_as_machine_enabled", return_value=True):
            status, _conf, note = _dispose_blank_or_noise(
                stock_amdstatus="HUMAN", feats=feats
            )
        self.assertEqual(status, "BLANK")
        self.assertEqual(note, "blank_silence")
        self.assertNotEqual(note, "stock_amd_trust_blank_capture")

    def test_checkbox_off_still_sends_blank_to_agent(self):
        with patch("app.ai.engine.is_blank_as_machine_enabled", return_value=False):
            status, _conf, note = _dispose_blank_or_noise(
                stock_amdstatus="NOTSURE", feats=_silent(0.0)
            )
        self.assertEqual(status, "HUMAN")
        self.assertEqual(note, "blank_to_agent")


if __name__ == "__main__":
    unittest.main()
