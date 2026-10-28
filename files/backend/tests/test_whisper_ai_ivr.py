"""Unit tests: AI assistant / IVR / recording cues → MACHINE; short hellos → HUMAN."""

from __future__ import annotations

import unittest

from app.ai.amd_decide import finalize_amd_decision, score_amd
from app.ai.whisper_amd import classify_transcript


class AiAssistantIvrCueTests(unittest.TestCase):
    def test_ai_assistant_recording_is_machine(self):
        status, conf, cue = classify_transcript("Hi. I'm an AI assistant recording.")
        self.assertEqual(status, "MACHINE")
        self.assertGreaterEqual(conf, 0.92)
        self.assertIn(cue, ("voicemail", "ivr_digits", "ivr"))

    def test_i_am_an_ai_assistant_is_machine(self):
        status, conf, cue = classify_transcript("Hello. I am an AI assistant.")
        self.assertEqual(status, "MACHINE")
        self.assertGreaterEqual(conf, 0.92)

    def test_virtual_assistant_is_machine(self):
        status, _, cue = classify_transcript("Hi, I am a virtual assistant.")
        self.assertEqual(status, "MACHINE")
        self.assertTrue(cue)

    def test_call_being_recorded_is_machine(self):
        status, _, _ = classify_transcript("This call is being recorded.")
        self.assertEqual(status, "MACHINE")

    def test_your_call_may_be_recorded_is_machine(self):
        status, _, _ = classify_transcript("Your call may be recorded for quality.")
        self.assertEqual(status, "MACHINE")

    def test_classic_leave_message_still_machine(self):
        status, _, cue = classify_transcript(
            "Please leave a message after the beep."
        )
        self.assertEqual(status, "MACHINE")
        self.assertEqual(cue, "voicemail")

    def test_ivr_press_one_is_machine(self):
        status, _, cue = classify_transcript("Press one for English.")
        self.assertEqual(status, "MACHINE")
        self.assertIn(cue, ("ivr_digits", "ivr", "voicemail"))

    def test_short_hello_remains_human(self):
        for text in ("hello", "Hello.", "hi", "Hi.", "yeah", "Yeah.", "yes"):
            status, conf, cue = classify_transcript(text)
            self.assertEqual(status, "HUMAN", msg=text)
            self.assertEqual(cue, "human_short", msg=text)
            self.assertGreaterEqual(conf, 0.90)

    def test_hi_plus_name_still_human(self):
        status, _, cue = classify_transcript("Hi, this is Mike.")
        self.assertEqual(status, "HUMAN")
        self.assertEqual(cue, "human_short")


class AmdDecideMachineOverrideTests(unittest.TestCase):
    def _base_feats(self) -> dict:
        return {
            "duration": 2.0,
            "speech_ratio": 0.4,
            "longest_burst_ms": 800.0,
            "peak": 0.15,
            "abs_speech_ms": 700.0,
            "beep": 0.0,
            "beep_conf": 0.0,
            "ringback": 0.0,
            "sit": 0.0,
            "asterisk_amd_human": 1.0,
            "asterisk_amd_machine": 0.0,
        }

    def test_whisper_ai_transcript_overrides_stock_human(self):
        feats = self._base_feats()
        silero = {"longest_speech_ms": 900.0, "num_segments": 1, "speech_ratio": 0.4}
        details: dict = {
            "ml": {
                "whisper": {
                    "whisper_ok": True,
                    # Stale wrong cue — fusion must reclassify transcript
                    "cue": "human_short",
                    "status": "HUMAN",
                    "confidence": 0.9,
                    "transcript": "Hi. I'm an AI assistant recording.",
                }
            }
        }

        status, conf = finalize_amd_decision(
            status="HUMAN",
            confidence=0.9,
            feats=feats,
            silero=silero,
            audio=None,
            sr=8000,
            details=details,
            looks_short_human_fn=lambda *_a, **_k: True,
            looks_digit_fn=lambda *_a, **_k: False,
            is_script_clip_fn=lambda *_a, **_k: False,
            is_front_script_fn=lambda *_a, **_k: False,
            is_blank_fn=lambda *_a, **_k: False,
            is_voip_fn=lambda *_a, **_k: False,
            dispose_blank_fn=lambda **_k: ("HUMAN", 0.5, "noop"),
        )
        self.assertEqual(status, "MACHINE")
        self.assertGreaterEqual(conf, 0.93)
        self.assertEqual(details.get("amd_stage"), "whisper_machine")
        self.assertTrue(details.get("whisper_machine_override"))

    def test_score_counts_whisper_machine(self):
        feats = self._base_feats()
        feats["asterisk_amd_human"] = 0.0
        score, note = score_amd(
            feats=feats,
            silero={"longest_speech_ms": 500.0, "num_segments": 1},
            whisper={"cue": "voicemail", "status": "MACHINE", "transcript": "ai"},
            looks_short_human=False,
            looks_digit=False,
            script_clip=False,
            front_script=False,
            dense_greeting=False,
            xgb_probs=None,
        )
        self.assertGreaterEqual(score, 70)
        self.assertIn("whisper_vm", note)


if __name__ == "__main__":
    unittest.main()
