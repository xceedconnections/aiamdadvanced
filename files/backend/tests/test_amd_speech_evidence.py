"""HUMAN needs speech; a greeting still playing at the cut is a machine.

Each case mirrors a production recording that was disposed the wrong way:
a live "Hello?" hung up as BLANK, a ringback tone sent to an agent, a
voicemail greeting passed as a live answer.
"""

from __future__ import annotations

import unittest

import numpy as np

from app.ai.amd_decide import finalize_amd_decision
from app.ai.engine import _detect_ringback, _is_voip_noise_burst
from app.ai.silero_vad import _speech_spans, _tail_speech_ms
from app.ai.whisper_amd import _select_speech


def _decide(*, silero: dict, whisper: dict, stock: str = "HUMAN", blank=("BLANK", 0.95, "blank_silence")):
    feats = {
        "duration": 3.0,
        "speech_ratio": 0.2,
        "longest_burst_ms": 400.0,
        "peak": 0.3,
        "abs_speech_ms": 300.0,
        "beep": 0.0,
        "beep_conf": 0.0,
        "ringback": 0.0,
        "sit": 0.0,
        "asterisk_amd_human": 1.0 if stock in ("HUMAN", "NOTSURE") else 0.0,
        "asterisk_amd_machine": 1.0 if stock == "MACHINE" else 0.0,
    }
    details: dict = {"ml": {"whisper": {"whisper_ok": True, **whisper}}}
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
        dispose_blank_fn=lambda **_k: blank,
    )
    return status, conf, details


def _silero(spans, clip_seconds=3.0):
    return {
        "ok": True,
        "spans": spans,
        "longest_speech_ms": max([(b - a) * 1000.0 for a, b in spans], default=0.0),
        "num_segments": len(spans),
        "speech_ratio": sum(b - a for a, b in spans) / clip_seconds,
        "tail_speech_ms": _tail_speech_ms(spans, clip_seconds),
    }


class SpeechEvidenceTests(unittest.TestCase):
    def test_hello_then_silence_is_human(self):
        status, _, details = _decide(
            silero=_silero([(1.3, 1.8)]),
            whisper={"transcript": "Hello?", "cue": "human_short", "status": "HUMAN"},
        )
        self.assertEqual(status, "HUMAN")
        self.assertEqual(details["amd_stage"], "whisper_human")

    def test_loud_clip_with_no_speech_is_blank_not_human(self):
        # Peak 0.3, stock AMD HUMAN, energy shaped like a hello — but nothing spoken.
        status, _, details = _decide(
            silero=_silero([]),
            whisper={"transcript": "", "cue": "no_speech"},
        )
        self.assertEqual(status, "BLANK")
        self.assertEqual(details["fuse_note"], "blank_no_speech")

    def test_no_speech_goes_to_agent_when_blank_checkbox_is_off(self):
        status, _, details = _decide(
            silero=_silero([]),
            whisper={"transcript": "", "cue": "no_speech"},
            blank=("HUMAN", 0.92, "blank_to_agent"),
        )
        self.assertEqual(status, "HUMAN")
        self.assertEqual(details["fuse_note"], "blank_to_agent")

    def test_unintelligible_speech_still_goes_to_agent(self):
        # Speech Whisper could not name is not evidence of a machine.
        status, _, details = _decide(
            silero=_silero([(1.0, 1.6)]),
            whisper={"transcript": "Other mountain virility.", "cue": "none"},
        )
        self.assertEqual(status, "HUMAN")

    def test_silero_unavailable_keeps_old_uncertain_to_agent(self):
        status, _, _ = _decide(
            silero={"ok": False},
            whisper={"transcript": "", "cue": "none"},
        )
        self.assertEqual(status, "HUMAN")


class LoneWordTests(unittest.TestCase):
    _digit = {"transcript": "Five.", "cue": "ivr_digits", "status": "MACHINE", "confidence": 0.93}

    def test_lone_word_heard_as_a_digit_is_not_a_number_readout(self):
        # A live "Hello?" that Whisper wrote down as "Five."
        status, _, details = _decide(
            silero=_silero([(1.2, 1.55)]), whisper=dict(self._digit), stock="NOTSURE"
        )
        self.assertEqual(status, "HUMAN")
        self.assertTrue(details["lone_word_not_digits"])

    def test_digit_fragment_touching_the_clip_start_stays_machine(self):
        status, _, _ = _decide(
            silero=_silero([(0.0, 0.4)]), whisper=dict(self._digit), stock="NOTSURE"
        )
        self.assertEqual(status, "MACHINE")

    def test_lone_digit_stays_machine_when_stock_amd_said_machine(self):
        status, _, _ = _decide(
            silero=_silero([(1.2, 1.55)]), whisper=dict(self._digit), stock="MACHINE"
        )
        self.assertEqual(status, "MACHINE")

    def test_number_readout_stays_machine(self):
        status, _, details = _decide(
            silero=_silero([(0.2, 3.0)]),
            whisper={
                "transcript": "1-9-7-2-2-7-6-5",
                "cue": "ivr_digits",
                "status": "MACHINE",
                "confidence": 0.93,
            },
            stock="NOTSURE",
        )
        self.assertEqual(status, "MACHINE")
        self.assertEqual(details["amd_stage"], "whisper_machine")


class GreetingStillPlayingTests(unittest.TestCase):
    def test_long_speech_running_into_the_cut_is_machine(self):
        # "This is Julie Gray at Keller Williams. I'm sorry I'm…" heard as nonsense
        status, _, details = _decide(
            silero=_silero([(1.3, 3.0)]),
            whisper={"transcript": "This is Uley 3 at Callowone and time.", "cue": "none"},
            stock="NOTSURE",
        )
        self.assertEqual(status, "MACHINE")
        self.assertEqual(details["amd_stage"], "speech_into_cut")

    def test_greeting_word_does_not_rescue_a_long_greeting(self):
        status, _, _ = _decide(
            silero=_silero([(0.4, 3.0)]),
            whisper={
                "transcript": "Hello. This is Kelly Blackburn, a Coldwell Banker.",
                "cue": "human_short",
                "status": "HUMAN",
            },
        )
        self.assertEqual(status, "MACHINE")

    def test_repeated_hello_at_the_cut_stays_human(self):
        status, _, _ = _decide(
            silero=_silero([(1.5, 3.0)]),
            whisper={"transcript": "Hello? Hello?", "cue": "human_short", "status": "HUMAN"},
        )
        self.assertEqual(status, "HUMAN")

    def test_speech_that_stopped_before_the_cut_is_not_a_tail(self):
        self.assertEqual(_tail_speech_ms([(0.2, 2.0)], 3.0), 0.0)
        self.assertAlmostEqual(_tail_speech_ms([(0.2, 0.6), (1.4, 2.95)], 3.0), 1550.0, places=3)


class StockMachineTests(unittest.TestCase):
    def test_stock_machine_without_a_live_greeting_is_machine(self):
        status, _, details = _decide(
            silero=_silero([(0.0, 0.4)]),
            whisper={"transcript": "Thank you.", "cue": "none"},
            stock="MACHINE",
        )
        self.assertEqual(status, "MACHINE")
        self.assertEqual(details["fuse_note"], "stock_amd_machine")

    def test_live_greeting_overrules_stock_machine(self):
        status, _, _ = _decide(
            silero=_silero([(0.0, 2.1)]),
            whisper={
                "transcript": "Southern Realty, this is Lisa.",
                "cue": "human_short",
                "status": "HUMAN",
            },
            stock="MACHINE",
        )
        self.assertEqual(status, "HUMAN")


class SileroSpanTests(unittest.TestCase):
    def test_single_frame_click_is_not_a_span(self):
        mask = np.zeros(60, dtype=bool)
        mask[10] = True
        self.assertEqual(_speech_spans(mask, 32.0), [])

    def test_short_pause_is_bridged_long_pause_is_not(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:20] = True
        mask[24:30] = True  # 128 ms pause → same island
        mask[60:70] = True  # ~1 s pause → new island
        spans = _speech_spans(mask, 32.0)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0][0], 0.32, places=3)
        self.assertAlmostEqual(spans[0][1], 0.96, places=3)

    def test_voip_shape_is_not_noise_once_silero_hears_a_word(self):
        # Silence, one loud word, silence: the shape of a VoIP noise dump and
        # of every live "Hello?" recorded while stock AMD is listening.
        feats = {
            "duration": 2.6,
            "peak": 0.25,
            "front_energy_ratio": 0.1,
            "zcr": 0.3,
            "longest_silence_ms": 1300.0,
            "longest_burst_ms": 400.0,
            "num_bursts": 1.0,
            "speech_ratio": 0.1,
        }
        self.assertTrue(_is_voip_noise_burst(feats, {"ok": True, "longest_speech_ms": 0.0}))
        self.assertFalse(
            _is_voip_noise_burst(
                feats, {"ok": True, "longest_speech_ms": 480.0, "avg_speech_prob": 0.9}
            )
        )


class _Segment:
    def __init__(self, text, *, no_speech_prob=0.05, avg_logprob=-0.3, compression_ratio=1.0):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio


class _FakeModel:
    """Stands in for faster-whisper: returns fixed segments, records its input."""

    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, clip, **kwargs):
        self.calls.append((len(clip), kwargs))
        return iter(self.segments), None


class WhisperGuardTests(unittest.TestCase):
    sr = 8000

    def _transcribe(self, segments, spans, seconds=3.0):
        from unittest.mock import patch

        from app.ai import whisper_amd

        model = _FakeModel(segments)
        audio = np.full(int(seconds * self.sr), 0.1, dtype=np.float32)
        with patch.object(whisper_amd, "_get_model", return_value=model):
            text, _info, meta = whisper_amd._transcribe_clip(
                audio, self.sr, 5.0, speech_spans=spans
            )
        return text, meta, model

    def test_whisper_is_not_run_when_nothing_was_spoken(self):
        text, meta, model = self._transcribe([_Segment("Thank you.")], spans=[])
        self.assertEqual(text, "")
        self.assertEqual(meta["cue"], "no_speech")
        self.assertEqual(model.calls, [])

    def test_no_prompt_is_given_to_whisper(self):
        _text, _meta, model = self._transcribe([_Segment("Hello?")], spans=[(1.0, 1.5)])
        kwargs = model.calls[0][1]
        self.assertNotIn("initial_prompt", kwargs)
        self.assertFalse(kwargs["vad_filter"])

    def test_more_words_than_the_speech_could_hold_is_discarded(self):
        text, meta, _ = self._transcribe(
            [_Segment("Thank you very much for your question.")], spans=[(0.0, 0.3)]
        )
        self.assertEqual(text, "")
        self.assertTrue(meta["implausible_word_rate"])

    def test_fast_real_greeting_is_kept(self):
        said = (
            "I'm sorry I can't take a call right now but if you leave me a "
            "message I'll get back to you soon as I can."
        )
        text, meta, _ = self._transcribe([_Segment(said)], spans=[(0.3, 5.0)], seconds=5.0)
        self.assertEqual(text, said)
        self.assertNotIn("implausible_word_rate", meta)

    def test_looping_decoder_output_is_dropped(self):
        text, _meta, _ = self._transcribe(
            [_Segment("G-W-W-W-W-W-W-W-W-W-W-W", compression_ratio=17.2)],
            spans=[(1.3, 2.9)],
        )
        self.assertEqual(text, "")


class WhisperInputTests(unittest.TestCase):
    def test_only_speech_islands_reach_whisper(self):
        sr = 8000
        clip = np.zeros(5 * sr, dtype=np.float32)
        clip[int(0.1 * sr) : int(0.4 * sr)] = 0.5
        clip[int(3.0 * sr) : int(4.0 * sr)] = 0.5
        out = _select_speech(clip, sr, [(0.1, 0.4), (3.0, 4.0)])
        # 0.3 s pad each side, clipped at the clip start: 0.7 s + 1.6 s
        self.assertAlmostEqual(len(out) / sr, 2.3, places=2)

    def test_close_islands_keep_the_audio_between_them(self):
        sr = 8000
        clip = np.zeros(3 * sr, dtype=np.float32)
        out = _select_speech(clip, sr, [(0.5, 1.0), (1.4, 2.0)])
        self.assertAlmostEqual(len(out) / sr, 2.1, places=2)


class LateRingbackTests(unittest.TestCase):
    def test_ring_after_three_seconds_of_dead_air_is_ringback(self):
        sr = 8000
        t = np.arange(5 * sr) / sr
        audio = np.zeros_like(t, dtype=np.float32)
        ring = t >= 3.1
        audio[ring] = 0.5 * (
            np.sin(2 * np.pi * 440 * t[ring]) + np.sin(2 * np.pi * 480 * t[ring])
        )
        detected, confidence = _detect_ringback(audio, sr)
        self.assertTrue(detected)
        self.assertGreaterEqual(confidence, 0.9)

    def test_late_voiced_word_is_not_ringback(self):
        sr = 8000
        t = np.arange(4 * sr) / sr
        audio = np.zeros_like(t, dtype=np.float32)
        # Long enough (0.9 s) that the loud-region test really runs on it.
        word = (t >= 2.0) & (t < 2.9)
        tw = t[word]
        envelope = np.sin(np.pi * (tw - 2.0) / 0.9) ** 2
        audio[word] = 0.4 * envelope * (
            np.sin(2 * np.pi * 454 * tw)
            + 0.45 * np.sin(2 * np.pi * 440 * tw)
            + 0.45 * np.sin(2 * np.pi * 480 * tw)
            + 0.20 * np.sin(2 * np.pi * 908 * tw)
        )
        detected, _ = _detect_ringback(audio, sr)
        self.assertFalse(detected)


if __name__ == "__main__":
    unittest.main()
