"""Custom MACHINE phrases: flexible, but not so flexible that filler matches."""

from __future__ import annotations

import unittest

from app.machine_phrases import (
    _match_phrase_flexible,
    _normalize_words,
    _phrase_keys,
)


class PhraseMatchingTests(unittest.TestCase):
    def test_whole_phrase_matches_anywhere(self):
        self.assertTrue(
            _match_phrase_flexible(
                "Hi, the person you are calling is not available right now.",
                "is not available",
            )
        )

    def test_meaningful_core_of_a_long_line_matches(self):
        self.assertTrue(
            _match_phrase_flexible(
                "It's been forwarded to an automatic voice message.",
                "your call has been forwarded to an automated voice messaging system",
            )
        )
        self.assertTrue(_match_phrase_flexible("Wait for the tone.", "at the tone"))

    def test_filler_core_does_not_match_live_speech(self):
        # "is not" / "you have" / "at the" / "your call" are in every conversation.
        self.assertFalse(_match_phrase_flexible("This is not a good time.", "is not available"))
        self.assertFalse(_match_phrase_flexible("Who do you have there?", "you have reached"))
        self.assertFalse(_match_phrase_flexible("I'm at the store.", "at the tone"))
        self.assertFalse(_match_phrase_flexible("Hello, your call?", "can't take your call"))
        self.assertFalse(
            _match_phrase_flexible("Is this the person you spoke to?", "the person you are calling")
        )

    def test_match_respects_word_boundaries(self):
        self.assertFalse(_match_phrase_flexible("That's the toner.", "the tone"))

    def test_contractions_are_equivalent(self):
        self.assertTrue(_match_phrase_flexible("We cannot take your call.", "can't take your call"))
        self.assertTrue(_match_phrase_flexible("You have reached Bob.", "you've reached"))

    def test_lone_filler_entry_is_ignored(self):
        # Left behind when the portal split "name, number, and message" on commas.
        self.assertEqual(_phrase_keys("number"), [])
        self.assertFalse(_match_phrase_flexible("What number are you calling?", "number"))
        self.assertTrue(_phrase_keys("voicemail"))


class PhraseListParsingTests(unittest.TestCase):
    def test_commas_stay_inside_a_phrase(self):
        self.assertEqual(
            _normalize_words("leave your name, number, and message\nat the tone"),
            ["leave your name, number, and message", "at the tone"],
        )

    def test_phrase_with_commas_still_matches_speech(self):
        self.assertTrue(
            _match_phrase_flexible(
                "Please leave your name number and message.",
                "leave your name, number, and message",
            )
        )


if __name__ == "__main__":
    unittest.main()
