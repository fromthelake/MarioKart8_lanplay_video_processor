import unittest

from mk8_local_play.name_unicode import (
    allowed_name_char_ratio,
    is_allowed_name_char,
    load_allowed_name_char_data,
    normalize_name_key,
    unknown_name_chars,
)
from mk8_local_play.ocr_name_matching import choose_canonical_name, preprocess_name, weighted_similarity
from mk8_local_play.ocr_scoreboard_consensus import weighted_name_vote_details


class UnicodeNameHandlingTests(unittest.TestCase):
    def test_normalize_name_key_preserves_unicode_letters(self):
        self.assertEqual(normalize_name_key(" ñïæłß "), "ñïæłss")
        self.assertEqual(normalize_name_key("ミii　テスト"), "ミii テスト")
        self.assertEqual(normalize_name_key("ÄBC あ"), "äbc あ")

    def test_allowed_character_data_loads_from_utf8_json(self):
        data = load_allowed_name_char_data()

        self.assertIn("latin_extended", data)
        self.assertIn("japanese", data)
        self.assertIn("special_symbols", data)
        self.assertTrue(is_allowed_name_char("ñ"))
        self.assertTrue(is_allowed_name_char("ß"))
        self.assertTrue(is_allowed_name_char("ł"))
        self.assertTrue(is_allowed_name_char("あ"))
        self.assertTrue(is_allowed_name_char("\ue000"))

    def test_allowed_character_helpers_are_deterministic(self):
        self.assertEqual(allowed_name_char_ratio("ñïæłß"), 1.0)
        self.assertEqual(unknown_name_chars("ñ🙂ß🙂"), "🙂")
        self.assertAlmostEqual(allowed_name_char_ratio("ñ🙂ß"), 2 / 3, places=6)

    def test_canonical_name_selection_does_not_penalize_non_ascii_names(self):
        self.assertEqual(
            choose_canonical_name(["ñïæłß", "ñïæłß", "niztB"]),
            "ñïæłß",
        )

    def test_weighted_name_vote_details_preserves_unicode_and_reports_unknowns(self):
        winner, confidence, details = weighted_name_vote_details(
            [
                ("ñïæłß", 88),
                ("ñïæłß", 84),
                ("🙂🙂🙂", 10),
            ]
        )

        self.assertEqual(winner, "ñïæłß")
        self.assertGreater(confidence, 0.8)
        self.assertEqual(details["allowed_ratio"], 100.0)
        self.assertEqual(details["unknown_chars"], "")
        self.assertEqual(details["flags"], "")

        _winner, _confidence, noisy_details = weighted_name_vote_details([("🙂🙂🙂", 10)])
        self.assertIn("unknown_chars", noisy_details["flags"])

    def test_matching_functions_do_not_collapse_unicode_to_ascii(self):
        self.assertEqual(preprocess_name("ñïæłß"), "ñïæłss")
        self.assertEqual(preprocess_name("カタカナ"), "カタカナ")
        self.assertEqual(weighted_similarity("ñïæłß", "ñïæłß"), 1.0)


if __name__ == "__main__":
    unittest.main()
