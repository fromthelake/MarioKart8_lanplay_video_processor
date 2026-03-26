import unittest

from mk8_local_play.ocr_scoreboard_consensus import map_total_rows_to_race_rows


class OcrScoreboardConsensusTests(unittest.TestCase):
    def test_duplicate_names_prefer_expected_total_continuity(self):
        score_rows = [
            {
                "PlayerName": "BaasBaas",
                "NameConfidence": 100.0,
                "CharacterIndex": 10,
                "Character": "Waluigi",
                "CharacterMatchConfidence": 100.0,
                "CharacterMatchMethod": "exact",
                "DetectedValue": 5,
                "DetectedSecondaryValue": 2,
                "DigitConfidence": 100.0,
            },
            {
                "PlayerName": "BaasBaas",
                "NameConfidence": 100.0,
                "CharacterIndex": 10,
                "Character": "Waluigi",
                "CharacterMatchConfidence": 100.0,
                "CharacterMatchMethod": "exact",
                "DetectedValue": 4,
                "DetectedSecondaryValue": 5,
                "DigitConfidence": 100.0,
            },
        ]
        total_rows = [
            {
                "PlayerName": "BaasBaas",
                "NameConfidence": 100.0,
                "CharacterIndex": 10,
                "Character": "Waluigi",
                "CharacterMatchConfidence": 100.0,
                "CharacterMatchMethod": "exact",
                "DetectedValue": 9,
                "DetectedValueSource": "ocr",
                "DigitConfidence": 100.0,
                "RowIndex": 0,
            },
            {
                "PlayerName": "BaasBaas",
                "NameConfidence": 100.0,
                "CharacterIndex": 10,
                "Character": "Waluigi",
                "CharacterMatchConfidence": 100.0,
                "CharacterMatchMethod": "exact",
                "DetectedValue": 7,
                "DetectedValueSource": "ocr",
                "DigitConfidence": 100.0,
                "RowIndex": 1,
            },
        ]

        mapped = map_total_rows_to_race_rows(
            score_rows,
            total_rows,
            preprocess_name=lambda value: str(value).strip().lower(),
            weighted_similarity=lambda left, right: 1.0 if left == right else 0.0,
            total_row_metrics=[],
        )

        self.assertEqual(mapped[0]["DetectedTotalScore"], 7)
        self.assertEqual(mapped[1]["DetectedTotalScore"], 9)
        self.assertEqual(mapped[0]["TotalScoreMappingMethod"], "duplicate_name_expected_total")
        self.assertEqual(mapped[1]["TotalScoreMappingMethod"], "duplicate_name_expected_total")


if __name__ == "__main__":
    unittest.main()
