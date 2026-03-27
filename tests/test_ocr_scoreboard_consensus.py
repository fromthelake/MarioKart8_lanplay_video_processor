import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

from mk8_local_play import ocr_scoreboard_consensus
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

    def test_build_position_signal_metrics_can_compare_black_white_path_without_changing_metrics(self):
        processed = np.zeros((720, 1280), dtype=np.uint8)
        baseline_metrics = [
            {
                "best_position_template": 1,
                "best_position_score": 0.91,
                "best_position_template_score": 0.91,
            },
            {
                "best_position_template": 2,
                "best_position_score": 0.88,
                "best_position_template_score": 0.88,
            },
        ]
        stats = defaultdict(float)
        with (
            mock.patch.object(ocr_scoreboard_consensus, "POSITION_TEMPLATE_USE_BLACK_WHITE_ENABLED", False),
            mock.patch.object(ocr_scoreboard_consensus, "POSITION_TEMPLATE_COMPARE_BLACK_WHITE_ENABLED", True),
            mock.patch.object(ocr_scoreboard_consensus, "extract_position_row_match_crops", return_value=[np.zeros((38, 58), dtype=np.uint8)] * 2),
            mock.patch.object(ocr_scoreboard_consensus, "load_position_row_templates", return_value=[np.zeros((36, 56), dtype=np.uint8)] * 12),
            mock.patch.object(ocr_scoreboard_consensus, "_build_position_signal_metrics_fast", return_value=baseline_metrics),
            mock.patch.object(ocr_scoreboard_consensus, "extract_position_tile_match_crops", return_value=[np.zeros((52, 52), dtype=np.uint8)] * 2),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "load_position_row_template_tiles",
                return_value={
                    "white": [(np.zeros((52, 52), dtype=np.uint8), None)] * 12,
                    "black": [(np.zeros((52, 52), dtype=np.uint8), None)] * 12,
                },
            ),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "_build_position_signal_metrics_black_white",
                return_value=baseline_metrics,
            ),
        ):
            metrics = ocr_scoreboard_consensus.build_position_signal_metrics(
                processed,
                stats=stats,
                stats_prefix="test",
                max_rows=2,
            )

        self.assertEqual(metrics, baseline_metrics)
        self.assertEqual(stats["test_position_template_compare_calls"], 1)
        self.assertEqual(stats["test_position_template_compare_rows"], 2)
        self.assertEqual(stats["test_position_template_compare_mismatches"], 0)

    def test_build_position_signal_metrics_can_use_black_white_path_authoritatively(self):
        processed = np.zeros((720, 1280), dtype=np.uint8)
        stats = defaultdict(float)
        expected_metrics = [
            {
                "best_position_template": 1,
                "best_position_score": 0.93,
                "best_position_template_score": 0.93,
            },
            {
                "best_position_template": 2,
                "best_position_score": 0.89,
                "best_position_template_score": 0.89,
            },
        ]
        with (
            mock.patch.object(ocr_scoreboard_consensus, "POSITION_TEMPLATE_USE_BLACK_WHITE_ENABLED", True),
            mock.patch.object(ocr_scoreboard_consensus, "extract_position_tile_match_crops", return_value=[np.zeros((52, 52), dtype=np.uint8)] * 2),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "load_position_row_template_tiles",
                return_value={
                    "white": [(np.zeros((52, 52), dtype=np.uint8), None)] * 12,
                    "black": [(np.zeros((52, 52), dtype=np.uint8), None)] * 12,
                },
            ),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "_build_position_signal_metrics_black_white",
                return_value=expected_metrics,
            ),
            mock.patch.object(ocr_scoreboard_consensus, "extract_position_row_match_crops") as legacy_crops,
            mock.patch.object(ocr_scoreboard_consensus, "load_position_row_templates") as legacy_templates,
        ):
            metrics = ocr_scoreboard_consensus.build_position_signal_metrics(
                processed,
                stats=stats,
                stats_prefix="test",
                max_rows=2,
            )

        legacy_crops.assert_not_called()
        legacy_templates.assert_not_called()
        self.assertEqual(metrics, expected_metrics)
        self.assertEqual(stats["test_position_metrics_black_white_calls"], 1)
        self.assertEqual(stats["test_position_metrics_rows_processed"], 2)


if __name__ == "__main__":
    unittest.main()
