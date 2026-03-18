import unittest

import pandas as pd

from mk8_local_play.extract_text import get_race_points
from mk8_local_play.low_res_identity import low_res_race_points
from mk8_local_play.ocr_session_validation import (
    assign_shared_positions_after_race,
    detect_connection_reset,
    detect_rebase_candidate,
    detect_total_score_order_violations,
    format_validation_status,
)


class OcrSessionValidationTests(unittest.TestCase):
    def test_race_points_support_small_player_tables_in_normal_and_low_res_paths(self):
        expected_tables = {
            6: [7, 5, 4, 3, 2, 1],
            5: [6, 4, 3, 2, 1],
            4: [4, 3, 2, 1],
            3: [3, 2, 1],
            2: [2, 1],
        }

        for player_count, expected in expected_tables.items():
            self.assertEqual(
                [get_race_points(position, player_count) for position in range(1, player_count + 1)],
                expected,
            )
            self.assertEqual(
                [low_res_race_points(position, player_count) for position in range(1, player_count + 1)],
                expected,
            )

    def test_format_validation_status_falls_back_to_humanized_text(self):
        self.assertEqual(format_validation_status("custom_status_code"), "Custom status code")

    def test_detect_rebase_candidate_requires_large_first_race_offset(self):
        prepared_rows = [
            {"detected_total": 30, "session_new_total": 15},
            {"detected_total": 28, "session_new_total": 12},
            {"detected_total": 26, "session_new_total": 10},
            {"detected_total": 22, "session_new_total": 8},
            {"detected_total": 18, "session_new_total": 7},
        ]

        self.assertTrue(detect_rebase_candidate(prepared_rows, race_index=0))
        self.assertFalse(detect_rebase_candidate(prepared_rows, race_index=1))

    def test_detect_connection_reset_requires_broad_total_drop(self):
        previous_validated_totals = {
            "p1": 50,
            "p2": 45,
            "p3": 40,
            "p4": 35,
            "p5": 30,
        }
        prepared_rows = [
            {"player_key": "p1", "detected_total": 10, "race_points": 5},
            {"player_key": "p2", "detected_total": 8, "race_points": 4},
            {"player_key": "p3", "detected_total": 6, "race_points": 3},
            {"player_key": "p4", "detected_total": 4, "race_points": 2},
            {"player_key": "p5", "detected_total": 2, "race_points": 1},
        ]

        self.assertTrue(detect_connection_reset(previous_validated_totals, prepared_rows))

    def test_detect_total_score_order_violations_flags_non_descending_totals(self):
        race_rows = pd.DataFrame(
            [
                {"DetectedPositionAfterRace": 1, "DetectedTotalScore": 30},
                {"DetectedPositionAfterRace": 2, "DetectedTotalScore": 25},
                {"DetectedPositionAfterRace": 3, "DetectedTotalScore": 27},
            ],
            index=[10, 11, 12],
        )

        violations = detect_total_score_order_violations(race_rows, remapped_totals_by_index={})

        self.assertEqual(violations, {11, 12})

    def test_assign_shared_positions_after_race_uses_competition_ranking(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "Demo", "RaceIDNumber": 1, "FixPlayerName": "A", "NewTotalScore": 30},
                {"RaceClass": "Demo", "RaceIDNumber": 1, "FixPlayerName": "B", "NewTotalScore": 30},
                {"RaceClass": "Demo", "RaceIDNumber": 1, "FixPlayerName": "C", "NewTotalScore": 20},
                {"RaceClass": "Demo", "RaceIDNumber": 1, "FixPlayerName": "D", "NewTotalScore": 10},
            ]
        )

        ranked = assign_shared_positions_after_race(df)

        self.assertEqual(ranked["PositionAfterRace"].tolist(), [1, 1, 3, 4])
