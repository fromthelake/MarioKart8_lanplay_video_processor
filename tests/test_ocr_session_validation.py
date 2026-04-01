import unittest

import pandas as pd

from mk8_local_play.extract_text import get_race_points
from mk8_local_play.low_res_identity import low_res_race_points
from mk8_local_play.ocr_session_validation import (
    apply_session_validation,
    assign_shared_positions_after_race,
    detect_connection_reset,
    detect_obvious_total_score_reset,
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

    def test_detect_connection_reset_can_be_reapplied_after_a_prior_reset(self):
        previous_displayed_totals = {
            "p1": 88,
            "p2": 82,
            "p3": 75,
            "p4": 69,
            "p5": 61,
        }
        prepared_rows = [
            {"player_key": "p1", "detected_total": 15, "race_points": 15},
            {"player_key": "p2", "detected_total": 12, "race_points": 12},
            {"player_key": "p3", "detected_total": 10, "race_points": 10},
            {"player_key": "p4", "detected_total": 9, "race_points": 9},
            {"player_key": "p5", "detected_total": 8, "race_points": 8},
        ]

        self.assertTrue(detect_connection_reset(previous_displayed_totals, prepared_rows))

    def test_detect_obvious_total_score_reset_when_totals_match_race_points_pattern(self):
        prepared_rows = [
            {"detected_total": 15, "session_new_total": 28, "race_points": 15},
            {"detected_total": 12, "session_new_total": 37, "race_points": 12},
            {"detected_total": 10, "session_new_total": 28, "race_points": 10},
            {"detected_total": 9, "session_new_total": 22, "race_points": 9},
            {"detected_total": 8, "session_new_total": 13, "race_points": 8},
            {"detected_total": 7, "session_new_total": 27, "race_points": 7},
            {"detected_total": 6, "session_new_total": 31, "race_points": 6},
            {"detected_total": 5, "session_new_total": 15, "race_points": 5},
            {"detected_total": 4, "session_new_total": 14, "race_points": 4},
            {"detected_total": 3, "session_new_total": 12, "race_points": 3},
            {"detected_total": 2, "session_new_total": 11, "race_points": 2},
            {"detected_total": 1, "session_new_total": 8, "race_points": 1},
        ]

        self.assertTrue(detect_obvious_total_score_reset(prepared_rows))

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

    def test_assign_shared_positions_after_race_uses_shared_rank_for_equal_totals(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "Demo", "RaceIDNumber": 1, "RacePosition": 2, "FixPlayerName": "A", "NewTotalScore": 30},
                {"RaceClass": "Demo", "RaceIDNumber": 1, "RacePosition": 4, "FixPlayerName": "B", "NewTotalScore": 30},
                {"RaceClass": "Demo", "RaceIDNumber": 1, "RacePosition": 3, "FixPlayerName": "C", "NewTotalScore": 20},
                {"RaceClass": "Demo", "RaceIDNumber": 1, "RacePosition": 1, "FixPlayerName": "D", "NewTotalScore": 10},
            ]
        )

        ranked = assign_shared_positions_after_race(df)

        self.assertEqual(ranked["PositionAfterRace"].tolist(), [1, 1, 3, 4])

    def test_assign_shared_positions_after_race_supports_three_way_first_place_tie(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "Demo", "RaceIDNumber": 12, "RacePosition": 1, "FixPlayerName": "Bas", "NewTotalScore": 101},
                {"RaceClass": "Demo", "RaceIDNumber": 12, "RacePosition": 2, "FixPlayerName": "Matthijs", "NewTotalScore": 101},
                {"RaceClass": "Demo", "RaceIDNumber": 12, "RacePosition": 3, "FixPlayerName": "Gianni", "NewTotalScore": 101},
                {"RaceClass": "Demo", "RaceIDNumber": 12, "RacePosition": 4, "FixPlayerName": "Menno", "NewTotalScore": 96},
            ]
        )

        ranked = assign_shared_positions_after_race(df)

        self.assertEqual(ranked["PositionAfterRace"].tolist(), [1, 1, 1, 4])

    def test_apply_session_validation_suppresses_order_violation_when_total_matches_expected(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "Demo",
                    "RaceIDNumber": 1,
                    "RacePosition": 1,
                    "FixPlayerName": "A",
                    "RacePoints": 7,
                    "DetectedRacePoints": 7,
                    "DetectedRacePointsSource": "ocr",
                    "DetectedOldTotalScore": 4,
                    "DetectedOldTotalScoreSource": "ocr",
                    "DetectedTotalScore": 11,
                    "DetectedTotalScoreSource": "ocr",
                    "DetectedPositionAfterRace": 1,
                    "PositionAfterRace": 1,
                    "ReviewReason": "",
                    "IsLowRes": False,
                    "NameConfidence": 100.0,
                    "DigitConsensus": 100.0,
                    "RowCountConfidence": 100.0,
                    "RaceScorePlayerCount": 2,
                    "TotalScorePlayerCount": 2,
                },
                {
                    "RaceClass": "Demo",
                    "RaceIDNumber": 1,
                    "RacePosition": 2,
                    "FixPlayerName": "B",
                    "RacePoints": 3,
                    "DetectedRacePoints": 3,
                    "DetectedRacePointsSource": "ocr",
                    "DetectedOldTotalScore": 7,
                    "DetectedOldTotalScoreSource": "ocr",
                    "DetectedTotalScore": 10,
                    "DetectedTotalScoreSource": "ocr",
                    "DetectedPositionAfterRace": 2,
                    "PositionAfterRace": 2,
                    "ReviewReason": "",
                    "IsLowRes": False,
                    "NameConfidence": 100.0,
                    "DigitConsensus": 100.0,
                    "RowCountConfidence": 100.0,
                    "RaceScorePlayerCount": 2,
                    "TotalScorePlayerCount": 2,
                },
            ]
        )

        validated = apply_session_validation(
            df,
            parse_detected_int=lambda value: None if pd.isna(value) else int(value),
            exact_total_score_fallback=lambda prepared_rows: {},
        )

        review_text = " | ".join(str(value) for value in validated["ReviewReason"].fillna("").tolist())
        self.assertNotIn("Scoreboard total order is not descending.", review_text)
