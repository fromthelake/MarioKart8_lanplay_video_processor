import unittest

import pandas as pd

from mk8_local_play.ocr_export import (
    _dedupe_review_reason_parts,
    _build_saved_file_lines,
    _select_most_used_character,
    build_player_count_summary_lines,
    build_final_standings_df,
    format_review_reason_for_export,
)


class OcrExportTests(unittest.TestCase):
    def test_dedupe_review_reason_parts_is_case_insensitive_and_strips_whitespace(self):
        value = " Low name confidence | low name confidence | Total mismatch "

        self.assertEqual(_dedupe_review_reason_parts(value), ["Low name confidence", "Total mismatch"])

    def test_format_review_reason_for_export_truncates_with_omitted_count(self):
        value = "one | two | three | four"

        self.assertEqual(
            format_review_reason_for_export(value, max_length=18),
            "one ... (+3 more)",
        )

    def test_select_most_used_character_breaks_ties_by_last_seen_race_then_name(self):
        player_rows = pd.DataFrame(
            [
                {"Character": "Luigi", "RaceIDNumber": 1},
                {"Character": "Yoshi", "RaceIDNumber": 2},
                {"Character": "Luigi", "RaceIDNumber": 3},
                {"Character": "Yoshi", "RaceIDNumber": 4},
            ]
        )

        self.assertEqual(_select_most_used_character(player_rows), "Yoshi")

    def test_build_final_standings_df_preserves_shared_positions(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "Demo", "RaceIDNumber": 1, "RacePosition": 1, "FixPlayerName": "Alpha", "NewTotalScore": 15, "Character": "Luigi"},
                {"RaceClass": "Demo", "RaceIDNumber": 1, "RacePosition": 2, "FixPlayerName": "Beta", "NewTotalScore": 12, "Character": "Yoshi"},
                {"RaceClass": "Demo", "RaceIDNumber": 2, "RacePosition": 1, "FixPlayerName": "Alpha", "NewTotalScore": 20, "Character": "Luigi"},
                {"RaceClass": "Demo", "RaceIDNumber": 2, "RacePosition": 2, "FixPlayerName": "Beta", "NewTotalScore": 20, "Character": "Yoshi"},
            ]
        )

        standings = build_final_standings_df(df)

        self.assertEqual(
            standings.to_dict(orient="records"),
            [
                {"VideoName": "Demo", "Races": 2, "Position": 1, "PlayerName": "Alpha", "TotalPoints": 20, "Character": "Luigi", "CharacterRosterName": "Luigi"},
                {"VideoName": "Demo", "Races": 2, "Position": 1, "PlayerName": "Beta", "TotalPoints": 20, "Character": "Yoshi", "CharacterRosterName": "Yoshi"},
            ],
        )

    def test_build_player_count_summary_lines_uses_compact_consistent_message(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "Demo", "RaceIDNumber": 1, "RaceScorePlayerCount": 12, "TotalScorePlayerCount": 12, "TrackName": "Mario Kart Stadium", "ReviewNeeded": False, "RowCountConfidence": 1.0},
                {"RaceClass": "Demo", "RaceIDNumber": 1, "RaceScorePlayerCount": 12, "TotalScorePlayerCount": 12, "TrackName": "Mario Kart Stadium", "ReviewNeeded": False, "RowCountConfidence": 1.0},
                {"RaceClass": "Demo", "RaceIDNumber": 2, "RaceScorePlayerCount": 12, "TotalScorePlayerCount": 12, "TrackName": "Water Park", "ReviewNeeded": False, "RowCountConfidence": 1.0},
                {"RaceClass": "Demo", "RaceIDNumber": 2, "RaceScorePlayerCount": 12, "TotalScorePlayerCount": 12, "TrackName": "Water Park", "ReviewNeeded": False, "RowCountConfidence": 1.0},
            ]
        )

        lines, summary = build_player_count_summary_lines(
            df,
            lambda *_args, **_kwargs: [],
            lambda count, singular, plural=None: singular if count == 1 else (plural or f"{singular}s"),
        )

        self.assertIn("Player count check", lines)
        self.assertIn("- Demo: 2 races | consistent at 2 players", lines)
        self.assertEqual(summary["Demo"]["player_count_summary"], "consistent (2 players)")

    def test_build_saved_file_lines_puts_paths_on_separate_lines(self):
        lines = _build_saved_file_lines(
            {
                "output_excel_path": "C:/Results.xlsx",
                "debug_output_excel_path": "C:/Debug.xlsx",
                "output_csv_path": "C:/Results.csv",
                "final_standings_csv_path": "C:/Final.csv",
                "debug_output_csv_path": "C:/Debug.csv",
            }
        )

        self.assertEqual(lines[:2], ["", "Saved files"])
        self.assertIn("- Results workbook", lines)
        self.assertIn("  C:/Results.xlsx", lines)
