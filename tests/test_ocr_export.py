import unittest

import pandas as pd

from mk8_local_play.ocr_export import (
    _dedupe_review_reason_parts,
    _select_most_used_character,
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
                {"VideoName": "Demo", "Races": 2, "Position": 1, "PlayerName": "Alpha", "TotalPoints": 20, "Character": "Luigi"},
                {"VideoName": "Demo", "Races": 2, "Position": 1, "PlayerName": "Beta", "TotalPoints": 20, "Character": "Yoshi"},
            ],
        )
