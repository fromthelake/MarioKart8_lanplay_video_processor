import unittest
from unittest.mock import patch

import pandas as pd

import mk8_local_play.ocr_name_matching as ocr_name_matching
from mk8_local_play.ocr_name_matching import merge_fragmented_identity_aliases, standardize_player_names


class TestOcrNameMatching(unittest.TestCase):
    def test_visual_similarity_can_separate_same_name_players(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "Poule_A",
                    "RaceIDNumber": 1,
                    "RacePosition": 1,
                    "PlayerName": "Pieter",
                    "CharacterIndex": 10,
                    "DetectedTotalScore": pd.NA,
                    "NameConfidence": 100.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "",
                    "ScoreLayoutId": "lan2_split_2p",
                },
                {
                    "RaceClass": "Poule_A",
                    "RaceIDNumber": 1,
                    "RacePosition": 2,
                    "PlayerName": "Pieter",
                    "CharacterIndex": 20,
                    "DetectedTotalScore": pd.NA,
                    "NameConfidence": 100.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "",
                    "ScoreLayoutId": "lan2_split_2p",
                },
                {
                    "RaceClass": "Poule_A",
                    "RaceIDNumber": 2,
                    "RacePosition": 1,
                    "PlayerName": "Pieter",
                    "CharacterIndex": 99,
                    "DetectedTotalScore": pd.NA,
                    "NameConfidence": 100.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "",
                    "ScoreLayoutId": "lan2_split_2p",
                },
                {
                    "RaceClass": "Poule_A",
                    "RaceIDNumber": 2,
                    "RacePosition": 2,
                    "PlayerName": "Pieter",
                    "CharacterIndex": 98,
                    "DetectedTotalScore": pd.NA,
                    "NameConfidence": 100.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "",
                    "ScoreLayoutId": "lan2_split_2p",
                },
            ]
        )

        visual_features = {0: "p1", 1: "p2", 2: "p1", 3: "p2"}

        def fake_visual_similarity(identity_visual_refs, row_visual_roi):
            return 1.0 if row_visual_roi in identity_visual_refs else 0.0

        with patch.object(ocr_name_matching, "_prepare_visual_identity_features", return_value=visual_features):
            with patch.object(ocr_name_matching, "_visual_similarity", side_effect=fake_visual_similarity):
                standardized = standardize_player_names(df, output_folder=".", write_debug_linking_excel=False)

        race1 = standardized[standardized["RaceIDNumber"] == 1].sort_values("RacePosition", kind="stable")
        race2 = standardized[standardized["RaceIDNumber"] == 2].sort_values("RacePosition", kind="stable")

        self.assertEqual(race1.iloc[0]["FixPlayerName"], race2.iloc[0]["FixPlayerName"])
        self.assertEqual(race1.iloc[1]["FixPlayerName"], race2.iloc[1]["FixPlayerName"])
        self.assertNotEqual(race2.iloc[0]["FixPlayerName"], race2.iloc[1]["FixPlayerName"])

    def test_unreliable_name_can_match_on_character_and_total_continuity(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 1,
                    "RacePosition": 6,
                    "PlayerName": "niablB",
                    "CharacterIndex": 80,
                    "DetectedTotalScore": 15,
                    "NameConfidence": 81.5,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                    "Character": "Mii",
                },
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 2,
                    "RacePosition": 7,
                    "PlayerName": "nizlB",
                    "CharacterIndex": 74,
                    "DetectedTotalScore": 21,
                    "NameConfidence": 86.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                    "Character": "Champion Link",
                },
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 3,
                    "RacePosition": 8,
                    "PlayerName": "nialbS",
                    "CharacterIndex": 80,
                    "DetectedTotalScore": 26,
                    "NameConfidence": 100.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                    "Character": "Mii",
                },
            ]
        )

        standardized = standardize_player_names(df, output_folder=".", write_debug_linking_excel=False)
        labels = standardized["FixPlayerName"].tolist()

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[0], labels[2])

    def test_fragmented_aliases_merge_when_video_exceeds_expected_player_count(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 1,
                    "RacePosition": 6,
                    "PlayerName": "ñíæłB",
                    "FixPlayerName": "ñíæłB",
                    "IdentityLabel": "ñíæłB",
                    "IdentityResolutionMethod": "new_identity",
                    "CharacterIndex": 80,
                    "NameConfidence": 81.5,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                },
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 2,
                    "RacePosition": 7,
                    "PlayerName": "ñîzlB",
                    "FixPlayerName": "ñíæłB",
                    "IdentityLabel": "ñíæłB",
                    "IdentityResolutionMethod": "name+character",
                    "CharacterIndex": 80,
                    "NameConfidence": 86.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                },
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 3,
                    "RacePosition": 4,
                    "PlayerName": "ñiæłß",
                    "FixPlayerName": "ñiæłß",
                    "IdentityLabel": "ñiæłß",
                    "IdentityResolutionMethod": "new_identity",
                    "CharacterIndex": 80,
                    "NameConfidence": 88.7,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                },
            ]
            + [
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": race_id,
                    "RacePosition": position,
                    "PlayerName": f"Player{position}",
                    "FixPlayerName": f"Player{position}",
                    "IdentityLabel": f"Player{position}",
                    "IdentityResolutionMethod": "name_only",
                    "CharacterIndex": position,
                    "NameConfidence": 100.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "",
                }
                for race_id in (1, 2, 3)
                for position in range(1, 13)
                if position != 6 and not (race_id == 2 and position == 7) and not (race_id == 3 and position == 4)
            ]
        )

        merged = merge_fragmented_identity_aliases(df)
        labels = sorted(merged["FixPlayerName"].dropna().unique().tolist())

        self.assertEqual(len(labels), 12)
        weird_rows = merged[merged["PlayerName"].astype(str).str.contains("ñ", na=False)]
        self.assertEqual(len(weird_rows["FixPlayerName"].unique()), 1)

    def test_small_unreliable_fragment_can_merge_despite_character_mismatch(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 1,
                    "RacePosition": 6,
                    "PlayerName": "ñíæłB",
                    "FixPlayerName": "ñíæłB",
                    "IdentityLabel": "ñíæłB",
                    "IdentityResolutionMethod": "new_identity",
                    "CharacterIndex": 80,
                    "NameConfidence": 81.5,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                },
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 2,
                    "RacePosition": 7,
                    "PlayerName": "ñîzlB",
                    "FixPlayerName": "ñíæłB",
                    "IdentityLabel": "ñíæłB",
                    "IdentityResolutionMethod": "name+character",
                    "CharacterIndex": 80,
                    "NameConfidence": 86.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                },
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 3,
                    "RacePosition": 8,
                    "PlayerName": "ñizlß",
                    "FixPlayerName": "ñíæłB",
                    "IdentityLabel": "ñíæłB",
                    "IdentityResolutionMethod": "name+character",
                    "CharacterIndex": 80,
                    "NameConfidence": 100.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "low_name_confidence",
                },
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 4,
                    "RacePosition": 5,
                    "PlayerName": "ńiæłß",
                    "FixPlayerName": "ñíæłB",
                    "IdentityLabel": "ñíæłB",
                    "IdentityResolutionMethod": "name+character",
                    "CharacterIndex": 80,
                    "NameConfidence": 94.5,
                    "NameAllowedCharRatio": 80.0,
                    "NameValidationFlags": "unknown_chars|low_name_confidence",
                },
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": 5,
                    "RacePosition": 4,
                    "PlayerName": "ńiæłß",
                    "FixPlayerName": "ńiæłß",
                    "IdentityLabel": "ńiæłß",
                    "IdentityResolutionMethod": "new_identity",
                    "CharacterIndex": 74,
                    "NameConfidence": 61.3,
                    "NameAllowedCharRatio": 80.0,
                    "NameValidationFlags": "unknown_chars|low_name_confidence",
                },
            ]
            + [
                {
                    "RaceClass": "Toernooi 1 - Ronde 1 - Groep 2",
                    "RaceIDNumber": race_id,
                    "RacePosition": position,
                    "PlayerName": f"Player{position}",
                    "FixPlayerName": f"Player{position}",
                    "IdentityLabel": f"Player{position}",
                    "IdentityResolutionMethod": "name_only",
                    "CharacterIndex": position,
                    "NameConfidence": 100.0,
                    "NameAllowedCharRatio": 100.0,
                    "NameValidationFlags": "",
                }
                for race_id in (1, 2, 3, 4, 5)
                for position in range(1, 13)
                if position != 6
                and not (race_id == 2 and position == 7)
                and not (race_id == 3 and position == 8)
                and not (race_id == 4 and position == 5)
                and not (race_id == 5 and position == 4)
            ]
        )

        merged = merge_fragmented_identity_aliases(df)
        weird_rows = merged[merged["PlayerName"].astype(str).str.contains("ñ|ń", na=False)]
        self.assertEqual(len(weird_rows["FixPlayerName"].unique()), 1)


if __name__ == "__main__":
    unittest.main()
