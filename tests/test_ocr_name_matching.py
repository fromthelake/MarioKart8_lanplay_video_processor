import unittest

import pandas as pd
from unittest import mock

from mk8_local_play.ocr_name_matching import (
    append_identity_ambiguity_review_notes,
    reconcile_connection_reset_identities,
    resolve_duplicate_name_identity_chains,
    standardize_player_names,
)


class OcrNameMatchingTests(unittest.TestCase):
    def test_resolve_duplicate_name_identity_chains_prefers_total_continuity(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 1, "FixPlayerName": "Same_2", "IdentityLabel": "Same_2", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 0, "DetectedTotalScore": 5, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 2, "FixPlayerName": "Same_1", "IdentityLabel": "Same_1", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 0, "DetectedTotalScore": 3, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 1, "FixPlayerName": "Same_2", "IdentityLabel": "Same_2", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 3, "DetectedTotalScore": 7, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 2, "FixPlayerName": "Same_1", "IdentityLabel": "Same_1", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 5, "DetectedTotalScore": 10, "IdentityResolutionMethod": "name+visual"},
            ]
        )

        resolved = resolve_duplicate_name_identity_chains(df)
        race1 = resolved.loc[resolved["RaceIDNumber"] == 1].sort_values("RacePosition")
        race2 = resolved.loc[resolved["RaceIDNumber"] == 2].sort_values("RacePosition")

        self.assertEqual(list(race1["FixPlayerName"]), ["Same_1", "Same_2"])
        self.assertEqual(list(race2["FixPlayerName"]), ["Same_2", "Same_1"])

    def test_append_identity_ambiguity_review_notes_sets_review(self):
        df = pd.DataFrame(
            [
                {
                    "IdentityAmbiguityDetected": True,
                    "IdentityAmbiguityNote": "Identity ambiguous with Same_2: final race could not be resolved uniquely for duplicate name/character/score chain; mapping carried forward from previous race.",
                    "ReviewReason": "",
                    "ReviewNeeded": False,
                }
            ]
        )

        updated = append_identity_ambiguity_review_notes(df)
        self.assertTrue(bool(updated.at[0, "ReviewNeeded"]))
        self.assertIn("Identity ambiguous", str(updated.at[0, "ReviewReason"]))

    def test_resolve_duplicate_name_identity_chains_marks_only_ambiguous_final_rows(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 1, "FixPlayerName": "Same_1", "IdentityLabel": "Same_1", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 0, "DetectedTotalScore": 7, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 2, "FixPlayerName": "Same_2", "IdentityLabel": "Same_2", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 0, "DetectedTotalScore": 5, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 3, "FixPlayerName": "Same_3", "IdentityLabel": "Same_3", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 0, "DetectedTotalScore": 3, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 1, "FixPlayerName": "Same_1", "IdentityLabel": "Same_1", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 7, "DetectedTotalScore": 9, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 2, "FixPlayerName": "Same_2", "IdentityLabel": "Same_2", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 5, "DetectedTotalScore": 6, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 3, "FixPlayerName": "Same_3", "IdentityLabel": "Same_3", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 3, "DetectedTotalScore": 4, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 3, "RacePosition": 1, "FixPlayerName": "Same_1", "IdentityLabel": "Same_1", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 9, "DetectedTotalScore": 12, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 3, "RacePosition": 2, "FixPlayerName": "Same_2", "IdentityLabel": "Same_2", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 6, "DetectedTotalScore": 8, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 3, "RacePosition": 3, "FixPlayerName": "Same_3", "IdentityLabel": "Same_3", "PlayerName": "Same", "CharacterIndex": 10, "DetectedOldTotalScore": 6, "DetectedTotalScore": 7, "IdentityResolutionMethod": "name+visual"},
            ]
        )

        resolved = resolve_duplicate_name_identity_chains(df)
        final_race = resolved.loc[resolved["RaceIDNumber"] == 3].sort_values("RacePosition")
        notes = {row["FixPlayerName"]: str(row.get("IdentityAmbiguityNote") or "") for _, row in final_race.iterrows()}

        self.assertEqual(notes["Same_1"], "")
        self.assertIn("Identity ambiguous with Same_3", notes["Same_2"])
        self.assertIn("Identity ambiguous with Same_2", notes["Same_3"])

    def test_standardize_player_names_preserves_case_distinct_players_seen_together(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 1, "PlayerName": "Floris", "CharacterIndex": 10, "DetectedTotalScore": 12, "ScoreLayoutId": "lan2_split_2p"},
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 2, "PlayerName": "floris", "CharacterIndex": 20, "DetectedTotalScore": 9, "ScoreLayoutId": "lan2_split_2p"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 1, "PlayerName": "Floris", "CharacterIndex": 10, "DetectedTotalScore": 27, "ScoreLayoutId": "lan2_split_2p"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 2, "PlayerName": "floris", "CharacterIndex": 20, "DetectedTotalScore": 18, "ScoreLayoutId": "lan2_split_2p"},
            ]
        )

        with (
            mock.patch("mk8_local_play.ocr_name_matching.find_score_bundle_anchor_path", return_value=None),
            mock.patch("mk8_local_play.ocr_name_matching._prepare_visual_identity_features", return_value={}),
        ):
            standardized = standardize_player_names(df, output_folder=".", write_debug_linking_excel=False)

        race1 = standardized.loc[standardized["RaceIDNumber"] == 1].sort_values("RacePosition")
        race2 = standardized.loc[standardized["RaceIDNumber"] == 2].sort_values("RacePosition")

        self.assertEqual(list(race1["FixPlayerName"]), ["Floris", "floris"])
        self.assertEqual(list(race2["FixPlayerName"]), ["Floris", "floris"])

    def test_standardize_player_names_allows_case_conflict_when_total_and_character_continue(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 3, "PlayerName": "Willemijn", "CharacterIndex": 62, "DetectedOldTotalScore": 62, "DetectedTotalScore": 72, "ScoreLayoutId": "lan2_split_2p"},
                {"RaceClass": "demo", "RaceIDNumber": 1, "RacePosition": 4, "PlayerName": "Willemijn", "CharacterIndex": 5, "DetectedOldTotalScore": 57, "DetectedTotalScore": 66, "ScoreLayoutId": "lan2_split_2p"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 2, "PlayerName": "willemijn", "CharacterIndex": 62, "DetectedOldTotalScore": 72, "DetectedTotalScore": 84, "ScoreLayoutId": "lan2_split_2p"},
                {"RaceClass": "demo", "RaceIDNumber": 2, "RacePosition": 11, "PlayerName": "Willemijn", "CharacterIndex": 5, "DetectedOldTotalScore": 66, "DetectedTotalScore": 68, "ScoreLayoutId": "lan2_split_2p"},
            ]
        )

        with (
            mock.patch("mk8_local_play.ocr_name_matching.find_score_bundle_anchor_path", return_value=None),
            mock.patch("mk8_local_play.ocr_name_matching._prepare_visual_identity_features", return_value={}),
        ):
            standardized = standardize_player_names(df, output_folder=".", write_debug_linking_excel=False)

        race1 = standardized.loc[standardized["RaceIDNumber"] == 1].sort_values("RacePosition")
        race2 = standardized.loc[standardized["RaceIDNumber"] == 2].sort_values("RacePosition")

        self.assertEqual(list(race1["FixPlayerName"]), ["Willemijn_1", "Willemijn_2"])
        self.assertEqual(list(race2["FixPlayerName"]), ["Willemijn_1", "Willemijn_2"])

    def test_reconcile_connection_reset_identities_can_merge_two_identity_swaps(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "demo", "RaceIDNumber": 12, "RacePosition": 10, "FixPlayerName": "Caitlin", "IdentityLabel": "Caitlin", "PlayerName": "Caitlin", "CharacterIndex": 29, "SessionResetDetected": False, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 12, "RacePosition": 4, "FixPlayerName": "ñiæłß", "IdentityLabel": "ñiæłß", "PlayerName": "ñiæłß", "CharacterIndex": 80, "SessionResetDetected": False, "IdentityResolutionMethod": "name+visual"},
                {"RaceClass": "demo", "RaceIDNumber": 13, "RacePosition": 12, "FixPlayerName": "queen opa", "IdentityLabel": "queen opa", "PlayerName": "queen opa", "CharacterIndex": 29, "SessionResetDetected": True, "IdentityResolutionMethod": "new_identity"},
                {"RaceClass": "demo", "RaceIDNumber": 13, "RacePosition": 5, "FixPlayerName": "ñíæłß", "IdentityLabel": "ñíæłß", "PlayerName": "ñíæłß", "CharacterIndex": 80, "SessionResetDetected": True, "IdentityResolutionMethod": "new_identity"},
            ]
        )

        updated = reconcile_connection_reset_identities(df)

        race13 = updated.loc[updated["RaceIDNumber"] == 13].sort_values("RacePosition")
        self.assertEqual(list(race13["FixPlayerName"]), ["ñiæłß", "Caitlin"])
        self.assertTrue(all(bool(value) for value in race13["IdentityRelinkDetected"]))
        self.assertTrue(all("connection_reset_relink" in str(value) for value in race13["IdentityResolutionMethod"]))


if __name__ == "__main__":
    unittest.main()
