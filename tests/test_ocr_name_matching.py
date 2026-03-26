import unittest

import pandas as pd

from mk8_local_play.ocr_name_matching import (
    append_identity_ambiguity_review_notes,
    resolve_duplicate_name_identity_chains,
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


if __name__ == "__main__":
    unittest.main()
