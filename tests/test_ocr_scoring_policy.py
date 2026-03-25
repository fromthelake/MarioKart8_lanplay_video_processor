import unittest

import pandas as pd

from mk8_local_play.ocr_scoring_policy import apply_temporary_player_drop_scoring_policy


def _build_policy_df(player_counts: list[int]) -> pd.DataFrame:
    rows = []
    player_names = [f"P{index}" for index in range(1, 13)]
    for race_id, player_count in enumerate(player_counts, start=1):
        for position in range(1, player_count + 1):
            rows.append(
                {
                    "RaceClass": "Demo",
                    "SessionIndex": 1,
                    "RaceIDNumber": race_id,
                    "RacePosition": position,
                    "FixPlayerName": player_names[position - 1],
                    "RacePoints": max(0, player_count - position + 1),
                    "OldTotalScore": 0,
                    "NewTotalScore": 0,
                    "SessionOldTotalScore": 0,
                    "SessionNewTotalScore": 0,
                    "PositionAfterRace": position,
                    "RaceScorePlayerCount": player_count,
                    "TotalScorePlayerCount": player_count,
                    "ReviewNeeded": False,
                    "ReviewReason": "",
                    "ReviewReasonCodes": "",
                }
            )
    return pd.DataFrame(rows)


class OcrScoringPolicyTests(unittest.TestCase):
    def test_intermediate_drop_is_excluded_when_later_higher_count_returns(self):
        df = _build_policy_df([7, 7, 5, 6, 6])

        adjusted = apply_temporary_player_drop_scoring_policy(df)

        counts = adjusted.groupby("RaceIDNumber", sort=True)["CountsTowardTotals"].first().to_dict()
        self.assertEqual(counts, {1: True, 2: True, 3: False, 4: True, 5: True})

    def test_only_highest_late_recovery_counts_in_degraded_chain(self):
        df = _build_policy_df([12, 10, 9, 11, 10, 8, 11, 12])

        adjusted = apply_temporary_player_drop_scoring_policy(df)

        counts = adjusted.groupby("RaceIDNumber", sort=True)["CountsTowardTotals"].first().to_dict()
        self.assertEqual(
            counts,
            {1: True, 2: False, 3: False, 4: False, 5: False, 6: False, 7: False, 8: True},
        )

    def test_terminal_lower_player_tail_stays_counted(self):
        df = _build_policy_df([12, 12, 11, 10, 9, 9])

        adjusted = apply_temporary_player_drop_scoring_policy(df)

        counts = adjusted.groupby("RaceIDNumber", sort=True)["CountsTowardTotals"].first().to_dict()
        self.assertEqual(counts, {1: True, 2: True, 3: True, 4: True, 5: True, 6: True})

    def test_recomputed_totals_ignore_excluded_races(self):
        df = _build_policy_df([7, 5, 6, 6])

        adjusted = apply_temporary_player_drop_scoring_policy(df)
        player_rows = adjusted.loc[adjusted["FixPlayerName"] == "P1"].sort_values("RaceIDNumber", kind="stable")

        self.assertEqual(player_rows["CountsTowardTotals"].tolist(), [True, False, True, True])
        self.assertEqual(player_rows["OldTotalScore"].tolist(), [0, 7, 7, 13])
        self.assertEqual(player_rows["NewTotalScore"].tolist(), [7, 7, 13, 19])


if __name__ == "__main__":
    unittest.main()
