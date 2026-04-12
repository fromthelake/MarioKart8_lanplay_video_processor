import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

from mk8_local_play import ocr_scoreboard_consensus
from mk8_local_play.ocr_scoreboard_consensus import map_total_rows_to_race_rows


class OcrScoreboardConsensusTests(unittest.TestCase):
    def test_player_identity_key_folds_accents(self):
        self.assertEqual(
            ocr_scoreboard_consensus.player_identity_key("Erík"),
            ocr_scoreboard_consensus.player_identity_key("Erik"),
        )

    def test_aligned_character_match_score_recovers_small_translation(self):
        template = np.zeros((48, 48, 3), dtype=np.uint8)
        template[12:36, 18:30] = (0, 0, 255)
        template[18:30, 22:26] = (255, 255, 255)
        alpha = np.zeros((48, 48), dtype=np.uint8)
        alpha[12:36, 18:30] = 255

        source = np.zeros((48, 48, 3), dtype=np.uint8)
        source[15:39, 21:33] = (0, 0, 255)
        source[21:33, 25:29] = (255, 255, 255)

        match = ocr_scoreboard_consensus.aligned_character_match_score(source, template, alpha)

        self.assertGreater(match["score"], 0.55)
        self.assertEqual(match["offset_x"], 3)
        self.assertEqual(match["offset_y"], 3)

    def test_masked_character_match_score_prefers_best_offset(self):
        template = np.zeros((48, 48, 3), dtype=np.uint8)
        template[:, :] = (0, 0, 255)
        alpha = np.full((48, 48), 255, dtype=np.uint8)

        source = np.zeros((52, 52, 3), dtype=np.uint8)
        source[2:50, 2:50] = (0, 0, 255)

        match = ocr_scoreboard_consensus.masked_character_match_score(source, template, alpha)

        self.assertGreater(match["score"], 0.99)
        self.assertEqual(match["offset_x"], 2)
        self.assertEqual(match["offset_y"], 2)

    def test_should_reject_character_match_as_mii_for_weak_ambiguous_match(self):
        self.assertTrue(
            ocr_scoreboard_consensus._should_reject_character_match_as_mii(
                [
                    {"CharacterMatchConfidence": 40.8, "CharacterRosterIndex": 5},
                    {"CharacterMatchConfidence": 40.4, "CharacterRosterIndex": 7},
                    {"CharacterMatchConfidence": 39.9, "CharacterRosterIndex": 12},
                    {"CharacterMatchConfidence": 39.1, "CharacterRosterIndex": 18},
                    {"CharacterMatchConfidence": 37.8, "CharacterRosterIndex": 24},
                ],
            )
        )
        self.assertFalse(
            ocr_scoreboard_consensus._should_reject_character_match_as_mii(
                [
                    {"CharacterMatchConfidence": 88.2, "CharacterRosterIndex": 5},
                    {"CharacterMatchConfidence": 82.5, "CharacterRosterIndex": 5},
                    {"CharacterMatchConfidence": 81.1, "CharacterRosterIndex": 5},
                    {"CharacterMatchConfidence": 78.9, "CharacterRosterIndex": 12},
                    {"CharacterMatchConfidence": 77.7, "CharacterRosterIndex": 14},
                ],
            )
        )

    def test_build_character_match_metrics_rejects_weak_open_set_match_to_mii(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        template_image = np.zeros((48, 48, 3), dtype=np.uint8)
        template_alpha = np.full((48, 48), 255, dtype=np.uint8)
        with (
            mock.patch.object(
                ocr_scoreboard_consensus,
                "load_character_templates",
                return_value=[
                    {
                        "character_index": 42,
                        "character_name": "Baby Peach",
                        "template_image": template_image,
                        "template_alpha": template_alpha,
                    },
                    {
                        "character_index": 43,
                        "character_name": "Baby Daisy",
                        "template_image": template_image,
                        "template_alpha": template_alpha,
                    },
                ],
            ),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "best_character_matches",
                return_value=[
                    {
                        "Character": "Baby Peach",
                        "CharacterIndex": 42,
                        "CharacterRosterIndex": 5,
                        "CharacterMatchConfidence": 40.8,
                        "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                        "CharacterMatchOffsetX": 0,
                        "CharacterMatchOffsetY": 0,
                    },
                    {
                        "Character": "Baby Daisy",
                        "CharacterIndex": 43,
                        "CharacterRosterIndex": 7,
                        "CharacterMatchConfidence": 40.4,
                        "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                        "CharacterMatchOffsetX": 0,
                        "CharacterMatchOffsetY": 0,
                    },
                    {
                        "Character": "Baby Rosalina",
                        "CharacterIndex": 44,
                        "CharacterRosterIndex": 9,
                        "CharacterMatchConfidence": 39.9,
                        "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                        "CharacterMatchOffsetX": 0,
                        "CharacterMatchOffsetY": 0,
                    },
                    {
                        "Character": "Link",
                        "CharacterIndex": 45,
                        "CharacterRosterIndex": 15,
                        "CharacterMatchConfidence": 39.1,
                        "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                        "CharacterMatchOffsetX": 0,
                        "CharacterMatchOffsetY": 0,
                    },
                    {
                        "Character": "Pink Gold Peach",
                        "CharacterIndex": 46,
                        "CharacterRosterIndex": 19,
                        "CharacterMatchConfidence": 37.8,
                        "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                        "CharacterMatchOffsetX": 0,
                        "CharacterMatchOffsetY": 0,
                    },
                ],
            ),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "_character_template_edge_agreement",
                return_value=0.20,
            ),
        ):
            metrics = ocr_scoreboard_consensus.build_character_match_metrics(frame)

        self.assertEqual(metrics[0]["Character"], "Mii")
        self.assertEqual(metrics[0]["CharacterIndex"], 80)
        self.assertEqual(metrics[0]["CharacterMatchMethod"], "open_set_mii_reject")

    def test_build_character_match_metrics_ignores_prior_for_duplicate_names_in_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        template_image = np.zeros((48, 48, 3), dtype=np.uint8)
        template_alpha = np.full((48, 48), 255, dtype=np.uint8)
        strong_yoshi_matches = [
            {
                "Character": "Yellow Yoshi",
                "CharacterIndex": 20,
                "CharacterRosterIndex": 8,
                "CharacterMatchConfidence": 57.9,
                "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                "CharacterMatchColorConfidence": 90.3,
                "CharacterMatchEdgeConfidence": 49.8,
                "CharacterMatchOffsetX": 3,
                "CharacterMatchOffsetY": 1,
            },
            {
                "Character": "Orange Yoshi",
                "CharacterIndex": 24,
                "CharacterRosterIndex": 8,
                "CharacterMatchConfidence": 53.7,
                "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                "CharacterMatchColorConfidence": 87.1,
                "CharacterMatchEdgeConfidence": 45.3,
                "CharacterMatchOffsetX": 3,
                "CharacterMatchOffsetY": 1,
            },
            {
                "Character": "Yoshi",
                "CharacterIndex": 16,
                "CharacterRosterIndex": 8,
                "CharacterMatchConfidence": 53.1,
                "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                "CharacterMatchColorConfidence": 80.6,
                "CharacterMatchEdgeConfidence": 46.2,
                "CharacterMatchOffsetX": 3,
                "CharacterMatchOffsetY": 1,
            },
        ]
        with (
            mock.patch.object(
                ocr_scoreboard_consensus,
                "load_character_templates",
                return_value=[
                    {
                        "character_index": 20,
                        "character_name": "Yellow Yoshi",
                        "template_image": template_image,
                        "template_alpha": template_alpha,
                    },
                    {
                        "character_index": 24,
                        "character_name": "Orange Yoshi",
                        "template_image": template_image,
                        "template_alpha": template_alpha,
                    },
                ],
            ),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "_eligible_character_shortlist_indices",
                return_value={20},
            ),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "_eligible_player_character_priors",
                return_value={
                    "arend": {
                        "CharacterIndex": 80,
                        "Character": "Mii",
                        "CharacterMatchConfidence": 75.0,
                        "seen_count": 5,
                        "fast_accepts_since_revalidation": 0,
                        "winner_counts": {"Mii": 5},
                        "closed_set_samples": 5,
                        "confidence_sum": 0.0,
                        "margin_sum": 0.0,
                        "spread_sum": 0.0,
                        "family_count_sum": 0.0,
                        "mii_likely": True,
                        "candidate_indices": [20, 24],
                    }
                },
            ),
            mock.patch.object(
                ocr_scoreboard_consensus,
                "best_character_matches",
                return_value=strong_yoshi_matches,
            ),
        ):
            metrics = ocr_scoreboard_consensus.build_character_match_metrics(
                frame,
                names=["Arend", "Arend"],
                name_confidences=[100, 100],
                video_context="VideoA",
                race_id_number=6,
            )

        self.assertEqual(metrics[0]["Character"], "Yellow Yoshi")
        self.assertNotEqual(metrics[0]["CharacterMatchMethod"], "character_prior_mii_likely")

    def test_map_total_rows_to_race_rows_keeps_race_character_when_present(self):
        score_rows = [
            {
                "PlayerName": "Wilco",
                "NameConfidence": 100.0,
                "CharacterIndex": 80,
                "Character": "Mii",
                "CharacterMatchConfidence": 40.0,
                "CharacterMatchMethod": "open_set_mii_reject",
                "DetectedValue": 6,
                "DetectedSecondaryValue": 44,
                "DigitConfidence": 100.0,
            }
        ]
        total_rows = [
            {
                "RowIndex": 0,
                "PlayerName": "Wilco",
                "NameConfidence": 100.0,
                "CharacterIndex": 73,
                "Character": "Isabelle",
                "CharacterMatchConfidence": 76.0,
                "CharacterMatchMethod": "alpha_aware_color_edge_template_local_search",
                "DetectedValue": 50,
                "DetectedValueSource": "ocr",
                "DigitConfidence": 100.0,
            }
        ]

        mapped = map_total_rows_to_race_rows(
            score_rows,
            total_rows,
            preprocess_name=lambda value: str(value or "").lower(),
            weighted_similarity=lambda left, right: 1.0 if left == right else 0.0,
            total_row_metrics=[{}],
        )

        self.assertEqual(mapped[0]["Character"], "Mii")
        self.assertEqual(mapped[0]["CharacterIndex"], 80)
        self.assertEqual(mapped[0]["CharacterMatchMethod"], "open_set_mii_reject")

    def test_eligible_character_shortlist_indices_ignore_future_races(self):
        shortlist_state = {42: 11, 43: 9, 44: 12}
        self.assertEqual(
            ocr_scoreboard_consensus._eligible_character_shortlist_indices(shortlist_state, 11),
            {43},
        )
        self.assertEqual(
            ocr_scoreboard_consensus._eligible_character_shortlist_indices(shortlist_state, 13),
            {42, 43, 44},
        )

    def test_eligible_player_character_priors_ignore_future_races(self):
        priors = {
            "ronald": {"CharacterIndex": 42, "last_race_id": 11},
            "amber": {"CharacterIndex": 43, "last_race_id": 9},
        }
        filtered = ocr_scoreboard_consensus._eligible_player_character_priors(priors, 11)
        self.assertNotIn("ronald", filtered)
        self.assertIn("amber", filtered)

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
        ):
            metrics = ocr_scoreboard_consensus.build_position_signal_metrics(
                processed,
                stats=stats,
                stats_prefix="test",
                max_rows=2,
            )

        self.assertEqual(metrics, expected_metrics)
        self.assertEqual(stats["test_position_metrics_black_white_calls"], 1)
        self.assertEqual(stats["test_position_metrics_rows_processed"], 2)

    def test_prior_state_is_mii_likely_for_unstable_closed_set_history(self):
        prior_state = {
            "winner_counts": {"Pink Yoshi": 2, "Red Yoshi": 1},
            "closed_set_samples": 3,
            "confidence_sum": 240.0,
            "margin_sum": 3.0,
            "spread_sum": 6.0,
            "family_count_sum": 12.0,
            "mii_likely": False,
        }

        self.assertTrue(ocr_scoreboard_consensus._prior_state_is_mii_likely(prior_state))

    def test_prior_state_is_not_mii_likely_for_stable_closed_set_history(self):
        prior_state = {
            "winner_counts": {"Inkling Boy": 3},
            "closed_set_samples": 3,
            "confidence_sum": 285.0,
            "margin_sum": 42.0,
            "spread_sum": 54.0,
            "family_count_sum": 3.0,
            "mii_likely": False,
        }

        self.assertFalse(ocr_scoreboard_consensus._prior_state_is_mii_likely(prior_state))

    def test_prior_state_is_mii_likely_for_unstable_two_sample_history(self):
        prior_state = {
            "winner_counts": {"Pink Yoshi": 1, "Red Yoshi": 1},
            "closed_set_samples": 2,
            "confidence_sum": 150.0,
            "margin_sum": 1.0,
            "spread_sum": 4.0,
            "family_count_sum": 8.0,
            "mii_likely": False,
        }

        self.assertTrue(ocr_scoreboard_consensus._prior_state_is_mii_likely(prior_state))

    def test_prior_state_is_not_mii_likely_for_stable_two_sample_history(self):
        prior_state = {
            "winner_counts": {"Inkling Boy": 2},
            "closed_set_samples": 2,
            "confidence_sum": 100.0,
            "margin_sum": 1.0,
            "spread_sum": 4.0,
            "family_count_sum": 8.0,
            "mii_likely": False,
        }

        self.assertFalse(ocr_scoreboard_consensus._prior_state_is_mii_likely(prior_state))

    def test_merge_prior_candidate_indices_preserves_order_and_filters_invalid(self):
        merged = ocr_scoreboard_consensus._merge_prior_candidate_indices(
            [16, 19, 80],
            [19, "22", 79, None, 56],
        )

        self.assertEqual(merged, [16, 19, 22, 56])


if __name__ == "__main__":
    unittest.main()
