import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

from mk8_local_play import extract_score_screen_selection


class ExtractScoreScreenSelectionTests(unittest.TestCase):
    def test_count_tie_aware_prefix_rows_accepts_tied_ranks(self):
        metrics = [
            {"best_position_template": 1, "best_position_score": 0.92, "coeff_ranked_templates": [1, 2, 3, 4, 5, 6]},
            {"best_position_template": 1, "best_position_score": 0.88, "coeff_ranked_templates": [1, 2, 3, 4, 5, 6]},
            {"best_position_template": 3, "best_position_score": 0.85, "coeff_ranked_templates": [3, 2, 1, 4, 5, 6]},
            {"best_position_template": 4, "best_position_score": 0.83, "coeff_ranked_templates": [4, 3, 2, 1, 5, 6]},
            {"best_position_template": 5, "best_position_score": 0.81, "coeff_ranked_templates": [5, 4, 3, 2, 1, 6]},
            {"best_position_template": 6, "best_position_score": 0.80, "coeff_ranked_templates": [6, 5, 4, 3, 2, 1]},
        ]

        visible_rows = extract_score_screen_selection._count_tie_aware_prefix_rows(metrics, 6)

        self.assertEqual(visible_rows, 6)

    def test_count_tie_aware_prefix_rows_rejects_rank_jump_beyond_row(self):
        metrics = [
            {"best_position_template": 1, "best_position_score": 0.92, "coeff_ranked_templates": [1, 2, 3, 4, 5, 6]},
            {"best_position_template": 3, "best_position_score": 0.88, "coeff_ranked_templates": [3, 4, 5, 6]},
            {"best_position_template": 3, "best_position_score": 0.85, "coeff_ranked_templates": [3, 2, 1, 4, 5, 6]},
        ]

        visible_rows = extract_score_screen_selection._count_tie_aware_prefix_rows(metrics, 3)

        self.assertEqual(visible_rows, 1)

    def test_match_twelfth_presence_uses_strongest_template_variant(self):
        gray_image = np.full((720, 1280), 255, dtype=np.uint8)
        templates = [(np.zeros((5, 5), dtype=np.uint8), None) for _ in range(9)]
        score_layout = mock.Mock()
        score_layout.twelfth_place_check_roi = (10, 10, 20, 20)

        with (
            mock.patch.object(
                extract_score_screen_selection,
                "preprocess_roi",
                side_effect=[np.full((20, 20), 255, dtype=np.uint8), np.full((20, 20), 255, dtype=np.uint8)],
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "match_template",
                side_effect=[0.31, 0.72],
            ),
        ):
            max_val = extract_score_screen_selection._match_twelfth_presence(
                gray_image,
                templates,
                score_layout,
                defaultdict(float),
            )

        self.assertAlmostEqual(max_val, 0.72)

    def test_expand_race_score_consensus_window_requires_twelfth_template_detection(self):
        result = {
            "twelfth_template_detected": False,
            "candidate": {
                "video_path": "dummy.mp4",
                "left": 0,
                "top": 0,
                "crop_width": 10,
                "crop_height": 10,
                "ocr_consensus_frames": 7,
            },
            "actual_race_score_frame": 100,
            "race_consensus_frames": ["keep"],
        }

        expanded = extract_score_screen_selection.expand_race_score_consensus_window(result, 12)

        self.assertEqual(expanded["race_consensus_frames"], ["keep"])


if __name__ == "__main__":
    unittest.main()
