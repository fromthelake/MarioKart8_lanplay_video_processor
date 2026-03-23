from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from mk8_local_play.extract_text import refine_black_blue_character_variants, rescue_placeholder_identity_names


class TestExtractTextRefinements(TestCase):
    def test_rescue_placeholder_identity_names_promotes_supported_name(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoA",
                    "RaceIDNumber": 1,
                    "RacePosition": 3,
                    "FixPlayerName": "PlayerNameMissing_3",
                    "IdentityLabel": "PlayerNameMissing_3",
                    "IdentityResolutionMethod": "new_identity",
                    "ScoreLayoutId": "lan2_split_2p",
                },
                {
                    "RaceClass": "VideoA",
                    "RaceIDNumber": 2,
                    "RacePosition": 4,
                    "FixPlayerName": "PlayerNameMissing_3",
                    "IdentityLabel": "PlayerNameMissing_3",
                    "IdentityResolutionMethod": "name+visual",
                    "ScoreLayoutId": "lan2_split_2p",
                },
                {
                    "RaceClass": "VideoA",
                    "RaceIDNumber": 3,
                    "RacePosition": 5,
                    "FixPlayerName": "PlayerNameMissing_3",
                    "IdentityLabel": "PlayerNameMissing_3",
                    "IdentityResolutionMethod": "name+visual",
                    "ScoreLayoutId": "lan2_split_2p",
                },
            ]
        )

        fake_layout = SimpleNamespace(
            player_name_coords=[((0, 0), (10, 10)) for _ in range(12)]
        )
        with patch("mk8_local_play.extract_text.find_score_bundle_anchor_path", return_value="dummy.png"):
            with patch("mk8_local_play.extract_text.cv2.imread", return_value=np.zeros((20, 20, 3), dtype=np.uint8)):
                with patch("mk8_local_play.extract_text.get_score_layout", return_value=fake_layout):
                    with patch(
                        "mk8_local_play.extract_text._generate_player_name_fallback_candidates",
                        side_effect=[
                            [("Lucas", 92, "a")],
                            [("Lucas", 88, "a")],
                            [("Lucas", 95, "a")],
                        ],
                    ):
                        rescued = rescue_placeholder_identity_names(df)

        self.assertEqual(set(rescued["FixPlayerName"]), {"Lucas"})
        self.assertTrue(all("placeholder_name_rescue" in str(value) for value in rescued["IdentityResolutionMethod"]))

    def test_refine_black_blue_character_variants_can_flip_to_blue(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "Poule_A",
                    "RaceIDNumber": 4,
                    "RacePosition": 1,
                    "Character": "Black Yoshi",
                    "CharacterIndex": 18,
                    "CharacterMatchConfidence": 70.0,
                    "CharacterMatchMethod": "alpha_aware_color_template_local_search",
                }
            ]
        )

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :] = (255, 0, 0)
        alpha = np.full((2, 2), 255, dtype=np.uint8)
        templates = [
            {
                "character_name": "Black Yoshi",
                "character_index": 18,
                "template_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "Blue Yoshi",
                "character_index": 14,
                "template_image": np.full((2, 2, 3), (255, 0, 0), dtype=np.uint8),
                "template_alpha": alpha,
            },
        ]

        with patch("mk8_local_play.extract_text.load_character_templates", return_value=templates):
            with patch("mk8_local_play.extract_text.find_score_bundle_anchor_path", return_value="dummy.png"):
                with patch("mk8_local_play.extract_text.cv2.imread", return_value=frame):
                    with patch("mk8_local_play.extract_text.character_row_roi", return_value=((0, 0), (2, 2))):
                        refined = refine_black_blue_character_variants(df, frames_folder=".")

        self.assertEqual(refined.iloc[0]["Character"], "Blue Yoshi")
        self.assertEqual(int(refined.iloc[0]["CharacterIndex"]), 14)
        self.assertIn("black_blue_chroma_refine", str(refined.iloc[0]["CharacterMatchMethod"]))
