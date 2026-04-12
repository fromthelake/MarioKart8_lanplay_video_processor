from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from mk8_local_play.extract_text import (
    apply_forced_ultra_low_res_override,
    apply_mii_character_fallback,
    build_character_variant_family_diagnostic_mask,
    build_grouped_race_images,
    character_variant_family_templates,
    diagnostic_character_variant_score,
    refine_character_variant_families,
    refine_black_blue_character_variants,
    resolve_character_variant_family_name,
    rescue_placeholder_identity_names,
)


class TestExtractTextRefinements(TestCase):
    def test_apply_forced_ultra_low_res_override_sets_is_low_res_for_forced_classes(self):
        df = pd.DataFrame(
            [
                {"RaceClass": "VideoA", "IsLowRes": False},
                {"RaceClass": "VideoB", "IsLowRes": False},
                {"RaceClass": "VideoC", "IsLowRes": True},
            ]
        )
        with patch("mk8_local_play.extract_text.forced_ultra_low_res_race_classes", return_value={"VideoA", "VideoC"}):
            updated = apply_forced_ultra_low_res_override(df)

        self.assertTrue(bool(updated.loc[updated["RaceClass"] == "VideoA", "IsLowRes"].iloc[0]))
        self.assertFalse(bool(updated.loc[updated["RaceClass"] == "VideoB", "IsLowRes"].iloc[0]))
        self.assertTrue(bool(updated.loc[updated["RaceClass"] == "VideoC", "IsLowRes"].iloc[0]))

    def test_apply_mii_character_fallback_rejects_unstable_closed_set_identity(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Wilco",
                    "Character": "Isabelle",
                    "CharacterIndex": 67,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                    "CharacterMatchRawBest": 42.0,
                    "CharacterMatchRawMargin": 0.4,
                    "CharacterMatchRawTop5Spread": 1.2,
                    "CharacterMatchRawTop5FamilyCount": 5,
                },
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Wilco",
                    "Character": "Champion Link",
                    "CharacterIndex": 69,
                    "CharacterMatchMethod": "open_set_mii_reject",
                    "CharacterMatchRawBest": 42.5,
                    "CharacterMatchRawMargin": 0.2,
                    "CharacterMatchRawTop5Spread": 1.0,
                    "CharacterMatchRawTop5FamilyCount": 5,
                },
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Wilco",
                    "Character": "Diddy Kong",
                    "CharacterIndex": 65,
                    "CharacterMatchMethod": "character_prior_mii_likely",
                    "CharacterMatchRawBest": 43.0,
                    "CharacterMatchRawMargin": 0.3,
                    "CharacterMatchRawTop5Spread": 1.5,
                    "CharacterMatchRawTop5FamilyCount": 4,
                },
            ]
        )

        refined = apply_mii_character_fallback(df)

        self.assertEqual(set(refined["Character"]), {"Mii"})
        self.assertTrue(
            all("mii_fallback_open_set_unstable_closed_set_identity" in str(value) for value in refined["CharacterMatchMethod"])
        )

    def test_apply_mii_character_fallback_keeps_unstable_closed_set_without_mii_signals(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Gianni",
                    "Character": "Pink Yoshi",
                    "CharacterIndex": 23,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search+variant_family_aligned_color_refine",
                    "CharacterMatchRawBest": 86.8,
                    "CharacterMatchRawMargin": 2.1,
                    "CharacterMatchRawTop5Spread": 5.1,
                    "CharacterMatchRawTop5FamilyCount": 4,
                },
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Gianni",
                    "Character": "Red Yoshi",
                    "CharacterIndex": 19,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search+variant_family_aligned_color_refine",
                    "CharacterMatchRawBest": 85.9,
                    "CharacterMatchRawMargin": 1.9,
                    "CharacterMatchRawTop5Spread": 5.0,
                    "CharacterMatchRawTop5FamilyCount": 4,
                },
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Gianni",
                    "Character": "Yellow Yoshi",
                    "CharacterIndex": 20,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search+variant_family_aligned_color_refine",
                    "CharacterMatchRawBest": 86.2,
                    "CharacterMatchRawMargin": 2.0,
                    "CharacterMatchRawTop5Spread": 5.3,
                    "CharacterMatchRawTop5FamilyCount": 4,
                },
            ]
        )

        refined = apply_mii_character_fallback(df)

        self.assertEqual(set(refined["Character"]), {"Pink Yoshi", "Red Yoshi", "Yellow Yoshi"})
        self.assertTrue(
            all("mii_fallback_open_set_unstable_closed_set_identity" not in str(value) for value in refined["CharacterMatchMethod"])
        )

    def test_apply_mii_character_fallback_keeps_stable_closed_set_identity(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoB",
                    "FixPlayerName": "BAwSer",
                    "Character": "Inkling Boy",
                    "CharacterIndex": 68,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                    "CharacterMatchRawBest": 61.8,
                    "CharacterMatchRawMargin": 5.4,
                    "CharacterMatchRawTop5Spread": 18.3,
                    "CharacterMatchRawTop5FamilyCount": 3,
                },
                {
                    "RaceClass": "VideoB",
                    "FixPlayerName": "BAwSer",
                    "Character": "Inkling Boy",
                    "CharacterIndex": 68,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                    "CharacterMatchRawBest": 60.9,
                    "CharacterMatchRawMargin": 7.0,
                    "CharacterMatchRawTop5Spread": 18.4,
                    "CharacterMatchRawTop5FamilyCount": 3,
                },
                {
                    "RaceClass": "VideoB",
                    "FixPlayerName": "BAwSer",
                    "Character": "Inkling Boy",
                    "CharacterIndex": 68,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                    "CharacterMatchRawBest": 62.1,
                    "CharacterMatchRawMargin": 6.1,
                    "CharacterMatchRawTop5Spread": 19.5,
                    "CharacterMatchRawTop5FamilyCount": 3,
                },
            ]
        )

        refined = apply_mii_character_fallback(df)

        self.assertEqual(list(refined["Character"]), ["Inkling Boy", "Inkling Boy", "Inkling Boy"])

    def test_apply_mii_character_fallback_rejects_too_few_closed_set_wins(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Wilco",
                    "Character": "Mii",
                    "CharacterIndex": 80,
                    "CharacterMatchMethod": "open_set_mii_reject",
                    "CharacterMatchRawBest": np.nan,
                    "CharacterMatchRawMargin": np.nan,
                    "CharacterMatchRawTop5Spread": np.nan,
                    "CharacterMatchRawTop5FamilyCount": np.nan,
                },
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Wilco",
                    "Character": "Metal Mario",
                    "CharacterIndex": 45,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                    "CharacterMatchRawBest": 51.5,
                    "CharacterMatchRawMargin": 4.0,
                    "CharacterMatchRawTop5Spread": 9.0,
                    "CharacterMatchRawTop5FamilyCount": 3,
                },
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Wilco",
                    "Character": "Metal Mario",
                    "CharacterIndex": 45,
                    "CharacterMatchMethod": "aligned_alpha_cutout_template_local_search",
                    "CharacterMatchRawBest": 50.7,
                    "CharacterMatchRawMargin": 3.5,
                    "CharacterMatchRawTop5Spread": 8.5,
                    "CharacterMatchRawTop5FamilyCount": 3,
                },
                {
                    "RaceClass": "VideoA",
                    "FixPlayerName": "Wilco",
                    "Character": "Mii",
                    "CharacterIndex": 80,
                    "CharacterMatchMethod": "open_set_mii_reject",
                    "CharacterMatchRawBest": np.nan,
                    "CharacterMatchRawMargin": np.nan,
                    "CharacterMatchRawTop5Spread": np.nan,
                    "CharacterMatchRawTop5FamilyCount": np.nan,
                },
            ]
        )

        refined = apply_mii_character_fallback(df)

        self.assertEqual(set(refined["Character"]), {"Mii"})

    def test_resolve_character_variant_family_name_includes_default_roster_members(self):
        self.assertEqual(resolve_character_variant_family_name("Shy Guy"), "Shy Guy")
        self.assertEqual(resolve_character_variant_family_name("Pink Shy Guy"), "Shy Guy")
        self.assertEqual(resolve_character_variant_family_name("Yoshi"), "Yoshi")
        self.assertEqual(resolve_character_variant_family_name("Birdo"), "Birdo")
        self.assertEqual(resolve_character_variant_family_name("Cat Peach"), "Peach")
        self.assertEqual(resolve_character_variant_family_name("Peachette"), "Peach")
        self.assertEqual(resolve_character_variant_family_name("Waluigi"), "")

    def test_character_variant_family_templates_include_default_and_colored_variants(self):
        templates = [
            {"character_name": "Shy Guy", "character_index": 27},
            {"character_name": "Pink Shy Guy", "character_index": 34},
            {"character_name": "Orange Shy Guy", "character_index": 35},
            {"character_name": "Yoshi", "character_index": 16},
            {"character_name": "Waluigi", "character_index": 26},
        ]

        family_templates = character_variant_family_templates(templates, "Pink Shy Guy")

        self.assertEqual(
            [template["character_name"] for template in family_templates],
            ["Shy Guy", "Pink Shy Guy", "Orange Shy Guy"],
        )

    def test_character_variant_family_templates_include_explicit_peach_family(self):
        templates = [
            {"character_name": "Peach", "character_index": 2},
            {"character_name": "Cat Peach", "character_index": 6},
            {"character_name": "Baby Peach", "character_index": 42},
            {"character_name": "Peachette", "character_index": 64},
            {"character_name": "Waluigi", "character_index": 26},
        ]

        family_templates = character_variant_family_templates(templates, "Cat Peach")

        self.assertEqual(
            [template["character_name"] for template in family_templates],
            ["Peach", "Cat Peach", "Baby Peach", "Peachette"],
        )

    def test_diagnostic_character_variant_score_prefers_matching_family_template(self):
        alpha = np.full((2, 2), 255, dtype=np.uint8)
        family_templates = [
            {
                "character_name": "Shy Guy",
                "character_index": 27,
                "template_image": np.full((2, 2, 3), (0, 0, 255), dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "Green Shy Guy",
                "character_index": 30,
                "template_image": np.full((2, 2, 3), (0, 255, 0), dtype=np.uint8),
                "template_alpha": alpha,
            },
        ]

        diagnostic_mask = build_character_variant_family_diagnostic_mask(family_templates)
        source = np.full((2, 2, 3), (0, 255, 0), dtype=np.uint8)
        shy_score = diagnostic_character_variant_score(source, family_templates[0]["template_image"], diagnostic_mask)
        green_score = diagnostic_character_variant_score(source, family_templates[1]["template_image"], diagnostic_mask)

        self.assertGreater(green_score, shy_score)

    def test_peach_family_diagnostic_mask_keeps_variant_specific_alpha_regions(self):
        alpha_full = np.full((2, 2), 255, dtype=np.uint8)
        alpha_cat = np.array([[255, 255], [0, 0]], dtype=np.uint8)
        family_templates = [
            {
                "character_name": "Peach",
                "character_index": 2,
                "template_image": np.full((2, 2, 3), (0, 200, 255), dtype=np.uint8),
                "template_alpha": alpha_full,
            },
            {
                "character_name": "Cat Peach",
                "character_index": 6,
                "template_image": np.full((2, 2, 3), (0, 150, 255), dtype=np.uint8),
                "template_alpha": alpha_cat,
            },
            {
                "character_name": "Baby Peach",
                "character_index": 42,
                "template_image": np.full((2, 2, 3), (0, 180, 255), dtype=np.uint8),
                "template_alpha": alpha_full,
            },
        ]

        diagnostic_mask = build_character_variant_family_diagnostic_mask(family_templates, family_name="Peach")

        self.assertTrue(bool(diagnostic_mask[1, 0]))

    def test_refine_character_variant_families_promotes_stable_diagnostic_winner(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoA",
                    "RaceIDNumber": 1,
                    "RacePosition": 1,
                    "FixPlayerName": "Shin",
                    "Character": "White Shy Guy",
                    "CharacterIndex": 32,
                    "CharacterMatchConfidence": 70.0,
                    "CharacterMatchMethod": "alpha_aware_color_template_local_search",
                },
                {
                    "RaceClass": "VideoA",
                    "RaceIDNumber": 2,
                    "RacePosition": 1,
                    "FixPlayerName": "Shin",
                    "Character": "Orange Shy Guy",
                    "CharacterIndex": 35,
                    "CharacterMatchConfidence": 72.0,
                    "CharacterMatchMethod": "alpha_aware_color_template_local_search",
                },
                {
                    "RaceClass": "VideoA",
                    "RaceIDNumber": 3,
                    "RacePosition": 1,
                    "FixPlayerName": "Shin",
                    "Character": "Yellow Shy Guy",
                    "CharacterIndex": 31,
                    "CharacterMatchConfidence": 71.0,
                    "CharacterMatchMethod": "alpha_aware_color_template_local_search",
                },
            ]
        )

        frame = np.full((4, 4, 3), (0, 255, 0), dtype=np.uint8)
        alpha = np.full((2, 2), 255, dtype=np.uint8)
        templates = [
            {
                "character_name": "Shy Guy",
                "character_index": 27,
                "template_image": np.full((2, 2, 3), (0, 0, 255), dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "Green Shy Guy",
                "character_index": 30,
                "template_image": np.full((2, 2, 3), (0, 255, 0), dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "White Shy Guy",
                "character_index": 32,
                "template_image": np.full((2, 2, 3), 255, dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "Orange Shy Guy",
                "character_index": 35,
                "template_image": np.full((2, 2, 3), (0, 128, 255), dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "Yellow Shy Guy",
                "character_index": 31,
                "template_image": np.full((2, 2, 3), (0, 255, 255), dtype=np.uint8),
                "template_alpha": alpha,
            },
        ]

        with patch("mk8_local_play.extract_text.load_character_templates", return_value=templates):
            with patch("mk8_local_play.extract_text.find_score_bundle_anchor_path", return_value="dummy.png"):
                with patch("mk8_local_play.extract_text.cv2.imread", return_value=frame):
                    with patch("mk8_local_play.extract_text.character_row_roi", return_value=((0, 0), (2, 2))):
                        refined = refine_character_variant_families(df, frames_folder=".")

        self.assertEqual(set(refined["Character"]), {"Green Shy Guy"})
        self.assertTrue(
            all("variant_family_aligned_color_refine" in str(value) for value in refined["CharacterMatchMethod"])
        )

    def test_refine_character_variant_families_can_stabilize_cat_peach_family(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoPeach",
                    "RaceIDNumber": 1,
                    "RacePosition": 1,
                    "FixPlayerName": "Amber",
                    "Character": "Baby Peach",
                    "CharacterIndex": 42,
                    "CharacterMatchConfidence": 71.0,
                    "CharacterMatchMethod": "alpha_aware_color_template_local_search",
                },
                {
                    "RaceClass": "VideoPeach",
                    "RaceIDNumber": 2,
                    "RacePosition": 1,
                    "FixPlayerName": "Amber",
                    "Character": "Peachette",
                    "CharacterIndex": 64,
                    "CharacterMatchConfidence": 72.0,
                    "CharacterMatchMethod": "alpha_aware_color_template_local_search",
                },
                {
                    "RaceClass": "VideoPeach",
                    "RaceIDNumber": 3,
                    "RacePosition": 1,
                    "FixPlayerName": "Amber",
                    "Character": "Cat Peach",
                    "CharacterIndex": 6,
                    "CharacterMatchConfidence": 73.0,
                    "CharacterMatchMethod": "alpha_aware_color_template_local_search",
                },
            ]
        )

        frame = np.full((4, 4, 3), (30, 140, 255), dtype=np.uint8)
        alpha = np.full((2, 2), 255, dtype=np.uint8)
        templates = [
            {
                "character_name": "Peach",
                "character_index": 2,
                "template_image": np.full((2, 2, 3), (180, 180, 255), dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "Cat Peach",
                "character_index": 6,
                "template_image": np.full((2, 2, 3), (30, 140, 255), dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "Baby Peach",
                "character_index": 42,
                "template_image": np.full((2, 2, 3), (170, 170, 255), dtype=np.uint8),
                "template_alpha": alpha,
            },
            {
                "character_name": "Peachette",
                "character_index": 64,
                "template_image": np.full((2, 2, 3), (200, 160, 255), dtype=np.uint8),
                "template_alpha": alpha,
            },
        ]

        with patch("mk8_local_play.extract_text.load_character_templates", return_value=templates):
            with patch("mk8_local_play.extract_text.find_score_bundle_anchor_path", return_value="dummy.png"):
                with patch("mk8_local_play.extract_text.cv2.imread", return_value=frame):
                    with patch("mk8_local_play.extract_text.character_row_roi", return_value=((0, 0), (2, 2))):
                        refined = refine_character_variant_families(df, frames_folder=".")

        self.assertEqual(set(refined["Character"]), {"Cat Peach"})
        self.assertTrue(
            all("variant_family_aligned_color_refine" in str(value) for value in refined["CharacterMatchMethod"])
        )

    def test_build_grouped_race_images_skips_incomplete_races_without_race_score(self):
        with patch("mk8_local_play.extract_text.iter_video_race_dirs", return_value=[("VideoA", 1, "Race_001"), ("VideoA", 2, "Race_002")]):
            with patch("mk8_local_play.extract_text.find_anchor_frame_path", side_effect=[None, None]):
                with patch(
                    "mk8_local_play.extract_text.find_score_bundle_anchor_path",
                    side_effect=[
                        "race1_score.jpg",
                        "race1_total.jpg",
                        None,
                        None,
                    ],
                ):
                    grouped = build_grouped_race_images("dummy")

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped[0][0], ("VideoA", 1))

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

    def test_rescue_placeholder_identity_names_can_force_choice_with_two_strong_hits(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoB",
                    "RaceIDNumber": 1,
                    "RacePosition": 2,
                    "FixPlayerName": "PlayerNameMissing_2",
                    "IdentityLabel": "PlayerNameMissing_2",
                    "IdentityResolutionMethod": "new_identity",
                    "ReviewReason": "",
                    "ScoreLayoutId": "lan2_split_2p",
                },
                {
                    "RaceClass": "VideoB",
                    "RaceIDNumber": 2,
                    "RacePosition": 2,
                    "FixPlayerName": "PlayerNameMissing_2",
                    "IdentityLabel": "PlayerNameMissing_2",
                    "IdentityResolutionMethod": "name+visual",
                    "ReviewReason": "",
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
                            [("Willemijn", 95, "a")],
                            [("Willemijn", 96, "a")],
                        ],
                    ):
                        rescued = rescue_placeholder_identity_names(df)

        self.assertEqual(set(rescued["FixPlayerName"]), {"Willemijn"})
        self.assertTrue(all("placeholder_name_forced_choice" in str(value) for value in rescued["IdentityResolutionMethod"]))
        self.assertTrue(all("forced_choice=1" in str(value) for value in rescued["ReviewReason"]))

    def test_rescue_placeholder_identity_names_can_force_choice_from_single_very_strong_hit(self):
        df = pd.DataFrame(
            [
                {
                    "RaceClass": "VideoC",
                    "RaceIDNumber": 1,
                    "RacePosition": 2,
                    "FixPlayerName": "PlayerNameMissing_2",
                    "IdentityLabel": "PlayerNameMissing_2",
                    "IdentityResolutionMethod": "new_identity",
                    "ReviewReason": "",
                    "ScoreLayoutId": "lan2_split_2p",
                },
                {
                    "RaceClass": "VideoC",
                    "RaceIDNumber": 2,
                    "RacePosition": 2,
                    "FixPlayerName": "PlayerNameMissing_2",
                    "IdentityLabel": "PlayerNameMissing_2",
                    "IdentityResolutionMethod": "name+visual",
                    "ReviewReason": "",
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
                            [("Christiaan", 98, "a")],
                            [("noise", 10, "a")],
                        ],
                    ):
                        rescued = rescue_placeholder_identity_names(df)

        self.assertEqual(set(rescued["FixPlayerName"]), {"Christiaan"})
        self.assertTrue(all("placeholder_name_forced_choice" in str(value) for value in rescued["IdentityResolutionMethod"]))
        self.assertTrue(all("forced_choice=1" in str(value) for value in rescued["ReviewReason"]))

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
