import unittest

import numpy as np

from tools.evaluate_character_roster_templates import (
    TemplateRecord,
    _rank_crop_against_templates,
    aligned_cutout_blend_score,
    aggregate_feature_matrices,
    build_stage_sequence,
    cutout_edge_agreement_score,
    compute_feature_matrices,
    generate_weight_grid,
    masked_cutout_color_score,
    stage_metrics,
)


def _solid_template(character_index: int, roster_index: int, name: str, color: tuple[int, int, int], *, full_mask: bool = True) -> TemplateRecord:
    image = np.full((48, 48, 3), color, dtype=np.uint8)
    alpha = np.full((48, 48), 255 if full_mask else 0, dtype=np.uint8)
    if not full_mask:
        alpha[8:40, 8:40] = 255
    return TemplateRecord(
        character_index=character_index,
        roster_index=roster_index,
        character_name=name,
        template_image=image,
        template_alpha=alpha,
    )


class EvaluateCharacterRosterTemplatesTests(unittest.TestCase):
    def test_cutout_scores_ignore_outer_background_pixels(self):
        template = np.full((48, 48, 3), (0, 0, 255), dtype=np.uint8)
        alpha = np.zeros((48, 48), dtype=np.uint8)
        alpha[8:40, 8:40] = 255

        source = np.full((48, 48, 3), (0, 255, 0), dtype=np.uint8)
        source[8:40, 8:40] = (0, 0, 255)
        source_alpha = np.full((48, 48), 255, dtype=np.uint8)

        cutout_score = masked_cutout_color_score(source, source_alpha, template, alpha)

        self.assertGreater(cutout_score, 0.99)

    def test_cutout_edge_score_tracks_inner_shape(self):
        template = np.zeros((48, 48, 3), dtype=np.uint8)
        template[10:38, 20:28] = (255, 255, 255)
        alpha = np.zeros((48, 48), dtype=np.uint8)
        alpha[8:40, 8:40] = 255

        source = np.zeros((48, 48, 3), dtype=np.uint8)
        source[10:38, 20:28] = (255, 255, 255)
        source_alpha = np.full((48, 48), 255, dtype=np.uint8)

        edge_score = cutout_edge_agreement_score(source, source_alpha, template, alpha)

        self.assertGreater(edge_score, 0.75)

    def test_generate_weight_grid_sums_to_one(self):
        grid = generate_weight_grid(3, step=0.5)
        self.assertIn((1.0, 0.0, 0.0), grid)
        self.assertIn((0.5, 0.5, 0.0), grid)
        self.assertTrue(all(abs(sum(item) - 1.0) < 1e-9 for item in grid))

    def test_compute_feature_matrices_preserves_self_match_identity(self):
        templates = [
            _solid_template(0, 0, "Red", (0, 0, 255)),
            _solid_template(1, 1, "Green", (0, 255, 0)),
            _solid_template(2, 2, "Blue", (255, 0, 0), full_mask=False),
        ]

        features = compute_feature_matrices(templates)
        stage = build_stage_sequence(features, weight_step=0.5)[-1]
        scores = aggregate_feature_matrices(features, stage.feature_names, stage.weights)
        correct_count, margins = stage_metrics(scores)

        self.assertEqual(correct_count, 3)
        self.assertTrue(np.all(margins > 0.0))

    def test_build_stage_sequence_improves_or_preserves_correct_self_matches(self):
        templates = [
            _solid_template(0, 0, "Red", (0, 0, 255)),
            _solid_template(1, 1, "DarkRed", (0, 0, 200)),
            _solid_template(2, 2, "Green", (0, 255, 0), full_mask=False),
        ]

        features = compute_feature_matrices(templates)
        stages = build_stage_sequence(features, weight_step=0.5)

        self.assertGreaterEqual(stages[-1].correct_count, stages[0].correct_count)
        self.assertEqual(stages[-1].total_count, 3)

    def test_rank_crop_assigns_runtime_and_cutout_ranks(self):
        templates = [
            _solid_template(0, 0, "Match", (0, 0, 255), full_mask=False),
            _solid_template(1, 1, "BackgroundTrap", (0, 255, 0), full_mask=True),
        ]
        crop = np.full((48, 48, 3), (0, 255, 0), dtype=np.uint8)
        crop[8:40, 8:40] = (0, 0, 255)

        ranked = _rank_crop_against_templates(crop, templates)
        runtime_ranks = [item["RuntimeRank"] for item in ranked]
        cutout_sorted = sorted(ranked, key=lambda item: item["CutoutRank"])

        self.assertEqual(runtime_ranks, [1, 2])
        self.assertEqual([item["CutoutRank"] for item in cutout_sorted], [1, 2])
        self.assertGreaterEqual(cutout_sorted[0]["CutoutBlendScore"], cutout_sorted[1]["CutoutBlendScore"])
        self.assertEqual([item["AlignedCutoutRank"] for item in sorted(ranked, key=lambda item: item["AlignedCutoutRank"])], [1, 2])

    def test_aligned_cutout_blend_recovers_small_translation(self):
        template = np.zeros((48, 48, 3), dtype=np.uint8)
        template[12:36, 18:30] = (0, 0, 255)
        template[18:30, 22:26] = (255, 255, 255)
        alpha = np.zeros((48, 48), dtype=np.uint8)
        alpha[12:36, 18:30] = 255
        record = TemplateRecord(0, 0, "Match", template, alpha)

        source = np.zeros((48, 48, 3), dtype=np.uint8)
        source[14:38, 20:32] = (0, 0, 255)
        source[20:32, 24:28] = (255, 255, 255)
        source_alpha = np.full((48, 48), 255, dtype=np.uint8)

        unaligned = _rank_crop_against_templates(source, [record])[0]
        aligned_blend, _, aligned_edge, dx, dy = aligned_cutout_blend_score(source, source_alpha, template, alpha, max_offset=4)

        self.assertGreater(aligned_blend, unaligned["CutoutBlendScore"])
        self.assertGreater(aligned_edge, unaligned["CutoutEdgeScore"])
        self.assertIn(dx, (2, 3))
        self.assertIn(dy, (2, 3))


if __name__ == "__main__":
    unittest.main()
