import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

import numpy as np

from mk8_local_play import extract_initial_scan


class ExtractInitialScanTests(unittest.TestCase):
    def test_initial_scan_template_indexes_match_extract_frame_load_order(self):
        self.assertEqual(extract_initial_scan.INITIAL_SCAN_TARGETS[1]["template_index"], 1)
        self.assertEqual(extract_initial_scan.INITIAL_SCAN_TARGETS[2]["template_index"], 2)
        self.assertEqual(extract_initial_scan.IGNORE_FRAME_TARGETS[0]["template_index"], 4)
        self.assertEqual(extract_initial_scan.IGNORE_FRAME_TARGETS[1]["template_index"], 5)
        self.assertEqual(extract_initial_scan.IGNORE_FRAME_TARGETS[2]["template_index"], 6)

    def test_match_score_target_layouts_does_not_report_blank_when_any_layout_matches(self):
        gray_image = np.full((720, 1280), 255, dtype=np.uint8)
        templates = []
        layouts = [
            SimpleNamespace(layout_id="blank_layout", score_anchor_roi=(0, 0, 10, 10)),
            SimpleNamespace(layout_id="match_layout", score_anchor_roi=(20, 20, 10, 10)),
        ]
        with (
            mock.patch.object(extract_initial_scan, "all_score_layouts", return_value=layouts),
            mock.patch.object(extract_initial_scan, "process_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8)),
            mock.patch.object(
                extract_initial_scan,
                "build_position_signal_metrics",
                side_effect=[
                    [],
                    [
                        {"best_position_template": 1, "best_position_score": 0.91},
                        {"best_position_template": 2, "best_position_score": 0.88},
                        {"best_position_template": 3, "best_position_score": 0.87},
                        {"best_position_template": 4, "best_position_score": 0.90},
                        {"best_position_template": 5, "best_position_score": 0.85},
                        {"best_position_template": 6, "best_position_score": 0.82},
                    ],
                ],
            ),
        ):
            max_val, rejected_as_blank, layout_id = extract_initial_scan._match_score_target_layouts(
                gray_image,
                templates,
                defaultdict(float),
            )

        self.assertAlmostEqual(max_val, 0.8716666666666667)
        self.assertFalse(rejected_as_blank)
        self.assertEqual(layout_id, "match_layout")

    def test_match_score_target_layouts_requires_average_threshold(self):
        gray_image = np.full((720, 1280), 255, dtype=np.uint8)
        templates = []
        layouts = [
            SimpleNamespace(layout_id="layout_a", score_anchor_roi=(0, 0, 52, 610)),
        ]
        with (
            mock.patch.object(extract_initial_scan, "all_score_layouts", return_value=layouts),
            mock.patch.object(extract_initial_scan, "POSITION_SCAN_MIN_AVG_COEFF", 0.8),
            mock.patch.object(extract_initial_scan, "process_image", return_value=np.zeros((10, 10, 3), dtype=np.uint8)),
            mock.patch.object(
                extract_initial_scan,
                "build_position_signal_metrics",
                return_value=[
                    {"best_position_template": 1, "best_position_score": 0.91},
                    {"best_position_template": 2, "best_position_score": 0.88},
                    {"best_position_template": 3, "best_position_score": 0.87},
                    {"best_position_template": 4, "best_position_score": 0.90},
                    {"best_position_template": 5, "best_position_score": 0.45},
                    {"best_position_template": 6, "best_position_score": 0.44},
                ],
            ),
        ):
            max_val, rejected_as_blank, layout_id = extract_initial_scan._match_score_target_layouts(
                gray_image,
                templates,
                defaultdict(float),
            )

        self.assertAlmostEqual(max_val, 0.0)
        self.assertFalse(rejected_as_blank)
        self.assertEqual(layout_id, extract_initial_scan.DEFAULT_SCORE_LAYOUT_ID)

    def test_match_initial_scan_target_accepts_alternate_roi(self):
        gray_image = np.full((720, 1280), 255, dtype=np.uint8)
        target = {
            "kind": "race",
            "label": "RaceNumber",
            "roi": (10, 10, 20, 20),
            "template_index": 2,
            "alternate_matches": ({"roi": (40, 40, 20, 20), "template_index": 2},),
        }
        templates = [
            (np.zeros((5, 5), dtype=np.uint8), None),
            (np.zeros((5, 5), dtype=np.uint8), None),
            (np.zeros((5, 5), dtype=np.uint8), None),
        ]
        blank_roi = np.zeros((20, 20), dtype=np.uint8)
        alternate_roi = np.full((20, 20), 255, dtype=np.uint8)
        with (
            mock.patch.object(extract_initial_scan, "INITIAL_SCAN_TARGETS", ({}, {}, target)),
            mock.patch.object(extract_initial_scan, "_expanded_roi", side_effect=[(10, 10, 20, 20), (40, 40, 20, 20)]),
            mock.patch.object(extract_initial_scan, "preprocess_roi", side_effect=[blank_roi, alternate_roi]),
            mock.patch.object(extract_initial_scan, "match_template", return_value=0.77),
        ):
            max_val, rejected_as_blank, processed_roi = extract_initial_scan._match_initial_scan_target(
                gray_image,
                target,
                templates,
                defaultdict(float),
            )

        self.assertAlmostEqual(max_val, 0.77)
        self.assertFalse(rejected_as_blank)
        self.assertTrue(np.array_equal(processed_roi, alternate_roi))

    def test_match_initial_scan_target_accepts_alternate_template_variant(self):
        gray_image = np.full((720, 1280), 255, dtype=np.uint8)
        target = {
            "kind": "race",
            "label": "RaceNumber",
            "roi": (10, 10, 20, 20),
            "template_index": 2,
            "alternate_matches": ({"roi": (40, 40, 20, 20), "template_index": 7},),
        }
        templates = [(np.zeros((5, 5), dtype=np.uint8), None) for _ in range(8)]
        blank_roi = np.zeros((20, 20), dtype=np.uint8)
        alternate_roi = np.full((20, 20), 255, dtype=np.uint8)
        with (
            mock.patch.object(extract_initial_scan, "INITIAL_SCAN_TARGETS", ({}, {}, target)),
            mock.patch.object(extract_initial_scan, "_expanded_roi", side_effect=[(10, 10, 20, 20), (40, 40, 20, 20)]),
            mock.patch.object(extract_initial_scan, "preprocess_roi", side_effect=[blank_roi, alternate_roi]),
            mock.patch.object(extract_initial_scan, "match_template", return_value=0.83),
        ):
            max_val, rejected_as_blank, processed_roi = extract_initial_scan._match_initial_scan_target(
                gray_image,
                target,
                templates,
                defaultdict(float),
            )

        self.assertAlmostEqual(max_val, 0.83)
        self.assertFalse(rejected_as_blank)
        self.assertTrue(np.array_equal(processed_roi, alternate_roi))
