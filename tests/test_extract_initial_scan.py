import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

import numpy as np

from mk8_local_play import extract_initial_scan


class ExtractInitialScanTests(unittest.TestCase):
    def test_match_score_target_layouts_does_not_report_blank_when_any_layout_matches(self):
        gray_image = np.full((720, 1280), 255, dtype=np.uint8)
        templates = [(np.ones((10, 10), dtype=np.uint8) * 255, None)]
        layouts = [
            SimpleNamespace(layout_id="blank_layout", score_anchor_roi=(0, 0, 10, 10)),
            SimpleNamespace(layout_id="match_layout", score_anchor_roi=(20, 20, 10, 10)),
        ]
        preprocess_outputs = [
            np.zeros((10, 10), dtype=np.uint8),
            np.ones((10, 10), dtype=np.uint8) * 255,
        ]
        with (
            mock.patch.object(extract_initial_scan, "all_score_layouts", return_value=layouts),
            mock.patch.object(extract_initial_scan, "preprocess_roi", side_effect=preprocess_outputs),
            mock.patch.object(extract_initial_scan, "match_template", return_value=0.42),
        ):
            max_val, rejected_as_blank, layout_id = extract_initial_scan._match_score_target_layouts(
                gray_image,
                templates,
                defaultdict(float),
            )

        self.assertAlmostEqual(max_val, 0.42)
        self.assertFalse(rejected_as_blank)
        self.assertEqual(layout_id, "match_layout")
