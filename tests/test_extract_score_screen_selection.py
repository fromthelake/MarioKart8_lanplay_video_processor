import unittest
from collections import defaultdict
from pathlib import Path
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

    def test_expand_race_score_consensus_window_skips_recapture_when_range_already_covered(self):
        result = {
            "twelfth_template_detected": True,
            "candidate": {
                "video_path": "dummy.mp4",
                "left": 0,
                "top": 0,
                "crop_width": 10,
                "crop_height": 10,
                "ocr_consensus_frames": 7,
            },
            "actual_race_score_frame": 100,
            "race_consensus_frames": [(93, "a"), (100, "b"), (106, "c")],
            "stats": defaultdict(float),
        }

        with (
            mock.patch.object(extract_score_screen_selection.cv2, "VideoCapture") as cap_mock,
            mock.patch.object(extract_score_screen_selection, "collect_frame_range_from_capture") as collect_mock,
        ):
            expanded = extract_score_screen_selection.expand_race_score_consensus_window(result, 12)

        self.assertEqual(expanded["race_consensus_frames"], [(93, "a"), (100, "b"), (106, "c")])
        cap_mock.assert_not_called()
        collect_mock.assert_not_called()

    def test_collect_frame_range_from_capture_uses_position_helper_when_stats_supplied(self):
        capture = mock.Mock()
        capture.get.return_value = 200
        stats = defaultdict(float)

        with (
            mock.patch.object(extract_score_screen_selection, "position_capture_for_read") as position_mock,
            mock.patch.object(
                extract_score_screen_selection,
                "read_video_frame",
                side_effect=[(True, np.zeros((10, 10, 3), dtype=np.uint8))] * 3,
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "crop_and_upscale_image",
                return_value=np.zeros((10, 10, 3), dtype=np.uint8),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "actual_frame_after_read",
                side_effect=[10, 11, 12],
            ),
        ):
            frames = extract_score_screen_selection.collect_frame_range_from_capture(
                capture,
                10,
                12,
                0,
                0,
                10,
                10,
                stats=stats,
            )

        self.assertEqual([frame_number for frame_number, _image in frames], [10, 11, 12])
        position_mock.assert_called_once()

    def test_record_score_capture_usage_tracks_overlap_and_same_run_unused_frames(self):
        stats = defaultdict(float)

        extract_score_screen_selection._record_score_capture_usage(
            stats,
            fps=30.0,
            actual_race_score_frame=100,
            actual_total_score_frame=200,
            actual_points_anchor_frame=300,
            race_consensus_frames=[(99, "a"), (100, "b"), (101, "c")],
            total_consensus_frames=[(199, "d"), (200, "e"), (201, "f")],
            points_context_frames=[(299, "g"), (300, "h"), (301, "i")],
        )

        self.assertEqual(int(stats["score_capture_frame_events_total"]), 12)
        self.assertEqual(int(stats["score_capture_unique_frames_total"]), 9)
        self.assertEqual(int(stats["score_capture_duplicate_frames_total"]), 3)
        self.assertEqual(int(stats["score_same_run_ocr_frames_total"]), 6)
        self.assertEqual(int(stats["score_same_run_ocr_unique_frames_total"]), 6)
        self.assertEqual(int(stats["score_persisted_ocr_frames_total"]), 8)
        self.assertEqual(int(stats["score_persisted_ocr_unique_frames_total"]), 7)
        self.assertEqual(int(stats["score_capture_frames_outside_same_run_cache_total"]), 3)
        self.assertAlmostEqual(float(stats["score_capture_duplicate_source_seconds_total"]), 0.1, places=6)
        self.assertAlmostEqual(float(stats["score_capture_outside_same_run_cache_source_seconds_total"]), 0.1, places=6)

    def test_remove_legacy_bundle_files_skips_glob_when_bundle_is_new(self):
        bundle_dir = mock.Mock(spec=Path)
        stats = defaultdict(float)

        removed = extract_score_screen_selection._remove_legacy_bundle_files(
            bundle_dir,
            ("frame_*",),
            stats=stats,
            preexisting=False,
        )

        self.assertEqual(removed, 0)
        bundle_dir.glob.assert_not_called()

    def test_remove_legacy_bundle_files_removes_matching_files_when_bundle_preexists(self):
        file_a = mock.Mock()
        file_b = mock.Mock()
        bundle_dir = mock.Mock(spec=Path)
        bundle_dir.glob.side_effect = [[file_a, file_b]]
        stats = defaultdict(float)

        removed = extract_score_screen_selection._remove_legacy_bundle_files(
            bundle_dir,
            ("frame_*",),
            stats=stats,
            preexisting=True,
        )

        self.assertEqual(removed, 2)
        self.assertEqual(file_a.unlink.call_count, 1)
        self.assertEqual(file_b.unlink.call_count, 1)
        self.assertEqual(int(stats["score_save_cleanup_removed"]), 2)
        self.assertEqual(int(stats["score_save_cleanup_runs"]), 1)

    def test_find_points_transition_frame_uses_multi_row_transition_and_centers_on_transition(self):
        capture = mock.Mock()
        stats = defaultdict(float)
        observations = [
            {
                "race_points": ["15", "12", "10", "9", "8", "7"],
                "total_points": ["303", "220", "198", "134", "248", "154"],
            },
            {
                "race_points": ["15", "12", "10", "9", "8", "7"],
                "total_points": ["303", "220", "198", "134", "249", "154"],
            },
            {
                "race_points": ["15", "11", "9", "9", "7", "6"],
                "total_points": ["309", "221", "198", "135", "249", "155"],
            },
        ]

        with (
            mock.patch.object(extract_score_screen_selection, "seek_to_frame"),
            mock.patch.object(
                extract_score_screen_selection,
                "read_video_frame",
                side_effect=[(True, np.zeros((10, 10, 3), dtype=np.uint8))] * len(observations),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "crop_and_upscale_image",
                return_value=np.zeros((10, 10, 3), dtype=np.uint8),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "extract_points_transition_observation",
                side_effect=observations,
            ),
        ):
            transition_frame, selected_points_anchor_frame = extract_score_screen_selection._find_points_transition_frame(
                capture,
                5895,
                5897,
                0,
                0,
                10,
                10,
                "lan2_split_2p",
                stats,
            )

        self.assertEqual(transition_frame, 5897)
        self.assertEqual(selected_points_anchor_frame, 5897)

    def test_find_total_score_stable_frame_uses_coarse_search_then_rewinds(self):
        capture = mock.Mock()
        stats = defaultdict(float)
        signatures = [
            None,
            (1, 2, 3),
            None,
            None,
            (4, 5, 6),
            (4, 5, 6),
            (4, 5, 6),
        ]

        with (
            mock.patch.object(extract_score_screen_selection, "fps_scaled_frames", return_value=3),
            mock.patch.object(extract_score_screen_selection, "seek_to_frame") as seek_mock,
            mock.patch.object(extract_score_screen_selection, "position_capture_for_read") as position_mock,
            mock.patch.object(
                extract_score_screen_selection,
                "read_video_frame",
                side_effect=[(True, np.zeros((10, 10, 3), dtype=np.uint8))] * len(signatures),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "crop_and_upscale_image",
                return_value=np.zeros((10, 10, 3), dtype=np.uint8),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "_extract_total_score_stable_signature",
                side_effect=signatures,
            ),
        ):
            stable_frame = extract_score_screen_selection._find_total_score_stable_frame(
                capture,
                100,
                30.0,
                0,
                0,
                10,
                10,
                "lan2_split_2p",
                stats,
            )

        self.assertEqual(stable_frame, 104)
        seek_targets = [call.args[1] for call in seek_mock.call_args_list]
        self.assertIn(100, seek_targets)
        position_targets = [call.args[1] for call in position_mock.call_args_list]
        self.assertIn(100, position_targets)
        self.assertIn(110, position_targets)

    def test_analyze_score_window_task_rejects_ignore_candidate_early(self):
        task = {
            "video_path": "dummy.mp4",
            "frame_number": 100,
            "fps": 30.0,
            "templates": [(np.zeros((5, 5), dtype=np.uint8), None)] * 9,
            "left": 0,
            "top": 0,
            "crop_width": 10,
            "crop_height": 10,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "score_layout_id": "lan2_split_2p",
            "ocr_consensus_frames": 7,
            "race_number": 1,
        }
        capture = mock.Mock()
        capture.isOpened.return_value = True

        with (
            mock.patch.object(extract_score_screen_selection.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(extract_score_screen_selection, "seek_to_frame"),
            mock.patch.object(
                extract_score_screen_selection,
                "read_video_frame",
                side_effect=[(True, np.zeros((10, 10, 3), dtype=np.uint8))],
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "crop_and_upscale_image",
                return_value=np.zeros((720, 1280, 3), dtype=np.uint8),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "_match_ignore_frame_target_detail",
                return_value={
                    "label": "IgnoreAlbumGallery",
                    "max_val": 0.95,
                    "match_threshold": 0.62,
                    "rejected_as_blank": False,
                },
            ),
            mock.patch.object(extract_score_screen_selection, "_fast_prefix_gate_score") as gate_mock,
        ):
            result = extract_score_screen_selection.analyze_score_window_task(task, lambda frame_number, fps: "00:00:00")

        self.assertEqual(result["race_score_frame"], 0)
        self.assertEqual(result["total_score_frame"], 0)
        self.assertTrue(result.get("ignored_candidate"))
        self.assertEqual(result.get("ignore_label"), "IgnoreAlbumGallery")
        gate_mock.assert_not_called()

    def test_analyze_score_window_task_precomputes_total_score_visible_players(self):
        task = {
            "video_path": "dummy.mp4",
            "frame_number": 100,
            "fps": 30.0,
            "templates": [(np.zeros((5, 5), dtype=np.uint8), None)] * 9,
            "left": 0,
            "top": 0,
            "crop_width": 10,
            "crop_height": 10,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "score_layout_id": "lan2_split_2p",
            "ocr_consensus_frames": 7,
            "race_number": 1,
        }
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 1000
        total_score_image = np.zeros((720, 1280, 3), dtype=np.uint8)

        with (
            mock.patch.object(extract_score_screen_selection.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(extract_score_screen_selection, "position_capture_for_read"),
            mock.patch.object(
                extract_score_screen_selection,
                "read_video_frame",
                return_value=(True, np.zeros((10, 10, 3), dtype=np.uint8)),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "crop_and_upscale_image",
                return_value=np.zeros((720, 1280, 3), dtype=np.uint8),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "_match_ignore_frame_target_detail",
                return_value={
                    "label": "",
                    "max_val": 0.0,
                    "match_threshold": 0.62,
                    "rejected_as_blank": True,
                },
            ),
            mock.patch.object(extract_score_screen_selection, "_fast_prefix_gate_score", return_value=True),
            mock.patch.object(
                extract_score_screen_selection,
                "_raw_fixed_grid_prefix_confirm",
                return_value=(True, 0.91),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "_find_points_transition_frame",
                return_value=(150, 148),
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "_find_total_score_stable_frame",
                return_value=200,
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "capture_export_frame",
                side_effect=[
                    (121, np.zeros((720, 1280, 3), dtype=np.uint8)),
                    (200, total_score_image),
                    (148, np.zeros((720, 1280, 3), dtype=np.uint8)),
                ],
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "collect_consensus_frames_from_capture",
                return_value=[],
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "collect_frame_range_from_capture",
                return_value=[],
            ),
            mock.patch.object(
                extract_score_screen_selection,
                "count_visible_position_rows",
                return_value=11,
            ) as count_mock,
        ):
            result = extract_score_screen_selection.analyze_score_window_task(
                task,
                lambda frame_number, fps: "00:00:00",
            )

        self.assertEqual(result["total_score_visible_players"], 11)
        count_mock.assert_called_once_with(total_score_image, "lan2_split_2p")


if __name__ == "__main__":
    unittest.main()
