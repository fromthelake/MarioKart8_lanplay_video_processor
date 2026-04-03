import shutil
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock
import shutil

from mk8_local_play.extract_common import (
    DEBUG_ROOT,
    EXPORT_IMAGE_FORMAT,
    choose_preferred_exported_image,
    debug_identity_workbook_path,
    debug_low_res_assignment_path,
    debug_low_res_resolution_path,
    debug_score_frame_path,
    normalize_export_image_format,
)
from mk8_local_play import main


TEST_TMP_ROOT = Path.cwd() / ".codex_tmp" / "unit_tests"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    return path


def _make_case_dir(case_name: str) -> Path:
    case_dir = TEST_TMP_ROOT / f"{case_name}_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


class MainSelectionHelpersTests(unittest.TestCase):
    def test_format_input_summary_lines_returns_compact_selection_block(self):
        lines = main._format_input_summary_lines(
            [
                ("Video_A", "Group 1/Video_A.mp4 (01:55:15)"),
                ("Video_B", "Group 1/Video_B.mp4 (01:54:00)"),
            ],
            2 * 3600,
        )

        self.assertEqual(lines[:2], ["Videos selected: 2", "Total source length: 02:00:00"])
        self.assertIn("Selection", lines)
        self.assertIn("01. Group 1/Video_A.mp4 (01:55:15)", lines)

    def test_summarize_pipeline_bottleneck_prefers_extract_when_it_dominates(self):
        self.assertEqual(
            main._summarize_pipeline_bottleneck(
                extract_duration_s=500.0,
                ocr_duration_s=120.0,
                total_processing_seconds=600.0,
            ),
            "Video loading and frame extraction",
        )

    def test_display_video_name_for_table_removes_extension_but_keeps_subfolders(self):
        self.assertEqual(
            main._display_video_name_for_table("2026-03-28/Kampioen_2026-03-27 21-50-56.mp4"),
            "2026-03-28/Kampioen_2026-03-27 21-50-56",
        )

    def test_overlap_scopes_use_equal_width_for_aligned_columns(self):
        self.assertEqual(len(main.OVERLAP_SCOPE), len(main.OVERLAP_SCOPE_OCR))
        self.assertTrue(main.OVERLAP_SCOPE.startswith("[Run - Overlap]"))

    def test_short_overlap_video_label_drops_subfolder_prefix(self):
        self.assertEqual(
            main._short_overlap_video_label("2026-03-28__Wild_2026-03-27_21-50-56"),
            "Wild_2026-03-27_21-50-56",
        )

    def test_format_overlap_race_event_uses_single_video_label_and_compact_queue_status(self):
        with mock.patch.object(main.LOGGER, "video_value", side_effect=lambda value, _identity: str(value)):
            formatted = main._format_overlap_race_event(
                "2026-03-28__Wild_2026-03-27_21-50-56",
                "queued",
                7,
                queued_for_video=1,
                queued_total=2,
            )

        self.assertIn("Wild_2026-03-27_21-50-56", formatted)
        self.assertIn("OCR queued", formatted)
        self.assertIn("R007", formatted)
        self.assertIn("Q  1 | GQ  2", formatted)
        self.assertEqual(formatted.count("Wild_2026-03-27_21-50-56"), 1)

    def test_format_overlap_ocr_detail_uses_compact_progress_layout(self):
        detail = main._format_overlap_ocr_detail(
            {
                "event": "progress",
                "completed": 5,
                "total": 7,
                "elapsed_s": 18.0,
                "race_id": 6,
                "track_name": "Wii Mushroom Gorge",
                "race_score_players": 11,
                "total_score_players": 11,
            },
            lambda seconds: "00:00:18",
        )

        self.assertEqual(
            detail,
            "OCR 05/07 ( 71%) | R006 | Wii Mushroom Gorge           | P 11/11 | 00:00:18",
        )

    def test_parse_args_accepts_multiple_video_paths(self):
        with mock.patch("sys.argv", ["mk8", "--all", "--subfolders", "--videos", "a.mp4", "b.mp4"]):
            args = main.parse_args()
        self.assertEqual(args.videos, ["a.mp4", "b.mp4"])
        self.assertTrue(args.all)
        self.assertTrue(args.subfolders)

    def test_normalize_export_image_format_defaults_to_png_and_accepts_jpeg_aliases(self):
        self.assertEqual(normalize_export_image_format(None), "png")
        self.assertEqual(normalize_export_image_format("png"), "png")
        self.assertEqual(normalize_export_image_format("jpg"), "jpg")
        self.assertEqual(normalize_export_image_format("jpeg"), "jpg")
        self.assertEqual(normalize_export_image_format("weird"), "png")

    def test_choose_preferred_exported_image_prefers_configured_default_mode(self):
        candidates = [
            Path("Demo_CaptureCard_Race+Race_001+2RaceScore.jpg"),
            Path("Demo_CaptureCard_Race+Race_001+2RaceScore.png"),
        ]
        expected_extension = ".jpg" if EXPORT_IMAGE_FORMAT == "jpg" else ".png"

        self.assertEqual(
            choose_preferred_exported_image(candidates),
            Path(f"Demo_CaptureCard_Race+Race_001+2RaceScore{expected_extension}"),
        )

    def test_build_video_identity_uses_stem_without_subfolders(self):
        case_dir = _make_case_dir("build_identity_root")
        try:
            input_dir = case_dir / "Input_Videos"
            video_path = _touch(input_dir / "Demo CaptureCard Race.mp4")
            with mock.patch.object(main, "INPUT_DIR", input_dir):
                self.assertEqual(
                    main.build_video_identity(video_path, include_subfolders=False),
                    "Demo CaptureCard Race",
                )
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_build_video_identity_sanitizes_subfolder_path(self):
        case_dir = _make_case_dir("build_identity_subfolders")
        try:
            input_dir = case_dir / "Input_Videos"
            video_path = _touch(input_dir / "Division A" / "Round 1" / "Demo CaptureCard Race.mp4")
            with mock.patch.object(main, "INPUT_DIR", input_dir):
                self.assertEqual(
                    main.build_video_identity(video_path, include_subfolders=True),
                    "Division_A__Round_1__Demo_CaptureCard_Race",
                )
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_debug_paths_follow_video_and_race_structure(self):
        expected_extension = ".jpg" if EXPORT_IMAGE_FORMAT == "jpg" else ".png"

        self.assertEqual(
            debug_score_frame_path("Demo_CaptureCard_Race", 7, "2RaceScore"),
            DEBUG_ROOT / "Score_Frames" / "Demo_CaptureCard_Race" / "Race_007" / "2RaceScore" / f"annotated_2RaceScore{expected_extension}",
        )
        self.assertEqual(
            debug_identity_workbook_path("Demo_CaptureCard_Race"),
            DEBUG_ROOT / "Identity_Linking" / "Demo_CaptureCard_Race" / "identity_linking.xlsx",
        )
        self.assertEqual(
            debug_low_res_assignment_path("Demo_CaptureCard_Race"),
            DEBUG_ROOT / "Low_Res" / "Demo_CaptureCard_Race" / "identity_assignment.csv",
        )
        self.assertEqual(
            debug_low_res_resolution_path("Demo_CaptureCard_Race"),
            DEBUG_ROOT / "Low_Res" / "Demo_CaptureCard_Race" / "identity_resolution.csv",
        )

    def test_selected_input_video_files_prefers_exact_filename_match(self):
        case_dir = _make_case_dir("select_exact")
        try:
            input_dir = case_dir / "Input_Videos"
            expected = _touch(input_dir / "Demo_CaptureCard_Race.mp4")
            _touch(input_dir / "Demo_CaptureCard_Race.webm")
            with mock.patch.object(main, "INPUT_DIR", input_dir):
                self.assertEqual(
                    main.selected_input_video_files("Demo_CaptureCard_Race.mp4"),
                    [expected],
                )
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_selected_input_video_files_prefers_exact_filename_before_relative_subfolder_match(self):
        case_dir = _make_case_dir("select_relative")
        try:
            input_dir = case_dir / "Input_Videos"
            expected_a = _touch(input_dir / "League 1" / "Demo_CaptureCard_Race.mp4")
            expected_b = _touch(input_dir / "League 2" / "Demo_CaptureCard_Race.mp4")
            with mock.patch.object(main, "INPUT_DIR", input_dir):
                self.assertEqual(
                    main.selected_input_video_files(
                        "League 1/Demo_CaptureCard_Race.mp4",
                        include_subfolders=True,
                    ),
                    [expected_a],
                )
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_discover_input_video_files_skips_corrupt_and_exclude_subfolders(self):
        case_dir = _make_case_dir("discover_skip_archives")
        try:
            input_dir = case_dir / "Input_Videos"
            expected = _touch(input_dir / "League 1" / "Demo_CaptureCard_Race.mp4")
            _touch(input_dir / "corrupt" / "RecoveredRace.mp4")
            _touch(input_dir / "League 1" / "exclude" / "SkippedRace.mp4")
            _touch(input_dir / "League 1" / "Archive" / "exclude" / "SkippedRaceToo.mp4")
            with mock.patch.object(main, "INPUT_DIR", input_dir):
                self.assertEqual(
                    main.discover_input_video_files(include_subfolders=True),
                    [expected],
                )
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_selected_race_classes_for_videos_follow_video_identity(self):
        case_dir = _make_case_dir("race_classes")
        try:
            input_dir = case_dir / "Input_Videos"
            video_paths = [
                _touch(input_dir / "Division A" / "Demo One.mp4"),
                _touch(input_dir / "Division B" / "Demo Two.mp4"),
            ]
            with mock.patch.object(main, "INPUT_DIR", input_dir):
                self.assertEqual(
                    main.selected_race_classes_for_videos(video_paths, include_subfolders=True),
                    ["Division_A__Demo_One", "Division_B__Demo_Two"],
                )
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_selected_input_video_files_accepts_multiple_explicit_relative_paths(self):
        case_dir = _make_case_dir("select_multiple_relative")
        try:
            input_dir = case_dir / "Input_Videos"
            expected_a = _touch(input_dir / "2026-03-28" / "Kampioen_2026-03-27 21-50-56.mp4")
            expected_b = _touch(input_dir / "2026-03-28" / "Talent_2026-03-27 21-50-56.mp4")
            with mock.patch.object(main, "INPUT_DIR", input_dir):
                self.assertEqual(
                    main.selected_input_video_files(
                        [
                            "2026-03-28/Kampioen_2026-03-27 21-50-56.mp4",
                            "2026-03-28/Talent_2026-03-27 21-50-56.mp4",
                        ],
                        include_subfolders=True,
                    ),
                    [expected_a, expected_b],
                )
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_selected_input_video_files_deduplicates_multi_match_results(self):
        case_dir = _make_case_dir("select_multiple_dedupe")
        try:
            input_dir = case_dir / "Input_Videos"
            expected = _touch(input_dir / "2026-03-28" / "Kampioen_2026-03-27 21-50-56.mp4")
            with mock.patch.object(main, "INPUT_DIR", input_dir):
                self.assertEqual(
                    main.selected_input_video_files(
                        [
                            "2026-03-28/Kampioen_2026-03-27 21-50-56.mp4",
                            "Kampioen_2026-03-27 21-50-56.mp4",
                        ],
                        include_subfolders=True,
                    ),
                    [expected],
                )
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_clear_output_results_preserves_gitkeep_files(self):
        case_dir = _make_case_dir("clear_output_results_gitkeep")
        try:
            output_dir = case_dir / "Output_Results"
            frames_dir = output_dir / "Frames"
            debug_dir = output_dir / "Debug"
            score_frames_dir = debug_dir / "Score_Frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            score_frames_dir.mkdir(parents=True, exist_ok=True)

            (frames_dir / ".gitkeep").write_text("")
            (debug_dir / ".gitkeep").write_text("")
            (frames_dir / "generated.txt").write_text("x")
            (score_frames_dir / "annotated.png").write_text("x")
            (output_dir / "result.csv").write_text("x")

            with (
                mock.patch.object(main, "OUTPUT_DIR", output_dir),
                mock.patch.object(main, "FRAMES_DIR", frames_dir),
                mock.patch.object(main, "DEBUG_DIR", debug_dir),
                mock.patch.object(main, "DEBUG_SCORE_FRAMES_DIR", score_frames_dir),
            ):
                self.assertTrue(main.clear_output_results(require_confirmation=False))

            self.assertTrue((frames_dir / ".gitkeep").exists())
            self.assertTrue((debug_dir / ".gitkeep").exists())
            self.assertFalse((frames_dir / "generated.txt").exists())
            self.assertFalse((score_frames_dir / "annotated.png").exists())
            self.assertFalse((output_dir / "result.csv").exists())
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_clear_output_results_for_videos_only_removes_selected_video_artifacts(self):
        case_dir = _make_case_dir("clear_output_results_for_videos")
        try:
            input_dir = case_dir / "Input_Videos"
            output_dir = case_dir / "Output_Results"
            frames_dir = output_dir / "Frames"
            debug_dir = output_dir / "Debug"
            score_frames_dir = debug_dir / "Score_Frames"
            selected_video = _touch(input_dir / "Demo_CaptureCard_Race.mp4")
            other_video = _touch(input_dir / "OtherRace.mp4")

            selected_label = "Demo_CaptureCard_Race"
            other_label = "OtherRace"
            (frames_dir / selected_label / "Race_001" / "0TrackName.jpg").parent.mkdir(parents=True, exist_ok=True)
            (frames_dir / selected_label / "Race_001" / "0TrackName.jpg").write_text("x")
            (frames_dir / other_label / "Race_001" / "0TrackName.jpg").parent.mkdir(parents=True, exist_ok=True)
            (frames_dir / other_label / "Race_001" / "0TrackName.jpg").write_text("x")
            (score_frames_dir / selected_label / "Race_001" / "annotated_2RaceScore.jpg").parent.mkdir(parents=True, exist_ok=True)
            (score_frames_dir / selected_label / "Race_001" / "annotated_2RaceScore.jpg").write_text("x")
            (debug_dir / "Identity_Linking" / selected_label / "identity_linking.xlsx").parent.mkdir(parents=True, exist_ok=True)
            (debug_dir / "Identity_Linking" / selected_label / "identity_linking.xlsx").write_text("x")
            (debug_dir / "Low_Res" / selected_label / "identity_assignment.csv").parent.mkdir(parents=True, exist_ok=True)
            (debug_dir / "Low_Res" / selected_label / "identity_assignment.csv").write_text("x")

            with (
                mock.patch.object(main, "INPUT_DIR", input_dir),
                mock.patch.object(main, "OUTPUT_DIR", output_dir),
                mock.patch.object(main, "FRAMES_DIR", frames_dir),
                mock.patch.object(main, "DEBUG_DIR", debug_dir),
                mock.patch.object(main, "DEBUG_SCORE_FRAMES_DIR", score_frames_dir),
            ):
                self.assertTrue(
                    main.clear_output_results_for_videos([selected_video], include_subfolders=False)
                )

            self.assertFalse((frames_dir / selected_label).exists())
            self.assertFalse((score_frames_dir / selected_label).exists())
            self.assertFalse((debug_dir / "Identity_Linking" / selected_label).exists())
            self.assertFalse((debug_dir / "Low_Res" / selected_label).exists())
            self.assertTrue((frames_dir / other_label).exists())
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_run_extract_selected_video_clears_scoped_outputs_before_launch(self):
        case_dir = _make_case_dir("run_extract_scoped_cleanup")
        try:
            input_dir = case_dir / "Input_Videos"
            video_path = _touch(input_dir / "Demo_CaptureCard_Race.mp4")
            with (
                mock.patch.object(main, "INPUT_DIR", input_dir),
                mock.patch.object(main, "ensure_runtime_or_raise"),
                mock.patch.object(main, "clear_output_results_for_videos") as cleanup_mock,
                mock.patch.object(main, "run_python_module") as run_module_mock,
            ):
                main.run_extract(selected_video=video_path.name)

            cleanup_mock.assert_called_once()
            cleanup_args, cleanup_kwargs = cleanup_mock.call_args
            self.assertEqual(cleanup_args[0], [video_path])
            self.assertFalse(cleanup_kwargs.get("include_subfolders", False))
            run_module_mock.assert_called_once_with(main.EXTRACT_MODULE, extra_args=["--video", video_path.name])
        finally:
            shutil.rmtree(case_dir, ignore_errors=True)

    def test_run_all_resets_logger_elapsed_time_before_start(self):
        original_start = main.LOGGER.start_time
        try:
            main.LOGGER.start_time = time.perf_counter() - 999.0
            with (
                mock.patch.object(main, "ensure_runtime_or_raise"),
                mock.patch.object(main, "selected_input_video_files", return_value=[]),
            ):
                with self.assertRaisesRegex(RuntimeError, "No supported videos found for selection"):
                    main.run_all(selection_mode=True)
            self.assertLess(main.LOGGER.elapsed_seconds(), 1.0)
        finally:
            main.LOGGER.start_time = original_start
