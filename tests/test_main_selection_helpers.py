import shutil
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

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
            DEBUG_ROOT / "Score_Frames" / "Demo_CaptureCard_Race" / "Race_007" / f"annotated_2RaceScore{expected_extension}",
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
                    [expected_a, expected_b],
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
