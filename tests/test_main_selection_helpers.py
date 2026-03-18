import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from mk8_local_play.extract_common import (
    EXPORT_IMAGE_FORMAT,
    choose_preferred_exported_image,
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
