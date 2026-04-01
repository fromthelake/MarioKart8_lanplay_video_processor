import unittest
from unittest import mock

from mk8_local_play import extract_frames


class ExtractFramesTests(unittest.TestCase):
    def test_prepare_video_context_uses_preflight_usable_total_frames_without_repair(self):
        class FakeCapture:
            def __init__(self, frame_count=207459, fps=30.0):
                self.frame_count = frame_count
                self.fps = fps

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == extract_frames.cv2.CAP_PROP_FRAME_COUNT:
                    return self.frame_count
                if prop == extract_frames.cv2.CAP_PROP_FPS:
                    return self.fps
                return 0

            def release(self):
                return None

        fake_frame = object()
        capture_factory = mock.Mock(side_effect=[FakeCapture(), FakeCapture(frame_count=207459, fps=30.0)])

        with mock.patch.object(extract_frames.cv2, "VideoCapture", capture_factory), \
             mock.patch.object(
                 extract_frames.video_io,
                 "sample_video_readability",
                 return_value={
                     "status": "tail_clamped",
                     "is_suspect": False,
                     "reason": "tail probe read failed at frame 207458; using 207458 readable frames",
                     "usable_total_frames": 207458,
                 },
             ), \
             mock.patch.object(extract_frames.video_io, "repair_video_if_needed", side_effect=lambda *args, **kwargs: args[0]) as repair_mock, \
             mock.patch.object(extract_frames.video_io, "seek_to_frame", return_value=True), \
             mock.patch.object(extract_frames.video_io, "read_video_frame", return_value=(True, fake_frame)), \
             mock.patch.object(extract_frames.video_io, "read_video_frame_with_timeout", return_value=(True, fake_frame, False)), \
             mock.patch.object(extract_frames, "determine_scaling", return_value=(1.0, 1.0, 0, 0, 1280, 720)), \
             mock.patch.object(extract_frames.initial_scan, "build_detection_segment_tasks", return_value=[]):
            context = extract_frames._prepare_video_context(
                video_path="Input_Videos/corrupt/corrupt_Kampioen_2026-03-27 21-50-56.mkv",
                folder_path="Input_Videos",
                include_subfolders=True,
                video_index=1,
                total_videos=1,
                template_load_time_s=0.0,
                templates=[],
                video_label="corrupt_Kampioen_2026-03-27 21-50-56",
            )

        self.assertIsNotNone(context)
        self.assertEqual(context["total_frames"], 207458)
        self.assertEqual(context["corrupt_check_status"], "tail_clamped")
        self.assertEqual(context["processing_video_path"], "Input_Videos/corrupt/corrupt_Kampioen_2026-03-27 21-50-56.mkv")
        self.assertEqual(repair_mock.call_count, 1)

    def test_prepare_video_context_carries_head_clamp_start_and_disables_parallel_scan(self):
        class FakeCapture:
            def __init__(self, frame_count=100, fps=30.0):
                self.frame_count = frame_count
                self.fps = fps

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == extract_frames.cv2.CAP_PROP_FRAME_COUNT:
                    return self.frame_count
                if prop == extract_frames.cv2.CAP_PROP_FPS:
                    return self.fps
                return 0

            def release(self):
                return None

        fake_frame = object()
        capture_factory = mock.Mock(side_effect=[FakeCapture(), FakeCapture(frame_count=100, fps=30.0)])

        with mock.patch.object(extract_frames.cv2, "VideoCapture", capture_factory), \
             mock.patch.object(
                 extract_frames.video_io,
                 "sample_video_readability",
                 return_value={
                     "status": "head_clamped",
                     "is_suspect": False,
                     "reason": "head probe read failed at frame 0; using 95 readable frames",
                     "usable_start_frame": 5,
                     "usable_total_frames": 95,
                 },
             ), \
             mock.patch.object(extract_frames.video_io, "repair_video_if_needed", side_effect=lambda *args, **kwargs: args[0]), \
             mock.patch.object(extract_frames.video_io, "seek_to_frame", return_value=True), \
             mock.patch.object(extract_frames.video_io, "read_video_frame", return_value=(True, fake_frame)), \
             mock.patch.object(extract_frames.video_io, "read_video_frame_with_timeout", return_value=(True, fake_frame, False)), \
             mock.patch.object(extract_frames, "determine_scaling", return_value=(1.0, 1.0, 0, 0, 1280, 720)), \
             mock.patch.object(extract_frames.initial_scan, "build_detection_segment_tasks", return_value=[]):
            context = extract_frames._prepare_video_context(
                video_path="Input_Videos/corrupt/demo.mkv",
                folder_path="Input_Videos",
                include_subfolders=True,
                video_index=1,
                total_videos=1,
                template_load_time_s=0.0,
                templates=[],
                video_label="demo",
            )

        self.assertIsNotNone(context)
        self.assertEqual(context["corrupt_check_status"], "head_clamped")
        self.assertEqual(context["usable_start_frame"], 5)
        self.assertEqual(context["total_frames"], 95)
        self.assertEqual(context["readable_end_frame"], 100)
        self.assertEqual(context["detection_segment_tasks"], [])

    def test_run_serial_initial_scan_starts_from_head_clamped_offset(self):
        class FakeCapture:
            def __init__(self):
                self.opened = True

            def isOpened(self):
                return self.opened

            def release(self):
                self.opened = False

        fake_capture = FakeCapture()
        processed_frames = []
        progress_updates = []

        class FakeProgress:
            def __init__(self, *args, **kwargs):
                self.last_percent = 0

            def update(self, completed, *args, **kwargs):
                progress_updates.append(completed)
                self.last_percent = 100 if completed >= 5 else 0

            def peak_lines(self):
                return []

        def _fake_process_frame(frame, frame_number, *_args, **_kwargs):
            processed_frames.append(frame_number)
            return 0

        context = {
            "processing_video_path": "Input_Videos/corrupt/demo.mkv",
            "video_index": 1,
            "total_videos": 1,
            "display_video_index": 1,
            "display_total_videos": 1,
            "video_stats": {"main_scan_loop_s": 0.0},
            "fps": 30.0,
            "total_frames": 5,
            "usable_start_frame": 3,
            "readable_end_frame": 8,
            "frame_skip": 1,
            "eof_guard_frames": 999,
            "video_label": "demo",
            "source_display_name": "demo.mkv",
            "median_scale_x": 1.0,
            "median_scale_y": 1.0,
            "median_left": 0,
            "median_top": 0,
            "median_crop_width": 1280,
            "median_crop_height": 720,
        }

        with mock.patch.object(extract_frames.cv2, "VideoCapture", return_value=fake_capture), \
             mock.patch.object(extract_frames, "ProgressPrinter", FakeProgress), \
             mock.patch.object(extract_frames.video_io, "seek_to_frame", return_value=True) as seek_mock, \
             mock.patch.object(extract_frames.video_io, "read_video_frame", return_value=(True, object())), \
             mock.patch.object(extract_frames.video_io, "read_video_frame_with_timeout", return_value=(True, object(), False)), \
             mock.patch.object(extract_frames.video_io, "advance_frames_by_grab", return_value=True), \
             mock.patch.object(extract_frames.video_io, "add_timing", return_value=None), \
             mock.patch.object(extract_frames.initial_scan, "process_frame", side_effect=_fake_process_frame), \
             mock.patch.object(extract_frames, "count_exported_detection_files", return_value={"track": 0, "race": 0}):
            result = extract_frames._run_serial_initial_scan(context, templates=[], csv_writer=None, metadata_writer=None)

        self.assertFalse(result["aborted"])
        self.assertEqual(processed_frames[0], 3)
        self.assertIn(0, progress_updates)
        self.assertIn(5, progress_updates)
        seek_mock.assert_any_call(fake_capture, 3, context["video_stats"])
