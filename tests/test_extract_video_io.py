import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from mk8_local_play import extract_video_io


class ExtractVideoIoTests(unittest.TestCase):
    def test_position_capture_for_read_uses_grab_for_small_forward_jump(self):
        class FakeCapture:
            def get(self, _prop):
                return 101

        stats = {"seek_calls": 0, "grab_calls": 0, "seek_time_s": 0.0, "grab_time_s": 0.0}
        with mock.patch.object(extract_video_io, "advance_frames_by_grab", return_value=True) as grab_mock, \
             mock.patch.object(extract_video_io, "seek_to_frame", return_value=True) as seek_mock:
            result = extract_video_io.position_capture_for_read(
                FakeCapture(),
                105,
                stats,
                max_forward_grab_frames=12,
            )

        self.assertTrue(result)
        grab_mock.assert_called_once()
        self.assertEqual(grab_mock.call_args[0][1], 4)
        seek_mock.assert_not_called()

    def test_position_capture_for_read_seeks_for_backward_or_large_jump(self):
        class FakeCapture:
            def __init__(self, pos):
                self.pos = pos

            def get(self, _prop):
                return self.pos

        stats = {"seek_calls": 0, "grab_calls": 0, "seek_time_s": 0.0, "grab_time_s": 0.0}
        with mock.patch.object(extract_video_io, "advance_frames_by_grab", return_value=True) as grab_mock, \
             mock.patch.object(extract_video_io, "seek_to_frame", return_value=True) as seek_mock:
            extract_video_io.position_capture_for_read(
                FakeCapture(110),
                105,
                stats,
                max_forward_grab_frames=12,
            )
            extract_video_io.position_capture_for_read(
                FakeCapture(100),
                120,
                stats,
                max_forward_grab_frames=12,
            )

        grab_mock.assert_not_called()
        self.assertEqual(seek_mock.call_count, 2)

    def test_update_repair_progress_state_resets_when_progress_advances(self):
        stalled_elapsed_s, progress_seconds = extract_video_io._update_repair_progress_state(
            12.0,
            10.0,
            8.0,
            2.0,
        )
        self.assertEqual(stalled_elapsed_s, 0.0)
        self.assertEqual(progress_seconds, 12.0)

    def test_update_repair_progress_state_accumulates_when_progress_is_flat(self):
        stalled_elapsed_s, progress_seconds = extract_video_io._update_repair_progress_state(
            10.1,
            10.0,
            8.0,
            2.0,
            epsilon_s=0.25,
        )
        self.assertEqual(stalled_elapsed_s, 10.0)
        self.assertEqual(progress_seconds, 10.0)

    def test_ffmpeg_remux_command_uses_stream_copy(self):
        command = extract_video_io._ffmpeg_remux_command(
            Path("Input_Videos/demo.mkv"),
            Path("Input_Videos/demo__remux.mkv"),
        )
        self.assertIn("-c", command)
        self.assertIn("copy", command)
        self.assertEqual(Path(command[-1]), Path("Input_Videos/demo__remux.mkv"))

    def test_compare_preflight_before_after_remux_returns_early_when_source_is_not_suspect(self):
        with mock.patch.object(
            extract_video_io,
            "sample_video_readability",
            return_value={"status": "checked", "is_suspect": False, "reason": "ok"},
        ) as sample_mock:
            result = extract_video_io.compare_preflight_before_after_remux(
                "Input_Videos/demo.mkv",
                120,
            )

        self.assertFalse(result["remux_attempted"])
        self.assertFalse(result["improved"])
        self.assertEqual(sample_mock.call_count, 1)
        self.assertIsNone(result["remux_preflight"])

    def test_compare_preflight_before_after_remux_reports_improvement_when_remux_clears_probe(self):
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "corrupt_demo.mkv"
            remux_path = Path(temp_dir) / "corrupt_demo__remux_probe.mkv"
            source_path.write_bytes(b"source")
            recorded_nominal_counts = []

            def _fake_sample(video_path, nominal_total_frames, **_kwargs):
                recorded_nominal_counts.append((Path(video_path).name, nominal_total_frames))
                if Path(video_path) == source_path:
                    return {"status": "suspect", "is_suspect": True, "reason": "probe failed at frame 10"}
                return {"status": "checked", "is_suspect": False, "reason": "all sampled frames read successfully"}

            def _fake_run(_command, _source_name, _stall_timeout_s, source_duration_s=None):
                remux_path.write_bytes(b"remuxed")

            with mock.patch.object(extract_video_io, "sample_video_readability", side_effect=_fake_sample) as sample_mock, \
                 mock.patch.object(extract_video_io, "read_nominal_frame_count", return_value=777) as frame_count_mock, \
                 mock.patch("mk8_local_play.extract_video_io.shutil.which", return_value="ffmpeg"), \
                 mock.patch.object(extract_video_io, "_run_ffmpeg_repair_command", side_effect=_fake_run) as run_mock:
                result = extract_video_io.compare_preflight_before_after_remux(
                    str(source_path),
                    900,
                    remuxed_path=remux_path,
                    keep_remuxed_file=True,
                )

        self.assertTrue(result["remux_attempted"])
        self.assertTrue(result["remux_created"])
        self.assertTrue(result["improved"])
        self.assertEqual(result["comparison_note"], "remux cleared the OpenCV preflight")
        self.assertEqual(sample_mock.call_count, 2)
        self.assertEqual(frame_count_mock.call_count, 1)
        self.assertEqual(run_mock.call_count, 1)
        self.assertEqual(result["remux_preflight"]["status"], "checked")
        self.assertEqual(recorded_nominal_counts, [("corrupt_demo.mkv", 900), ("corrupt_demo__remux_probe.mkv", 777)])

    def test_tail_probe_clamp_returns_usable_frame_count_for_terminal_frame_failure(self):
        self.assertEqual(
            extract_video_io._tail_probe_clamp(207459, 207458, 207457),
            207458,
        )

    def test_tail_probe_clamp_ignores_midstream_failure(self):
        self.assertIsNone(extract_video_io._tail_probe_clamp(207459, 120000, 119999))

    def test_find_last_readable_frame_binary_searches_boundary(self):
        class FakeCapture:
            pass

        def _fake_timed_probe_read(_capture, frame_number, _timeout_s):
            if frame_number <= 207457:
                return True, object(), False, frame_number
            return False, None, False, 207457

        with mock.patch.object(extract_video_io, "_timed_probe_read", side_effect=_fake_timed_probe_read):
            last_good = extract_video_io._find_last_readable_frame(FakeCapture(), 207450, 207458, 5.0)

        self.assertEqual(last_good, 207457)

    def test_sample_video_readability_clamps_tail_failure_instead_of_marking_video_suspect(self):
        class FakeCapture:
            def __init__(self):
                self.pos = 10

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == extract_video_io.cv2.CAP_PROP_FRAME_COUNT:
                    return 10
                if prop == extract_video_io.cv2.CAP_PROP_POS_FRAMES:
                    return self.pos
                return 0

            def release(self):
                return None

        with mock.patch.object(extract_video_io.cv2, "VideoCapture", return_value=FakeCapture()), \
             mock.patch.object(extract_video_io, "_sample_probe_frames", return_value=[9]), \
             mock.patch.object(extract_video_io, "_timed_probe_read", return_value=(False, None, False, 8)), \
             mock.patch.object(extract_video_io, "_find_last_readable_frame", return_value=8):
            result = extract_video_io.sample_video_readability("demo.mkv", 10)

        self.assertEqual(result["status"], "tail_clamped")
        self.assertFalse(result["is_suspect"])
        self.assertEqual(result["usable_total_frames"], 9)

    def test_sample_video_readability_clamps_head_failure_instead_of_marking_video_suspect(self):
        class FakeCapture:
            def __init__(self):
                self.pos = 0

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == extract_video_io.cv2.CAP_PROP_FRAME_COUNT:
                    return 10
                if prop == extract_video_io.cv2.CAP_PROP_POS_FRAMES:
                    return self.pos
                return 0

            def release(self):
                return None

        with mock.patch.object(extract_video_io.cv2, "VideoCapture", return_value=FakeCapture()), \
             mock.patch.object(extract_video_io, "_sample_probe_frames", return_value=[0, 9]), \
             mock.patch.object(extract_video_io, "_timed_probe_read", return_value=(False, None, False, 0)), \
             mock.patch.object(extract_video_io, "_find_first_readable_frame", return_value=2):
            result = extract_video_io.sample_video_readability("demo.mkv", 10)

        self.assertEqual(result["status"], "head_clamped")
        self.assertFalse(result["is_suspect"])
        self.assertEqual(result["usable_start_frame"], 2)
        self.assertEqual(result["usable_total_frames"], 8)

    def test_sample_video_readability_keeps_midstream_failure_suspect(self):
        class FakeCapture:
            def __init__(self):
                self.pos = 5

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == extract_video_io.cv2.CAP_PROP_FRAME_COUNT:
                    return 10
                if prop == extract_video_io.cv2.CAP_PROP_POS_FRAMES:
                    return self.pos
                return 0

            def release(self):
                return None

        with mock.patch.object(extract_video_io.cv2, "VideoCapture", return_value=FakeCapture()), \
             mock.patch.object(extract_video_io, "_sample_probe_frames", return_value=[5]), \
             mock.patch.object(extract_video_io, "_timed_probe_read", return_value=(False, None, False, 4)):
            result = extract_video_io.sample_video_readability("demo.mkv", 10)

        self.assertEqual(result["status"], "suspect")
        self.assertTrue(result["is_suspect"])

    def test_repair_video_if_needed_prefers_mp4_remux_before_transcode(self):
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "suspect_input.mkv"
            source_path.write_bytes(b"source")
            source_text = str(source_path)
            remux_working = source_path.with_name(f"{source_path.stem}__remuxing.mp4")

            def _fake_run(command, _source_name, _stall_timeout_s, source_duration_s=None):
                if "copy" in command:
                    remux_working.write_bytes(b"remux")
                    return
                raise AssertionError("Transcode should not run when remux clears preflight")

            with mock.patch("mk8_local_play.extract_video_io.shutil.which", return_value="ffmpeg"), \
                 mock.patch.object(extract_video_io, "read_nominal_frame_count", return_value=123), \
                 mock.patch.object(extract_video_io, "sample_video_readability", return_value={"status": "checked", "is_suspect": False, "reason": "ok"}), \
                 mock.patch.object(extract_video_io, "_run_ffmpeg_repair_command", side_effect=_fake_run):
                repaired = extract_video_io.repair_video_if_needed(
                    source_text,
                    200,
                    {"status": "suspect", "is_suspect": True, "reason": "probe failed"},
                    duration_s=10.0,
                    stats={},
                )

            self.assertTrue(repaired.endswith(".mp4"))
            self.assertFalse(Path(source_text).exists())
            self.assertTrue(Path(repaired).exists())

    def test_repair_video_if_needed_falls_back_to_transcode_when_remux_stays_suspect(self):
        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "suspect_input.mkv"
            source_path.write_bytes(b"source")
            source_text = str(source_path)
            remux_working = source_path.with_name(f"{source_path.stem}__remuxing.mp4")
            transcode_working = source_path.with_name(f"{source_path.stem}__repairing.mp4")
            call_kinds = []

            def _fake_run(command, _source_name, _stall_timeout_s, source_duration_s=None):
                if "copy" in command:
                    call_kinds.append("remux")
                    remux_working.write_bytes(b"remux")
                    return
                call_kinds.append("transcode")
                transcode_working.write_bytes(b"transcode")

            sample_results = [
                {"status": "suspect", "is_suspect": True, "reason": "still suspect"},
            ]

            with mock.patch("mk8_local_play.extract_video_io.shutil.which", return_value="ffmpeg"), \
                 mock.patch.object(extract_video_io, "read_nominal_frame_count", return_value=123), \
                 mock.patch.object(extract_video_io, "sample_video_readability", side_effect=sample_results), \
                 mock.patch.object(extract_video_io, "_run_ffmpeg_repair_command", side_effect=_fake_run):
                repaired = extract_video_io.repair_video_if_needed(
                    source_text,
                    200,
                    {"status": "suspect", "is_suspect": True, "reason": "probe failed"},
                    duration_s=10.0,
                    stats={},
                )

            self.assertEqual(call_kinds, ["remux", "transcode"])
            self.assertTrue(repaired.endswith(".mp4"))
            self.assertTrue(Path(repaired).exists())
