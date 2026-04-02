import unittest
from collections import defaultdict
from unittest import mock

from mk8_local_play import extract_frames


class ExtractFramesTests(unittest.TestCase):
    def test_print_extract_profiler_summary_reports_capture_usage_lines(self):
        stats = defaultdict(float)
        stats["score_capture_frame_events_total"] = 12
        stats["score_capture_unique_frames_total"] = 9
        stats["score_capture_duplicate_frames_total"] = 3
        stats["score_capture_duplicate_source_seconds_total"] = 0.1
        stats["score_capture_race_consensus_frames"] = 3
        stats["score_capture_total_consensus_frames"] = 3
        stats["score_capture_points_context_frames"] = 3
        stats["score_capture_race_anchor_frames"] = 1
        stats["score_capture_total_anchor_frames"] = 1
        stats["score_capture_points_anchor_frames"] = 1
        stats["score_same_run_ocr_frames_total"] = 6
        stats["score_same_run_ocr_unique_frames_total"] = 6
        stats["score_persisted_ocr_frames_total"] = 8
        stats["score_persisted_ocr_unique_frames_total"] = 8
        stats["score_capture_frames_outside_same_run_cache_total"] = 3
        stats["score_capture_outside_same_run_cache_source_seconds_total"] = 0.1

        with mock.patch.object(extract_frames.LOGGER, "summary_block") as summary_mock:
            extract_frames.print_extract_profiler_summary("demo.mp4", stats)

        summary_lines = summary_mock.call_args.args[1]
        joined = "\n".join(summary_lines)
        self.assertIn("score capture events/unique frames: 12/9", joined)
        self.assertIn("score capture overlap: 3 duplicate frame reads (0.10s source)", joined)
        self.assertIn("same-run OCR frame inputs: 6 (6 unique)", joined)
        self.assertIn("persisted rerun OCR frame inputs: 8 (8 unique)", joined)
        self.assertIn("captured frames outside same-run in-memory OCR cache: 3 (0.10s source)", joined)

    def test_print_extract_profiler_summary_reports_backlog_and_lock_wait_lines(self):
        stats = defaultdict(float)
        stats["score_ready_results_max"] = 3
        stats["score_out_of_order_results"] = 5
        stats["score_flush_io_lock_wait_s"] = 1.25
        stats["score_flush_io_lock_acquires"] = 8
        stats["score_callback_io_lock_wait_s"] = 0.5
        stats["score_callback_io_lock_acquires"] = 4

        with mock.patch.object(extract_frames.LOGGER, "summary_block") as summary_mock:
            extract_frames.print_extract_profiler_summary("demo.mp4", stats)

        summary_lines = summary_mock.call_args.args[1]
        joined = "\n".join(summary_lines)
        self.assertIn("parallel score result backlog: max ready 3 | out-of-order completions 5", joined)
        self.assertIn("flush IO lock wait: 1.25s across 8 acquires", joined)
        self.assertIn("callback IO lock wait: 0.50s across 4 acquires", joined)

    def test_print_extract_profiler_summary_reports_positioning_lines(self):
        stats = defaultdict(float)
        stats["position_calls"] = 10
        stats["position_noop_calls"] = 2
        stats["position_forward_grab_calls"] = 3
        stats["position_forward_grab_frames"] = 9
        stats["position_seek_fallback_calls"] = 5
        stats["seek_calls"] = 5
        stats["seek_forward_calls"] = 3
        stats["seek_backward_calls"] = 2
        stats["seek_short_calls"] = 1
        stats["seek_medium_calls"] = 2
        stats["seek_long_calls"] = 2
        stats["seek_frame_distance_total"] = 450
        stats["seek_calls__total_stable_rewind"] = 3
        stats["seek_backward_calls__total_stable_rewind"] = 2
        stats["seek_long_calls__total_stable_rewind"] = 1
        stats["seek_frame_distance_total__total_stable_rewind"] = 320

        with mock.patch.object(extract_frames.LOGGER, "summary_block") as summary_mock:
            extract_frames.print_extract_profiler_summary("demo.mp4", stats)

        summary_lines = summary_mock.call_args.args[1]
        joined = "\n".join(summary_lines)
        self.assertIn("capture positioning: 10 calls | no-op 2 | grab-advance 3 (9 frames) | seek fallback 5", joined)
        self.assertIn("seek profile: forward 3 | backward 2 | short 1 | medium 2 | long 2 | distance 450 frames", joined)
        self.assertIn("seek hotspots: total_stable_rewind 3c/2b/1l/320f", joined)

    def test_process_score_candidates_queues_ocr_immediately_for_completed_races(self):
        callback_payloads = []

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, futures_in_submit_order):
                self._futures = list(futures_in_submit_order)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, _fn, *_args, **_kwargs):
                return self._futures.pop(0)

        def _build_result(race_number, total_visible_players):
            return {
                "candidate": {
                    "race_number": race_number,
                    "score_layout_id": "lan",
                },
                "race_score_frame": 100 + race_number,
                "total_score_frame": 200 + race_number,
                "actual_race_score_frame": 100 + race_number,
                "actual_total_score_frame": 200 + race_number,
                "actual_points_anchor_frame": 300 + race_number,
                "race_score_image": object(),
                "total_score_image": object(),
                "race_consensus_frames": [(100 + race_number, object())],
                "total_consensus_frames": [(200 + race_number, object())],
                "points_context_frames": [(300 + race_number, object())],
                "total_score_visible_players": total_visible_players,
                "twelfth_template_detected": False,
                "debug_rows": [],
                "stats": defaultdict(float),
            }

        analyze_results = {
            1: _build_result(1, 12),
            2: _build_result(2, 11),
        }

        def _fake_analyze(task, _frame_to_timecode):
            return analyze_results[int(task["race_number"])]

        futures = [FakeFuture(analyze_results[1]), FakeFuture(analyze_results[2])]

        with mock.patch.object(extract_frames.score_screen_selection, "analyze_score_window_task", side_effect=_fake_analyze), \
             mock.patch.object(extract_frames.score_screen_selection, "refine_race_score_result_for_expected_players", side_effect=lambda result, _expected_players: result), \
             mock.patch.object(extract_frames.score_screen_selection, "expand_race_score_consensus_window", side_effect=lambda result, _expected_players: result), \
             mock.patch.object(extract_frames.score_screen_selection, "is_static_gallery_race_bundle", return_value=(False, None)), \
             mock.patch.object(extract_frames.score_screen_selection, "save_score_frames", return_value=True), \
             mock.patch.object(extract_frames.video_io, "add_timing", return_value=None), \
             mock.patch.object(extract_frames, "ThreadPoolExecutor", return_value=FakeExecutor(futures)), \
             mock.patch.object(extract_frames, "as_completed", return_value=[futures[1], futures[0]]):
            extract_frames.process_score_candidates(
                video_path="Input_Videos/demo.mkv",
                video_label="demo",
                video_source_path="2026-03-28/demo.mkv",
                score_candidates=[
                    {"race_number": 1, "frame_number": 100},
                    {"race_number": 2, "frame_number": 200},
                ],
                templates=[],
                fps=30.0,
                csv_writer=mock.Mock(),
                scale_x=1.0,
                scale_y=1.0,
                left=0,
                top=0,
                crop_width=1280,
                crop_height=720,
                stats=defaultdict(float),
                metadata_writer=mock.Mock(),
                per_race_complete_callback=lambda payload: callback_payloads.append(dict(payload)),
                analysis_workers_override=2,
            )

        self.assertEqual(
            callback_payloads,
            [
                {"video_label": "demo", "race_number": 2, "ocr_revision": 1},
                {"video_label": "demo", "race_number": 1, "ocr_revision": 1},
            ],
        )

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

    def test_parallel_video_scan_starts_total_score_before_all_scans_complete(self):
        class FakeFuture:
            def __init__(self, result, done=False):
                self._result = result
                self._done = done

            def done(self):
                return self._done

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, kind, events, prepare_results=None, scan_futures=None, score_results=None):
                self.kind = kind
                self.events = events
                self.prepare_results = list(prepare_results or [])
                self.scan_futures = list(scan_futures or [])
                self.score_results = dict(score_results or {})

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, *args, **kwargs):
                if self.kind == "prepare":
                    video_path = args[0]
                    self.events.append(("prepare_submit", video_path))
                    return FakeFuture(self.prepare_results.pop(0), done=True)
                context = args[0]
                label = context["video_label"]
                if self.kind == "scan":
                    self.events.append(("scan_submit", label))
                    return self.scan_futures.pop(0)
                self.events.append(("score_submit", label))
                return FakeFuture(self.score_results[label], done=True)

        events = []
        scan_result_1 = {
            "aborted": False,
            "capture_poisoned": False,
            "score_candidates": [{"race_number": 1}],
            "scan_track_count": 1,
            "scan_race_count": 1,
            "scan_progress": None,
        }
        scan_result_2 = {
            "aborted": False,
            "capture_poisoned": False,
            "score_candidates": [{"race_number": 1}],
            "scan_track_count": 1,
            "scan_race_count": 1,
            "scan_progress": None,
        }
        scan_future_1 = FakeFuture(scan_result_1, done=True)
        scan_future_2 = FakeFuture(scan_result_2, done=False)

        score_results = {
            "video-1": {
                "exported_counts": {"score": 1, "track": 1, "race": 1},
                "per_video_summary": {"display_video_index": 1, "video_name": "video1.mp4"},
            },
            "video-2": {
                "exported_counts": {"score": 1, "track": 1, "race": 1},
                "per_video_summary": {"display_video_index": 2, "video_name": "video2.mp4"},
            },
        }

        context_1 = {
            "video_path": "Input_Videos/video1.mp4",
            "video_name": "video1.mp4",
            "video_label": "video-1",
            "source_display_name": "2026-03-28/video1.mp4",
            "display_video_index": 1,
            "display_total_videos": 2,
            "video_index": 1,
            "total_videos": 2,
            "video_start": 0.0,
            "video_stats": defaultdict(float, {"main_scan_loop_s": 1.0}),
            "fps": 30.0,
            "total_frames": 300,
            "detection_segment_tasks": [object()],
        }

        context_2 = {
            "video_path": "Input_Videos/video2.mp4",
            "video_name": "video2.mp4",
            "video_label": "video-2",
            "source_display_name": "2026-03-28/video2.mp4",
            "display_video_index": 2,
            "display_total_videos": 2,
            "video_index": 2,
            "total_videos": 2,
            "video_start": 0.0,
            "video_stats": defaultdict(float, {"main_scan_loop_s": 1.0}),
            "fps": 30.0,
            "total_frames": 300,
            "detection_segment_tasks": [object()],
        }

        executors = [
            FakeExecutor("prepare", events, prepare_results=[context_1, context_2]),
            FakeExecutor("scan", events, scan_futures=[scan_future_1, scan_future_2]),
            FakeExecutor("score", events, score_results=score_results),
        ]

        def _fake_executor_factory(*args, **kwargs):
            return executors.pop(0)

        def _fake_sleep(_seconds):
            if not scan_future_2._done:
                events.append(("second_scan_released", "video-2"))
                scan_future_2._done = True

        def _fake_wait(_futures, return_when=None):
            _fake_sleep(0.0)
            return (set(), set())

        with mock.patch.object(extract_frames, "build_workflow_video_plan", return_value=([
                {"video_path": context_1["video_path"], "display_video_index": 1, "video_label": "video-1", "source_display_name": context_1["source_display_name"]},
                {"video_path": context_2["video_path"], "display_video_index": 2, "video_label": "video-2", "source_display_name": context_2["source_display_name"]},
            ], 10.0)), \
             mock.patch.object(extract_frames, "as_completed", side_effect=lambda futures: list(futures)), \
             mock.patch.object(extract_frames, "ThreadPoolExecutor", side_effect=_fake_executor_factory), \
             mock.patch.object(extract_frames, "wait", side_effect=_fake_wait), \
             mock.patch.object(extract_frames, "time") as time_mock, \
             mock.patch.object(extract_frames.LOGGER, "log"), \
             mock.patch.object(extract_frames.LOGGER, "summary_block"), \
             mock.patch.object(extract_frames.LOGGER, "peak_lines", return_value=[]), \
             mock.patch.object(extract_frames.LOGGER.resources, "sample", return_value=None):
            time_mock.time.return_value = 5.0
            time_mock.sleep.side_effect = _fake_sleep
            result = extract_frames._extract_frames_parallel_video_scan(
                video_paths=["Input_Videos/video1.mp4", "Input_Videos/video2.mp4"],
                folder_path="Input_Videos",
                include_subfolders=True,
                templates=[],
                template_load_time_s=0.0,
                csv_writer=None,
                metadata_writer=None,
                metadata_context=None,
                per_video_complete_callback=None,
                per_race_complete_callback=None,
                total_source_seconds=10.0,
                return_frame_cache=False,
                phase_start_time=0.0,
            )

        self.assertEqual(result["summary"]["total_score_screens"], 2)
        self.assertLess(events.index(("score_submit", "video-1")), events.index(("second_scan_released", "video-2")))
