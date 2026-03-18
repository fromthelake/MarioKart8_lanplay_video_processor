import cv2
import numpy as np
import os
import argparse
import csv
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .app_runtime import load_app_config
from .console_logging import LOGGER
from .data_paths import resolve_asset_file
from .project_paths import PROJECT_ROOT
from . import extract_initial_scan as initial_scan
from . import extract_score_screen_selection as score_screen_selection
from . import extract_video_io as video_io
from .extract_common import (
    GPU_RUNTIME,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    build_video_identity,
    count_exported_detection_files,
    crop_and_upscale_image,
    determine_scaling,
    frame_to_timecode,
    load_videos_from_folder,
    relative_video_path,
)

# Record the start time
start_run_time = time.time()
APP_CONFIG = load_app_config()

SCORE_ANALYSIS_WORKERS = APP_CONFIG.score_analysis_workers
INITIAL_SCAN_WORKERS = APP_CONFIG.pass1_scan_workers
INITIAL_SCAN_WINDOW_STEPS = 2
INITIAL_SCAN_SEGMENT_OVERLAP_FRAMES = APP_CONFIG.pass1_segment_overlap_frames
INITIAL_SCAN_MIN_SEGMENT_FRAMES = APP_CONFIG.pass1_min_segment_frames
INITIAL_SCAN_PROGRESS_REPORT_SECONDS = 2.0
INITIAL_SCAN_EOF_GUARD_SECONDS = 10.0
INITIAL_SCAN_EOF_GUARD_PROGRESS = 0.99
INITIAL_SCAN_EOF_READ_TIMEOUT_SECONDS = 10.0
INITIAL_SCAN_FFPROBE_COUNTFRAME_MIN_DELTA_FRAMES = 30

CONSENSUS_FRAME_CACHE = {}


class NullCsvWriter:
    """Drop debug CSV rows when debug output is disabled."""

    def writerow(self, _row):
        return None


class MetadataCsvWriter:
    """Write exported frame metadata when enabled."""

    def __init__(self, csv_writer):
        self._csv_writer = csv_writer

    def writerow(self, row):
        self._csv_writer.writerow(row)


class ProgressPrinter:
    """Print throttled progress updates for long-running stages."""

    def __init__(self, scope, total_units, percent_step=5, min_interval_s=3.0):
        self.scope = scope
        self.total_units = max(1, int(total_units))
        self.percent_step = max(1, int(percent_step))
        self.min_interval_s = float(min_interval_s)
        self.last_percent = -1
        self.last_print_time = 0.0
        self.start_time = time.perf_counter()
        self.phase_peak = {
            "cpu_percent": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
            "gpu_percent": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
        }

    def update(self, completed_units, detail=""):
        percent = min(100, int((max(0, completed_units) / self.total_units) * 100))
        now = time.perf_counter()
        should_print = percent >= 100 or self.last_percent < 0
        if not should_print and percent >= self.last_percent + self.percent_step:
            should_print = True
        if not should_print and now - self.last_print_time >= self.min_interval_s:
            should_print = True
        if not should_print:
            return
        snapshot = LOGGER.resources.sample()
        self._update_phase_peak(snapshot)
        resource_text = LOGGER.resource_text(snapshot)
        detail_suffix = f" | {detail}" if detail else ""
        LOGGER.log(
            self.scope,
            f"{percent:3d}% | {min(completed_units, self.total_units):,}/{self.total_units:,} | "
            f"{resource_text}{detail_suffix}",
        )
        self.last_percent = percent
        self.last_print_time = now

    def _update_phase_peak(self, snapshot):
        for field in ("cpu_percent", "ram_used_gb", "gpu_percent", "vram_used_gb"):
            current = getattr(snapshot, field)
            peak_value = self.phase_peak[field]
            if current is not None and (peak_value is None or current > peak_value):
                self.phase_peak[field] = current
        for field in ("ram_total_gb", "vram_total_gb"):
            current = getattr(snapshot, field)
            if current is not None:
                self.phase_peak[field] = current

    def peak_lines(self):
        lines = []
        if self.phase_peak["cpu_percent"] is not None:
            lines.append(f"Peak CPU: {self.phase_peak['cpu_percent']:.0f}%")
        if self.phase_peak["ram_used_gb"] is not None and self.phase_peak["ram_total_gb"] is not None:
            lines.append(f"Peak RAM: {self.phase_peak['ram_used_gb']:.1f} / {self.phase_peak['ram_total_gb']:.1f} GB")
        if self.phase_peak["gpu_percent"] is not None:
            lines.append(f"Peak GPU: {self.phase_peak['gpu_percent']:.0f}%")
        if self.phase_peak["vram_used_gb"] is not None and self.phase_peak["vram_total_gb"] is not None:
            lines.append(f"Peak VRAM: {self.phase_peak['vram_used_gb']:.1f} / {self.phase_peak['vram_total_gb']:.1f} GB")
        return lines


def format_duration(seconds):
    """Format seconds as HH:MM:SS for status messages."""
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def print_timing_summary(video_name, stats):
    """Print a user-facing summary of where time went."""
    total_time = float(stats.get("video_total_s", 0.0))
    if total_time <= 0:
        return

    major_buckets = [
        ("calibration", float(stats.get("scaling_scan_s", 0.0))),
        ("initial result-screen scan", float(stats.get("main_scan_loop_s", 0.0))),
        ("score screen selection", float(stats.get("score_candidate_pass_s", 0.0))),
        ("frame export", float(stats.get("output_frame_capture_s", 0.0))),
        ("video seek/read/grab", float(stats.get("seek_time_s", 0.0) + stats.get("read_time_s", 0.0) + stats.get("grab_time_s", 0.0))),
    ]
    ranked = [(label, value) for label, value in major_buckets if value > 0.0]
    ranked.sort(key=lambda item: item[1], reverse=True)

    lines = []
    for label, value in ranked[:3]:
        percent = (value / total_time) * 100 if total_time > 0 else 0.0
        lines.append(f"{label}: {format_duration(value)} ({percent:.0f}%)")
    LOGGER.summary_block(f"[{video_name} - Debug Timing]", lines, color_name="dim")


def collect_consensus_frames(video_path, video_label, center_frame, fps, left, top, crop_width, crop_height, bundle_kind):
    """Collect nearby upscaled frames for in-memory OCR consensus during --all runs."""
    radius = max(0, APP_CONFIG.ocr_consensus_frames // 2)
    cache_key = (video_label, int(bundle_kind[0]), bundle_kind[1])
    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        return

    total_frames = int(local_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bundled_frames = []
    start_frame = max(0, center_frame - radius)
    end_frame = min(total_frames, center_frame + radius + 1)
    local_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in range(start_frame, end_frame):
        ret, frame = local_cap.read()
        if not ret:
            continue
        bundled_frames.append(crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT))
    local_cap.release()
    if bundled_frames:
        CONSENSUS_FRAME_CACHE[cache_key] = bundled_frames


def process_score_candidates(video_path, video_label, video_source_path, score_candidates, templates, fps, csv_writer, scale_x, scale_y, left, top,
                             crop_width, crop_height, stats, metadata_writer, video_index=None, total_videos=None, progress=None):
    """Second pass over recorded score candidates."""
    if not score_candidates:
        return

    stage_start = time.perf_counter()
    tasks = [
        {
            "video_path": video_path,
            "race_number": candidate["race_number"],
            "frame_number": candidate["frame_number"],
            "fps": fps,
            "templates": templates,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "left": left,
            "top": top,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "ocr_consensus_frames": APP_CONFIG.ocr_consensus_frames,
            "score_layout_id": candidate.get("score_layout_id"),
        }
        for candidate in score_candidates
    ]

    worker_count = min(SCORE_ANALYSIS_WORKERS, len(tasks))
    local_progress = progress or ProgressPrinter(
        f"[Video {video_index}/{total_videos} - Total Score Screen]" if video_index is not None else "[Total Score Screen]",
        len(tasks),
        percent_step=5,
        min_interval_s=2.0,
    )
    if worker_count == 1:
        results = []
        for completed_count, task in enumerate(tasks, start=1):
            result = score_screen_selection.analyze_score_window_task(
                task,
                frame_to_timecode,
            )
            results.append(result)
            if result["total_score_frame"] > 0:
                LOGGER.log(
                    "",
                    f"Race {task['race_number']:03} | total score screen found at source {frame_to_timecode(result['total_score_frame'], fps)}",
                    color_name="green",
                )
            local_progress.update(
                completed_count,
                f"Completed windows: {completed_count}/{len(tasks)} | Last completed: race {task['race_number']:03}/{len(tasks):03}",
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    score_screen_selection.analyze_score_window_task,
                    task,
                    frame_to_timecode,
                ): task
                for task in tasks
            }
            results = []
            for completed_count, future in enumerate(as_completed(future_map), start=1):
                task = future_map[future]
                result = future.result()
                results.append(result)
                if result["total_score_frame"] > 0:
                    LOGGER.log(
                        "",
                        f"Race {task['race_number']:03} | total score screen found at source {frame_to_timecode(result['total_score_frame'], fps)}",
                        color_name="green",
                    )
                local_progress.update(
                    completed_count,
                    f"Completed windows: {completed_count}/{len(tasks)} | Last completed: race {task['race_number']:03}/{len(tasks):03}",
                )

    previous_total_score_players = None
    for result in sorted(results, key=lambda item: item["candidate"]["race_number"]):
        expected_players = previous_total_score_players if int(result["candidate"].get("race_number", 0) or 0) >= 2 else None
        result = score_screen_selection.refine_race_score_result_for_expected_players(result, expected_players)
        for key, value in result["stats"].items():
            stats[key] += value
        for row in result["debug_rows"]:
            csv_writer.writerow(row)
        if result["race_score_frame"] <= 0 or result["total_score_frame"] <= 0:
            continue
        score_screen_selection.save_score_frames(
            video_path,
            video_label,
            result["candidate"]["race_number"],
            result["race_score_frame"],
            result["total_score_frame"],
            result.get("actual_race_score_frame"),
            result.get("actual_total_score_frame"),
            result.get("race_score_image"),
            result.get("total_score_image"),
            result.get("race_consensus_frames", []),
            result.get("total_consensus_frames", []),
            fps,
            metadata_writer,
            CONSENSUS_FRAME_CACHE,
            frame_to_timecode,
            video_source_path=video_source_path,
            score_layout_id=result["candidate"].get("score_layout_id"),
        )
        total_score_image = result.get("total_score_image")
        if total_score_image is not None:
            previous_total_score_players = score_screen_selection.count_visible_position_rows(
                total_score_image,
                result["candidate"].get("score_layout_id"),
            )
    video_io.add_timing(stats, "score_candidate_pass_s", stage_start)

def extract_frames(return_frame_cache=False, selected_videos=None, include_subfolders=False):
    """Main function to process videos and apply template matching."""
    phase_start_time = time.time()
    CONSENSUS_FRAME_CACHE.clear()
    empty_summary = {
        "duration_s": 0.0,
        "total_source_seconds": 0.0,
        "track_screens": 0,
        "race_numbers": 0,
        "total_score_screens": 0,
        "per_video_summaries": [],
    }
    LOGGER.log("[Extract - Phase Start]", "Extract race and score screens", color_name="cyan")

    folder_path = os.path.join(PROJECT_ROOT, 'Input_Videos')

    template_paths = [
        (str(resolve_asset_file('templates', 'Score_template.png')), None),
        (str(resolve_asset_file('templates', 'Trackname_template.png')), None),
        (str(resolve_asset_file('templates', 'Race_template.png')), None),
        (str(resolve_asset_file('templates', '12th_pos_template.png')), None)
    ]

    templates = []
    template_load_start = time.perf_counter()
    for template_path, _ in template_paths:
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if template is None:
            LOGGER.log("[Extract - Phase Start]", f"Template image could not be loaded: {template_path}", color_name="red")
            return {"frame_bundle_cache": {}, "summary": empty_summary} if return_frame_cache else {"summary": empty_summary}
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            _, alpha_mask = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
        elif len(template.shape) == 3 and template.shape[2] == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
            alpha_mask = None
        elif len(template.shape) == 2:
            template_binary = template
            alpha_mask = None
        else:
            LOGGER.log("[Extract - Phase Start]", f"Template image has an unexpected number of channels: {template_path}", color_name="red")
            return {"frame_bundle_cache": {}, "summary": empty_summary} if return_frame_cache else {"summary": empty_summary}
        templates.append((template_binary, alpha_mask))
    template_load_time_s = time.perf_counter() - template_load_start

    csv_output_path = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug', 'debug_max_val.csv')
    metadata_output_path = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug', 'exported_frame_metadata.csv')
    video_paths = load_videos_from_folder(folder_path, include_subfolders=include_subfolders)
    if selected_videos:
        selected_names = {str(name).replace("\\", "/").lower() for name in selected_videos}
        filtered_video_paths = []
        for path in video_paths:
            path_obj = Path(path)
            basename = path_obj.name.lower()
            relative_name = relative_video_path(path_obj, folder_path).lower()
            if basename in selected_names or relative_name in selected_names:
                filtered_video_paths.append(path)
        video_paths = filtered_video_paths
    if not video_paths:
        LOGGER.log("[Extract - Phase Start]", "No videos found in Input_Videos", color_name="red")
        return {"frame_bundle_cache": {}, "summary": empty_summary} if return_frame_cache else {"summary": empty_summary}

    if APP_CONFIG.write_debug_csv:
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        csv_context = open(csv_output_path, mode='w', newline='')
        csv_writer = csv.writer(csv_context, delimiter=';')
        csv_writer.writerow(["Video", "Target", "Frame Number", "Max Value", "Timecode"])
        metadata_context = open(metadata_output_path, mode='w', newline='')
        metadata_writer = MetadataCsvWriter(csv.writer(metadata_context, delimiter=';'))
        metadata_writer.writerow(
            [
                "VideoLabel",
                "Video",
                "Race",
                "Kind",
                "Requested Frame",
                "Requested Timecode",
                "Actual Frame",
                "Actual Timecode",
                "Score Layout",
                "Bundle Path",
                "Anchor Path",
            ]
        )
    else:
        csv_context = None
        csv_writer = NullCsvWriter()
        metadata_context = None
        metadata_writer = None

    total_source_seconds = 0.0
    for video_path in video_paths:
        probe = cv2.VideoCapture(video_path)
        if probe.isOpened():
            fps = probe.get(cv2.CAP_PROP_FPS) or 1
            frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            total_source_seconds += frames / max(fps, 1)
        probe.release()
    LOGGER.log("[Extract - Settings]", f"GPU acceleration: {GPU_RUNTIME['backend']} | Initial scan workers: {INITIAL_SCAN_WORKERS}", color_name="cyan")

    try:
        total_videos = len(video_paths)
        total_score_screens_found = 0
        total_track_screens_found = 0
        total_race_numbers_found = 0
        per_video_summaries = []
        for video_index, video_path in enumerate(video_paths, start=1):
            video_start = time.perf_counter()
            video_stats = defaultdict(float)
            video_stats["template_load_s"] = template_load_time_s
            capture_poisoned = False

            probe = cv2.VideoCapture(video_path)
            if not probe.isOpened():
                LOGGER.log(f"[Video {video_index}/{total_videos} - Start]", f"Could not open video: {video_path}", color_name="red")
                continue
            nominal_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            nominal_fps = probe.get(cv2.CAP_PROP_FPS) or 1
            nominal_duration_s = nominal_total_frames / max(nominal_fps, 1)
            corrupt_check_status = "checked"
            preflight_result = video_io.sample_video_readability(video_path, nominal_total_frames, stats=video_stats)
            corrupt_check_status = str(preflight_result.get("status", "checked"))
            probe.release()
            if preflight_result is not None and preflight_result.get("is_suspect"):
                LOGGER.log(
                    f"[Video {video_index}/{total_videos} - Start]",
                    f"Corrupt preflight flagged file: {preflight_result.get('reason', 'sample probe failed')}",
                    color_name="yellow",
                )
            processing_video_path = video_io.repair_video_if_needed(
                video_path,
                nominal_total_frames,
                preflight_result,
                duration_s=nominal_duration_s,
                stats=video_stats,
            )

            processing_readable_frames = nominal_total_frames

            cap = cv2.VideoCapture(processing_video_path)
            if not cap.isOpened():
                LOGGER.log(
                    f"[Video {video_index}/{total_videos} - Start]",
                    f"Could not open video: {processing_video_path}",
                    color_name="red",
                )
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = int(3 * max(1, int(fps)))
            score_candidates = []
            runtime_state = {
                "last_track_frame": 0,
                "last_race_frame": 0,
                "next_race_number": 1,
                "capture": cap,
            }
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processing_is_suspect = bool(preflight_result and preflight_result.get("is_suspect") and processing_video_path == video_path)
            scan_worker_count = 1 if processing_is_suspect else INITIAL_SCAN_WORKERS
            if processing_is_suspect:
                LOGGER.log(
                    f"[Video {video_index}/{total_videos} - Start]",
                    "Using conservative single-process scan for suspect video",
                    color_name="yellow",
                )
            sample_frames = np.linspace(0, total_frames - 1, 19).astype(int)
            scales = []
            video_name = os.path.basename(processing_video_path)
            video_label = build_video_identity(Path(video_path), input_root=folder_path, include_subfolders=include_subfolders)
            source_display_name = relative_video_path(Path(video_path), folder_path) if include_subfolders else os.path.basename(video_path)
            LOGGER.log(
                f"[Video {video_index}/{total_videos} - Start]",
                f"{source_display_name if include_subfolders else video_name} | Source length: {format_duration(total_frames / max(fps, 1))}",
                color_name="cyan",
            )
            stage_start = time.perf_counter()
            eof_guard_frames = max(frame_skip, int(fps * INITIAL_SCAN_EOF_GUARD_SECONDS))
            for frame_num in sample_frames:
                video_io.seek_to_frame(cap, frame_num, video_stats)
                read_timeout_s = (
                    INITIAL_SCAN_EOF_READ_TIMEOUT_SECONDS
                    if total_frames - int(frame_num) <= eof_guard_frames
                    else None
                )
                if read_timeout_s is None:
                    ret, frame = video_io.read_video_frame(cap, video_stats)
                    timed_out = False
                else:
                    ret, frame, timed_out = video_io.read_video_frame_with_timeout(cap, video_stats, read_timeout_s)
                if timed_out:
                    LOGGER.log(
                        f"[Video {video_index}/{total_videos} - Start]",
                        (
                            f"Aborting video after frame-read stall during scaling scan "
                            f"(requested frame {int(frame_num)}/{total_frames})"
                        ),
                        color_name="yellow",
                    )
                    capture_poisoned = True
                    break
                if not ret:
                    continue
                scale_x, scale_y, left, top, crop_width, crop_height = determine_scaling(frame)
                scales.append((scale_x, scale_y, left, top, crop_width, crop_height))
            video_io.add_timing(video_stats, "scaling_scan_s", stage_start)

            if not scales:
                if capture_poisoned:
                    LOGGER.log(
                        f"[Video {video_index}/{total_videos} - Start]",
                        f"Skipping poisoned capture after read timeout: {processing_video_path}",
                        color_name="yellow",
                    )
                else:
                    LOGGER.log(
                        f"[Video {video_index}/{total_videos} - Start]",
                        f"No valid frames found for scaling: {processing_video_path}",
                        color_name="red",
                    )
                    cap.release()
                continue

            median_scale_x = np.median([s[0] for s in scales])
            median_scale_y = np.median([s[1] for s in scales])
            median_left = int(np.median([s[2] for s in scales]))
            median_top = int(np.median([s[3] for s in scales]))
            median_crop_width = int(np.median([s[4] for s in scales]))
            median_crop_height = int(np.median([s[5] for s in scales]))

            video_io.seek_to_frame(cap, 0, video_stats)
            detection_segment_tasks = initial_scan.build_detection_segment_tasks(
                processing_video_path,
                video_label,
                source_display_name if include_subfolders else os.path.basename(video_path),
                total_frames,
                fps,
                templates,
                median_scale_x,
                median_scale_y,
                median_left,
                median_top,
                median_crop_width,
                median_crop_height,
                scan_worker_count,
                APP_CONFIG.write_debug_csv,
                INITIAL_SCAN_SEGMENT_OVERLAP_FRAMES,
                INITIAL_SCAN_MIN_SEGMENT_FRAMES,
                INITIAL_SCAN_WINDOW_STEPS,
                INITIAL_SCAN_PROGRESS_REPORT_SECONDS,
            )

            LOGGER.log(f"[Video {video_index}/{total_videos} - Scan - Phase Start]", "", color_name="cyan")
            scan_progress = None
            scan_track_count = 0
            scan_race_count = 0
            if not detection_segment_tasks:
                frame_count = 0
                stage_start = time.perf_counter()
                scan_progress = ProgressPrinter(
                    f"[Video {video_index}/{total_videos} - Scan]",
                    total_frames,
                    percent_step=5,
                    min_interval_s=2.0,
                )
                scan_progress.update(0)
                while cap.isOpened() and frame_count < total_frames:
                    remaining_frames = max(0, total_frames - frame_count)
                    if (
                        total_frames > 0
                        and frame_count / total_frames >= INITIAL_SCAN_EOF_GUARD_PROGRESS
                        and remaining_frames <= eof_guard_frames
                    ):
                        LOGGER.log(
                            f"[Video {video_index}/{total_videos} - Scan]",
                            (
                                f"Stopping scan near EOF to avoid decoder stalls "
                                f"({remaining_frames} frames remaining, "
                                f"{remaining_frames / max(fps, 1):.1f}s tail)"
                            ),
                            color_name="yellow",
                        )
                        frame_count = total_frames
                        break
                    window_interrupted = False

                    for _ in range(INITIAL_SCAN_WINDOW_STEPS):
                        read_timeout_s = (
                            INITIAL_SCAN_EOF_READ_TIMEOUT_SECONDS
                            if remaining_frames <= eof_guard_frames
                            else None
                        )
                        if read_timeout_s is None:
                            ret, frame = video_io.read_video_frame(cap, video_stats)
                            timed_out = False
                        else:
                            ret, frame, timed_out = video_io.read_video_frame_with_timeout(cap, video_stats, read_timeout_s)
                        if timed_out:
                            LOGGER.log(
                                f"[Video {video_index}/{total_videos} - Scan]",
                                (
                                    f"Aborting video after frame-read stall near EOF "
                                    f"({remaining_frames} frames remaining, "
                                    f"{remaining_frames / max(fps, 1):.1f}s tail)"
                                ),
                                color_name="yellow",
                            )
                            frame_count = total_frames
                            window_interrupted = True
                            capture_poisoned = True
                            break
                        if not ret:
                            window_interrupted = True
                            break

                        frames_to_skip = initial_scan.process_frame(
                            frame,
                            frame_count,
                            processing_video_path,
                            video_label,
                            source_display_name if include_subfolders else os.path.basename(video_path),
                            templates,
                            fps,
                            csv_writer,
                            median_scale_x,
                            median_scale_y,
                            median_left,
                            median_top,
                            median_crop_width,
                            median_crop_height,
                            video_stats,
                            runtime_state,
                            score_candidates,
                            metadata_writer,
                        )
                        if frames_to_skip > 0:
                            frame_count += frames_to_skip + frame_skip
                            if frame_count < total_frames:
                                video_io.seek_to_frame(cap, frame_count, video_stats)
                            window_interrupted = True
                            scan_progress.update(min(frame_count, total_frames), f"Score candidates: {len(score_candidates)}")
                            break

                        if not video_io.advance_frames_by_grab(cap, frame_skip - 1, video_stats):
                            window_interrupted = True
                            frame_count = total_frames
                            break

                        frame_count += frame_skip
                        if frame_count >= total_frames:
                            window_interrupted = True
                            break

                        scan_progress.update(frame_count, f"Score candidates: {len(score_candidates)}")

                    if window_interrupted and frame_count >= total_frames:
                        break
                if scan_progress.last_percent < 100:
                    scan_progress.update(total_frames, f"Score candidates: {len(score_candidates)}")
                video_io.add_timing(video_stats, "main_scan_loop_s", stage_start)
                pre_pass2_counts = count_exported_detection_files(processing_video_path if not include_subfolders else video_label)
                scan_track_count = pre_pass2_counts["track"]
                scan_race_count = pre_pass2_counts["race"]
            else:
                stage_start = time.perf_counter()
                scan_progress = ProgressPrinter(
                    f"[Video {video_index}/{total_videos} - Scan]",
                    total_frames,
                    percent_step=5,
                    min_interval_s=1.0,
                )
                scan_progress.update(0)
                segment_results = initial_scan.run_parallel_detection_segments(detection_segment_tasks, scan_progress)
                video_io.add_timing(video_stats, "main_scan_loop_s", stage_start)

                merged_score_detections = []
                merged_track_detections = []
                merged_race_detections = []
                merged_debug_rows = []

                for result in sorted(segment_results, key=lambda item: item["segment_index"]):
                    for key, value in result["stats"].items():
                        video_stats[key] += value
                    merged_score_detections.extend(result["score_detections"])
                    merged_track_detections.extend(result["track_detections"])
                    merged_race_detections.extend(result["race_detections"])
                    merged_debug_rows.extend(result["debug_rows"])

                min_gap_frames = int(fps * 20)
                merged_score_detections = initial_scan.merge_nearby_detections(merged_score_detections, min_gap_frames)
                merged_track_detections = initial_scan.merge_nearby_detections(merged_track_detections, min_gap_frames)
                merged_race_detections = initial_scan.merge_nearby_detections(merged_race_detections, min_gap_frames)

                score_frame_numbers = [item["frame_number"] for item in merged_score_detections]
                score_candidates = [
                    {"race_number": index + 1, "frame_number": item["frame_number"]}
                    for index, item in enumerate(merged_score_detections)
                ]

                if APP_CONFIG.write_debug_csv and merged_debug_rows:
                    csv_writer.writerows(merged_debug_rows)

                auxiliary_detections = [
                    {"kind": "track", **item}
                    for item in merged_track_detections
                ]
                auxiliary_detections.extend({"kind": "race", **item} for item in merged_race_detections)
                auxiliary_detections.sort(key=lambda item: item["frame_number"])
                if auxiliary_detections:
                    LOGGER.log(f"[Video {video_index}/{total_videos} - Scan - Confirmed Results]", "", color_name="cyan")
                scan_track_count = len(merged_track_detections)
                scan_race_count = len(merged_race_detections)
                initial_scan.save_auxiliary_detection_frames(
                    cap,
                    processing_video_path,
                    video_label,
                    source_display_name if include_subfolders else os.path.basename(video_path),
                    auxiliary_detections,
                    score_frame_numbers,
                    median_left,
                    median_top,
                    median_crop_width,
                    median_crop_height,
                    fps,
                    video_stats,
                    metadata_writer,
                )
            if capture_poisoned:
                LOGGER.log(
                    f"[Video {video_index}/{total_videos} - Complete]",
                    f"{video_name} | Aborted after timed-out read to avoid reusing a poisoned decoder",
                    color_name="yellow",
                )
                video_stats["video_total_s"] = time.perf_counter() - video_start
                continue
            scan_duration = float(video_stats.get("main_scan_loop_s", 0.0))
            scan_speed = total_frames / scan_duration if scan_duration > 0 else 0.0
            scan_summary_lines = [
                f"Duration: {format_duration(scan_duration)}",
                f"Source length: {format_duration(total_frames / max(fps, 1))}",
                f"Frames scanned: {total_frames:,}",
                f"Scan speed: {scan_speed:,.0f} frames/s",
                f"Track screens found: {scan_track_count}",
                f"Race numbers found: {scan_race_count}",
                f"Total score screens queued: {len(score_candidates)}",
            ]
            if scan_progress is not None:
                scan_summary_lines.extend(scan_progress.peak_lines())
            LOGGER.summary_block(
                f"[Video {video_index}/{total_videos} - Scan - Phase Complete]",
                scan_summary_lines,
                color_name="green",
            )
            LOGGER.log(f"[Video {video_index}/{total_videos} - Total Score Screen - Phase Start]", "", color_name="cyan")
            total_score_progress = ProgressPrinter(
                f"[Video {video_index}/{total_videos} - Total Score Screen]",
                max(1, len(score_candidates)),
                percent_step=5,
                min_interval_s=2.0,
            )
            total_score_progress.update(0)
            process_score_candidates(
                processing_video_path,
                video_label,
                source_display_name if include_subfolders else os.path.basename(video_path),
                score_candidates,
                templates,
                fps,
                csv_writer,
                median_scale_x,
                median_scale_y,
                median_left,
                median_top,
                median_crop_width,
                median_crop_height,
                video_stats,
                metadata_writer,
                video_index=video_index,
                total_videos=total_videos,
                progress=total_score_progress,
            )
            exported_counts = count_exported_detection_files(processing_video_path if not include_subfolders else video_label)
            total_score_screens_found += exported_counts["score"]
            total_track_screens_found += exported_counts["track"]
            total_race_numbers_found += exported_counts["race"]
            if not capture_poisoned:
                cap.release()
            video_stats["video_total_s"] = time.perf_counter() - video_start
            if total_score_progress.last_percent < 100:
                total_score_progress.update(len(score_candidates))
            LOGGER.summary_block(
                f"[Video {video_index}/{total_videos} - Total Score Screen - Phase Complete]",
                [
                    f"Duration: {format_duration(video_stats.get('score_candidate_pass_s', 0.0))}",
                    f"Total score screens found: {exported_counts['total']}",
                    *total_score_progress.peak_lines(),
                ] if total_score_progress.peak_lines() else [
                    f"Duration: {format_duration(video_stats.get('score_candidate_pass_s', 0.0))}",
                    f"Total score screens found: {exported_counts['total']}",
                ],
                color_name="green",
            )
            LOGGER.log(
                f"[Video {video_index}/{total_videos} - Complete]",
                f"{video_name} | Processing duration: {format_duration(video_stats['video_total_s'])} | "
                f"Source length: {format_duration(total_frames / max(fps, 1))} | Track screens: {exported_counts['track']} | "
                f"Race numbers: {exported_counts['race']} | Total score screens: {exported_counts['total']}",
                color_name="green",
            )
            if video_index < total_videos:
                LOGGER.blank_lines(2)
            per_video_summaries.append(
                {
                    "video_name": video_name,
                    "source_length_s": total_frames / max(fps, 1),
                    "processing_duration_s": float(video_stats["video_total_s"]),
                    "scan_duration_s": float(video_stats.get("main_scan_loop_s", 0.0)),
                    "total_score_duration_s": float(video_stats.get("score_candidate_pass_s", 0.0)),
                    "corrupt_check_duration_s": float(video_stats.get("corrupt_check_duration_s", 0.0)),
                    "corrupt_check_status": corrupt_check_status,
                    "repair_duration_s": float(video_stats.get("repair_duration_s", 0.0)),
                    "repair_created": bool(video_stats.get("repair_created", 0)),
                    "track_screens": exported_counts["track"],
                    "race_numbers": exported_counts["race"],
                    "total_score_screens": exported_counts["total"],
                }
            )
    finally:
        if csv_context is not None:
            csv_context.close()
        if metadata_context is not None:
            metadata_context.close()

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - phase_start_time

    # Convert elapsed time to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # Print the elapsed time in mm:ss format
    extract_summary = {
        "duration_s": elapsed_time,
        "total_source_seconds": total_source_seconds,
        "track_screens": total_track_screens_found,
        "race_numbers": total_race_numbers_found,
        "total_score_screens": total_score_screens_found,
        "corrupt_check_duration_s": sum(float(item.get("corrupt_check_duration_s", 0.0)) for item in per_video_summaries),
        "repair_duration_s": sum(float(item.get("repair_duration_s", 0.0)) for item in per_video_summaries),
        "repair_count": sum(1 for item in per_video_summaries if item.get("repair_created")),
        "per_video_summaries": per_video_summaries,
    }
    extract_lines = [
        f"Duration: {format_duration(elapsed_time)}",
        f"Source video total: {format_duration(total_source_seconds)}",
        f"Track screens found: {total_track_screens_found}",
        f"Race numbers found: {total_race_numbers_found}",
        f"Total score screens found: {total_score_screens_found}",
    ]
    extract_lines.extend(LOGGER.peak_lines())
    LOGGER.summary_block("[Extract - Phase Complete]", extract_lines, color_name="green")
    if return_frame_cache:
        return {
            "frame_bundle_cache": {key: value[:] for key, value in CONSENSUS_FRAME_CACHE.items()},
            "summary": extract_summary,
        }
    return {"summary": extract_summary}


def main():
    parser = argparse.ArgumentParser(description="Extract Mario Kart 8 race and score screens")
    parser.add_argument("--video", help="Process only a specific video filename")
    args = parser.parse_args()
    extract_frames(return_frame_cache=False, selected_videos=[args.video] if args.video else None)

if __name__ == "__main__":
    main()
