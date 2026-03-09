import cv2
import numpy as np
import os
import argparse
from glob import glob
import csv
import time
import multiprocessing as mp
from PIL import Image, ImageEnhance, ImageFilter
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from queue import Empty, Queue
from app_runtime import detect_gpu_runtime, load_app_config
from console_logging import LOGGER

# Record the start time
start_run_time = time.time()
APP_CONFIG = load_app_config()

# Global parameter for frame skip value
FRAME_SKIP = int(3 * 30)  # Skip 3 seconds (assuming 30 FPS)
LastTrackNameFrame = 0
LastRaceNumberFrame = 0
RaceCount = 1
SCORE_ANALYSIS_WORKERS = APP_CONFIG.score_analysis_workers
PASS1_SCAN_WORKERS = APP_CONFIG.pass1_scan_workers
PASS1_WINDOW_STEPS = 2
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
VERTICAL_DILATE_KERNEL = np.ones((2, 1), np.uint8)
PASS1_SEGMENT_OVERLAP_FRAMES = APP_CONFIG.pass1_segment_overlap_frames
PASS1_MIN_SEGMENT_FRAMES = APP_CONFIG.pass1_min_segment_frames
PASS1_PROGRESS_REPORT_SECONDS = 2.0
GPU_RUNTIME = detect_gpu_runtime(APP_CONFIG)

# Pass-two ROIs are evaluated on a fixed 1280x720 working image, so these bounds
# can be precomputed once instead of rebuilt for every frame.
PASS2_SCORE_ROI = (290, 32, 102, 660)
PASS2_12TH_ROI = (313, 632, 651, 88)
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


def update_segment_progress(progress_queue, segment_index, frame_number, scan_start, scan_end, emit_start, emit_end, force=False):
    """Send best-effort in-segment progress updates from pass-one workers."""
    if progress_queue is None:
        return
    completed = max(0, min(max(frame_number, emit_start), emit_end) - emit_start)
    total = max(1, emit_end - emit_start)
    progress_queue.put(
        {
            "segment_index": segment_index,
            "completed_frames": completed,
            "total_frames": total,
            "force": force,
        }
    )


def add_timing(stats, key, start_time):
    """Accumulate elapsed time for a named timing bucket."""
    stats[key] += time.perf_counter() - start_time


def seek_to_frame(cap, frame_number, stats):
    """Seek to a frame and record the operation cost."""
    start_time = time.perf_counter()
    result = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    add_timing(stats, "seek_time_s", start_time)
    stats["seek_calls"] += 1
    return result


def read_video_frame(cap, stats):
    """Read a frame and record the operation cost."""
    start_time = time.perf_counter()
    ret, frame = cap.read()
    add_timing(stats, "read_time_s", start_time)
    stats["read_calls"] += 1
    return ret, frame


def grab_video_frame(cap, stats):
    """Advance one frame without decoding it for analysis."""
    start_time = time.perf_counter()
    ret = cap.grab()
    add_timing(stats, "grab_time_s", start_time)
    stats["grab_calls"] += 1
    return ret


def advance_frames_by_grab(cap, frames_to_advance, stats):
    """Advance by grabbing frames, avoiding full reads inside local scan windows."""
    for _ in range(max(0, frames_to_advance)):
        if not grab_video_frame(cap, stats):
            return False
    return True


def print_timing_summary(video_name, stats):
    """Print a user-facing summary of where time went."""
    total_time = float(stats.get("video_total_s", 0.0))
    if total_time <= 0:
        return

    major_buckets = [
        ("calibration", float(stats.get("scaling_scan_s", 0.0))),
        ("pass 1 scan", float(stats.get("main_scan_loop_s", 0.0))),
        ("pass 2 score selection", float(stats.get("score_candidate_pass_s", 0.0))),
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


def print_detection_event(video_path, label, race_number, timecode):
    """Print concise user-facing detection updates."""
    LOGGER.log("", f"Race {race_number:03} | {label} found at source {timecode}", color_name="green")


def actual_frame_after_read(cap):
    """Best-effort actual decoded frame index after a successful read()."""
    return max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)


def log_exported_frame(metadata_writer, video_path, race_number, kind, requested_frame, actual_frame, fps):
    """Record requested and actual decoded frames for each exported screenshot."""
    if metadata_writer is None:
        return
    metadata_writer.writerow(
        [
            os.path.basename(video_path),
            f"{race_number:03}",
            kind,
            int(requested_frame),
            frame_to_timecode(requested_frame, fps),
            int(actual_frame),
            frame_to_timecode(actual_frame, fps),
        ]
    )


def collect_consensus_frames(video_path, center_frame, fps, left, top, crop_width, crop_height, bundle_kind):
    """Collect nearby upscaled frames for in-memory OCR consensus during --all runs."""
    radius = max(0, APP_CONFIG.ocr_consensus_frames // 2)
    cache_key = (os.path.splitext(os.path.basename(video_path))[0], int(bundle_kind[0]), bundle_kind[1])
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


def collect_consensus_frames_from_capture(local_cap, center_frame, left, top, crop_width, crop_height):
    """Collect nearby upscaled frames from an already-open capture for in-memory OCR consensus."""
    radius = max(0, APP_CONFIG.ocr_consensus_frames // 2)
    total_frames = int(local_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bundled_frames = []
    start_frame = max(0, center_frame - radius)
    end_frame = min(total_frames, center_frame + radius + 1)
    local_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _frame_number in range(start_frame, end_frame):
        ret, frame = local_cap.read()
        if not ret:
            continue
        bundled_frames.append(crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT))
    return bundled_frames


def enhance_export_frame(upscaled_image, scale_x, scale_y):
    if scale_x > 1.3 and scale_y >= 1.3:
        if isinstance(upscaled_image, np.ndarray):
            upscaled_image = Image.fromarray(upscaled_image)
        contrast_enhancer = ImageEnhance.Contrast(upscaled_image)
        high_contrast_image = contrast_enhancer.enhance(1.70)
        sharpness_enhancer = ImageEnhance.Sharpness(high_contrast_image)
        sharpened_image = sharpness_enhancer.enhance(1.23)
        return np.array(sharpened_image)
    return upscaled_image


def capture_export_frame(local_cap, target_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats):
    seek_to_frame(local_cap, target_frame, stats)
    ret, frame = read_video_frame(local_cap, stats)
    if not ret:
        return None, None
    actual_frame = actual_frame_after_read(local_cap)
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
    return actual_frame, enhance_export_frame(upscaled_image, scale_x, scale_y)


def analyze_score_window(video_path, frame_number, fps, templates, csv_writer, scale_x, scale_y, left, top,
                         crop_width, crop_height, stats):
    """Analyze a detected score-screen window and return the selected frame numbers."""
    score_detail_start = time.perf_counter()
    start_frame = frame_number - int(3 * fps)
    end_frame = frame_number + int(13 * fps)
    race_score_frame = 0
    total_score_frame = 0
    player12 = 0
    check_player_12 = 0

    detail_frame_number = start_frame
    seek_to_frame(cap, detail_frame_number, stats)
    template_binary, alpha_mask = templates[0]

    while detail_frame_number < end_frame:
        ret, frame = read_video_frame(cap, stats)
        if not ret:
            break

        frame_prepare_start = time.perf_counter()
        gray_image, crop_upscale_time, grayscale_time = crop_to_gray_and_upscale_image(
            frame,
            left,
            top,
            crop_width,
            crop_height,
            TARGET_WIDTH,
            TARGET_HEIGHT,
        )
        stats["score_detail_crop_upscale_s"] += crop_upscale_time
        stats["score_detail_grayscale_s"] += grayscale_time

        stage_start = time.perf_counter()
        roi_x, roi_y, roi_width, roi_height = PASS2_SCORE_ROI
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        add_timing(stats, "score_detail_score_roi_extract_s", stage_start)

        stage_start = time.perf_counter()
        processed_roi = preprocess_roi(roi, 0)
        add_timing(stats, "score_detail_score_preprocess_s", stage_start)
        add_timing(stats, "score_detail_frame_prepare_s", frame_prepare_start)

        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            detail_frame_number += 1
            timecode = frame_to_timecode(detail_frame_number, fps)
            csv_writer.writerow([os.path.basename(video_path), "Score", detail_frame_number, 0, timecode])
            if race_score_frame != 0:
                total_score_frame = detail_frame_number - int(2.7 * fps)
                break
            continue

        stage_start = time.perf_counter()
        max_val = match_template(processed_roi, template_binary, alpha_mask)
        add_timing(stats, "score_detail_match_score_s", stage_start)

        timecode = frame_to_timecode(detail_frame_number, fps)
        csv_writer.writerow([os.path.basename(video_path), "Score", detail_frame_number, max_val, timecode])

        if max_val > 0.3 and not np.isinf(max_val) and race_score_frame == 0:
            race_score_frame = detail_frame_number + int(0.6 * fps)
            check_player_12 = 1
            continue

        if max_val > 0.3 and not np.isinf(max_val) and check_player_12 == 1:
            stage_start = time.perf_counter()
            roi_x2, roi_y2, roi_width2, roi_height2 = PASS2_12TH_ROI
            roi2 = gray_image[roi_y2:roi_y2 + roi_height2, roi_x2:roi_x2 + roi_width2]
            add_timing(stats, "score_detail_12th_roi_extract_s", stage_start)

            stage_start = time.perf_counter()
            processed_roi2 = preprocess_roi(roi2, 0)
            add_timing(stats, "score_detail_12th_preprocess_s", stage_start)

            template_binary2, alpha_mask2 = templates[3]
            stage_start = time.perf_counter()
            max_val2 = match_template(processed_roi2, template_binary2, alpha_mask2)
            add_timing(stats, "score_detail_match_12th_s", stage_start)

            if max_val2 > 0.4 and not np.isinf(max_val2):
                player12 = 1

            if player12 == 1 and max_val2 < 0.1:
                race_score_frame = detail_frame_number + int(16)
                detail_frame_number += int(3.9 * fps)
                seek_to_frame(cap, detail_frame_number, stats)
                check_player_12 = 2
                continue

        if max_val <= 0 and race_score_frame != 0:
            total_score_frame = detail_frame_number - int(2.7 * fps)
            break

        detail_frame_number += 1

    add_timing(stats, "score_detail_total_s", score_detail_start)
    return race_score_frame, total_score_frame


def analyze_score_window_task(task):
    """Analyze a score candidate using an isolated VideoCapture for parallel pass-2 work."""
    video_path = task["video_path"]
    frame_number = task["frame_number"]
    fps = task["fps"]
    templates = task["templates"]
    left = task["left"]
    top = task["top"]
    crop_width = task["crop_width"]
    crop_height = task["crop_height"]
    scale_x = task["scale_x"]
    scale_y = task["scale_y"]

    start_frame = frame_number - int(3 * fps)
    end_frame = frame_number + int(13 * fps)
    race_score_frame = 0
    total_score_frame = 0
    player12 = 0
    check_player_12 = 0
    debug_rows = []
    stats = defaultdict(float)

    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        return {"candidate": task, "race_score_frame": 0, "total_score_frame": 0, "debug_rows": [], "stats": stats}

    detail_frame_number = start_frame
    seek_to_frame(local_cap, detail_frame_number, stats)
    template_binary, alpha_mask = templates[0]

    while detail_frame_number < end_frame:
        ret, frame = read_video_frame(local_cap, stats)
        if not ret:
            break

        frame_prepare_start = time.perf_counter()
        gray_image, crop_upscale_time, grayscale_time = crop_to_gray_and_upscale_image(
            frame,
            left,
            top,
            crop_width,
            crop_height,
            TARGET_WIDTH,
            TARGET_HEIGHT,
        )
        stats["score_detail_crop_upscale_s"] += crop_upscale_time
        stats["score_detail_grayscale_s"] += grayscale_time

        stage_start = time.perf_counter()
        roi_x, roi_y, roi_width, roi_height = PASS2_SCORE_ROI
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        add_timing(stats, "score_detail_score_roi_extract_s", stage_start)

        stage_start = time.perf_counter()
        processed_roi = preprocess_roi(roi, 0)
        add_timing(stats, "score_detail_score_preprocess_s", stage_start)
        add_timing(stats, "score_detail_frame_prepare_s", frame_prepare_start)

        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            detail_frame_number += 1
            timecode = frame_to_timecode(detail_frame_number, fps)
            debug_rows.append([os.path.basename(video_path), "Score", detail_frame_number, 0, timecode])
            if race_score_frame != 0:
                total_score_frame = detail_frame_number - int(2.7 * fps)
                break
            continue

        stage_start = time.perf_counter()
        max_val = match_template(processed_roi, template_binary, alpha_mask)
        add_timing(stats, "score_detail_match_score_s", stage_start)
        timecode = frame_to_timecode(detail_frame_number, fps)
        debug_rows.append([os.path.basename(video_path), "Score", detail_frame_number, max_val, timecode])

        if max_val > 0.3 and not np.isinf(max_val) and race_score_frame == 0:
            race_score_frame = detail_frame_number + int(0.6 * fps)
            check_player_12 = 1
            continue

        if max_val > 0.3 and not np.isinf(max_val) and check_player_12 == 1:
            stage_start = time.perf_counter()
            roi_x2, roi_y2, roi_width2, roi_height2 = PASS2_12TH_ROI
            roi2 = gray_image[roi_y2:roi_y2 + roi_height2, roi_x2:roi_x2 + roi_width2]
            add_timing(stats, "score_detail_12th_roi_extract_s", stage_start)

            stage_start = time.perf_counter()
            processed_roi2 = preprocess_roi(roi2, 0)
            add_timing(stats, "score_detail_12th_preprocess_s", stage_start)
            template_binary2, alpha_mask2 = templates[3]

            stage_start = time.perf_counter()
            max_val2 = match_template(processed_roi2, template_binary2, alpha_mask2)
            add_timing(stats, "score_detail_match_12th_s", stage_start)

            if max_val2 > 0.4 and not np.isinf(max_val2):
                player12 = 1

            if player12 == 1 and max_val2 < 0.1:
                race_score_frame = detail_frame_number + int(16)
                detail_frame_number += int(3.9 * fps)
                seek_to_frame(local_cap, detail_frame_number, stats)
                check_player_12 = 2
                continue

        if max_val <= 0 and race_score_frame != 0:
            total_score_frame = detail_frame_number - int(2.7 * fps)
            break

        detail_frame_number += 1

    race_score_image = None
    total_score_image = None
    actual_race_score_frame = None
    actual_total_score_frame = None
    race_consensus_frames = []
    total_consensus_frames = []
    if race_score_frame > 0 and total_score_frame > 0:
        export_stage_start = time.perf_counter()
        actual_race_score_frame, race_score_image = capture_export_frame(
            local_cap, race_score_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats
        )
        if actual_race_score_frame is not None:
            race_consensus_frames = collect_consensus_frames_from_capture(
                local_cap, actual_race_score_frame, left, top, crop_width, crop_height
            )
        actual_total_score_frame, total_score_image = capture_export_frame(
            local_cap, total_score_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats
        )
        if actual_total_score_frame is not None:
            total_consensus_frames = collect_consensus_frames_from_capture(
                local_cap, actual_total_score_frame, left, top, crop_width, crop_height
            )
        add_timing(stats, "output_frame_capture_s", export_stage_start)

    local_cap.release()
    return {
        "candidate": task,
        "race_score_frame": race_score_frame,
        "total_score_frame": total_score_frame,
        "actual_race_score_frame": actual_race_score_frame,
        "actual_total_score_frame": actual_total_score_frame,
        "race_score_image": race_score_image,
        "total_score_image": total_score_image,
        "race_consensus_frames": race_consensus_frames,
        "total_consensus_frames": total_consensus_frames,
        "debug_rows": debug_rows,
        "stats": stats,
    }


def save_score_frames(video_path, race_number, race_score_frame, total_score_frame, actual_race_score_frame,
                      actual_total_score_frame, race_score_image, total_score_image, race_consensus_frames,
                      total_consensus_frames, fps, stats, metadata_writer):
    """Save the selected race score and total score frames from worker-prepared output."""
    if race_score_image is None or total_score_image is None:
        return False
    script_dir = os.path.dirname(__file__)
    output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')
    os.makedirs(output_folder, exist_ok=True)
    frame_filename = os.path.join(
        output_folder,
        f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{race_number:03}+2RaceScore.png"
    )
    cv2.imwrite(frame_filename, race_score_image)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    if race_consensus_frames:
        CONSENSUS_FRAME_CACHE[(video_stem, int(race_number), "RaceScore")] = list(race_consensus_frames)
    log_exported_frame(
        metadata_writer, video_path, race_number, "RaceScore", race_score_frame, actual_race_score_frame, fps
    )

    frame_filename = os.path.join(
        output_folder,
        f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{race_number:03}+3TotalScore.png"
    )
    cv2.imwrite(frame_filename, total_score_image)
    if total_consensus_frames:
        CONSENSUS_FRAME_CACHE[(video_stem, int(race_number), "TotalScore")] = list(total_consensus_frames)
    log_exported_frame(
        metadata_writer, video_path, race_number, "TotalScore", total_score_frame, actual_total_score_frame, fps
    )
    return True


def process_score_candidates(video_path, score_candidates, templates, fps, csv_writer, scale_x, scale_y, left, top,
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
            result = analyze_score_window_task(task)
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
                executor.submit(analyze_score_window_task, task): task
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

    for result in sorted(results, key=lambda item: item["candidate"]["race_number"]):
        for key, value in result["stats"].items():
            stats[key] += value
        for row in result["debug_rows"]:
            csv_writer.writerow(row)
        if result["race_score_frame"] <= 0 or result["total_score_frame"] <= 0:
            continue
        save_score_frames(
            video_path,
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
            stats,
            metadata_writer,
        )
    add_timing(stats, "score_candidate_pass_s", stage_start)

def calculate_sum_intensity(gray_image):
    """Calculate the sum of pixel intensities for rows and columns."""
    sum_row_intensity = np.sum(gray_image, axis=1)
    sum_col_intensity = np.sum(gray_image, axis=0)
    return sum_row_intensity, sum_col_intensity

def find_borders(sum_row_intensity, sum_col_intensity, threshold=25000):
    """Find the borders of the active game area based on intensity sums."""
    top = next((i for i, val in enumerate(sum_row_intensity) if val > threshold), 0)
    bottom = next((i for i, val in enumerate(reversed(sum_row_intensity)) if val > threshold), 0)
    bottom = len(sum_row_intensity) - bottom
    left = next((i for i, val in enumerate(sum_col_intensity) if val > threshold), 0)
    right = next((i for i, val in enumerate(reversed(sum_col_intensity)) if val > threshold), 0)
    right = len(sum_col_intensity) - right
    return top, left, bottom, right

def determine_scaling(image):
    """Determine the scaling factors and crop dimensions for the image."""
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sum_row_intensity, sum_col_intensity = calculate_sum_intensity(gray_frame)
    top, left, bottom, right = find_borders(sum_row_intensity, sum_col_intensity, threshold=15000)

    height, width = image.shape[:2]
    crop_width = right - left
    crop_height = bottom - top
    scale_x = 1280 / crop_width
    scale_y = 720 / crop_height

    return scale_x, scale_y, left, top, crop_width, crop_height


def gpu_resize(image, width, height, interpolation=cv2.INTER_LINEAR):
    if not GPU_RUNTIME["enabled"]:
        return cv2.resize(image, (width, height), interpolation=interpolation)
    if GPU_RUNTIME["backend"] == "cuda":
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(image)
            resized = cv2.cuda.resize(gpu_mat, (width, height), interpolation=interpolation)
            return resized.download()
        except Exception:
            return cv2.resize(image, (width, height), interpolation=interpolation)
    if GPU_RUNTIME["backend"] == "opencl":
        try:
            return cv2.resize(cv2.UMat(image), (width, height), interpolation=interpolation).get()
        except Exception:
            return cv2.resize(image, (width, height), interpolation=interpolation)
    try:
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(image)
        resized = cv2.cuda.resize(gpu_mat, (width, height), interpolation=interpolation)
        return resized.download()
    except Exception:
        return cv2.resize(image, (width, height), interpolation=interpolation)


def gpu_cvt_color(image, code):
    if not GPU_RUNTIME["enabled"]:
        return cv2.cvtColor(image, code)
    if GPU_RUNTIME["backend"] == "cuda":
        try:
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(image)
            converted = cv2.cuda.cvtColor(gpu_mat, code)
            return converted.download()
        except Exception:
            return cv2.cvtColor(image, code)
    if GPU_RUNTIME["backend"] == "opencl":
        try:
            return cv2.cvtColor(cv2.UMat(image), code).get()
        except Exception:
            return cv2.cvtColor(image, code)
    return cv2.cvtColor(image, code)


def match_template(processed_roi, template_binary, alpha_mask=None):
    if GPU_RUNTIME["enabled"] and alpha_mask is None and GPU_RUNTIME["backend"] == "cuda":
        try:
            matcher = cv2.cuda.createTemplateMatching(processed_roi.dtype, cv2.TM_CCOEFF_NORMED)
            roi_gpu = cv2.cuda_GpuMat()
            template_gpu = cv2.cuda_GpuMat()
            roi_gpu.upload(processed_roi)
            template_gpu.upload(template_binary)
            res = matcher.match(roi_gpu, template_gpu).download()
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return max_val
        except Exception:
            pass
    if GPU_RUNTIME["enabled"] and alpha_mask is None and GPU_RUNTIME["backend"] == "opencl":
        try:
            res = cv2.matchTemplate(cv2.UMat(processed_roi), cv2.UMat(template_binary), cv2.TM_CCOEFF_NORMED).get()
            _, max_val, _, _ = cv2.minMaxLoc(res)
            return max_val
        except Exception:
            pass

    if alpha_mask is None:
        res = cv2.matchTemplate(processed_roi, template_binary, cv2.TM_CCOEFF_NORMED)
    else:
        res = cv2.matchTemplate(processed_roi, template_binary, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val

def load_videos_from_folder(folder_path):
    """Load video file paths from the specified folder."""
    video_extensions = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"]
    video_paths = []
    for extension in video_extensions:
        video_paths.extend(glob(os.path.join(folder_path, extension)))
    return video_paths


def count_exported_detection_files(video_path):
    script_dir = os.path.dirname(__file__)
    output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    return {
        "track": len(glob(os.path.join(output_folder, f"{video_stem}+Race_*+0TrackName.png"))),
        "race": len(glob(os.path.join(output_folder, f"{video_stem}+Race_*+1RaceNumber.png"))),
        "score": len(glob(os.path.join(output_folder, f"{video_stem}+Race_*+2RaceScore.png"))),
        "total": len(glob(os.path.join(output_folder, f"{video_stem}+Race_*+3TotalScore.png"))),
    }

def crop_and_upscale_image(image, left, top, crop_width, crop_height, target_width, target_height):
    """Crop the image to the detected borders and upscale to the target size."""
    cropped_image = image[top:top + crop_height, left:left + crop_width]
    upscaled_image = gpu_resize(cropped_image, target_width, target_height, interpolation=cv2.INTER_LINEAR)
    return upscaled_image


def crop_to_gray_and_upscale_image(image, left, top, crop_width, crop_height, target_width, target_height):
    """Crop first, convert to grayscale, then resize to reduce pass-2 image work."""
    stage_start = time.perf_counter()
    cropped_image = image[top:top + crop_height, left:left + crop_width]
    gray_image = gpu_cvt_color(cropped_image, cv2.COLOR_BGR2GRAY)
    grayscale_time = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    upscaled_gray_image = gpu_resize(gray_image, target_width, target_height, interpolation=cv2.INTER_LINEAR)
    crop_upscale_time = time.perf_counter() - stage_start
    return upscaled_gray_image, crop_upscale_time, grayscale_time


def preprocess_roi(roi, process_type):
    """Preprocess the Region of Interest (ROI) based on the process type."""
    if process_type == 0:
        _, binary_section = cv2.threshold(roi, 180, 255, cv2.THRESH_BINARY)
        modified_binary_section = binary_section.copy()
        step_height = max(1, roi.shape[0] // 12)
        for i in range(12):
            sub_region_y_start = i * step_height
            sub_region_y_end = min((i + 1) * step_height, roi.shape[0])
            sub_region = binary_section[sub_region_y_start:sub_region_y_end, :]
            if sub_region.size == 0:
                continue
            white_pixels = cv2.countNonZero(sub_region)
            black_pixels = sub_region.size - white_pixels
            if white_pixels > black_pixels:
                _, binary_section_sub = cv2.threshold(sub_region, 120, 255, cv2.THRESH_BINARY)
                sub_region_copy = binary_section_sub.copy()
                sub_region_copy = cv2.dilate(sub_region_copy, VERTICAL_DILATE_KERNEL, iterations=1)
                sub_region_copy = cv2.bitwise_not(sub_region_copy)
                modified_binary_section[sub_region_y_start:sub_region_y_end, :] = sub_region_copy
            else:
                dilated_section = cv2.dilate(sub_region, VERTICAL_DILATE_KERNEL, iterations=1)
                modified_binary_section[sub_region_y_start:sub_region_y_end, :] = dilated_section
    else:
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        if len(blurred.shape) == 3 and blurred.shape[2] == 3:
            gray_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        else:
            gray_blurred = blurred
        thresholded = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        modified_binary_section = cv2.filter2D(thresholded, -1, kernel)
    return modified_binary_section

def frame_to_timecode(frame_number, fps):
    """Convert frame number to timecode in HH:MM:SS format."""
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def process_frame(frame, frame_number, video_path, templates, fps, csv_writer, scale_x, scale_y, left, top,
                  crop_width, crop_height, stats, score_candidates, metadata_writer):
    """Process a single video frame and apply template matching."""
    global LastTrackNameFrame
    global LastRaceNumberFrame
    global RaceCount
    global cap
    global FRAME_SKIP

    process_frame_start = time.perf_counter()

    # Crop and upscale the image using the calculated scaling factors
    stage_start = time.perf_counter()
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)


    # Convert the image to grayscale
    gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    add_timing(stats, "initial_frame_prepare_s", stage_start)

    for i in range(3):
        if i == 0:
            # Define the ROI for RaceScore
            roi_x, roi_y, roi_width, roi_height = 315, 57, 52, 610
            TargetColumn = "Score"
        elif i == 1:
            # Define the ROI for TrackName
            roi_x, roi_y, roi_width, roi_height = 141, 607, 183, 101
            TargetColumn = "TrackName"
        else:
            # Define the ROI for RaceNumber
            roi_x, roi_y, roi_width, roi_height = 640, 590, 144, 48
            TargetColumn = "RaceNumber"

        # Extend ROI by 25 pixels in each direction
        roi_x = max(roi_x - 25, 0)
        roi_y = max(roi_y - 25, 0)
        roi_width = min(roi_width + 50, gray_image.shape[1] - roi_x)
        roi_height = min(roi_height + 50, gray_image.shape[0] - roi_y)
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Preprocess the ROI
        stage_start = time.perf_counter()
        processed_roi = preprocess_roi(roi, i)
        add_timing(stats, "initial_roi_preprocess_s", stage_start)

        # Skip the frame if 97% or more of the pixels in the ROI are black
        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            timecode = frame_to_timecode(frame_number, fps)
            csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, 0, timecode])
            #we need to ensure all templates are checked before quitting for black screens.
            if i == 2:
                return 0
            else:
                continue

        template_binary, alpha_mask = templates[i]
        if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
            processed_roi = cv2.resize(processed_roi, (max(template_binary.shape[1], processed_roi.shape[1]),
                                                       max(template_binary.shape[0], processed_roi.shape[0])),
                                       interpolation=cv2.INTER_LINEAR)

        stage_start = time.perf_counter()
        max_val = match_template(processed_roi, template_binary, alpha_mask)
        add_timing(stats, "initial_match_s", stage_start)

        if i == 0 and max_val > 0.3 and not np.isinf(max_val):
            timecode = frame_to_timecode(frame_number, fps)
            csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, max_val, timecode])
            score_candidates.append({
                "race_number": RaceCount,
                "frame_number": frame_number,
            })

            #we can skip 20 seconds knowing a new game will not start within 20 seconds from end score screen.
            frames_to_skip = int(fps * 20)
            RaceCount += 1
            add_timing(stats, "process_frame_total_s", process_frame_start)
            return frames_to_skip
        elif i == 1 and max_val > 0.6 and not np.isinf(max_val):
            if LastTrackNameFrame < max(1, frame_number - int(fps * 20)):
                stage_start = time.perf_counter()
                seek_to_frame(cap, frame_number + int(fps * 1), stats)
                ret, frame = read_video_frame(cap, stats)
                if not ret:
                    break
                actual_track_frame = actual_frame_after_read(cap)
                upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
                LastTrackNameFrame = frame_number
                timecode = frame_to_timecode(frame_number, fps)
                print_detection_event(video_path, "track screen", RaceCount, timecode)
                script_dir = os.path.dirname(__file__)  # Directory of the script
                output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')
                os.makedirs(output_folder, exist_ok=True)
                frame_filename = os.path.join(output_folder,
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{RaceCount:03}+0TrackName.png")
                cv2.imwrite(frame_filename, upscaled_image)
                log_exported_frame(
                    metadata_writer,
                    video_path,
                    RaceCount,
                    "TrackName",
                    frame_number + int(fps * 1),
                    actual_track_frame,
                    fps,
                )
                seek_to_frame(cap, frame_number, stats)
                ret, frame = read_video_frame(cap, stats)
                if not ret:
                    break
                add_timing(stats, "output_frame_capture_s", stage_start)
            csv_writer.writerow([os.path.basename(video_path), "TrackName", frame_number, max_val, timecode])
            add_timing(stats, "process_frame_total_s", process_frame_start)
            return 0
        elif i == 2 and max_val > 0.6 and not np.isinf(max_val):
            if LastRaceNumberFrame < max(1, frame_number - int(fps * 20)):
                LastRaceNumberFrame = frame_number
                timecode = frame_to_timecode(frame_number, fps)
                print_detection_event(video_path, "race number", RaceCount, timecode)
                script_dir = os.path.dirname(__file__)  # Directory of the script
                output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')

                os.makedirs(output_folder, exist_ok=True)
                frame_filename = os.path.join(output_folder,
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{RaceCount:03}+1RaceNumber.png")
                cv2.imwrite(frame_filename, upscaled_image)
                log_exported_frame(
                    metadata_writer,
                    video_path,
                    RaceCount,
                    "RaceNumber",
                    frame_number,
                    frame_number,
                    fps,
                )
                stats["output_frame_capture_s"] += 0
            csv_writer.writerow([os.path.basename(video_path), "RaceNumber", frame_number, max_val, timecode])
            frames_to_skip = int(fps * 60)
            add_timing(stats, "process_frame_total_s", process_frame_start)
            return frames_to_skip

        if i == 0:
            timecode = frame_to_timecode(frame_number, fps)
            csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, max_val, timecode])
        elif i == 1:
            timecode = frame_to_timecode(frame_number, fps)
            csv_writer.writerow([os.path.basename(video_path), "TrackName", frame_number, max_val, timecode])
        else:
            timecode = frame_to_timecode(frame_number, fps)
            csv_writer.writerow([os.path.basename(video_path), "RaceNumber", frame_number, max_val, timecode])
    add_timing(stats, "process_frame_total_s", process_frame_start)
    return 0


def process_segment_frame(frame, frame_number, video_path, templates, fps, scale_x, scale_y, left, top,
                          crop_width, crop_height, stats, state, debug_rows, emit_results):
    """Process a single frame for parallel pass-one scanning without side effects."""
    process_frame_start = time.perf_counter()
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
    gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    stats["initial_frame_prepare_s"] += time.perf_counter() - process_frame_start

    for i in range(3):
        if i == 0:
            roi_x, roi_y, roi_width, roi_height = 315, 57, 52, 610
            target_column = "Score"
        elif i == 1:
            roi_x, roi_y, roi_width, roi_height = 141, 607, 183, 101
            target_column = "TrackName"
        else:
            roi_x, roi_y, roi_width, roi_height = 640, 590, 144, 48
            target_column = "RaceNumber"

        roi_x = max(roi_x - 25, 0)
        roi_y = max(roi_y - 25, 0)
        roi_width = min(roi_width + 50, gray_image.shape[1] - roi_x)
        roi_height = min(roi_height + 50, gray_image.shape[0] - roi_y)
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        stage_start = time.perf_counter()
        processed_roi = preprocess_roi(roi, i)
        add_timing(stats, "initial_roi_preprocess_s", stage_start)

        timecode = frame_to_timecode(frame_number, fps) if emit_results else ""
        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            if emit_results:
                debug_rows.append([os.path.basename(video_path), "Score", frame_number, 0, timecode])
            if i == 2:
                add_timing(stats, "process_frame_total_s", process_frame_start)
                return 0
            continue

        template_binary, alpha_mask = templates[i]
        if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
            processed_roi = cv2.resize(
                processed_roi,
                (max(template_binary.shape[1], processed_roi.shape[1]), max(template_binary.shape[0], processed_roi.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )

        stage_start = time.perf_counter()
        max_val = match_template(processed_roi, template_binary, alpha_mask)
        add_timing(stats, "initial_match_s", stage_start)

        if emit_results:
            debug_rows.append([os.path.basename(video_path), target_column if i else "Score", frame_number, max_val, timecode])

        if i == 0 and max_val > 0.3 and not np.isinf(max_val):
            if emit_results:
                state["score_detections"].append({"frame_number": frame_number, "confidence": max_val})
            add_timing(stats, "process_frame_total_s", process_frame_start)
            return int(fps * 20)

        if i == 1 and max_val > 0.6 and not np.isinf(max_val):
            if state["last_track_frame"] < max(1, frame_number - int(fps * 20)):
                state["last_track_frame"] = frame_number
                if emit_results:
                    save_frame_number = frame_number + int(fps * 1)
                    saved_image = None
                    if state["cap"] is not None:
                        seek_to_frame(state["cap"], save_frame_number, stats)
                        ret, save_frame = read_video_frame(state["cap"], stats)
                        if ret:
                            saved_image = crop_and_upscale_image(
                                save_frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT
                            )
                        seek_to_frame(state["cap"], frame_number, stats)
                        read_video_frame(state["cap"], stats)
                    state["track_detections"].append(
                        {
                            "frame_number": frame_number,
                            "save_frame_number": save_frame_number,
                            "confidence": max_val,
                            "saved_image": saved_image,
                        }
                    )
            add_timing(stats, "process_frame_total_s", process_frame_start)
            return 0

        if i == 2 and max_val > 0.6 and not np.isinf(max_val):
            if state["last_race_frame"] < max(1, frame_number - int(fps * 20)):
                state["last_race_frame"] = frame_number
                if emit_results:
                    state["race_detections"].append(
                        {
                            "frame_number": frame_number,
                            "save_frame_number": frame_number,
                            "confidence": max_val,
                            "saved_image": upscaled_image.copy(),
                        }
                    )
            add_timing(stats, "process_frame_total_s", process_frame_start)
            return int(fps * 60)

    add_timing(stats, "process_frame_total_s", process_frame_start)
    return 0


def scan_pass1_segment(task):
    """Scan one overlapped pass-one segment in a separate process."""
    video_path = task["video_path"]
    fps = task["fps"]
    scan_start = task["scan_start"]
    scan_end = task["scan_end"]
    emit_start = task["emit_start"]
    emit_end = task["emit_end"]
    templates = task["templates"]
    scale_x = task["scale_x"]
    scale_y = task["scale_y"]
    left = task["left"]
    top = task["top"]
    crop_width = task["crop_width"]
    crop_height = task["crop_height"]
    include_debug_rows = task["include_debug_rows"]
    progress_queue = task.get("progress_queue")

    local_cap = cv2.VideoCapture(video_path)
    stats = defaultdict(float)
    state = {
        "last_track_frame": 0,
        "last_race_frame": 0,
        "score_detections": [],
        "track_detections": [],
        "race_detections": [],
        "cap": local_cap,
    }
    debug_rows = []

    if not local_cap.isOpened():
        return {
            "segment_index": task["segment_index"],
            "stats": stats,
            "debug_rows": [],
            "score_detections": [],
            "track_detections": [],
            "race_detections": [],
        }

    seek_to_frame(local_cap, scan_start, stats)
    frame_count = scan_start
    last_progress_report = time.perf_counter()
    update_segment_progress(progress_queue, task["segment_index"], frame_count, scan_start, scan_end, emit_start, emit_end, force=True)

    while local_cap.isOpened() and frame_count < scan_end:
        window_interrupted = False
        for _ in range(PASS1_WINDOW_STEPS):
            ret, frame = read_video_frame(local_cap, stats)
            if not ret:
                window_interrupted = True
                break

            emit_results = emit_start <= frame_count < emit_end
            frames_to_skip = process_segment_frame(
                frame,
                frame_count,
                video_path,
                templates,
                fps,
                scale_x,
                scale_y,
                left,
                top,
                crop_width,
                crop_height,
                stats,
                state,
                debug_rows,
                emit_results and include_debug_rows,
            )

            if frames_to_skip > 0:
                frame_count += frames_to_skip + FRAME_SKIP
                if frame_count < scan_end:
                    seek_to_frame(local_cap, frame_count, stats)
                window_interrupted = True
                break

            if not advance_frames_by_grab(local_cap, FRAME_SKIP - 1, stats):
                frame_count = scan_end
                window_interrupted = True
                break

            frame_count += FRAME_SKIP
            if frame_count >= scan_end:
                window_interrupted = True
                break

            now = time.perf_counter()
            if now - last_progress_report >= PASS1_PROGRESS_REPORT_SECONDS:
                update_segment_progress(progress_queue, task["segment_index"], frame_count, scan_start, scan_end, emit_start, emit_end)
                last_progress_report = now

        if window_interrupted and frame_count >= scan_end:
            break

    local_cap.release()
    update_segment_progress(progress_queue, task["segment_index"], emit_end, scan_start, scan_end, emit_start, emit_end, force=True)
    return {
        "segment_index": task["segment_index"],
        "stats": dict(stats),
        "debug_rows": debug_rows,
        "score_detections": state["score_detections"],
        "track_detections": state["track_detections"],
        "race_detections": state["race_detections"],
    }


def choose_pass1_segment_count(total_frames, fps, requested_workers):
    """Use multiprocessing only for sufficiently long videos."""
    if requested_workers <= 1 or total_frames < PASS1_MIN_SEGMENT_FRAMES:
        return 1
    segment_count = int(total_frames // PASS1_MIN_SEGMENT_FRAMES)
    return max(1, min(requested_workers, segment_count if segment_count > 0 else 1))


def build_pass1_segment_tasks(video_path, total_frames, fps, templates, scale_x, scale_y, left, top, crop_width,
                              crop_height, requested_workers, include_debug_rows):
    """Split pass-one scanning into overlapped segments with non-overlapping emit ranges."""
    segment_count = choose_pass1_segment_count(total_frames, fps, requested_workers)
    if segment_count == 1:
        return []

    overlap_frames = PASS1_SEGMENT_OVERLAP_FRAMES
    segment_size = (total_frames + segment_count - 1) // segment_count
    tasks = []
    for segment_index in range(segment_count):
        emit_start = segment_index * segment_size
        emit_end = min(total_frames, emit_start + segment_size)
        scan_start = max(0, emit_start - overlap_frames)
        scan_end = min(total_frames, emit_end + overlap_frames)
        tasks.append(
            {
                "segment_index": segment_index,
                "video_path": video_path,
                "fps": fps,
                "scan_start": scan_start,
                "scan_end": scan_end,
                "emit_start": emit_start,
                "emit_end": emit_end,
                "templates": templates,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "left": left,
                "top": top,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "include_debug_rows": include_debug_rows,
            }
        )
    return tasks


def merge_nearby_detections(detections, min_gap_frames):
    """Merge nearby detections by keeping the stronger confidence within each window."""
    if not detections:
        return []

    merged = []
    for detection in sorted(detections, key=lambda item: item["frame_number"]):
        if not merged:
            merged.append(detection)
            continue
        previous = merged[-1]
        if detection["frame_number"] - previous["frame_number"] <= min_gap_frames:
            if detection["confidence"] > previous["confidence"]:
                merged[-1] = detection
            continue
        merged.append(detection)
    return merged


def assign_race_number(frame_number, score_frame_numbers):
    """Map a detection before a score screen to its race number."""
    return bisect_left(score_frame_numbers, frame_number) + 1


def save_auxiliary_detection_frames(video_path, detections, score_frame_numbers, scale_x, scale_y, left, top,
                                    crop_width, crop_height, fps, stats, metadata_writer):
    """Save merged track-name and race-number frames after pass-one merging."""
    if not detections:
        return

    script_dir = os.path.dirname(__file__)
    output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')
    os.makedirs(output_folder, exist_ok=True)

    for detection in detections:
        race_number = assign_race_number(detection["frame_number"], score_frame_numbers)
        stage_start = time.perf_counter()
        if detection.get("saved_image") is not None:
            upscaled_image = detection["saved_image"]
            actual_saved_frame = detection["save_frame_number"]
        else:
            seek_to_frame(cap, detection["save_frame_number"], stats)
            ret, frame = read_video_frame(cap, stats)
            if not ret:
                continue
            actual_saved_frame = actual_frame_after_read(cap)
            upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)

        if detection["kind"] == "track":
            timecode = frame_to_timecode(detection["frame_number"], fps)
            print_detection_event(video_path, "track screen", race_number, timecode)
            suffix = "0TrackName"
            kind = "TrackName"
        else:
            timecode = frame_to_timecode(detection["frame_number"], fps)
            print_detection_event(video_path, "race number", race_number, timecode)
            suffix = "1RaceNumber"
            kind = "RaceNumber"

        frame_filename = os.path.join(
            output_folder,
            f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{race_number:03}+{suffix}.png"
        )
        cv2.imwrite(frame_filename, upscaled_image)
        log_exported_frame(
            metadata_writer,
            video_path,
            race_number,
            kind,
            detection["save_frame_number"],
            actual_saved_frame,
            fps,
        )
        add_timing(stats, "output_frame_capture_s", stage_start)


def run_parallel_pass1_segments(segment_tasks, progress=None):
    """Prefer processes for CPU-bound pass-one work, but fall back to threads if blocked."""
    def _drain_progress(progress_queue, segment_state):
        changed = False
        while True:
            try:
                message = progress_queue.get_nowait()
            except Empty:
                break
            segment_state[message["segment_index"]] = (
                message["completed_frames"],
                message["total_frames"],
            )
            changed = True
        if changed and progress is not None:
            completed_frames = sum(item[0] for item in segment_state.values())
            total_frames = sum(item[1] for item in segment_state.values())
            progress.total_units = max(1, total_frames)
            progress.update(completed_frames)

    def _run_with_executor(executor_factory, progress_queue):
        segment_state = {task["segment_index"]: (0, max(1, task["emit_end"] - task["emit_start"])) for task in segment_tasks}
        task_payloads = [{**task, "progress_queue": progress_queue} for task in segment_tasks]
        with executor_factory(max_workers=len(segment_tasks)) as executor:
            pending = {
                executor.submit(scan_pass1_segment, task): task
                for task in task_payloads
            }
            results = []
            while pending:
                done, _ = wait(pending.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
                _drain_progress(progress_queue, segment_state)
                for future in done:
                    task = pending.pop(future)
                    results.append(future.result())
                    if progress is not None:
                        _drain_progress(progress_queue, segment_state)
                        pass
            _drain_progress(progress_queue, segment_state)
            return results

    try:
        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            return _run_with_executor(ProcessPoolExecutor, progress_queue)
    except (PermissionError, OSError) as exc:
        LOGGER.log("[Extract - Settings]", "Process pool unavailable, falling back to threads", color_name="yellow")
        progress_queue = Queue()
        return _run_with_executor(ThreadPoolExecutor, progress_queue)

def extract_frames(return_frame_cache=False, selected_videos=None):
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

    script_dir = os.path.dirname(__file__)  # Directory of the script
    folder_path = os.path.join(script_dir, 'Input_Videos')

    template_paths = [
        (os.path.join(script_dir, 'Find_Templates', 'Score_template.png'), None),
        (os.path.join(script_dir, 'Find_Templates', 'Trackname_template.png'), None),
        (os.path.join(script_dir, 'Find_Templates', 'Race_template.png'), None),
        (os.path.join(script_dir, 'Find_Templates', '12th_pos_template.png'), None)
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

    csv_output_path = os.path.join(script_dir, 'Output_Results', 'Debug', 'debug_max_val.csv')
    metadata_output_path = os.path.join(script_dir, 'Output_Results', 'Debug', 'exported_frame_metadata.csv')
    video_paths = load_videos_from_folder(folder_path)
    if selected_videos:
        selected_names = {str(name).lower() for name in selected_videos}
        video_paths = [path for path in video_paths if os.path.basename(path).lower() in selected_names]
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
            ["Video", "Race", "Kind", "Requested Frame", "Requested Timecode", "Actual Frame", "Actual Timecode"]
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
    LOGGER.log("[Extract - Settings]", f"GPU acceleration: {GPU_RUNTIME['backend']} | Scan workers: {PASS1_SCAN_WORKERS}", color_name="cyan")

    try:
        total_videos = len(video_paths)
        total_score_screens_found = 0
        total_track_screens_found = 0
        total_race_numbers_found = 0
        per_video_summaries = []
        for video_index, video_path in enumerate(video_paths, start=1):
            global LastTrackNameFrame
            global LastRaceNumberFrame
            global RaceCount
            global cap
            global FRAME_SKIP

            LastTrackNameFrame = 0
            LastRaceNumberFrame = 0
            RaceCount = 1
            video_start = time.perf_counter()
            video_stats = defaultdict(float)
            video_stats["template_load_s"] = template_load_time_s
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                LOGGER.log(f"[Video {video_index}/{total_videos} - Start]", f"Could not open video: {video_path}", color_name="red")
                continue
            #Determine the FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            FRAME_SKIP = int(3 * int(fps))
            score_candidates = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = np.linspace(0, total_frames - 1, 19).astype(int)
            scales = []
            video_name = os.path.basename(video_path)
            LOGGER.log(
                f"[Video {video_index}/{total_videos} - Start]",
                f"{video_name} | Source length: {format_duration(total_frames / max(fps, 1))}",
                color_name="cyan",
            )
            stage_start = time.perf_counter()
            for frame_num in sample_frames:
                seek_to_frame(cap, frame_num, video_stats)
                ret, frame = read_video_frame(cap, video_stats)
                if not ret:
                    continue
                scale_x, scale_y, left, top, crop_width, crop_height = determine_scaling(frame)
                scales.append((scale_x, scale_y, left, top, crop_width, crop_height))
            add_timing(video_stats, "scaling_scan_s", stage_start)

            if not scales:
                LOGGER.log(f"[Video {video_index}/{total_videos} - Start]", f"No valid frames found for scaling: {video_path}", color_name="red")
                continue

            median_scale_x = np.median([s[0] for s in scales])
            median_scale_y = np.median([s[1] for s in scales])
            median_left = int(np.median([s[2] for s in scales]))
            median_top = int(np.median([s[3] for s in scales]))
            median_crop_width = int(np.median([s[4] for s in scales]))
            median_crop_height = int(np.median([s[5] for s in scales]))

            seek_to_frame(cap, 0, video_stats)
            segment_tasks = build_pass1_segment_tasks(
                video_path,
                total_frames,
                fps,
                templates,
                median_scale_x,
                median_scale_y,
                median_left,
                median_top,
                median_crop_width,
                median_crop_height,
                PASS1_SCAN_WORKERS,
                APP_CONFIG.write_debug_csv,
            )

            LOGGER.log(f"[Video {video_index}/{total_videos} - Scan - Phase Start]", "", color_name="cyan")
            scan_progress = None
            if not segment_tasks:
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
                    window_interrupted = False

                    for _ in range(PASS1_WINDOW_STEPS):
                        ret, frame = read_video_frame(cap, video_stats)
                        if not ret:
                            window_interrupted = True
                            break

                        frames_to_skip = process_frame(frame, frame_count, video_path, templates, fps, csv_writer,
                                                       median_scale_x, median_scale_y, median_left, median_top,
                                                       median_crop_width, median_crop_height, video_stats, score_candidates,
                                                       metadata_writer)
                        if frames_to_skip > 0:
                            frame_count += frames_to_skip + FRAME_SKIP
                            if frame_count < total_frames:
                                seek_to_frame(cap, frame_count, video_stats)
                            window_interrupted = True
                            scan_progress.update(min(frame_count, total_frames), f"Races found: {len(score_candidates)}")
                            break

                        if not advance_frames_by_grab(cap, FRAME_SKIP - 1, video_stats):
                            window_interrupted = True
                            frame_count = total_frames
                            break

                        frame_count += FRAME_SKIP
                        if frame_count >= total_frames:
                            window_interrupted = True
                            break

                        scan_progress.update(frame_count, f"Races found: {len(score_candidates)}")

                    if window_interrupted and frame_count >= total_frames:
                        break
                if scan_progress.last_percent < 100:
                    scan_progress.update(total_frames, f"Races found: {len(score_candidates)}")
                add_timing(video_stats, "main_scan_loop_s", stage_start)
            else:
                stage_start = time.perf_counter()
                scan_progress = ProgressPrinter(
                    f"[Video {video_index}/{total_videos} - Scan]",
                    total_frames,
                    percent_step=5,
                    min_interval_s=1.0,
                )
                scan_progress.update(0)
                segment_results = run_parallel_pass1_segments(segment_tasks, scan_progress)
                add_timing(video_stats, "main_scan_loop_s", stage_start)

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
                merged_score_detections = merge_nearby_detections(merged_score_detections, min_gap_frames)
                merged_track_detections = merge_nearby_detections(merged_track_detections, min_gap_frames)
                merged_race_detections = merge_nearby_detections(merged_race_detections, min_gap_frames)

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
                save_auxiliary_detection_frames(
                    video_path,
                    auxiliary_detections,
                    score_frame_numbers,
                    median_scale_x,
                    median_scale_y,
                    median_left,
                    median_top,
                    median_crop_width,
                    median_crop_height,
                    fps,
                    video_stats,
                    metadata_writer,
                )
            pre_pass2_counts = count_exported_detection_files(video_path)
            scan_duration = float(video_stats.get("main_scan_loop_s", 0.0))
            scan_speed = total_frames / scan_duration if scan_duration > 0 else 0.0
            scan_summary_lines = [
                f"Duration: {format_duration(scan_duration)}",
                f"Source length: {format_duration(total_frames / max(fps, 1))}",
                f"Frames scanned: {total_frames:,}",
                f"Scan speed: {scan_speed:,.0f} frames/s",
                f"Track screens found: {pre_pass2_counts['track']}",
                f"Race numbers found: {pre_pass2_counts['race']}",
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
                video_path,
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
            exported_counts = count_exported_detection_files(video_path)
            total_score_screens_found += exported_counts["score"]
            total_track_screens_found += exported_counts["track"]
            total_race_numbers_found += exported_counts["race"]
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
