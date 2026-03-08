import cv2
import numpy as np
import os
from glob import glob
import csv
import time
from PIL import Image, ImageEnhance, ImageFilter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from app_runtime import load_app_config

# Record the start time
start_run_time = time.time()
APP_CONFIG = load_app_config()

print("Extract Frames Started - Calculating black borders")

# Global parameter for frame skip value
FRAME_SKIP = int(3 * 30)  # Skip 3 seconds (assuming 30 FPS)
LastTrackNameFrame = 0
LastRaceNumberFrame = 0
RaceCount = 1
SCORE_ANALYSIS_WORKERS = APP_CONFIG.score_analysis_workers
PASS1_WINDOW_STEPS = 2


class NullCsvWriter:
    """Drop debug CSV rows when debug output is disabled."""

    def writerow(self, _row):
        return None


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
    """Print a compact timing summary for a processed video."""
    summary_order = [
        "video_total_s",
        "scaling_scan_s",
        "main_scan_loop_s",
        "process_frame_total_s",
        "initial_frame_prepare_s",
        "initial_roi_preprocess_s",
        "initial_match_s",
        "score_detail_total_s",
        "score_detail_frame_prepare_s",
        "score_detail_match_score_s",
        "score_detail_match_12th_s",
        "output_frame_capture_s",
        "seek_time_s",
        "grab_time_s",
        "read_time_s",
    ]
    print(f"Timing Summary: {video_name}")
    for key in summary_order:
        if key in stats:
            print(f"  {key}: {stats[key]:.3f}")
    print(f"  seek_calls: {int(stats.get('seek_calls', 0))}")
    print(f"  grab_calls: {int(stats.get('grab_calls', 0))}")
    print(f"  read_calls: {int(stats.get('read_calls', 0))}")


def analyze_score_window(video_path, frame_number, fps, templates, csv_writer, scale_x, scale_y, left, top,
                         crop_width, crop_height, stats):
    """Analyze a detected score-screen window and return the selected frame numbers."""
    target_width, target_height = 1280, 720
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

        timecode = frame_to_timecode(detail_frame_number, fps)
        stage_start = time.perf_counter()
        upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)
        gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
        roi_x, roi_y, roi_width, roi_height = 315, 57, 52, 610
        roi_x = max(roi_x - 25, 0)
        roi_y = max(roi_y - 25, 0)
        roi_width = min(roi_width + 50, gray_image.shape[1] - roi_x)
        roi_height = min(roi_height + 50, gray_image.shape[0] - roi_y)
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        processed_roi = preprocess_roi(roi, 0)
        add_timing(stats, "score_detail_frame_prepare_s", stage_start)

        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            detail_frame_number += 1
            csv_writer.writerow([os.path.basename(video_path), "Score", detail_frame_number, 0, timecode])
            if race_score_frame != 0:
                total_score_frame = detail_frame_number - int(2.7 * fps)
                break
            continue

        if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
            processed_roi = cv2.resize(
                processed_roi,
                (max(template_binary.shape[1], processed_roi.shape[1]),
                 max(template_binary.shape[0], processed_roi.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )

        stage_start = time.perf_counter()
        res = cv2.matchTemplate(processed_roi, template_binary, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        add_timing(stats, "score_detail_match_score_s", stage_start)

        csv_writer.writerow([os.path.basename(video_path), "Score", detail_frame_number, max_val, timecode])

        if max_val > 0.3 and not np.isinf(max_val) and race_score_frame == 0:
            race_score_frame = detail_frame_number + int(0.6 * fps)
            check_player_12 = 1
            continue

        if max_val > 0.3 and not np.isinf(max_val) and check_player_12 == 1:
            roi_x2, roi_y2, roi_width2, roi_height2 = 338, 657, 601, 39
            roi_x2 = max(roi_x2 - 25, 0)
            roi_y2 = max(roi_y2 - 25, 0)
            roi_width2 = min(roi_width2 + 50, gray_image.shape[1] - roi_x2)
            roi_height2 = min(roi_height2 + 50, gray_image.shape[0] - roi_y2)
            roi2 = gray_image[roi_y2:roi_y2 + roi_height2, roi_x2:roi_x2 + roi_width2]
            processed_roi2 = preprocess_roi(roi2, 0)

            template_binary2, alpha_mask2 = templates[3]
            if processed_roi2.shape[0] < template_binary2.shape[0] or processed_roi2.shape[1] < template_binary2.shape[1]:
                processed_roi2 = cv2.resize(
                    processed_roi2,
                    (max(template_binary2.shape[1], processed_roi2.shape[1]),
                     max(template_binary2.shape[0], processed_roi2.shape[0])),
                    interpolation=cv2.INTER_LINEAR,
                )

            stage_start = time.perf_counter()
            res = cv2.matchTemplate(processed_roi2, template_binary2, cv2.TM_CCOEFF_NORMED, mask=alpha_mask2)
            _, max_val2, _, _ = cv2.minMaxLoc(res)
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

    target_width, target_height = 1280, 720
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

        timecode = frame_to_timecode(detail_frame_number, fps)
        stage_start = time.perf_counter()
        upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)
        gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
        roi_x, roi_y, roi_width, roi_height = 315, 57, 52, 610
        roi_x = max(roi_x - 25, 0)
        roi_y = max(roi_y - 25, 0)
        roi_width = min(roi_width + 50, gray_image.shape[1] - roi_x)
        roi_height = min(roi_height + 50, gray_image.shape[0] - roi_y)
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        processed_roi = preprocess_roi(roi, 0)
        add_timing(stats, "score_detail_frame_prepare_s", stage_start)

        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            detail_frame_number += 1
            debug_rows.append([os.path.basename(video_path), "Score", detail_frame_number, 0, timecode])
            if race_score_frame != 0:
                total_score_frame = detail_frame_number - int(2.7 * fps)
                break
            continue

        if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
            processed_roi = cv2.resize(
                processed_roi,
                (max(template_binary.shape[1], processed_roi.shape[1]),
                 max(template_binary.shape[0], processed_roi.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )

        stage_start = time.perf_counter()
        res = cv2.matchTemplate(processed_roi, template_binary, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        add_timing(stats, "score_detail_match_score_s", stage_start)
        debug_rows.append([os.path.basename(video_path), "Score", detail_frame_number, max_val, timecode])

        if max_val > 0.3 and not np.isinf(max_val) and race_score_frame == 0:
            race_score_frame = detail_frame_number + int(0.6 * fps)
            check_player_12 = 1
            continue

        if max_val > 0.3 and not np.isinf(max_val) and check_player_12 == 1:
            roi_x2, roi_y2, roi_width2, roi_height2 = 338, 657, 601, 39
            roi_x2 = max(roi_x2 - 25, 0)
            roi_y2 = max(roi_y2 - 25, 0)
            roi_width2 = min(roi_width2 + 50, gray_image.shape[1] - roi_x2)
            roi_height2 = min(roi_height2 + 50, gray_image.shape[0] - roi_y2)
            roi2 = gray_image[roi_y2:roi_y2 + roi_height2, roi_x2:roi_x2 + roi_width2]
            processed_roi2 = preprocess_roi(roi2, 0)
            template_binary2, alpha_mask2 = templates[3]

            if processed_roi2.shape[0] < template_binary2.shape[0] or processed_roi2.shape[1] < template_binary2.shape[1]:
                processed_roi2 = cv2.resize(
                    processed_roi2,
                    (max(template_binary2.shape[1], processed_roi2.shape[1]),
                     max(template_binary2.shape[0], processed_roi2.shape[0])),
                    interpolation=cv2.INTER_LINEAR,
                )

            stage_start = time.perf_counter()
            res = cv2.matchTemplate(processed_roi2, template_binary2, cv2.TM_CCOEFF_NORMED, mask=alpha_mask2)
            _, max_val2, _, _ = cv2.minMaxLoc(res)
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

    local_cap.release()
    return {
        "candidate": task,
        "race_score_frame": race_score_frame,
        "total_score_frame": total_score_frame,
        "debug_rows": debug_rows,
        "stats": stats,
    }


def save_score_frames(video_path, race_number, race_score_frame, total_score_frame, scale_x, scale_y, left, top,
                      crop_width, crop_height, fps, stats):
    """Save the selected race score and total score frames."""
    target_width, target_height = 1280, 720
    stage_start = time.perf_counter()

    seek_to_frame(cap, race_score_frame, stats)
    ret, frame = read_video_frame(cap, stats)
    if not ret:
        return False
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)

    if scale_x > 1.3 and scale_y >= 1.3:
        if isinstance(upscaled_image, np.ndarray):
            upscaled_image = Image.fromarray(upscaled_image)
        contrast_enhancer = ImageEnhance.Contrast(upscaled_image)
        high_contrast_image = contrast_enhancer.enhance(1.70)
        sharpness_enhancer = ImageEnhance.Sharpness(high_contrast_image)
        sharpened_image = sharpness_enhancer.enhance(1.23)
        upscaled_image = np.array(sharpened_image)

    script_dir = os.path.dirname(__file__)
    output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')
    os.makedirs(output_folder, exist_ok=True)
    frame_filename = os.path.join(
        output_folder,
        f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{race_number:03}+2RaceScore.png"
    )
    cv2.imwrite(frame_filename, upscaled_image)

    timecode = frame_to_timecode(total_score_frame, fps)
    print(f"Video: {os.path.basename(video_path)}, TotalScoreFrame {race_number:03} at Timecode: {timecode}")

    seek_to_frame(cap, total_score_frame, stats)
    ret, frame = read_video_frame(cap, stats)
    if not ret:
        return False
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)

    if scale_x > 1.3 and scale_y >= 1.3:
        if isinstance(upscaled_image, np.ndarray):
            upscaled_image = Image.fromarray(upscaled_image)
        contrast_enhancer = ImageEnhance.Contrast(upscaled_image)
        high_contrast_image = contrast_enhancer.enhance(1.70)
        sharpness_enhancer = ImageEnhance.Sharpness(high_contrast_image)
        sharpened_image = sharpness_enhancer.enhance(1.23)
        upscaled_image = np.array(sharpened_image)

    frame_filename = os.path.join(
        output_folder,
        f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{race_number:03}+3TotalScore.png"
    )
    cv2.imwrite(frame_filename, upscaled_image)
    add_timing(stats, "output_frame_capture_s", stage_start)
    return True


def process_score_candidates(video_path, score_candidates, templates, fps, csv_writer, scale_x, scale_y, left, top,
                             crop_width, crop_height, stats):
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
            "left": left,
            "top": top,
            "crop_width": crop_width,
            "crop_height": crop_height,
        }
        for candidate in score_candidates
    ]

    worker_count = min(SCORE_ANALYSIS_WORKERS, len(tasks))
    if worker_count == 1:
        results = [analyze_score_window_task(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(analyze_score_window_task, tasks))

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
            scale_x,
            scale_y,
            left,
            top,
            crop_width,
            crop_height,
            fps,
            stats,
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

def load_videos_from_folder(folder_path):
    """Load video file paths from the specified folder."""
    video_extensions = ["*.mp4", "*.mkv", "*.mov", "*.avi", "*.webm"]
    video_paths = []
    for extension in video_extensions:
        video_paths.extend(glob(os.path.join(folder_path, extension)))
    return video_paths

def crop_and_upscale_image(image, left, top, crop_width, crop_height, target_width, target_height):
    """Crop the image to the detected borders and upscale to the target size."""
    cropped_image = image[top:top + crop_height, left:left + crop_width]
    upscaled_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return upscaled_image


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
            black_pixels = np.count_nonzero(sub_region == 0)
            white_pixels = np.count_nonzero(sub_region == 255)
            if white_pixels > black_pixels:
                _, binary_section_sub = cv2.threshold(sub_region, 120, 255, cv2.THRESH_BINARY)
                sub_region_copy = binary_section_sub.copy()
                kernel = np.ones((2, 1), np.uint8)
                sub_region_copy = cv2.dilate(sub_region_copy, kernel, iterations=1)
                sub_region_copy = cv2.bitwise_not(sub_region_copy)
                modified_binary_section[sub_region_y_start:sub_region_y_end, :] = sub_region_copy
            else:
                kernel = np.ones((2, 1), np.uint8)
                dilated_section = cv2.dilate(sub_region, kernel, iterations=1)
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
                  crop_width, crop_height, stats, score_candidates):
    """Process a single video frame and apply template matching."""
    global LastTrackNameFrame
    global LastRaceNumberFrame
    global RaceCount
    global cap
    global FRAME_SKIP

    process_frame_start = time.perf_counter()

    # Crop and upscale the image using the calculated scaling factors
    target_width, target_height = 1280, 720
    stage_start = time.perf_counter()
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)


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

        # Convert frame number to timecode
        timecode = frame_to_timecode(frame_number, fps)

        # Skip the frame if 97% or more of the pixels in the ROI are black
        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
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
        res = cv2.matchTemplate(processed_roi, template_binary, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        add_timing(stats, "initial_match_s", stage_start)

        if i == 0 and max_val > 0.3 and not np.isinf(max_val):
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
                upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, target_width, target_height)
                LastTrackNameFrame = frame_number
                print(f"Video: {os.path.basename(video_path)}, TrackName {RaceCount:03} Timecode: {timecode}, Max Value: {max_val}")
                script_dir = os.path.dirname(__file__)  # Directory of the script
                output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')
                os.makedirs(output_folder, exist_ok=True)
                frame_filename = os.path.join(output_folder,
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{RaceCount:03}+0TrackName.png")
                cv2.imwrite(frame_filename, upscaled_image)
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
                print(f"Video: {os.path.basename(video_path)}, RaceNumber {RaceCount:03} at Timecode: {timecode}, Max Value: {max_val}")
                script_dir = os.path.dirname(__file__)  # Directory of the script
                output_folder = os.path.join(script_dir, 'Output_Results', 'Frames')

                os.makedirs(output_folder, exist_ok=True)
                frame_filename = os.path.join(output_folder,
                                              f"{os.path.splitext(os.path.basename(video_path))[0]}+Race_{RaceCount:03}+1RaceNumber.png")
                cv2.imwrite(frame_filename, upscaled_image)
                stats["output_frame_capture_s"] += 0
            csv_writer.writerow([os.path.basename(video_path), "RaceNumber", frame_number, max_val, timecode])
            frames_to_skip = int(fps * 60)
            add_timing(stats, "process_frame_total_s", process_frame_start)
            return frames_to_skip

        if i == 0:
            csv_writer.writerow([os.path.basename(video_path), "Score", frame_number, max_val, timecode])
        elif i == 1:
            csv_writer.writerow([os.path.basename(video_path), "TrackName", frame_number, max_val, timecode])
        else:
            csv_writer.writerow([os.path.basename(video_path), "RaceNumber", frame_number, max_val, timecode])
    add_timing(stats, "process_frame_total_s", process_frame_start)
    return 0

def main():
    """Main function to process videos and apply template matching."""

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
            print(f"Error: Template image at '{template_path}' could not be loaded.")
            return
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            _, alpha_mask = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
        elif len(template.shape) == 3 and template.shape[2] == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
            alpha_mask = np.ones(template_binary.shape, dtype=np.uint8) * 255
        elif len(template.shape) == 2:
            template_binary = template
            alpha_mask = np.ones(template_binary.shape, dtype=np.uint8) * 255
        else:
            print(f"Error: Template image at '{template_path}' has an unexpected number of channels.")
            return
        templates.append((template_binary, alpha_mask))
    template_load_time_s = time.perf_counter() - template_load_start

    csv_output_path = os.path.join(script_dir, 'Output_Results', 'Debug', 'debug_max_val.csv')
    video_paths = load_videos_from_folder(folder_path)
    if not video_paths:
        print("No videos found in the specified folder. Exiting.")
        return

    if APP_CONFIG.write_debug_csv:
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        csv_context = open(csv_output_path, mode='w', newline='')
        csv_writer = csv.writer(csv_context, delimiter=';')
        csv_writer.writerow(["Video", "Target", "Frame Number", "Max Value", "Timecode"])
    else:
        csv_context = None
        csv_writer = NullCsvWriter()

    try:
        for video_path in video_paths:
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
                print(f"Error: Could not open video {video_path}")
                continue
            #Determine the FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            FRAME_SKIP = int(3 * int(fps))
            score_candidates = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_frames = np.linspace(0, total_frames - 1, 19).astype(int)
            scales = []

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
                print(f"Error: No valid frames found for scaling in video {video_path}.")
                continue

            median_scale_x = np.median([s[0] for s in scales])
            median_scale_y = np.median([s[1] for s in scales])
            median_left = int(np.median([s[2] for s in scales]))
            median_top = int(np.median([s[3] for s in scales]))
            median_crop_width = int(np.median([s[4] for s in scales]))
            median_crop_height = int(np.median([s[5] for s in scales]))

            print(f"Median scale_x: {median_scale_x}, Median scale_y: {median_scale_y}")
            print(f"Median left: {median_left}, Median top: {median_top}")
            print(f"Median crop_width: {median_crop_width}, Median crop_height: {median_crop_height}")

            seek_to_frame(cap, 0, video_stats)
            frame_count = 0

            stage_start = time.perf_counter()
            while cap.isOpened() and frame_count < total_frames:
                window_interrupted = False

                for _ in range(PASS1_WINDOW_STEPS):
                    ret, frame = read_video_frame(cap, video_stats)
                    if not ret:
                        window_interrupted = True
                        break

                    frames_to_skip = process_frame(frame, frame_count, video_path, templates, fps, csv_writer,
                                                   median_scale_x, median_scale_y, median_left, median_top,
                                                   median_crop_width, median_crop_height, video_stats, score_candidates)

                    if frames_to_skip > 0:
                        frame_count += frames_to_skip + FRAME_SKIP
                        if frame_count < total_frames:
                            seek_to_frame(cap, frame_count, video_stats)
                        window_interrupted = True
                        break

                    if not advance_frames_by_grab(cap, FRAME_SKIP - 1, video_stats):
                        window_interrupted = True
                        frame_count = total_frames
                        break

                    frame_count += FRAME_SKIP
                    if frame_count >= total_frames:
                        window_interrupted = True
                        break

                if window_interrupted and frame_count >= total_frames:
                    break
            add_timing(video_stats, "main_scan_loop_s", stage_start)

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
            )
            cap.release()
            video_stats["video_total_s"] = time.perf_counter() - video_start
            print_timing_summary(os.path.basename(video_path), video_stats)
    finally:
        if csv_context is not None:
            csv_context.close()

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_run_time

    # Convert elapsed time to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # Print the elapsed time in mm:ss format
    print(f"Runtime was: {minutes:02}:{seconds:02}")

if __name__ == "__main__":
    main()
