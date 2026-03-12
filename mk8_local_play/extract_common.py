import time
import os
import re
from glob import glob
from pathlib import Path

import cv2
import numpy as np

from .app_runtime import detect_gpu_runtime, load_app_config
from .project_paths import PROJECT_ROOT

APP_CONFIG = load_app_config()
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
VERTICAL_DILATE_KERNEL = np.ones((2, 1), np.uint8)
GPU_RUNTIME = detect_gpu_runtime(APP_CONFIG)


def calculate_sum_intensity(gray_image):
    """Return row/column intensity sums used to locate the active game area."""
    sum_row_intensity = np.sum(gray_image, axis=1)
    sum_col_intensity = np.sum(gray_image, axis=0)
    return sum_row_intensity, sum_col_intensity


def find_borders(sum_row_intensity, sum_col_intensity, threshold=25000):
    """Find the non-black content bounds inside a captured frame."""
    top = next((i for i, val in enumerate(sum_row_intensity) if val > threshold), 0)
    bottom = next((i for i, val in enumerate(reversed(sum_row_intensity)) if val > threshold), 0)
    bottom = len(sum_row_intensity) - bottom
    left = next((i for i, val in enumerate(sum_col_intensity) if val > threshold), 0)
    right = next((i for i, val in enumerate(reversed(sum_col_intensity)) if val > threshold), 0)
    right = len(sum_col_intensity) - right
    return top, left, bottom, right


def determine_scaling(image):
    """Determine how a source frame maps onto the fixed 1280x720 working canvas."""
    gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sum_row_intensity, sum_col_intensity = calculate_sum_intensity(gray_frame)
    top, left, bottom, right = find_borders(sum_row_intensity, sum_col_intensity, threshold=15000)

    crop_width = right - left
    crop_height = bottom - top
    scale_x = TARGET_WIDTH / crop_width
    scale_y = TARGET_HEIGHT / crop_height

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


def build_video_identity(video_path, input_root=None, include_subfolders=False):
    """Return the stable video identifier used in exported frame names and OCR grouping."""
    video_path = Path(video_path)
    if not include_subfolders:
        return video_path.stem
    root_path = Path(input_root) if input_root is not None else video_path.parent
    try:
        relative_path = video_path.relative_to(root_path)
    except ValueError:
        relative_path = video_path if not video_path.is_absolute() else Path(video_path.name)
    path_without_suffix = relative_path.with_suffix("")
    sanitized_parts = [
        re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("._-") or "part"
        for part in path_without_suffix.parts
    ]
    return "__".join(sanitized_parts)


def relative_video_path(video_path, input_root):
    video_path = Path(video_path)
    return str(video_path.relative_to(Path(input_root))).replace("\\", "/")


def load_videos_from_folder(folder_path, *, include_subfolders=False):
    """Load supported video files from an input folder."""
    video_extensions = {".mp4", ".mkv", ".mkv", ".mov", ".avi", ".webm"}
    root = Path(folder_path)
    iterator = root.rglob("*") if include_subfolders else root.iterdir()
    return [str(path) for path in sorted(iterator) if path.is_file() and path.suffix.lower() in video_extensions]


def count_exported_detection_files(video_path_or_label):
    """Count exported frame types for one source video or precomputed video label."""
    output_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Frames')
    path_obj = Path(str(video_path_or_label))
    video_stem = path_obj.stem if path_obj.suffix else str(video_path_or_label)
    return {
        "track": len(glob(os.path.join(output_folder, f"{video_stem}+Race_*+0TrackName.png"))),
        "race": len(glob(os.path.join(output_folder, f"{video_stem}+Race_*+1RaceNumber.png"))),
        "score": len(glob(os.path.join(output_folder, f"{video_stem}+Race_*+2RaceScore.png"))),
        "total": len(glob(os.path.join(output_folder, f"{video_stem}+Race_*+3TotalScore.png"))),
    }


def crop_and_upscale_image(image, left, top, crop_width, crop_height, target_width, target_height):
    """Crop the active play area and resize it onto the fixed working canvas."""
    cropped_image = image[top:top + crop_height, left:left + crop_width]
    upscaled_image = gpu_resize(cropped_image, target_width, target_height, interpolation=cv2.INTER_LINEAR)
    return upscaled_image


def crop_to_gray_and_upscale_image(image, left, top, crop_width, crop_height, target_width, target_height):
    """Crop first, grayscale second, then resize for cheaper pass-two matching."""
    stage_start = time.perf_counter()
    cropped_image = image[top:top + crop_height, left:left + crop_width]
    gray_image = gpu_cvt_color(cropped_image, cv2.COLOR_BGR2GRAY)
    grayscale_time = time.perf_counter() - stage_start
    stage_start = time.perf_counter()
    upscaled_gray_image = gpu_resize(gray_image, target_width, target_height, interpolation=cv2.INTER_LINEAR)
    crop_upscale_time = time.perf_counter() - stage_start
    return upscaled_gray_image, crop_upscale_time, grayscale_time


def preprocess_roi(roi, process_type):
    """Prepare an ROI for template matching using the per-target preprocessing path."""
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
    """Convert a frame number into a whole-second HH:MM:SS label."""
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"
