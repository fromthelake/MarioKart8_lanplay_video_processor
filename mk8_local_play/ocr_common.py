import csv
from pathlib import Path

import cv2
import numpy as np

from .score_layouts import DEFAULT_SCORE_LAYOUT_ID

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720


def calculate_sum_intensity(gray_image: np.ndarray):
    """Return row/column intensity sums used to find the active game area."""
    sum_row_intensity = np.sum(gray_image, axis=1)
    sum_col_intensity = np.sum(gray_image, axis=0)
    return sum_row_intensity, sum_col_intensity


def find_borders(sum_row_intensity: np.ndarray, sum_col_intensity: np.ndarray, threshold: int = 15000):
    """Find non-black content borders inside a captured frame."""
    top = next((i for i, val in enumerate(sum_row_intensity) if val > threshold), 0)
    bottom = next((i for i, val in enumerate(reversed(sum_row_intensity)) if val > threshold), 0)
    bottom = len(sum_row_intensity) - bottom
    left = next((i for i, val in enumerate(sum_col_intensity) if val > threshold), 0)
    right = next((i for i, val in enumerate(reversed(sum_col_intensity)) if val > threshold), 0)
    right = len(sum_col_intensity) - right
    return top, left, bottom, right


def determine_scaling(frame: np.ndarray):
    """Determine the crop bounds and upscale factor for OCR frames."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sum_row_intensity, sum_col_intensity = calculate_sum_intensity(gray_frame)
    top, left, bottom, right = find_borders(sum_row_intensity, sum_col_intensity)
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    return left, top, crop_width, crop_height


def crop_and_upscale_frame(frame: np.ndarray) -> np.ndarray:
    """Crop the detected game area and resize it to the fixed OCR working size."""
    left, top, crop_width, crop_height = determine_scaling(frame)
    cropped_frame = frame[top:top + crop_height, left:left + crop_width]
    return cv2.resize(cropped_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)


def load_exported_frame_metadata(base_dir: Path):
    """Load exported frame metadata written during extraction, if present."""
    metadata_path = base_dir / "Output_Results" / "Debug" / "exported_frame_metadata.csv"
    if not metadata_path.exists():
        return {}
    metadata_index = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            video_name = str(row.get("Video", "")).strip()
            race_number_text = str(row.get("Race", "")).strip()
            kind = str(row.get("Kind", "")).strip()
            if not video_name or not race_number_text or not kind:
                continue
            try:
                race_number = int(race_number_text)
            except ValueError:
                continue
            metadata_index[(video_name, race_number, kind)] = {
                "video": video_name,
                "race": race_number,
                "kind": kind,
                "requested_frame": int(row.get("Requested Frame", 0) or 0),
                "actual_frame": int(row.get("Actual Frame", 0) or 0),
                "score_layout_id": str(row.get("Score Layout", "")).strip() or DEFAULT_SCORE_LAYOUT_ID,
            }
    return metadata_index


def find_metadata_entry(metadata_index, race_class: str, race_id_number: int, kind: str):
    """Look up metadata for one exported frame bundle."""
    for (video_name, race_number, entry_kind), value in metadata_index.items():
        if race_number != race_id_number or entry_kind != kind:
            continue
        if Path(video_name).stem == race_class or str(video_name) == race_class:
            return value
    return None


def load_consensus_frames(image_path: str, metadata_entry, input_videos_folder: Path,
                          consensus_size: int, in_memory_frames=None):
    """Reload neighbouring frames around an exported frame for OCR voting."""
    if in_memory_frames:
        return in_memory_frames
    fallback_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if fallback_image is None:
        return []
    if metadata_entry is None:
        return [fallback_image]

    video_value = str(metadata_entry["video"])
    video_path = Path(video_value)
    if not video_path.is_absolute():
        video_path = input_videos_folder / video_value
    if not video_path.exists():
        return [fallback_image]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [fallback_image]

    radius = max(0, consensus_size // 2)
    actual_frame = int(metadata_entry["actual_frame"])
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, actual_frame - radius)
    end_frame = min(total_frames, actual_frame + radius + 1)
    frames = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _frame_number in range(start_frame, end_frame):
        ret, frame = capture.read()
        if not ret:
            continue
        frames.append(crop_and_upscale_frame(frame))
    capture.release()
    return frames or [fallback_image]
