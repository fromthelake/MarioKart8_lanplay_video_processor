import os
import argparse
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw
import pytesseract
import pandas as pd
from typing import List, Dict, Tuple
import logging
from PIL import Image
import time
import sys
import difflib
import openpyxl
import textdistance
import re
import threading
from datetime import datetime
from jellyfish import soundex
from collections import defaultdict, Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from app_runtime import configure_tesseract, load_app_config
from console_logging import LOGGER
from track_metadata import load_track_tuples

# Record the start time
start_run_time = time.time()
APP_CONFIG = load_app_config()
OCR_WORKERS = APP_CONFIG.ocr_workers
OCR_CONSENSUS_FRAMES = APP_CONFIG.ocr_consensus_frames
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


PLAYER_NAME_COORDS = [
    ((428, 52), (620, 96)), ((428, 104), (620, 148)),
    ((428, 156), (620, 200)), ((428, 208), (620, 252)),
    ((428, 260), (620, 304)), ((428, 312), (620, 356)),
    ((428, 364), (620, 408)), ((428, 416), (620, 460)),
    ((428, 468), (620, 512)), ((428, 520), (620, 564)),
    ((428, 572), (620, 617)), ((428, 624), (620, 669))
]


class OcrProfiler:
    def __init__(self):
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"calls": 0, "seconds": 0.0})

    def record(self, label: str, duration_s: float) -> None:
        with self._lock:
            self._stats[label]["calls"] += 1
            self._stats[label]["seconds"] += duration_s

    def summary_lines(self) -> List[str]:
        with self._lock:
            stats = {key: value.copy() for key, value in self._stats.items()}
        if not stats:
            return ["Tesseract calls: none recorded"]
        total_calls = sum(item["calls"] for item in stats.values())
        total_seconds = sum(item["seconds"] for item in stats.values())
        lines = [f"Tesseract calls: {total_calls} | OCR engine time: {total_seconds:.2f}s"]
        for label, item in sorted(stats.items(), key=lambda pair: pair[1]["seconds"], reverse=True):
            avg_ms = (item["seconds"] / max(1, item["calls"])) * 1000.0
            lines.append(
                f"- {label}: {item['calls']} calls | {item['seconds']:.2f}s total | {avg_ms:.1f} ms/call"
            )
        return lines


OCR_PROFILER = OcrProfiler()


class ProgressPrinter:
    """Print throttled progress updates for OCR/export stages."""

    def __init__(self, scope: str, total_units: int, percent_step: int = 10, min_interval_s: float = 2.0):
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

    def update(self, completed_units: int, detail: str = "") -> None:
        percent = min(100, int((max(0, completed_units) / self.total_units) * 100))
        now = time.perf_counter()
        should_print = percent >= 100 or self.last_percent < 0
        if not should_print and percent >= self.last_percent + self.percent_step:
            should_print = True
        if not should_print and now - self.last_print_time >= self.min_interval_s:
            should_print = True
        if not should_print:
            return
        detail_suffix = f" | {detail}" if detail else ""
        snapshot = LOGGER.resources.sample()
        self._update_phase_peak(snapshot)
        resource_text = LOGGER.resource_text(snapshot)
        LOGGER.log(
            self.scope,
            f"{completed_units}/{self.total_units} ({percent}%) | {resource_text}{detail_suffix}",
            color_name="magenta",
        )
        self.last_percent = percent
        self.last_print_time = now

    def heartbeat(self, completed_units: int, detail: str = "") -> None:
        now = time.perf_counter()
        if now - self.last_print_time < self.min_interval_s:
            return
        percent = min(100, int((max(0, completed_units) / self.total_units) * 100))
        snapshot = LOGGER.resources.sample()
        self._update_phase_peak(snapshot)
        resource_text = LOGGER.resource_text(snapshot)
        detail_suffix = f" | {detail}" if detail else ""
        LOGGER.log(
            self.scope,
            f"{completed_units}/{self.total_units} ({percent}%) | {resource_text}{detail_suffix}",
            color_name="magenta",
        )
        self.last_print_time = now

    def _update_phase_peak(self, snapshot) -> None:
        for field in ("cpu_percent", "ram_used_gb", "gpu_percent", "vram_used_gb"):
            current = getattr(snapshot, field)
            peak_value = self.phase_peak[field]
            if current is not None and (peak_value is None or current > peak_value):
                self.phase_peak[field] = current
        for field in ("ram_total_gb", "vram_total_gb"):
            current = getattr(snapshot, field)
            if current is not None:
                self.phase_peak[field] = current

    def peak_lines(self) -> list[str]:
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


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"

def apply_threshold(image: np.ndarray, threshold: int = 205) -> np.ndarray:
    """Apply a binary threshold to the image."""
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def apply_inversion(image: np.ndarray) -> np.ndarray:
    """Invert the binary image."""
    return cv2.bitwise_not(image)

def preprocess_image_v_channel(image: np.ndarray, threshold_value: int = 205) -> np.ndarray:
    """Preprocess the image using the V channel from the HSV color space."""
    if len(image.shape) == 2:  # If the image is grayscale, convert it to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv_image)
    _, binary_v = cv2.threshold(v, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_v

def get_race_points(position: int, num_players: int) -> int:
    """Return the points based on position and number of players."""
    # Define point tables for different player counts
    points_table = {
        12: [15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        11: [13, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        10: [12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        9: [11, 9, 8, 6, 5, 4, 3, 2, 1, 0],
        8: [10, 8, 6, 5, 4, 3, 2, 1, 0],
        7: [9, 7, 5, 4, 3, 2, 1, 0],
    }

    # Default to 12 players if not explicitly defined
    if num_players not in points_table:
        num_players = 12

    # Get the points for the given position
    if 1 <= position <= len(points_table[num_players]):
        return points_table[num_players][position - 1]
    else:
        return 0  # No points for invalid positions

def crop_and_process_image(frame: np.ndarray, coordinates: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                           image_type: str) -> List[np.ndarray]:
    """Crop and process specific regions of an image based on coordinates and image type."""
    cropped_images = []
    for (x1, y1), (x2, y2) in coordinates:
        section_img = frame[y1:y2, x1:x2]
        threshold_value = 205
        binary_section = preprocess_image_v_channel(section_img, threshold_value)
        black_pixels = np.count_nonzero(binary_section == 0)
        white_pixels = np.count_nonzero(binary_section == 255)

        if white_pixels > black_pixels:
            inverted_section = cv2.bitwise_not(binary_section)
            section_img = Image.fromarray(inverted_section)
        else:
            section_img = Image.fromarray(binary_section)

        # Apply dilation if the image_type is "race_points" or "total_points"
        if image_type in ["race_points", "total_points"]:
            # Convert the PIL image to a NumPy array (OpenCV image)
            section_img_np = np.array(section_img)
            if len(section_img_np.shape) == 3 and section_img_np.shape[2] == 3:
                # Convert to grayscale if it is an RGB image
                gray_image = cv2.cvtColor(section_img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = section_img_np

            # Define the kernel for dilation
            kernel = np.ones((2, 2), np.uint8)
            # Apply dilation
            dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
            # Convert the dilated image back to a PIL image
            section_img = Image.fromarray(cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2RGB))

        cropped_images.append(np.array(section_img))
    return cropped_images

def process_image(image_source) -> np.ndarray:
    """Process an image by cropping and modifying specified regions based on coordinates."""
    coordinates = {
        "player_position": [
            ((310, 52), (371, 96)), ((310, 104), (371, 148)),
            ((310, 156), (371, 200)), ((310, 208), (371, 252)),
            ((310, 260), (371, 304)), ((310, 312), (371, 356)),
            ((310, 364), (371, 408)), ((310, 416), (371, 460)),
            ((310, 468), (371, 512)), ((310, 520), (371, 564)),
            ((310, 572), (371, 617)), ((310, 624), (371, 669))
        ],
        "player_name": [
            ((428, 52), (620, 96)), ((428, 104), (620, 148)),
            ((428, 156), (620, 200)), ((428, 208), (620, 252)),
            ((428, 260), (620, 304)), ((428, 312), (620, 356)),
            ((428, 364), (620, 408)), ((428, 416), (620, 460)),
            ((428, 468), (620, 512)), ((428, 520), (620, 564)),
            ((428, 572), (620, 617)), ((428, 624), (620, 669))
        ],
        "race_points": [
            ((825, 52), (861, 96)), ((825, 104), (861, 148)),
            ((825, 156), (861, 200)), ((825, 208), (861, 252)),
            ((825, 260), (861, 304)), ((825, 312), (861, 356)),
            ((825, 364), (861, 408)), ((825, 416), (861, 460)),
            ((825, 468), (861, 512)), ((825, 520), (861, 564)),
            ((825, 572), (861, 617)), ((825, 624), (861, 669))
        ],
        "total_points": [
            ((910, 52), (973, 96)), ((910, 104), (973, 148)),
            ((910, 156), (973, 200)), ((910, 208), (973, 252)),
            ((910, 260), (973, 304)), ((910, 312), (973, 356)),
            ((910, 364), (973, 408)), ((910, 416), (973, 460)),
            ((910, 468), (973, 512)), ((910, 520), (973, 564)),
            ((910, 572), (973, 617)), ((910, 624), (973, 669))
        ]
    }

    if isinstance(image_source, str):
        image = cv2.imread(image_source, cv2.IMREAD_COLOR)
        image_path = image_source
    else:
        image = image_source
        image_path = "<array>"
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Preprocess the image using the V channel
    processed_image = preprocess_image_v_channel(image)

    # Convert to 3-channel if processed_image is single-channel
    if len(processed_image.shape) == 2:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    # Process each region type and its coordinates
    for region_type, coord_list in coordinates.items():
        rois = crop_and_process_image(processed_image, coord_list, region_type)
        for roi, ((x1, y1), (x2, y2)) in zip(rois, coord_list):
            # Ensure roi is 3-channel
            if len(roi.shape) == 2:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            processed_image[y1:y2, x1:x2] = roi
    return processed_image

def is_white_box(image: Image.Image, top_left: Tuple[int, int], box_size: Tuple[int, int] = (3, 2)) -> bool:
    """Determine if a 3x2 pixel box is white based on RGB values."""
    x, y = top_left
    width, height = box_size
    white_pixels = 0
    total_pixels = width * height

    for i in range(width):
        for j in range(height):
            r, g, b = image.getpixel((x + i, y + j))
            if r > 180 and g > 180 and b > 180:
                white_pixels += 1

    return white_pixels >= total_pixels / 2

def identify_digit(image: Image.Image, box_top_left: Tuple[int, int], red_pixels: Dict[str, Tuple[int, int]]) -> int:
    """Identify the digit in a cropped region of an image based on pixel analysis."""
    white_pixels = {label: is_white_box(image, (box_top_left[0] + x, box_top_left[1] + y))
                    for label, (x, y) in red_pixels.items()}

    digit_patterns = [
        (8, {"top_middle", "left_middle", "right_middle", "center", "right_bottom", "left_bottom", "middle_bottom_edge"}),
        (0, {"top_middle", "left_middle", "right_middle", "left_bottom", "right_bottom", "middle_bottom_edge"}),
        (6, {"top_middle", "left_middle", "center", "right_bottom", "left_bottom", "middle_bottom_edge"}),
        (9, {"top_middle", "left_middle", "right_middle", "center", "right_bottom", "middle_bottom_edge"}),
        (2, {"top_middle", "right_middle", "center", "left_bottom", "middle_bottom_edge"}),
        (3, {"top_middle", "right_middle", "center", "right_bottom", "middle_bottom_edge"}),
        (5, {"top_middle", "left_middle", "center", "right_bottom", "middle_bottom_edge"}),
        (4, {"right_middle", "left_middle", "center", "right_bottom"}),
        (7, {"top_middle", "right_middle", "right_bottom"}),
        (1, {"middle_middle", "middle_bottom"})
    ]

    for digit, pattern in digit_patterns:
        if all(white_pixels.get(label, False) for label in pattern):
            return digit

    return -1

def detect_digits_in_image(image: Image.Image, start_coords: List[Tuple[int, int]], row_offset: int,
                           box_dims: Tuple[int, int], red_pixels: Dict[str, Tuple[int, int]],
                           num_rows: int, boxes_per_row: int) -> List[str]:
    """Detect digits in an image based on given coordinates and draw red pixels."""
    coordinate_set = []
    draw = ImageDraw.Draw(image)

    for i in range(num_rows):
        y_offset = i * row_offset
        row_number = ""

        for j in range(boxes_per_row):
            start_x, start_y = start_coords[j]
            top_left = (start_x, start_y + y_offset)
            digit = identify_digit(image, top_left, red_pixels)
            if digit != -1:
                row_number += str(digit)
            # Draw a red rectangle at the center of the digit box
            for label, (x, y) in red_pixels.items():
                rect_top_left = (top_left[0] + x, top_left[1] + y)
                rect_bottom_right = (rect_top_left[0] + 3, rect_top_left[1] + 2)
                draw.rectangle([rect_top_left, rect_bottom_right], outline='red', fill='red')

        coordinate_set.append(row_number)

    return coordinate_set

def scale_coords(coords, scale_factor):
    """Scale coordinates by a given factor."""
    return [(x * scale_factor, y * scale_factor) for x, y in coords]

def scale_pixel_positions(pixels, scale_factor):
    """Scale pixel positions by a given factor."""
    return {label: (x * scale_factor, y * scale_factor) for label, (x, y) in pixels.items()}


def extract_text_with_confidence(image_source, coordinates: Dict[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]],
                                 lang: str, config: str) -> Tuple[Dict[str, List[str]], List[int]]:
    """Extract text and confidence scores from image ROIs using Tesseract OCR."""
    extracted_text = {}
    confidence_scores = []
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_source}")
    else:
        image = image_source
        if image is None:
            raise ValueError("Image source array is None")

    for region_type, coord_list in coordinates.items():
        region_text = []
        region_confidence = []
        for (x1, y1), (x2, y2) in coord_list:
            roi = image[y1:y2, x1:x2]
            data = run_tesseract_image_to_data(roi, lang, config, f"{region_type}_roi")

            # Combine all words with a space
            words_in_roi = []
            confidences_in_roi = []
            for idx, conf in enumerate(data["conf"]):
                if conf >= 0 and data["text"][idx].strip():  # Only valid words
                    words_in_roi.append(data["text"][idx].strip())
                    confidences_in_roi.append(conf)

            # Combine words into a single string and calculate average confidence
            combined_text = " ".join(words_in_roi)
            #print(f"text: {data["text"]} conf:{data["conf"]}")
            average_confidence = sum(confidences_in_roi) // len(confidences_in_roi) if confidences_in_roi else 0

            region_text.append(combined_text)
            region_confidence.append(average_confidence)

        extracted_text[region_type] = region_text  # List of combined texts for each ROI
        confidence_scores.extend(region_confidence)

    return extracted_text, confidence_scores


def run_tesseract_image_to_data(image: np.ndarray, lang: str, config: str, profile_label: str):
    start_time = time.perf_counter()
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
    OCR_PROFILER.record(profile_label, time.perf_counter() - start_time)
    return data


def extract_player_names_batched(
    image: np.ndarray,
    coord_list: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    lang: str = "eng",
    config: str = "--psm 6",
) -> Tuple[List[str], List[int]]:
    rois = []
    widths = []
    heights = []
    for (x1, y1), (x2, y2) in coord_list:
        roi = image[y1:y2, x1:x2]
        rois.append(roi)
        heights.append(max(1, roi.shape[0]))
        widths.append(max(1, roi.shape[1]))

    separator_height = 10
    canvas_width = max(widths)
    canvas_height = sum(heights) + separator_height * (len(rois) - 1)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    row_ranges = []
    cursor = 0
    for roi in rois:
        height, width = roi.shape[:2]
        canvas[cursor:cursor + height, 0:width] = roi
        row_ranges.append((cursor, cursor + height))
        cursor += height + separator_height

    data = run_tesseract_image_to_data(canvas, lang, config, "player_name_batch")
    texts_by_row = [[] for _ in rois]
    confidences_by_row = [[] for _ in rois]
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        conf = data["conf"][index]
        if conf < 0 or not text:
            continue
        y = int(data["top"][index])
        height = int(data["height"][index])
        center_y = y + max(1, height // 2)
        for row_index, (start_y, end_y) in enumerate(row_ranges):
            if start_y <= center_y < end_y:
                texts_by_row[row_index].append(text)
                confidences_by_row[row_index].append(conf)
                break

    extracted_names = []
    confidence_scores = []
    for row_index, ((x1, y1), (x2, y2)) in enumerate(coord_list):
        combined_text = " ".join(texts_by_row[row_index]).strip()
        if confidences_by_row[row_index]:
            average_confidence = int(sum(confidences_by_row[row_index]) // len(confidences_by_row[row_index]))
        else:
            average_confidence = 0

        stripped_name = re.sub(r"[^a-zA-Z0-9]", "", combined_text)
        if average_confidence < 85 or len(stripped_name) < 3 or len(set(stripped_name)) < 3:
            roi = image[y1:y2, x1:x2]
            fallback_data = run_tesseract_image_to_data(roi, lang, "--psm 7", "player_name_row_fallback")
            words_in_roi = []
            confidences_in_roi = []
            for text_index, fallback_conf in enumerate(fallback_data["conf"]):
                fallback_text = fallback_data["text"][text_index].strip()
                if fallback_conf >= 0 and fallback_text:
                    words_in_roi.append(fallback_text)
                    confidences_in_roi.append(fallback_conf)
            combined_text = " ".join(words_in_roi).strip()
            average_confidence = int(sum(confidences_in_roi) // len(confidences_in_roi)) if confidences_in_roi else 0

        extracted_names.append(combined_text)
        confidence_scores.append(average_confidence)

    return extracted_names, confidence_scores


tracks_list = load_track_tuples()

def match_track_name(track_name: str, track_list: List[Tuple[int, str, str, int, str]]) -> str:
    """Match a track name (in English or Dutch) to its English equivalent."""
    best_match_score = 0
    best_match_english_name = track_name  # Default to input if no good match is found

    # Iterate through tracks_list
    for track in track_list:
        english_name = track[1]
        dutch_name = track[2]

        # Compute similarity scores
        score_english = difflib.SequenceMatcher(None, track_name, english_name).ratio()
        score_dutch = difflib.SequenceMatcher(None, track_name, dutch_name).ratio()

        # Debugging output
        #print(f"Checking track: {track_name}")
        #print(f"Against English: {english_name}, Score: {score_english}")
        #print(f"Against Dutch: {dutch_name}, Score: {score_dutch}")

        # Update the best match if the current score is higher
        if score_english > best_match_score:
            best_match_score = score_english
            best_match_english_name = english_name

        if score_dutch > best_match_score:
            best_match_score = score_dutch
            best_match_english_name = english_name

    # Debugging final match
    return best_match_english_name


def get_cup_name(track_name: str, track_list: List[Tuple[int, str, str, int, str]]) -> str:
    """Get the Cup Name corresponding to a track."""
    for track in track_list:
        if track_name == track[1]:  # Match English name
            return track[4]
    return ""

def preprocess_name(name):
    """Preprocess the name by removing special characters, normalizing case and whitespace."""
    name = re.sub(r'\W+', ' ', name)  # Remove non-alphanumeric characters
    name = name.strip().lower()  # Normalize case and strip leading/trailing whitespace
    name = re.sub(r'\s+', ' ', name)  # Normalize multiple spaces to single space
    return name

def soundex_similarity(name1, name2):
    """Calculate Soundex similarity as binary (0 or 1)."""
    return 1 if soundex(name1) == soundex(name2) else 0

def weighted_similarity(name1, name2):
    """Calculate a weighted similarity score based on difflib, string length, Jaro-Winkler, and Soundex."""
    # Preprocess names
    name1 = preprocess_name(name1)
    name2 = preprocess_name(name2)

    # Difflib similarity ratio
    difflib_score = difflib.SequenceMatcher(None, name1, name2).ratio()

    # String length difference (normalized)
    length_score = 1 - abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)

    # Jaro-Winkler similarity ratio
    jaro_winkler_score = textdistance.jaro_winkler(name1, name2)

    # Soundex similarity (binary)
    soundex_score = soundex_similarity(name1, name2)

    # Weights for each score
    weights = {
        'difflib': 0.3,
        'length': 0.4,
        'jaro_winkler': 0.3,
        'soundex': 0.0
    }

    # Calculate weighted score
    weighted_score = (weights['difflib'] * difflib_score +
                      weights['length'] * length_score +
                      weights['jaro_winkler'] * jaro_winkler_score +
                      weights['soundex'] * soundex_score)

    return weighted_score

def match_names(previous_names_with_indices, current_names):
    """Match names between a list of previous names with indices and current names based on exact match and weighted similarity."""
    matches = {}
    used_previous_names = set()
    used_current_names = set()
    exact_match_rows = set()

    previous_names = [name for name, _ in previous_names_with_indices]

    # Step 1: Exact Match
    for curr_name in current_names:
        if curr_name in previous_names:
            idx = previous_names.index(curr_name)
            row_idx = previous_names_with_indices[idx][1]
            matches[curr_name] = (curr_name, row_idx)
            used_previous_names.add(curr_name)
            used_current_names.add(curr_name)
            exact_match_rows.add(row_idx)

    # Step 2: Similarity Scores for remaining names, excluding exact match rows
    similarity_scores = []
    for curr_name in current_names:
        if curr_name not in used_current_names:
            for prev_name, prev_idx in previous_names_with_indices:
                if prev_name not in used_previous_names and prev_idx not in exact_match_rows:
                    score = weighted_similarity(prev_name, curr_name)
                    similarity_scores.append((score, prev_name, curr_name, prev_idx))

    # Sort similarity scores in descending order
    similarity_scores.sort(reverse=True, key=lambda x: x[0])

    # Match based on similarity scores
    for score, prev_name, curr_name, prev_idx in similarity_scores:
        if curr_name not in used_current_names and prev_name not in used_previous_names:
            matches[curr_name] = (prev_name, prev_idx)
            used_current_names.add(curr_name)
            used_previous_names.add(prev_name)

    # Handle remaining unmatched names by assigning them in order
    remaining_current_names = [name for name in current_names if name not in used_current_names]
    remaining_previous_names = [name for name, idx in previous_names_with_indices if name not in used_previous_names]

    for curr_name, prev_name in zip(remaining_current_names, remaining_previous_names):
        prev_idx = previous_names.index(prev_name)
        matches[curr_name] = (prev_name, prev_idx)

    # Add unmatched current names as-is if they have no match at all
    for curr_name in remaining_current_names[len(remaining_previous_names):]:
        matches[curr_name] = (curr_name, None)

    return matches

def build_name_links(df, output_folder):
    """Build name links across races for each RaceClass."""
    name_links = defaultdict(list)
    all_player_names_df = {}

    for race_class, group in df.groupby('RaceClass'):
        races = sorted(group['RaceIDNumber'].unique())
        num_races = len(races)

        max_players = group.groupby('RaceIDNumber').size().max()

        player_names_df = pd.DataFrame(index=range(max_players), columns=range(num_races))

        initial_race_id = races[0]
        initial_names = group[group['RaceIDNumber'] == initial_race_id]['PlayerName'].tolist()

        # Ensure the lengths match by padding or truncating
        player_names_df[0] = initial_names + [None] * (len(player_names_df) - len(initial_names))

        if num_races < 2:
            single_race_df = group[['PlayerName', 'RaceIDNumber']].copy()
            single_race_df.columns = ['PlayerName', 0]  # Rename columns to fit expected format
            single_race_df = single_race_df.set_index('PlayerName').T
            all_player_names_df[race_class] = single_race_df
            continue

        for col_idx, race_id in enumerate(races[1:], start=1):
            current_race_names = group[group['RaceIDNumber'] == race_id]['PlayerName'].tolist()

            # Collect all names from previous rounds along with their indices
            all_previous_names_with_indices = [(player_names_df.at[row_idx, prev_col], row_idx)
                                               for prev_col in range(col_idx)
                                               for row_idx in range(max_players)
                                               if pd.notna(player_names_df.at[row_idx, prev_col])]

            # Match names using the improved match_names function
            matches = match_names(all_previous_names_with_indices, current_race_names)

            used_rows_exact = set()  # Track rows used by exact matches
            used_rows_similarity = set()  # Track rows used by similarity matches

            # Assign exact matched names to the DataFrame using the row indices
            for curr_name in current_race_names:
                if curr_name in matches:
                    matched_name, row_idx = matches[curr_name]
                    if row_idx is not None and row_idx not in used_rows_exact:
                        player_names_df.at[row_idx, col_idx] = curr_name
                        used_rows_exact.add(row_idx)

            # Assign similarity matched names to remaining available rows
            for curr_name in current_race_names:
                if curr_name in matches and curr_name not in player_names_df.iloc[:, col_idx].dropna().values:
                    matched_name, row_idx = matches[curr_name]
                    if row_idx is not None and row_idx not in used_rows_exact and row_idx not in used_rows_similarity:
                        for i in range(max_players):
                            if i not in used_rows_exact and i not in used_rows_similarity and pd.isna(player_names_df.at[i, col_idx]):
                                player_names_df.at[i, col_idx] = curr_name
                                used_rows_similarity.add(i)
                                break

            # Fill remaining unmatched names
            used_names = set(player_names_df.iloc[:, col_idx].dropna().values)
            for row_idx, name in enumerate(current_race_names):
                if name not in used_names:
                    for i in range(max_players):
                        if i not in used_rows_exact and i not in used_rows_similarity and pd.isna(player_names_df.at[i, col_idx]):
                            player_names_df.at[i, col_idx] = name
                            used_names.add(name)
                            used_rows_similarity.add(i)
                            break

        for idx in range(max_players):
            name_link = player_names_df.loc[idx].dropna().tolist()
            if name_link:
                most_common_name = Counter(name_link).most_common(1)[0][0]
                for name in name_link:
                    name_links[name].append(most_common_name)

        if APP_CONFIG.write_debug_linking_excel:
            output_path = os.path.join(output_folder, f'linking_{race_class}.xlsx')
            player_names_df.to_excel(output_path, index=False)

        all_player_names_df[race_class] = player_names_df

    return name_links, all_player_names_df



def choose_canonical_name(name_link, group):
    """Choose a canonical name using all linked observations and OCR confidence."""
    candidates = [normalize_name_for_vote(name) for name in name_link if normalize_name_for_vote(name)]
    if not candidates:
        return ""

    candidate_counts = Counter(candidates)
    confidence_lookup = defaultdict(list)
    for candidate in candidates:
        matching_rows = group[group["PlayerName"] == candidate]
        if not matching_rows.empty:
            confidence_lookup[candidate].extend(matching_rows["NameConfidence"].tolist())

    best_name = candidates[0]
    best_score = float("-inf")
    unique_candidates = sorted(candidate_counts)
    for candidate in unique_candidates:
        support_score = 0.0
        for observed_name in candidates:
            weight = 1.0 + (sum(confidence_lookup.get(observed_name, [0])) / max(1, len(confidence_lookup.get(observed_name, [])))) / 100.0
            support_score += weighted_similarity(candidate, observed_name) * weight
        quality = len(set(re.sub(r"[^a-zA-Z0-9]", "", candidate)))
        confidence_bonus = sum(confidence_lookup.get(candidate, [0])) / max(1, len(confidence_lookup.get(candidate, [])))
        score = support_score + candidate_counts[candidate] * 2.5 + quality * 0.3 + confidence_bonus / 100.0
        if score > best_score:
            best_score = score
            best_name = candidate
    return best_name


def standardize_names(player_names_df, group):
    """Determine a canonical name for each linked row using all linked evidence."""
    name_mapping = {}
    standardized_names = {}

    for idx in range(len(player_names_df)):
        name_link = player_names_df.loc[idx].dropna().tolist()
        if name_link:
            most_common_name = choose_canonical_name(name_link, group)
            for name in name_link:
                name_mapping[name] = (most_common_name, idx)  # Store the name and the row index
                standardized_names[idx] = most_common_name  # Store the standardized name by row index

    return name_mapping, standardized_names

def standardize_player_names(df, output_folder):
    """Standardize player names within each RaceClass."""
    standardized_names = pd.DataFrame()
    name_mapping = {}
    standardized_names_dict = {}
    name_links, all_player_names_df = build_name_links(df, output_folder)

    for race_class, player_names_df in all_player_names_df.items():

        # Handle cases where there is only one race
        if player_names_df.shape[1] < 2:
            group = df[df['RaceClass'] == race_class].copy()
            group.loc[:, 'FixPlayerName'] = group['PlayerName']
            standardized_names = pd.concat([standardized_names, group], ignore_index=True)
            continue

        group = df[df['RaceClass'] == race_class].copy()
        local_name_mapping, local_standardized_names_dict = standardize_names(player_names_df, group)
        name_mapping.update(local_name_mapping)
        standardized_names_dict.update(local_standardized_names_dict)

        def get_standardized_name(row):
            player_name = row['PlayerName']
            if player_name in local_name_mapping:
                return standardized_names_dict[local_name_mapping[player_name][1]]
            else:
                return player_name  # Fallback to original name if not found

        group.loc[:, 'FixPlayerName'] = group.apply(get_standardized_name, axis=1)

        standardized_names = pd.concat([standardized_names, group], ignore_index=True)


    return standardized_names


def parse_detected_int(value: str) -> int | None:
    if value is None:
        return None
    stripped = re.sub(r"[^0-9]", "", str(value))
    if not stripped:
        return None
    return int(stripped)


def calculate_sum_intensity(gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.sum(gray_image, axis=1), np.sum(gray_image, axis=0)


def find_borders(sum_row_intensity: np.ndarray, sum_col_intensity: np.ndarray, threshold: int = 15000) -> Tuple[int, int, int, int]:
    top = next((i for i, val in enumerate(sum_row_intensity) if val > threshold), 0)
    bottom = next((i for i, val in enumerate(reversed(sum_row_intensity)) if val > threshold), 0)
    bottom = len(sum_row_intensity) - bottom
    left = next((i for i, val in enumerate(sum_col_intensity) if val > threshold), 0)
    right = next((i for i, val in enumerate(reversed(sum_col_intensity)) if val > threshold), 0)
    right = len(sum_col_intensity) - right
    return top, left, bottom, right


def determine_scaling(frame: np.ndarray) -> Tuple[int, int, int, int]:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sum_row_intensity, sum_col_intensity = calculate_sum_intensity(gray_frame)
    top, left, bottom, right = find_borders(sum_row_intensity, sum_col_intensity)
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    return left, top, crop_width, crop_height


def crop_and_upscale_frame(frame: np.ndarray) -> np.ndarray:
    left, top, crop_width, crop_height = determine_scaling(frame)
    cropped = frame[top:top + crop_height, left:left + crop_width]
    return cv2.resize(cropped, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)


def load_exported_frame_metadata(base_dir: Path) -> Dict[Tuple[str, int, str], Dict[str, int | str]]:
    metadata_path = base_dir / "Output_Results" / "Debug" / "exported_frame_metadata.csv"
    if not metadata_path.exists():
        return {}

    metadata_index = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            try:
                key = (row["Video"], int(row["Race"]), row["Kind"])
                metadata_index[key] = {
                    "video": row["Video"],
                    "race": int(row["Race"]),
                    "kind": row["Kind"],
                    "requested_frame": int(row["Requested Frame"]),
                    "actual_frame": int(row["Actual Frame"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return metadata_index


def load_consensus_frames(image_path: str, metadata_entry: Dict[str, int | str] | None, input_videos_folder: Path,
                          consensus_size: int, in_memory_frames: List[np.ndarray] | None = None) -> List[np.ndarray]:
    if in_memory_frames:
        return in_memory_frames
    fallback_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if fallback_image is None:
        return []
    if metadata_entry is None:
        return [fallback_image]

    video_path = input_videos_folder / str(metadata_entry["video"])
    if not video_path.exists():
        return [fallback_image]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [fallback_image]

    radius = max(0, consensus_size // 2)
    actual_frame = int(metadata_entry["actual_frame"])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampled_frames = []
    start_frame = max(0, actual_frame - radius)
    end_frame = min(total_frames, actual_frame + radius + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            continue
        sampled_frames.append(crop_and_upscale_frame(frame))

    cap.release()
    return sampled_frames or [fallback_image]


def find_metadata_entry(metadata_index: Dict[Tuple[str, int, str], Dict[str, int | str]], race_class: str,
                        race_id_number: int, kind: str) -> Dict[str, int | str] | None:
    for (video_name, race_number, entry_kind), value in metadata_index.items():
        if race_number == race_id_number and entry_kind == kind and Path(video_name).stem == race_class:
            return value
    return None


def score_digit_layout(scale_factor: int = 5):
    start_coords_run1 = scale_coords([(830, 71), (843, 71)], scale_factor)
    red_pixels_run1 = scale_pixel_positions(
        {
            "top_middle": (7, 2), "left_middle": (2, 5), "middle_middle": (7, 5),
            "right_middle": (11, 5), "left_bottom": (2, 13), "middle_bottom": (7, 13),
            "right_bottom": (11, 13), "middle_bottom_edge": (7, 17), "center": (7, 9)
        },
        scale_factor,
    )

    start_coords_run2 = scale_coords([(916, 66), (933, 66), (950, 66)], scale_factor)
    red_pixels_run2 = scale_pixel_positions(
        {
            "top_middle": (8, 2), "left_middle": (2, 7), "middle_middle": (8, 7),
            "right_middle": (13, 7), "left_bottom": (2, 16), "middle_bottom": (8, 16),
            "right_bottom": (13, 16), "middle_bottom_edge": (8, 21), "center": (8, 11)
        },
        scale_factor,
    )
    return {
        "race_points": (start_coords_run1, 52 * scale_factor, (13 * scale_factor, 19 * scale_factor), red_pixels_run1, 12, 2),
        "total_points": (start_coords_run2, 52 * scale_factor, (16 * scale_factor, 24 * scale_factor), red_pixels_run2, 12, 3),
    }


def extract_scoreboard_observation(frame_image: np.ndarray, annotate_path: str | None = None) -> Dict[str, object]:
    processed_img = process_image(frame_image)
    processed_img_pil = Image.fromarray(processed_img).convert('RGB')
    scale_factor = 5
    scaled_image = processed_img_pil.resize(
        (processed_img_pil.width * scale_factor, processed_img_pil.height * scale_factor), Image.NEAREST
    )
    layout = score_digit_layout(scale_factor)
    race_points = detect_digits_in_image(scaled_image, *layout["race_points"])
    total_points = detect_digits_in_image(scaled_image, *layout["total_points"])

    scaled_image_resized = scaled_image.resize((processed_img_pil.width, processed_img_pil.height), Image.NEAREST)
    annotated_image = cv2.cvtColor(np.array(scaled_image_resized), cv2.COLOR_RGB2BGR)
    if annotate_path:
        scaled_image_resized.save(annotate_path)

    names, confidence_scores = extract_player_names_batched(annotated_image, PLAYER_NAME_COORDS)
    valid_rows = []
    for index, player_name in enumerate(names):
        stripped_name = re.sub(r'[^a-zA-Z0-9]', '', player_name)
        confidence = confidence_scores[index] if index < len(confidence_scores) else 0
        valid_rows.append(
            len(stripped_name) >= 3 and len(set(stripped_name)) >= 3
            or bool(race_points[index])
            or bool(total_points[index])
            or confidence >= 35
        )

    visible_rows = 0
    for index, row_present in enumerate(valid_rows, start=1):
        if row_present:
            visible_rows = index

    template_row_confidence = max(0.0, min(1.0, visible_rows / 12.0))
    return {
        "names": names,
        "name_confidences": confidence_scores,
        "race_points": race_points,
        "total_points": total_points,
        "visible_rows": visible_rows,
        "template_row_confidence": template_row_confidence,
    }


def normalize_name_for_vote(name: str) -> str:
    text = "" if name is None else str(name)
    return re.sub(r'\s+', ' ', text.strip())


def weighted_vote(values: List[Tuple[object, float]]) -> Tuple[object, float]:
    score_by_value = defaultdict(float)
    total_weight = 0.0
    for value, weight in values:
        if value in (None, ""):
            continue
        numeric_weight = max(0.0, float(weight))
        score_by_value[value] += numeric_weight
        total_weight += numeric_weight
    if not score_by_value:
        return None, 0.0
    best_value, best_weight = max(score_by_value.items(), key=lambda item: item[1])
    confidence = best_weight / total_weight if total_weight > 0 else 0.0
    return best_value, confidence


def build_consensus_rows(observations: List[Dict[str, object]], visible_rows: int, points_key: str) -> List[Dict[str, object]]:
    rows = []
    for row_index in range(max(visible_rows, 1)):
        name_votes = []
        point_votes = []
        for observation in observations:
            name = normalize_name_for_vote(observation["names"][row_index]) if row_index < len(observation["names"]) else ""
            name_conf = observation["name_confidences"][row_index] if row_index < len(observation["name_confidences"]) else 0
            name_votes.append((name, max(1.0, float(name_conf))))
            point_votes.append((parse_detected_int(observation[points_key][row_index]), 1.0))

        player_name, name_confidence = weighted_vote(name_votes)
        detected_value, point_confidence = weighted_vote(point_votes)
        stripped_name = re.sub(r'[^a-zA-Z0-9]', '', str(player_name or ''))
        if len(stripped_name) < 3 or len(set(stripped_name)) < 3:
            if detected_value is None:
                continue

        rows.append(
            {
                "RowIndex": row_index,
                "PlayerName": player_name or "",
                "NameConfidence": round(name_confidence * 100, 1),
                "DetectedValue": detected_value,
                "DigitConfidence": round(point_confidence * 100, 1),
            }
        )
    return rows


def map_total_rows_to_race_rows(score_rows: List[Dict[str, object]], total_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    mapped_rows = []
    if not score_rows:
        return mapped_rows
    if not total_rows:
        for score_row in score_rows:
            mapped_rows.append(
                {
                    "RacePosition": len(mapped_rows) + 1,
                    "PlayerName": score_row["PlayerName"],
                    "DetectedRacePoints": score_row["DetectedValue"],
                    "DetectedTotalScore": None,
                    "NameConfidence": score_row["NameConfidence"],
                    "DigitConsensus": score_row["DigitConfidence"],
                    "TotalScoreMappingMethod": "missing_total_rows",
                }
            )
        return mapped_rows

    candidate_matches = []
    for score_index, score_row in enumerate(score_rows):
        score_name = str(score_row["PlayerName"] or "")
        normalized_score_name = preprocess_name(score_name)
        for total_index, total_row in enumerate(total_rows):
            total_name = str(total_row["PlayerName"] or "")
            normalized_total_name = preprocess_name(total_name)
            if not normalized_score_name or not normalized_total_name:
                continue
            similarity = 1.0 if normalized_score_name == normalized_total_name else weighted_similarity(score_name, total_name)
            confidence_floor = min(float(score_row["NameConfidence"]), float(total_row["NameConfidence"])) / 100.0
            combined_score = (similarity * 0.8) + (confidence_floor * 0.2)
            if similarity >= 0.72 or normalized_score_name == normalized_total_name:
                candidate_matches.append((combined_score, similarity, score_index, total_index))

    assigned_score_indices = set()
    assigned_total_indices = set()
    matched_totals_by_score_index: Dict[int, Tuple[int, str]] = {}
    for _, similarity, score_index, total_index in sorted(candidate_matches, reverse=True):
        if score_index in assigned_score_indices or total_index in assigned_total_indices:
            continue
        mapping_method = "name_exact" if similarity >= 0.999 else "name_fuzzy"
        matched_totals_by_score_index[score_index] = (total_index, mapping_method)
        assigned_score_indices.add(score_index)
        assigned_total_indices.add(total_index)

    remaining_total_indices = [index for index in range(len(total_rows)) if index not in assigned_total_indices]
    remaining_pointer = 0
    for score_index, score_row in enumerate(score_rows):
        matched_total_index = None
        mapping_method = "row_fallback"
        if score_index in matched_totals_by_score_index:
            matched_total_index, mapping_method = matched_totals_by_score_index[score_index]
        elif remaining_pointer < len(remaining_total_indices):
            matched_total_index = remaining_total_indices[remaining_pointer]
            remaining_pointer += 1

        total_score = None
        total_digit_confidence = 0.0
        if matched_total_index is not None:
            total_row = total_rows[matched_total_index]
            total_score = total_row["DetectedValue"]
            total_digit_confidence = float(total_row["DigitConfidence"])

        mapped_rows.append(
            {
                "RacePosition": len(mapped_rows) + 1,
                "PlayerName": score_row["PlayerName"],
                "DetectedRacePoints": score_row["DetectedValue"],
                "DetectedTotalScore": total_score,
                "NameConfidence": score_row["NameConfidence"],
                "DigitConsensus": round((float(score_row["DigitConfidence"]) + total_digit_confidence) / 2.0, 1),
                "TotalScoreMappingMethod": mapping_method,
            }
        )
    return mapped_rows


def build_consensus_observation(frames: List[np.ndarray], total_frames: List[np.ndarray], annotate_path: str | None = None) -> Dict[str, object]:
    if not frames:
        return {"rows": [], "visible_rows": 0, "row_count_confidence": 0.0, "name_confidence": 0.0, "digit_consensus": 0.0}

    score_observations = []
    total_observations = []
    for index, frame in enumerate(frames):
        score_observations.append(extract_scoreboard_observation(frame, annotate_path if index == len(frames) // 2 else None))
    for frame in total_frames:
        total_observations.append(extract_scoreboard_observation(frame))
    if not total_observations:
        total_observations = score_observations

    visible_votes = Counter(observation["visible_rows"] for observation in score_observations if observation["visible_rows"] > 0)
    visible_rows = visible_votes.most_common(1)[0][0] if visible_votes else 0
    row_count_confidence = (visible_votes[visible_rows] / len(score_observations)) if visible_rows and score_observations else 0.0
    total_visible_votes = Counter(observation["visible_rows"] for observation in total_observations if observation["visible_rows"] > 0)
    total_visible_rows = total_visible_votes.most_common(1)[0][0] if total_visible_votes else visible_rows

    score_rows = build_consensus_rows(score_observations, visible_rows, "race_points")
    total_rows = build_consensus_rows(total_observations, total_visible_rows, "total_points")
    rows = map_total_rows_to_race_rows(score_rows, total_rows)
    name_confidences = [float(row["NameConfidence"]) / 100.0 for row in rows if row.get("NameConfidence") is not None]
    digit_confidences = [float(row["DigitConsensus"]) / 100.0 for row in rows if row.get("DigitConsensus") is not None]

    return {
        "rows": rows,
        "visible_rows": len(rows),
        "score_visible_rows": visible_rows,
        "total_visible_rows": total_visible_rows,
        "row_count_confidence": round(row_count_confidence * 100, 1),
        "name_confidence": round((sum(name_confidences) / len(name_confidences)) * 100, 1) if name_confidences else 0.0,
        "digit_consensus": round((sum(digit_confidences) / len(digit_confidences)) * 100, 1) if digit_confidences else 0.0,
    }


def build_race_warning_messages(expected_players: int | None, race_score_players: int, total_score_players: int,
                                row_count_confidence: float) -> List[str]:
    messages = []
    if expected_players is not None and race_score_players != expected_players:
        messages.append(f"{expected_players} players expected, but only {race_score_players} were found")
    if total_score_players and total_score_players != race_score_players:
        messages.append("player count does not match between the race score screen and total score screen")
    if row_count_confidence < 60:
        messages.append("player count could not be read with enough confidence")
    return messages


def exact_total_score_fallback(prepared_rows: List[Dict[str, object]]) -> Dict[int, int]:
    detected_totals = [row["detected_total"] for row in prepared_rows if row["detected_total"] is not None]
    expected_totals = [row["session_new_total"] for row in prepared_rows]
    if not detected_totals or len(detected_totals) != len(expected_totals):
        return {}
    if Counter(int(value) for value in detected_totals) != Counter(int(value) for value in expected_totals):
        return {}

    remapped = {}
    for row in prepared_rows:
        remapped[row["index"]] = int(row["session_new_total"])
    return remapped


def should_start_new_session(session_totals: Dict[str, int], detected_totals: List[int | float | str]) -> bool:
    parsed_detected_totals = [int(value) for value in detected_totals if pd.notna(value)]
    if not parsed_detected_totals:
        return False

    previous_totals = [int(value) for value in session_totals.values() if int(value) > 0]
    if len(previous_totals) < max(4, len(parsed_detected_totals) // 3):
        return False

    previous_max = max(previous_totals)
    previous_median = float(np.median(previous_totals))
    current_max = max(parsed_detected_totals)
    current_median = float(np.median(parsed_detected_totals))
    low_total_count = sum(1 for value in parsed_detected_totals if value <= 20)

    enough_history = previous_max >= 30 and previous_median >= 20
    broad_drop = low_total_count >= max(3, len(parsed_detected_totals) // 2)
    lower_than_previous = current_max <= previous_max * 0.55 and current_median <= previous_median * 0.6
    return enough_history and broad_drop and lower_than_previous


def process_race_group(grouped_item, text_detected_folder, metadata_index, input_videos_folder, in_memory_frame_bundles=None):
    """Process a single race group and return extracted rows."""
    race_start_time = time.perf_counter()
    (race_class, race_id_number), images = grouped_item
    if len(images) < 2:
        return {"rows": [], "summary": None, "duration_s": time.perf_counter() - race_start_time}

    track_name_image = None
    race_score_image = None
    total_score_image = None

    for frame_content, image_path in images:
        if frame_content == "0TrackName":
            track_name_image = image_path
        elif frame_content == "2RaceScore":
            race_score_image = image_path
        elif frame_content == "3TotalScore":
            total_score_image = image_path

    if not track_name_image or not race_score_image:
        return {"rows": [], "summary": None, "duration_s": time.perf_counter() - race_start_time}

    results = []
    track_name_img = cv2.imread(track_name_image)
    coordinates = {"TrackName": [((319, 633), (925, 685))]}
    track_name_data, _ = extract_text_with_confidence(track_name_img, coordinates, 'eng', '--psm 7')

    raw_track_name_text = " ".join(track_name_data['TrackName']).strip()
    track_name_text = match_track_name(raw_track_name_text, tracks_list)

    race_metadata = find_metadata_entry(metadata_index, race_class, race_id_number, "RaceScore")
    total_metadata = find_metadata_entry(metadata_index, race_class, race_id_number, "TotalScore") if total_score_image else None

    annotate_path = None
    if APP_CONFIG.write_debug_score_images:
        annotate_path = os.path.join(text_detected_folder, f'annotated_{os.path.basename(race_score_image)}')

    race_bundle_key = (race_class, race_id_number, "RaceScore")
    total_bundle_key = (race_class, race_id_number, "TotalScore")
    race_frames = load_consensus_frames(
        race_score_image,
        race_metadata,
        input_videos_folder,
        OCR_CONSENSUS_FRAMES,
        in_memory_frames=(in_memory_frame_bundles or {}).get(race_bundle_key),
    )
    total_frames = load_consensus_frames(
        total_score_image or race_score_image,
        total_metadata,
        input_videos_folder,
        OCR_CONSENSUS_FRAMES,
        in_memory_frames=(in_memory_frame_bundles or {}).get(total_bundle_key),
    )
    consensus = build_consensus_observation(race_frames, total_frames, annotate_path)
    num_players = len(consensus["rows"])
    race_score_players = int(consensus.get("score_visible_rows", num_players))
    total_score_players = int(consensus.get("total_visible_rows", num_players))
    race_warning_messages = build_race_warning_messages(None, race_score_players, total_score_players, consensus["row_count_confidence"])

    for row in consensus["rows"]:
        race_position = row["RacePosition"]
        race_points_fix = get_race_points(race_position, num_players)
        review_reasons = []
        if row["NameConfidence"] < 45:
            review_reasons.append("low_name_confidence")
        if row["DigitConsensus"] < 55:
            review_reasons.append("low_digit_consensus")
        if consensus["row_count_confidence"] < 60:
            review_reasons.append("unstable_row_count")

        results.append([
            race_class,
            race_id_number,
            track_name_text,
            race_position,
            row["PlayerName"],
            race_points_fix,
            row["DetectedRacePoints"],
            row["DetectedTotalScore"],
            row["NameConfidence"],
            row["DigitConsensus"],
            consensus["row_count_confidence"],
            race_score_players,
            total_score_players,
            row.get("TotalScoreMappingMethod", ""),
            ";".join(review_reasons),
        ])

    summary = {
        "race_class": race_class,
        "race_id_number": race_id_number,
        "track_name": track_name_text,
        "race_score_players": race_score_players,
        "total_score_players": total_score_players,
        "warning_messages": race_warning_messages,
    }
    return {"rows": results, "summary": summary, "duration_s": time.perf_counter() - race_start_time}


def process_images_in_folder(folder_path: str, in_memory_frame_bundles=None, selected_race_classes=None):
    phase_start_time = time.time()
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    # Check if the folder is empty
    if not image_files:
        LOGGER.log("[OCR - Read text from image - Phase Start]", "The Frames folder is empty. Run extraction first.", color_name="red")
        sys.exit(1)  # Exit the script with a non-zero status

    script_dir = os.path.dirname(__file__)  # Directory of the script
    base_dir = Path(script_dir)
    text_detected_folder = os.path.join(script_dir, 'Output_Results', 'Debug', 'Score_Frames')
    if APP_CONFIG.write_debug_score_images and not os.path.exists(text_detected_folder):
        os.makedirs(text_detected_folder)

    linking_data_folder = os.path.join(script_dir, 'Output_Results', 'Debug')
    if APP_CONFIG.write_debug_linking_excel and not os.path.exists(linking_data_folder):
        os.makedirs(linking_data_folder)

    metadata_index = load_exported_frame_metadata(base_dir)
    input_videos_folder = base_dir / "Input_Videos"

    grouped_images = {}
    selected_classes = {str(item) for item in selected_race_classes} if selected_race_classes else None
    for image_file in image_files:
        parts = image_file.split('+')
        if len(parts) < 3:
            logging.warning(f"Skipping file with unexpected format: {image_file}")
            continue

        race_class = parts[0]
        if selected_classes is not None and race_class not in selected_classes:
            continue
        try:
            race_id_number = int(parts[1][-3:])
        except ValueError:
            logging.warning(f"Skipping file with invalid race ID number: {image_file}")
            continue

        frame_content = parts[2].replace('.png', '')
        if frame_content == "1RaceNumber":
            continue

        key = (race_class, race_id_number)
        if key not in grouped_images:
            grouped_images[key] = []
        grouped_images[key].append((frame_content, os.path.join(folder_path, image_file)))

    sorted_grouped_images = sorted(grouped_images.items(), key=lambda item: item[0])
    LOGGER.log("[OCR - Read text from image - Phase Start]", "", color_name="magenta")
    LOGGER.log(
        "[OCR - Settings]",
        f"OCR workers: {OCR_WORKERS} | Consensus frames: {OCR_CONSENSUS_FRAMES} | Input race groups: {len(sorted_grouped_images)}",
        color_name="magenta",
    )
    results = []
    race_summaries = []
    per_video_ocr_durations = defaultdict(float)
    race_totals_by_class = {}
    for (race_class, _race_id), _images in sorted_grouped_images:
        race_totals_by_class[race_class] = race_totals_by_class.get(race_class, 0) + 1
    progress = ProgressPrinter("[OCR]", len(sorted_grouped_images), percent_step=5, min_interval_s=2.0)

    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
        future_map = {
            executor.submit(
                process_race_group,
                item,
                text_detected_folder,
                metadata_index,
                input_videos_folder,
                in_memory_frame_bundles,
            ): item
            for item in sorted_grouped_images
        }
        pending = set(future_map.keys())
        completed_count = 0
        while pending:
            done, pending = wait(pending, timeout=3.0, return_when=FIRST_COMPLETED)
            if not done:
                progress.heartbeat(completed_count, "Still processing OCR races")
                continue
            for future in done:
                completed_count += 1
                race_result = future.result()
                race_class, race_id_number = future_map[future][0]
                matching_summary = race_result["summary"]
                per_video_ocr_durations[race_class] += float(race_result.get("duration_s", 0.0))
                results.extend(race_result["rows"])
                if race_result["summary"] is not None:
                    race_summaries.append(race_result["summary"])
                progress.update(completed_count)
                if matching_summary is not None:
                    LOGGER.log(
                        "",
                        f"Video: {race_class} | Race: {race_id_number:03}/{race_totals_by_class.get(race_class, race_id_number):03} | Track: {matching_summary['track_name']}",
                        color_name="magenta",
                    )
                    LOGGER.log(
                        "",
                        f"Players: race score {matching_summary['race_score_players']} | total score {matching_summary['total_score_players']}",
                        color_name="magenta",
                    )
                else:
                    LOGGER.log(
                        "",
                        f"Video: {race_class} | Race: {race_id_number:03}/{race_totals_by_class.get(race_class, race_id_number):03}",
                        color_name="magenta",
                    )
                if matching_summary is not None:
                    for warning_message in matching_summary["warning_messages"]:
                        LOGGER.log(
                            f"[OCR - Warning]",
                            f"Video: {race_class} | Race: {race_id_number:03} | Track: {matching_summary['track_name']} | {warning_message}",
                            color_name="yellow",
                        )

    df = pd.DataFrame(results, columns=[
        "RaceClass", "RaceIDNumber", "TrackName", "RacePosition", "PlayerName",
        "RacePoints", "DetectedRacePoints", "DetectedTotalScore", "NameConfidence",
        "DigitConsensus", "RowCountConfidence", "RaceScorePlayerCount", "TotalScorePlayerCount",
        "TotalScoreMappingMethod", "ReviewReason"
    ])
    df = df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable").reset_index(drop=True)
    if df.empty:
        LOGGER.log("[OCR - Phase Complete]", "No races were extracted", color_name="yellow")
        return {"duration_s": 0.0, "output_excel_path": None, "per_video_durations": {}}

    # Add the CupName column
    df['CupName'] = df['TrackName'].apply(lambda name: get_cup_name(name, tracks_list))

    # Add the TrackID column
    df['TrackID'] = df['TrackName'].apply(lambda name: next((track[0] for track in tracks_list if track[1] == name), None))

    df = standardize_player_names(df, linking_data_folder)

    df["ReviewNeeded"] = False
    df["SessionIndex"] = 1
    df["SessionOldTotalScore"] = 0
    df["SessionNewTotalScore"] = 0
    df['OldTotalScore'] = 0
    df['NewTotalScore'] = 0
    df["ScoreValidationStatus"] = "computed_only"

    for race_class, race_group in df.groupby("RaceClass", sort=False):
        tournament_totals = defaultdict(int)
        session_totals = defaultdict(int)
        session_index = 1
        race_ids = sorted(race_group["RaceIDNumber"].unique())
        expected_players = int(race_group.groupby("RaceIDNumber").size().mode().iloc[0])

        for race_id in race_ids:
            race_mask = (df["RaceClass"] == race_class) & (df["RaceIDNumber"] == race_id)
            race_rows = df[race_mask].sort_values("RacePosition")
            detected_totals = [value for value in race_rows["DetectedTotalScore"].tolist() if pd.notna(value)]
            if race_id != race_ids[0] and should_start_new_session(session_totals, detected_totals):
                session_index += 1
                session_totals = defaultdict(int)

            prepared_rows = []
            for index, row in race_rows.iterrows():
                player_key = row["FixPlayerName"]
                old_total = tournament_totals[player_key]
                session_old_total = session_totals[player_key]
                race_points = int(row["RacePoints"])
                prepared_rows.append(
                    {
                        "index": index,
                        "player_key": player_key,
                        "old_total": old_total,
                        "session_old_total": session_old_total,
                        "race_points": race_points,
                        "new_total": old_total + race_points,
                        "session_new_total": session_old_total + race_points,
                        "detected_race": parse_detected_int(row["DetectedRacePoints"]),
                        "detected_total": parse_detected_int(row["DetectedTotalScore"]),
                    }
                )

            remapped_totals_by_index = exact_total_score_fallback(prepared_rows)

            for prepared_row, (_, row) in zip(prepared_rows, race_rows.iterrows()):
                index = prepared_row["index"]
                player_key = prepared_row["player_key"]
                old_total = prepared_row["old_total"]
                session_old_total = prepared_row["session_old_total"]
                race_points = prepared_row["race_points"]
                new_total = prepared_row["new_total"]
                session_new_total = prepared_row["session_new_total"]
                detected_race = prepared_row["detected_race"]
                detected_total = remapped_totals_by_index.get(index, prepared_row["detected_total"])
                review_reasons = [reason for reason in str(row["ReviewReason"]).split(";") if reason]
                score_status = "computed_only"

                if detected_race is not None:
                    if detected_race == race_points:
                        score_status = "race_points_match"
                    else:
                        review_reasons.append("race_points_mismatch")
                        score_status = "race_points_mismatch"

                if detected_total is not None:
                    if detected_total == session_new_total:
                        score_status = "validated"
                    else:
                        review_reasons.append("total_score_mismatch")
                        score_status = "total_score_mismatch"

                if row["NameConfidence"] < 45:
                    review_reasons.append("low_name_confidence")
                if row["DigitConsensus"] < 55:
                    review_reasons.append("low_digit_consensus")
                if row["RowCountConfidence"] < 60:
                    review_reasons.append("unstable_row_count")
                if int(row["RaceScorePlayerCount"]) != expected_players:
                    review_reasons.append("player_count_mismatch")
                if int(row["TotalScorePlayerCount"]) != int(row["RaceScorePlayerCount"]):
                    review_reasons.append("race_total_player_count_mismatch")

                review_reasons = sorted(set(review_reasons))
                df.at[index, "SessionIndex"] = session_index
                df.at[index, "SessionOldTotalScore"] = session_old_total
                df.at[index, "SessionNewTotalScore"] = session_new_total
                df.at[index, "OldTotalScore"] = old_total
                df.at[index, "NewTotalScore"] = new_total
                if index in remapped_totals_by_index:
                    df.at[index, "DetectedTotalScore"] = detected_total
                    existing_mapping_method = str(df.at[index, "TotalScoreMappingMethod"]).strip()
                    df.at[index, "TotalScoreMappingMethod"] = f"{existing_mapping_method}+score_fallback" if existing_mapping_method else "score_fallback"
                df.at[index, "ReviewReason"] = ";".join(review_reasons)
                df.at[index, "ReviewNeeded"] = bool(review_reasons)
                df.at[index, "ScoreValidationStatus"] = score_status

                tournament_totals[player_key] = new_total
                session_totals[player_key] = session_new_total

    desired_order = [
        "RaceClass", "RaceIDNumber", "TrackName", "TrackID", "CupName", "RacePosition", "PlayerName",
        "FixPlayerName", "RacePoints", "DetectedRacePoints", "DetectedTotalScore",
        "RaceScorePlayerCount", "TotalScorePlayerCount", "TotalScoreMappingMethod",
        "SessionIndex", "SessionOldTotalScore", "SessionNewTotalScore",
        "OldTotalScore", "NewTotalScore", "NameConfidence", "DigitConsensus", "RowCountConfidence",
        "ScoreValidationStatus", "ReviewNeeded", "ReviewReason"
    ]
    df = df[desired_order]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel_path = os.path.normpath(os.path.join(folder_path, '..', f"{timestamp}_Tournament_Results.xlsx"))
    df.to_excel(output_excel_path, index=False)
    race_count = int(df[["RaceClass", "RaceIDNumber"]].drop_duplicates().shape[0])
    per_video_summary = {}
    lines = [f"Duration: {format_duration(time.time() - phase_start_time)}", f"Races processed: {race_count}"]
    lines.extend(progress.peak_lines())
    lines.extend(["", "OCR call profile"])
    lines.extend(OCR_PROFILER.summary_lines())
    lines.extend(["", "Per-video player count summary"])
    for race_class, race_group in df.groupby("RaceClass", sort=False):
        race_count_for_class = int(race_group["RaceIDNumber"].nunique())
        player_count_distribution = race_group.groupby("RaceIDNumber").size().value_counts().sort_index(ascending=False)
        dominant_players = int(race_group.groupby("RaceIDNumber").size().mode().iloc[0])
        review_row_count = int(race_group["ReviewNeeded"].sum())
        review_race_count = int(race_group.loc[race_group["ReviewNeeded"], "RaceIDNumber"].nunique())
        inconsistent_races = []
        for race_id, race_rows in race_group.groupby("RaceIDNumber", sort=True):
            race_score_players = int(race_rows["RaceScorePlayerCount"].iloc[0])
            total_score_players = int(race_rows["TotalScorePlayerCount"].iloc[0])
            track_name = str(race_rows["TrackName"].iloc[0])
            messages = build_race_warning_messages(dominant_players, race_score_players, total_score_players, float(race_rows["RowCountConfidence"].iloc[0]))
            if messages:
                inconsistent_races.append((int(race_id), track_name, messages))
        per_video_summary[race_class] = {
            "race_count": race_count_for_class,
            "dominant_players": dominant_players,
            "review_row_count": review_row_count,
            "review_race_count": review_race_count,
            "player_count_distribution": {int(player_count): int(count) for player_count, count in player_count_distribution.items()},
        }

        if not inconsistent_races:
            lines.append(f"- {race_class}: {race_count_for_class} {pluralize(race_count_for_class, 'race')} | Player count was consistent ({dominant_players} players)")
        else:
            distribution_text = ", ".join(f"{player_count} players x {count}" for player_count, count in player_count_distribution.items())
            lines.append(f"- {race_class}: {race_count_for_class} {pluralize(race_count_for_class, 'race')} | Player count was not consistent")
            lines.append(f"  Most races showed {dominant_players} players")
            lines.append(f"  Summary: {distribution_text}")
            lines.append("  Please review these races:")
            for race_id, track_name, messages in inconsistent_races:
                for message in messages:
                    lines.append(f"  - Race {race_id:03} | Track: {track_name} | {message}")
    lines.extend(["", "Output workbook:", output_excel_path])
    LOGGER.summary_block("[OCR - Phase Complete]", lines, color_name="green")
    return {
        "duration_s": time.time() - phase_start_time,
        "output_excel_path": output_excel_path,
        "race_count": race_count,
        "per_video_durations": dict(per_video_ocr_durations),
        "per_video_summary": per_video_summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Mario Kart 8 extracted frames")
    parser.add_argument("--video", help="Process only a specific video filename or race class stem")
    args = parser.parse_args()
    configure_tesseract(pytesseract, APP_CONFIG)

    # Folder path to the images
    script_dir = os.path.dirname(__file__)  # Directory of the script
    folder_path = os.path.join(script_dir, 'Output_Results', 'Frames')
    selected_race_classes = [Path(args.video).stem] if args.video else None
    process_images_in_folder(folder_path, selected_race_classes=selected_race_classes)
