import os
import argparse
import csv
import cv2
import numpy as np
import pytesseract
import pandas as pd
from typing import List, Dict, Tuple
import logging
import time
import difflib
import openpyxl
import re
import threading
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from .app_runtime import configure_tesseract, load_app_config
from .extract_common import build_video_identity
from .console_logging import LOGGER
from .ocr_export import build_completion_payload
from .ocr_name_matching import (
    append_identity_relink_review_notes,
    compact_identity_labels,
    preprocess_name,
    reconcile_connection_reset_identities,
    standardize_player_names,
    weighted_similarity,
)
from .ocr_common import find_metadata_entry, load_consensus_frames, load_exported_frame_metadata
from .low_res_identity import apply_low_res_identity_pipeline, is_low_res_height
from .ocr_scoreboard_consensus import (
    TOTAL_SCORE_CONSENSUS_WINDOW_SIZE,
    build_consensus_observation,
    build_race_warning_messages,
    character_shortlist_summary_lines,
    exact_total_score_fallback,
    observation_stage_summary_lines,
    parse_detected_int,
    reset_character_shortlist_state,
    reset_observation_stage_stats,
)
from .ocr_session_validation import apply_session_validation
from .project_paths import PROJECT_ROOT
from .track_metadata import load_track_tuples

POSITION_TEMPLATE_COEFF_COLUMNS = [f"PositionTemplate{template_index:02}_Coeff" for template_index in range(1, 13)]

# Record the start time
start_run_time = time.time()
APP_CONFIG = load_app_config()
OCR_WORKERS = APP_CONFIG.ocr_workers
OCR_CONSENSUS_FRAMES = APP_CONFIG.ocr_consensus_frames
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE = max(
    0,
    min(100, int(os.environ.get("MK8_PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE", "80"))),
)
PLAYER_NAME_BATCH_SEPARATOR_HEIGHT = max(
    0,
    int(os.environ.get("MK8_PLAYER_NAME_BATCH_SEPARATOR_HEIGHT", "10")),
)
PLAYER_NAME_BATCH_HORIZONTAL_PADDING = max(
    0,
    int(os.environ.get("MK8_PLAYER_NAME_BATCH_HORIZONTAL_PADDING", "0")),
)
PLAYER_NAME_BATCH_VERTICAL_PADDING = max(
    0,
    int(os.environ.get("MK8_PLAYER_NAME_BATCH_VERTICAL_PADDING", "0")),
)
PLAYER_NAME_BATCH_CONFIG = os.environ.get("MK8_PLAYER_NAME_BATCH_CONFIG", "--psm 6")
PLAYER_NAME_ROW_FALLBACK_ENABLED = os.environ.get("MK8_PLAYER_NAME_ROW_FALLBACK_ENABLED", "0").lower() not in {"0", "false", "no"}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
PLAYER_NAME_FALLBACK_STATS = defaultdict(int)


def reset_player_name_fallback_stats() -> None:
    PLAYER_NAME_FALLBACK_STATS.clear()


def player_name_fallback_summary_lines() -> List[str]:
    if not PLAYER_NAME_FALLBACK_STATS:
        return []
    total = int(PLAYER_NAME_FALLBACK_STATS.get("fallback_rows", 0))
    return [
        "Player-name fallback activity",
        f"- fallback rows: {total}",
        f"- low batch confidence: {int(PLAYER_NAME_FALLBACK_STATS.get('reason_low_confidence', 0))}",
        f"- short batch text: {int(PLAYER_NAME_FALLBACK_STATS.get('reason_short_text', 0))}",
        f"- low batch character diversity: {int(PLAYER_NAME_FALLBACK_STATS.get('reason_low_diversity', 0))}",
        f"- fallback improved confidence: {int(PLAYER_NAME_FALLBACK_STATS.get('fallback_improved_confidence', 0))}",
        f"- fallback kept/better text length: {int(PLAYER_NAME_FALLBACK_STATS.get('fallback_kept_or_improved_length', 0))}",
    ]


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

@lru_cache(maxsize=128)
def resolve_video_dimensions(video_value: str, input_videos_folder: str) -> tuple[int | None, int | None]:
    video_path = Path(str(video_value))
    if not video_path.is_absolute():
        video_path = Path(input_videos_folder) / video_path
    if not video_path.exists():
        return None, None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None, None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return (width or None, height or None)


@lru_cache(maxsize=128)
def resolve_video_path_for_race_class(race_class: str, input_videos_folder: str) -> str | None:
    input_root = Path(input_videos_folder)
    for video_path in input_root.rglob('*'):
        if not video_path.is_file():
            continue
        if build_video_identity(video_path, input_root=input_root, include_subfolders=True) == race_class:
            return str(video_path)
    return None


def resolve_low_res_metadata(metadata_entry, input_videos_folder: Path, race_class: str) -> tuple[int | None, int | None, bool]:
    video_value = None
    if metadata_entry is not None:
        video_value = str(metadata_entry.get('video', '')).strip() or None
    if not video_value:
        video_value = resolve_video_path_for_race_class(race_class, str(input_videos_folder))
    if not video_value:
        return None, None, False
    width, height = resolve_video_dimensions(video_value, str(input_videos_folder))
    return width, height, is_low_res_height(height, APP_CONFIG.low_res_max_source_height)


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
    config: str = PLAYER_NAME_BATCH_CONFIG,
) -> Tuple[List[str], List[int]]:
    rois = []
    widths = []
    heights = []
    for (x1, y1), (x2, y2) in coord_list:
        roi = image[y1:y2, x1:x2]
        rois.append(roi)
        heights.append(max(1, roi.shape[0]))
        widths.append(max(1, roi.shape[1]))

    separator_height = PLAYER_NAME_BATCH_SEPARATOR_HEIGHT
    horizontal_padding = PLAYER_NAME_BATCH_HORIZONTAL_PADDING
    vertical_padding = PLAYER_NAME_BATCH_VERTICAL_PADDING
    canvas_width = max(widths) + horizontal_padding * 2
    canvas_height = sum(height + vertical_padding * 2 for height in heights) + separator_height * (len(rois) - 1)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    row_ranges = []
    cursor = 0
    for roi in rois:
        height, width = roi.shape[:2]
        start_y = cursor + vertical_padding
        end_y = start_y + height
        start_x = horizontal_padding
        end_x = start_x + width
        canvas[start_y:end_y, start_x:end_x] = roi
        row_ranges.append((start_y, end_y))
        cursor = end_y + vertical_padding + separator_height

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
        low_confidence = average_confidence < PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE
        short_text = len(stripped_name) < 3
        low_diversity = len(set(stripped_name)) < 3
        if PLAYER_NAME_ROW_FALLBACK_ENABLED and (low_confidence or short_text or low_diversity):
            PLAYER_NAME_FALLBACK_STATS["fallback_rows"] += 1
            if low_confidence:
                PLAYER_NAME_FALLBACK_STATS["reason_low_confidence"] += 1
            if short_text:
                PLAYER_NAME_FALLBACK_STATS["reason_short_text"] += 1
            if low_diversity:
                PLAYER_NAME_FALLBACK_STATS["reason_low_diversity"] += 1
            original_text = combined_text
            original_confidence = average_confidence
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
            if average_confidence > original_confidence:
                PLAYER_NAME_FALLBACK_STATS["fallback_improved_confidence"] += 1
            if len(re.sub(r"[^a-zA-Z0-9]", "", combined_text)) >= len(re.sub(r"[^a-zA-Z0-9]", "", original_text)):
                PLAYER_NAME_FALLBACK_STATS["fallback_kept_or_improved_length"] += 1

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

    source_video_width, source_video_height, is_low_res = resolve_low_res_metadata(race_metadata, input_videos_folder, race_class)

    annotate_path = None
    total_annotate_path = None
    if APP_CONFIG.write_debug_score_images:
        annotate_path = os.path.join(text_detected_folder, f'annotated_{os.path.basename(race_score_image)}')
        if total_score_image:
            total_annotate_path = os.path.join(text_detected_folder, f'annotated_{os.path.basename(total_score_image)}')

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
        TOTAL_SCORE_CONSENSUS_WINDOW_SIZE,
        in_memory_frames=(in_memory_frame_bundles or {}).get(total_bundle_key),
    )
    consensus = build_consensus_observation(
        race_frames,
        total_frames,
        extract_player_names_batched,
        preprocess_name,
        weighted_similarity,
        annotate_path,
        total_annotate_path,
        video_context=race_class,
        is_low_res=is_low_res,
    )
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
            row.get("Character", ""),
            row.get("CharacterIndex"),
            row.get("CharacterMatchConfidence", 0.0),
            row.get("CharacterMatchMethod", ""),
            race_points_fix,
            row["DetectedRacePoints"],
            row.get("DetectedRacePointsSource", ""),
            row.get("DetectedOldTotalScore"),
            row.get("DetectedOldTotalScoreSource", ""),
            row["DetectedTotalScore"],
            row.get("DetectedTotalScoreSource", ""),
            row.get("DetectedNewTotalScore"),
            row.get("DetectedNewTotalScoreSource", ""),
            row.get("PositionAfterRace"),
            *[row.get(column_name) for column_name in POSITION_TEMPLATE_COEFF_COLUMNS],
            row["NameConfidence"],
            row["DigitConsensus"],
            consensus["row_count_confidence"],
            race_score_players,
            total_score_players,
            consensus.get("legacy_score_visible_rows", race_score_players),
            consensus.get("legacy_total_visible_rows", total_score_players),
            consensus.get("legacy_row_count_confidence", consensus["row_count_confidence"]),
            consensus.get("score_count_votes", ""),
            consensus.get("total_count_votes", ""),
            consensus.get("legacy_score_count_votes", ""),
            consensus.get("legacy_total_count_votes", ""),
            consensus.get("score_row_metrics_summary", ""),
            consensus.get("total_row_metrics_summary", ""),
            consensus.get("race_score_recovery_used", False),
            consensus.get("race_score_recovery_source", ""),
            consensus.get("race_score_recovery_count", race_score_players),
            row.get("TotalScoreMappingMethod", ""),
            source_video_width,
            source_video_height,
            is_low_res,
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
    reset_observation_stage_stats()
    reset_character_shortlist_state()
    reset_player_name_fallback_stats()
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

    if not image_files:
        LOGGER.log("[OCR - Read text from image - Phase Start]", "The Frames folder is empty. Run extraction first.", color_name="red")
        raise RuntimeError("The Frames folder is empty. Run extraction first.")

    base_dir = Path(PROJECT_ROOT)
    text_detected_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug', 'Score_Frames')
    if APP_CONFIG.write_debug_score_images and not os.path.exists(text_detected_folder):
        os.makedirs(text_detected_folder)

    linking_data_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug')
    if APP_CONFIG.write_debug_linking_excel and not os.path.exists(linking_data_folder):
        os.makedirs(linking_data_folder)

    metadata_index = load_exported_frame_metadata(base_dir)
    input_videos_folder = base_dir / "Input_Videos"

    grouped_images = {}
    selected_classes = {str(item) for item in selected_race_classes} if selected_race_classes else None
    # OCR runs per race bundle, not per image. Grouping the exported frames up front
    # lets the later stages reason about track name, race score, and total score together.
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
    if selected_classes is not None:
        LOGGER.log(
            "[OCR - Settings]",
            f"Selection scope: {len(selected_classes)} video classes",
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
        # Each race group is independent, so the safest parallelism boundary is one
        # worker per race bundle.
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
        "Character", "CharacterIndex", "CharacterMatchConfidence", "CharacterMatchMethod",
        "RacePoints", "DetectedRacePoints", "DetectedRacePointsSource", "DetectedOldTotalScore", "DetectedOldTotalScoreSource", "DetectedTotalScore", "DetectedTotalScoreSource", "DetectedNewTotalScore", "DetectedNewTotalScoreSource", "PositionAfterRace",
        *POSITION_TEMPLATE_COEFF_COLUMNS,
        "NameConfidence",
        "DigitConsensus", "RowCountConfidence", "RaceScorePlayerCount", "TotalScorePlayerCount",
        "LegacyRaceScorePlayerCount", "LegacyTotalScorePlayerCount", "LegacyRowCountConfidence",
        "RaceScoreCountVotes", "TotalScoreCountVotes", "LegacyRaceScoreCountVotes", "LegacyTotalScoreCountVotes",
        "RaceScoreRowSignals", "TotalScoreRowSignals",
        "RaceScoreRecoveryUsed", "RaceScoreRecoverySource", "RaceScoreRecoveryCount",
        "TotalScoreMappingMethod", "SourceVideoWidth", "SourceVideoHeight", "IsLowRes", "ReviewReason"
    ])
    df = df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable").reset_index(drop=True)
    if df.empty:
        LOGGER.log("[OCR - Phase Complete]", "No races were extracted", color_name="yellow")
        return {"duration_s": 0.0, "output_excel_path": None, "per_video_durations": {}}

    # Add the CupName column
    df['CupName'] = df['TrackName'].apply(lambda name: get_cup_name(name, tracks_list))

    # Add the TrackID column
    df['TrackID'] = df['TrackName'].apply(lambda name: next((track[0] for track in tracks_list if track[1] == name), None))

    low_res_mask = df["IsLowRes"].fillna(False).astype(bool)
    standardized_frames = []
    high_res_df = df.loc[~low_res_mask].copy()
    if not high_res_df.empty:
        standardized_frames.append(standardize_player_names(high_res_df, linking_data_folder, APP_CONFIG.write_debug_linking_excel))
    low_res_df = df.loc[low_res_mask].copy()
    if not low_res_df.empty:
        frames_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Frames')
        for race_class, low_res_group in low_res_df.groupby('RaceClass', sort=False):
            standardized_frames.append(
                apply_low_res_identity_pipeline(
                    low_res_group,
                    frames_folder,
                    race_class,
                    write_debug_outputs=APP_CONFIG.write_debug_csv,
                    debug_dir=linking_data_folder,
                )
            )
    df = pd.concat(standardized_frames, ignore_index=True) if standardized_frames else df.copy()
    df = df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable").reset_index(drop=True)

    df = apply_session_validation(df, parse_detected_int, exact_total_score_fallback)
    df = reconcile_connection_reset_identities(df)
    df = compact_identity_labels(df)
    df = df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable").reset_index(drop=True)
    df = apply_session_validation(df, parse_detected_int, exact_total_score_fallback)
    df = append_identity_relink_review_notes(df)

    completion_payload = build_completion_payload(
        df,
        folder_path,
        phase_start_time,
        progress.peak_lines(),
        OCR_PROFILER.summary_lines()
        + observation_stage_summary_lines()
        + character_shortlist_summary_lines()
        + player_name_fallback_summary_lines(),
        per_video_ocr_durations,
        build_race_warning_messages,
        pluralize,
        format_duration,
    )
    LOGGER.summary_block("[OCR - Phase Complete]", completion_payload["lines"], color_name="green")
    return {
        "duration_s": completion_payload["duration_s"],
        "output_excel_path": completion_payload["output_excel_path"],
        "race_count": completion_payload["race_count"],
        "per_video_durations": completion_payload["per_video_durations"],
        "per_video_summary": completion_payload["per_video_summary"],
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="OCR Mario Kart 8 extracted frames")
    parser.add_argument("--video", help="Process only a specific video filename or race class stem")
    parser.add_argument(
        "--race-class",
        dest="race_classes",
        action="append",
        help="Process only a specific race class identifier; may be supplied multiple times",
    )
    args = parser.parse_args()
    configure_tesseract(pytesseract, APP_CONFIG)

    folder_path = os.path.join(PROJECT_ROOT, 'Output_Results', 'Frames')
    selected_race_classes = list(args.race_classes or [])
    if args.video:
        selected_race_classes.append(Path(args.video).stem)
    process_images_in_folder(folder_path, selected_race_classes=selected_race_classes)


if __name__ == "__main__":
    main()
