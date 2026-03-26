import difflib
import os
import re
import time
import threading
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import easyocr
except Exception:
    easyocr = None

try:
    import torch
except Exception:
    torch = None

from .app_runtime import easyocr_gpu_enabled as runtime_easyocr_gpu_enabled, load_app_config
from .data_paths import resolve_asset_file
from .extract_common import EXPORT_IMAGE_FORMAT
from .game_catalog import load_game_catalog
from .name_unicode import (
    allowed_name_char_ratio,
    collapse_name_whitespace,
    distinct_visible_name_count,
    normalize_name_key,
    unknown_name_chars,
    visible_name_length,
)
from .score_layouts import DEFAULT_SCORE_LAYOUT_ID, get_score_layout

APP_CONFIG = load_app_config()
_DIGIT_EASYOCR_READER = None
_DIGIT_EASYOCR_LOCK = threading.Lock()
PLAYER_NAME_COORDS = get_score_layout(DEFAULT_SCORE_LAYOUT_ID).player_name_coords
BASE_POSITION_STRIP_ROI = get_score_layout(DEFAULT_SCORE_LAYOUT_ID).base_position_strip_roi
POSITION_STRIP_OFFSET_X = -2
POSITION_STRIP_OFFSET_Y = -2
POSITION_STRIP_PADDING_X = 2
POSITION_STRIP_PADDING_Y = 2
POSITION_ROW_PADDING_X = 1
POSITION_ROW_PADDING_TOP = 1
POSITION_ROW_PADDING_BOTTOM = 4
POSITION_TEMPLATE_FILENAME = "Score_template.png"
POSITION_TEMPLATE_WIDTH = 56
POSITION_TEMPLATE_HEIGHT = 36
POSITION_TEMPLATE_ROW_STARTS = [0, 50, 102, 154, 206, 258, 310, 362, 414, 466, 518, 570]
POSITION_FALSE_NEGATIVE_WEIGHT = 2.0
POSITION_PRESENT_COEFF_THRESHOLD = 0.60
POSITION_PRESENT_ROW1_COEFF_THRESHOLD = 0.40
POSITION_TEMPLATE_FAST_PATH_ENABLED = os.environ.get("MK8_POSITION_TEMPLATE_FAST_PATH_ENABLED", "1").lower() not in {"0", "false", "no"}
POSITION_TEMPLATE_BEAM_WIDTH = max(1, int(os.environ.get("MK8_POSITION_TEMPLATE_BEAM_WIDTH", "3")))
CHARACTER_ROI_LEFT = get_score_layout(DEFAULT_SCORE_LAYOUT_ID).character_roi_left
CHARACTER_TEMPLATE_SIZE = 48
CHARACTER_ROW_START = 49
CHARACTER_ROW_STEP = 52
CHARACTER_ROW_PADDING_TOP = 2
CHARACTER_ROW_PADDING_BOTTOM = 2
EXCLUDED_CHARACTER_TEMPLATE_INDICES = {79, 80, 81}
OBSERVATION_STAGE_STATS = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
CALL_MATRIX_STATS = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
CHARACTER_SHORTLIST_BY_VIDEO = defaultdict(set)
CHARACTER_SHORTLIST_LOCK = threading.Lock()
CHARACTER_SHORTLIST_STATS = defaultdict(int)
PLAYER_CHARACTER_PRIORS = defaultdict(dict)
CHARACTER_SHORTLIST_MIN_CONFIDENCE = 78.0
CHARACTER_SHORTLIST_MIN_MARGIN = 5.0
CHARACTER_PRIOR_CONFIRM_MIN_CONFIDENCE = 76.0
CHARACTER_PRIOR_STABLE_MIN_SEEN = 2
CHARACTER_PRIOR_MAX_FAST_ACCEPTS = 6
LOW_RES_ROW12_CHARACTER_FALLBACK_MIN_CONFIDENCE = APP_CONFIG.low_res_row12_character_fallback_min_confidence
LOW_RES_ROW12_CHARACTER_FALLBACK_MIN_POSITION_SCORE = APP_CONFIG.low_res_row12_character_fallback_min_position_score
ULTRA_LOW_RES_ROW_LEFT = CHARACTER_ROI_LEFT
ULTRA_LOW_RES_ROW_RIGHT = PLAYER_NAME_COORDS[0][1][0]
ULTRA_LOW_RES_ROW_MIN_STDDEV = APP_CONFIG.ultra_low_res_row_min_stddev
ULTRA_LOW_RES_ROW_MIN_EDGE_DENSITY = APP_CONFIG.ultra_low_res_row_min_edge_density
TOTAL_SCORE_RACE_POINTS_ENABLED = os.environ.get("MK8_TOTAL_SCORE_RACE_POINTS_ENABLED", "0").lower() not in {"0", "false", "no"}
DIGIT_OCR_FALLBACK_ENABLED = os.environ.get("MK8_DIGIT_OCR_FALLBACK_ENABLED", "0").lower() not in {"0", "false", "no"}


def easyocr_gpu_enabled() -> bool:
    runtime_config = load_app_config()
    return runtime_easyocr_gpu_enabled(runtime_config)


def record_observation_stage(label: str, duration_s: float) -> None:
    stats = OBSERVATION_STAGE_STATS[label]
    stats["calls"] += 1
    stats["seconds"] += duration_s


def observation_stage_summary_lines() -> List[str]:
    if not OBSERVATION_STAGE_STATS:
        return []
    total_seconds = sum(item["seconds"] for item in OBSERVATION_STAGE_STATS.values())
    rows = []
    for label, item in sorted(OBSERVATION_STAGE_STATS.items(), key=lambda pair: pair[1]["seconds"], reverse=True):
        avg_ms = (item["seconds"] / max(1, item["calls"])) * 1000.0
        rows.append((str(label), f"{int(item['calls']):>5}", f"{item['seconds']:>10.2f}", f"{avg_ms:>9.1f}"))
    headers = ("Stage", "Calls", "Total (s)", "Avg (ms)")
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        f"Observation stage profile (cumulative across all observations): {total_seconds:.2f}s",
        "  "
        + "  ".join(
            header.ljust(widths[index]) if index == 0 else header.rjust(widths[index])
            for index, header in enumerate(headers)
        ),
        "  "
        + "  ".join(
            ("-" * widths[index]).ljust(widths[index]) if index == 0 else ("-" * widths[index]).rjust(widths[index])
            for index in range(len(headers))
        ),
    ]
    for row in rows:
        lines.append(
            "  "
            + "  ".join(
                row[index].ljust(widths[index]) if index == 0 else row[index].rjust(widths[index])
                for index in range(len(headers))
            )
        )
    return lines


def reset_observation_stage_stats() -> None:
    OBSERVATION_STAGE_STATS.clear()


def record_call_matrix(bundle_kind: str, field_name: str, method_name: str, duration_s: float) -> None:
    stats = CALL_MATRIX_STATS[(str(bundle_kind), str(field_name), str(method_name))]
    stats["calls"] += 1
    stats["seconds"] += float(duration_s)


def reset_call_matrix_stats() -> None:
    CALL_MATRIX_STATS.clear()


def call_matrix_summary_lines(colorize=None) -> List[str]:
    if not CALL_MATRIX_STATS:
        return []

    entries = []
    for (bundle_kind, field_name, method_name), item in CALL_MATRIX_STATS.items():
        calls = int(item["calls"])
        seconds = float(item["seconds"])
        avg_ms = (seconds / max(1, calls)) * 1000.0
        entries.append(
            {
                "bundle": bundle_kind,
                "field": field_name,
                "method": method_name,
                "calls": calls,
                "seconds": seconds,
                "avg_ms": avg_ms,
            }
        )

    ranked_entries = sorted(entries, key=lambda item: item["seconds"], reverse=True)
    highlight_colors: dict[tuple[str, str, str], str] = {}
    if ranked_entries:
        highlight_colors[(ranked_entries[0]["bundle"], ranked_entries[0]["field"], ranked_entries[0]["method"])] = "red"
    if len(ranked_entries) > 1:
        highlight_colors[(ranked_entries[1]["bundle"], ranked_entries[1]["field"], ranked_entries[1]["method"])] = "yellow"

    lines = ["OCR Call Matrix"]
    bundle_width = max(len("Bundle"), max(len(str(entry["bundle"])) for entry in entries))
    field_width = max(len("Field"), max(len(str(entry["field"])) for entry in entries))
    method_width = max(len("Method"), max(len(str(entry["method"])) for entry in entries))
    detail_header = (
        f"{'Bundle':<{bundle_width}}  {'Field':<{field_width}}  {'Method':<{method_width}}  "
        f"{'Calls':>5}  {'Total (s)':>10}  {'Avg (ms)':>9}"
    )
    detail_divider = (
        f"{'-' * bundle_width}  {'-' * field_width}  {'-' * method_width}  "
        f"{'-' * 5}  {'-' * 10}  {'-' * 9}"
    )
    lines.append(detail_header)
    lines.append(detail_divider)
    bundle_order = {"2RaceScore": 0, "3TotalScore": 1}
    for entry in sorted(
        entries,
        key=lambda item: (bundle_order.get(str(item["bundle"]), 99), str(item["bundle"]), -float(item["seconds"]), str(item["field"]), str(item["method"])),
    ):
        row = (
            f"{entry['bundle']:<{bundle_width}}  {entry['field']:<{field_width}}  {entry['method']:<{method_width}}  "
            f"{entry['calls']:>5d}  {entry['seconds']:>10.2f}  {entry['avg_ms']:>9.1f}"
        )
        row_color = highlight_colors.get((entry["bundle"], entry["field"], entry["method"]))
        lines.append(colorize(row, row_color) if colorize and row_color else row)
    lines.append("")

    field_totals = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    for entry in entries:
        field_totals[entry["field"]]["calls"] += entry["calls"]
        field_totals[entry["field"]]["seconds"] += entry["seconds"]
    lines.append("Totals By Field")
    summary_field_width = max(len("Field"), max(len(str(field_name)) for field_name in field_totals))
    summary_header = f"{'Field':<{summary_field_width}}  {'Calls':>5}  {'Total (s)':>10}  {'Avg (ms)':>9}"
    summary_divider = f"{'-' * summary_field_width}  {'-' * 5}  {'-' * 10}  {'-' * 9}"
    lines.append(summary_header)
    lines.append(summary_divider)
    for field_name, item in sorted(field_totals.items(), key=lambda pair: pair[1]["seconds"], reverse=True):
        calls = int(item["calls"])
        seconds = float(item["seconds"])
        avg_ms = (seconds / max(1, calls)) * 1000.0
        lines.append(f"{field_name:<{summary_field_width}}  {calls:>5d}  {seconds:>10.2f}  {avg_ms:>9.1f}")
    lines.append("")

    bundle_totals = defaultdict(lambda: {"calls": 0, "seconds": 0.0})
    for entry in entries:
        bundle_totals[entry["bundle"]]["calls"] += entry["calls"]
        bundle_totals[entry["bundle"]]["seconds"] += entry["seconds"]
    lines.append("Totals By Bundle")
    bundle_summary_width = max(len("Bundle"), max(len(str(bundle_name)) for bundle_name in bundle_totals))
    bundle_header = f"{'Bundle':<{bundle_summary_width}}  {'Calls':>5}  {'Total (s)':>10}  {'Avg (ms)':>9}"
    bundle_divider = f"{'-' * bundle_summary_width}  {'-' * 5}  {'-' * 10}  {'-' * 9}"
    lines.append(bundle_header)
    lines.append(bundle_divider)
    for bundle_name, item in sorted(bundle_totals.items(), key=lambda pair: (bundle_order.get(str(pair[0]), 99), str(pair[0]))):
        calls = int(item["calls"])
        seconds = float(item["seconds"])
        avg_ms = (seconds / max(1, calls)) * 1000.0
        lines.append(f"{bundle_name:<{bundle_summary_width}}  {calls:>5d}  {seconds:>10.2f}  {avg_ms:>9.1f}")
    return lines


def character_shortlist_summary_lines() -> List[str]:
    if not CHARACTER_SHORTLIST_STATS:
        return []
    return [
        "Character shortlist activity",
        f"- prior rows accepted: {CHARACTER_SHORTLIST_STATS['prior_accepts']}",
        f"- shortlist rows accepted: {CHARACTER_SHORTLIST_STATS['shortlist_accepts']}",
        f"- prior rows escalated to shortlist/full search: {CHARACTER_SHORTLIST_STATS['prior_fallbacks']}",
        f"- shortlist rows escalated to full search: {CHARACTER_SHORTLIST_STATS['shortlist_fallbacks']}",
        f"- full-search rows: {CHARACTER_SHORTLIST_STATS['full_search_rows']}",
        f"- shortlist expansions: {CHARACTER_SHORTLIST_STATS['shortlist_expansions']}",
    ]


def reset_character_shortlist_state() -> None:
    with CHARACTER_SHORTLIST_LOCK:
        CHARACTER_SHORTLIST_BY_VIDEO.clear()
        PLAYER_CHARACTER_PRIORS.clear()
    CHARACTER_SHORTLIST_STATS.clear()


def player_identity_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def is_risky_character_family(character_name: str) -> bool:
    normalized = str(character_name or "").lower()
    return "yoshi" in normalized or "birdo" in normalized


def character_margin_threshold(character_name: str) -> float:
    return 3.5 if is_risky_character_family(character_name) else 3.0


def character_confidence_threshold(character_name: str) -> float:
    return 74.5 if is_risky_character_family(character_name) else CHARACTER_PRIOR_CONFIRM_MIN_CONFIDENCE


def position_strip_roi(score_layout_id: str | None = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the adjusted position-strip ROI after applying the current offset and padding."""
    (base_x1, base_y1), (base_x2, base_y2) = get_score_layout(score_layout_id).base_position_strip_roi
    shifted_x1 = base_x1 + POSITION_STRIP_OFFSET_X
    shifted_y1 = base_y1 + POSITION_STRIP_OFFSET_Y
    shifted_x2 = base_x2 + POSITION_STRIP_OFFSET_X
    shifted_y2 = base_y2 + POSITION_STRIP_OFFSET_Y
    return (
        (shifted_x1 - POSITION_STRIP_PADDING_X, shifted_y1 - POSITION_STRIP_PADDING_Y),
        (shifted_x2 + POSITION_STRIP_PADDING_X, shifted_y2 + POSITION_STRIP_PADDING_Y),
    )


def character_row_roi(row_index: int, score_layout_id: str | None = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    row_top = CHARACTER_ROW_START + (row_index * CHARACTER_ROW_STEP)
    character_roi_left = get_score_layout(score_layout_id).character_roi_left
    return (
        (character_roi_left, row_top - CHARACTER_ROW_PADDING_TOP),
        (character_roi_left + CHARACTER_TEMPLATE_SIZE, row_top + CHARACTER_TEMPLATE_SIZE + CHARACTER_ROW_PADDING_BOTTOM),
    )


def ultra_low_res_combined_row_roi(row_index: int, score_layout_id: str | None = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    player_name_coords = get_score_layout(score_layout_id).player_name_coords
    (cx1, cy1), (_cx2, cy2) = character_row_roi(row_index, score_layout_id=score_layout_id)
    (_nx1, ny1), (nx2, ny2) = player_name_coords[row_index]
    return (
        (int(get_score_layout(score_layout_id).character_roi_left), int(min(cy1, ny1))),
        (int(max(player_name_coords[0][1][0], nx2)), int(max(cy2, ny2))),
    )


def normalize_binary_foreground(binary_image: np.ndarray) -> np.ndarray:
    """Keep the foreground polarity consistent across OCR regions and template matching."""
    black_pixels = int(np.count_nonzero(binary_image == 0))
    white_pixels = int(np.count_nonzero(binary_image == 255))
    if white_pixels > black_pixels:
        return cv2.bitwise_not(binary_image)
    return binary_image


def position_template_row_windows() -> List[Tuple[int, int]]:
    """Return the fixed row windows used by the template strip itself."""
    return [(start_y, start_y + POSITION_TEMPLATE_HEIGHT) for start_y in POSITION_TEMPLATE_ROW_STARTS]


def split_strip_into_rows(strip_image: np.ndarray) -> List[np.ndarray]:
    """Split a position strip using the same row windows as the position templates."""
    row_images = []
    strip_height = strip_image.shape[0]
    for start_y, end_y in position_template_row_windows():
        clipped_start = max(0, min(strip_height, start_y))
        clipped_end = max(clipped_start, min(strip_height, end_y))
        row_images.append(strip_image[clipped_start:clipped_end, :])
    return row_images


def normalize_position_rows(strip_image: np.ndarray) -> List[np.ndarray]:
    """Normalize foreground polarity per scoreboard row instead of per full strip.

    This mirrors how player names and score cells are handled: each ROI decides its own
    inversion based on its own black/white balance. Empty bottom rows should not be able
    to inherit the polarity choice from busy rows above them.
    """
    return [normalize_binary_foreground(row_image) for row_image in split_strip_into_rows(strip_image)]


def slice_position_templates(template_binary: np.ndarray) -> List[np.ndarray]:
    """Slice the template strip using fixed designer-approved row windows.

    Each template row keeps the full 56 px width and a fixed 36 px height.
    The top positions come from the manually tuned row starts provided for
    Score_template.png: 0, 50, 102, 154, ... with 52 px spacing after row 1.
    """
    template_rows = []
    image_height, image_width = template_binary.shape[:2]
    crop_width = min(POSITION_TEMPLATE_WIDTH, image_width)
    for start_y in POSITION_TEMPLATE_ROW_STARTS:
        end_y = min(image_height, start_y + POSITION_TEMPLATE_HEIGHT)
        row_crop = template_binary[start_y:end_y, :crop_width]
        template_rows.append(normalize_binary_foreground(row_crop))
    return template_rows


def combine_position_rows(row_images: List[np.ndarray], score_layout_id: str | None = None) -> np.ndarray:
    """Rebuild the full strip after per-row normalization using the template row windows."""
    if not row_images:
        return np.zeros((0, 0), dtype=np.uint8)
    strip_width = row_images[0].shape[1]
    strip_height = position_strip_roi(score_layout_id=score_layout_id)[1][1] - position_strip_roi(score_layout_id=score_layout_id)[0][1]
    combined = np.zeros((strip_height, strip_width), dtype=np.uint8)
    for row_image, (start_y, end_y) in zip(row_images, position_template_row_windows()):
        clipped_start = max(0, min(strip_height, start_y))
        clipped_end = max(clipped_start, min(strip_height, end_y))
        height = min(row_image.shape[0], clipped_end - clipped_start)
        combined[clipped_start:clipped_start + height, :] = row_image[:height, :]
    return combined


def extract_position_row_match_crops(processed_image: np.ndarray, score_layout_id: str | None = None) -> List[np.ndarray]:
    """Extract per-row position ROIs based on the template row windows.

    The core template area is 56x36. We keep just 1 px padding around it so the
    matcher can adjust slightly inside a tight 58x38 ROI.
    """
    (x1, y1), (x2, y2) = position_strip_roi(score_layout_id=score_layout_id)
    image_height, image_width = processed_image.shape[:2]
    if len(processed_image.shape) == 3:
        grayscale_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale_image = processed_image

    row_crops = []
    for row_start_offset, row_end_offset in position_template_row_windows():
        row_start = y1 + int(row_start_offset)
        row_end = y1 + int(row_end_offset)
        crop_x1 = max(0, x1 - POSITION_ROW_PADDING_X)
        crop_y1 = max(0, row_start - POSITION_ROW_PADDING_TOP)
        crop_x2 = min(image_width, x2 + POSITION_ROW_PADDING_X)
        crop_y2 = min(image_height, row_end + POSITION_ROW_PADDING_BOTTOM)
        row_crop = grayscale_image[crop_y1:crop_y2, crop_x1:crop_x2]
        _, row_crop = cv2.threshold(row_crop, 180, 255, cv2.THRESH_BINARY)
        row_crops.append(normalize_binary_foreground(row_crop))
    return row_crops


@lru_cache(maxsize=1)
def load_position_row_templates() -> List[np.ndarray]:
    """Slice the score-screen position strip template into 12 fixed row templates."""
    template_path = resolve_asset_file("templates", POSITION_TEMPLATE_FILENAME)
    template_image = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
    if template_image is None:
        raise FileNotFoundError(f"Position template image not found: {template_path}")
    _, template_binary = cv2.threshold(template_image, 180, 255, cv2.THRESH_BINARY)
    return slice_position_templates(template_binary)


@lru_cache(maxsize=1)
def load_character_templates() -> List[Dict[str, object]]:
    """Load full-color character icon templates from the catalog-backed asset folder."""
    catalog = load_game_catalog()
    templates = []
    for character in catalog.characters:
        if int(character.character_index) in EXCLUDED_CHARACTER_TEMPLATE_INDICES:
            continue
        template_path = resolve_asset_file("character", f"{character.character_index}.png")
        if not template_path.exists():
            continue
        template_image = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if template_image is None:
            continue
        if len(template_image.shape) == 2:
            template_image = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGRA)
        elif template_image.shape[2] == 3:
            alpha_channel = np.full(template_image.shape[:2], 255, dtype=np.uint8)
            template_image = np.dstack((template_image, alpha_channel))
        resized_template = cv2.resize(
            template_image,
            (CHARACTER_TEMPLATE_SIZE, CHARACTER_TEMPLATE_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        template_rgb = resized_template[:, :, :3]
        template_alpha = resized_template[:, :, 3]
        templates.append(
            {
                "character_index": int(character.character_index),
                "character_name": str(character.name_uk),
                "template_image": template_rgb,
                "template_alpha": template_alpha,
            }
        )
    return templates


def shortlist_character_templates(templates: List[Dict[str, object]], allowed_indices: set[int]) -> List[Dict[str, object]]:
    if not allowed_indices:
        return templates
    shortlisted = [template for template in templates if int(template["character_index"]) in allowed_indices]
    return shortlisted or templates


def best_character_matches(row_roi: np.ndarray, templates: List[Dict[str, object]], limit: int = 2) -> List[Dict[str, object]]:
    ranked_matches = []
    for template in templates:
        match_result = masked_character_match_score(
            row_roi,
            template["template_image"],
            template["template_alpha"],
        )
        ranked_matches.append(
            {
                "Character": template["character_name"],
                "CharacterIndex": template["character_index"],
                "CharacterMatchConfidence": round(float(match_result["score"]) * 100.0, 1),
                "CharacterMatchMethod": "alpha_aware_color_template_local_search",
            }
        )
    ranked_matches.sort(key=lambda item: item["CharacterMatchConfidence"], reverse=True)
    return ranked_matches[:limit]


def update_player_character_prior(video_context: str | None, player_name: str, match: Dict[str, object]) -> None:
    if not video_context:
        return
    player_key = player_identity_key(player_name)
    if len(player_key) < 3:
        return
    character_index = match.get("CharacterIndex")
    if character_index is None:
        return
    with CHARACTER_SHORTLIST_LOCK:
        player_priors = PLAYER_CHARACTER_PRIORS[str(video_context)]
        existing = player_priors.get(player_key)
        if existing and int(existing["CharacterIndex"]) == int(character_index):
            existing["seen_count"] = int(existing.get("seen_count", 1)) + 1
            if match.get("CharacterMatchMethod") == "character_prior_confirm":
                existing["fast_accepts_since_revalidation"] = int(existing.get("fast_accepts_since_revalidation", 0)) + 1
            else:
                existing["fast_accepts_since_revalidation"] = 0
            existing["CharacterMatchConfidence"] = float(match.get("CharacterMatchConfidence", existing.get("CharacterMatchConfidence", 0.0)))
            existing["Character"] = str(match.get("Character", existing.get("Character", "")))
        else:
            player_priors[player_key] = {
                "CharacterIndex": int(character_index),
                "Character": str(match.get("Character", "")),
                "CharacterMatchConfidence": float(match.get("CharacterMatchConfidence", 0.0)),
                "seen_count": 1,
                "fast_accepts_since_revalidation": 0,
            }


def _template_match_score(source_image: np.ndarray, template_image: np.ndarray) -> float:
    source_height, source_width = source_image.shape[:2]
    template_height, template_width = template_image.shape[:2]
    if source_height < template_height or source_width < template_width:
        source_image = cv2.resize(source_image, (template_width, template_height), interpolation=cv2.INTER_NEAREST)
    result = cv2.matchTemplate(source_image, template_image, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, _ = cv2.minMaxLoc(result)
    return float(max_value)


def _best_template_overlap_metrics(source_image: np.ndarray, template_image: np.ndarray) -> Dict[str, float]:
    """Measure binary overlap quality for the best template placement inside the ROI.

    White IoU is intentionally foreground-centric:
    - template white over ROI white: good
    - template white over ROI black: bad
    - template black over ROI white: bad
    - template black over ROI black: ignored for the main overlap score

    This keeps empty rows from scoring artificially high just because most of the
    background is black.
    """
    source_height, source_width = source_image.shape[:2]
    template_height, template_width = template_image.shape[:2]
    if source_height < template_height or source_width < template_width:
        source_image = cv2.resize(source_image, (template_width, template_height), interpolation=cv2.INTER_NEAREST)
        source_height, source_width = source_image.shape[:2]

    template_white = template_image > 200
    best_metrics = None
    for offset_y in range(source_height - template_height + 1):
        for offset_x in range(source_width - template_width + 1):
            window = source_image[offset_y:offset_y + template_height, offset_x:offset_x + template_width]
            row_white = window > 200
            true_positive = int(np.count_nonzero(template_white & row_white))
            false_positive = int(np.count_nonzero((~template_white) & row_white))
            false_negative = int(np.count_nonzero(template_white & (~row_white)))
            union = true_positive + false_positive + false_negative
            weighted_union = true_positive + false_positive + (POSITION_FALSE_NEGATIVE_WEIGHT * false_negative)
            white_iou = true_positive / union if union else 0.0
            weighted_white_iou = true_positive / weighted_union if weighted_union else 0.0
            white_f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) if (2 * true_positive + false_positive + false_negative) else 0.0
            metrics = {
                "white_iou": float(white_iou),
                "weighted_white_iou": float(weighted_white_iou),
                "white_f1": float(white_f1),
                "tp": int(true_positive),
                "fp": int(false_positive),
                "fn": int(false_negative),
                "offset_x": int(offset_x),
                "offset_y": int(offset_y),
            }
            if best_metrics is None or metrics["weighted_white_iou"] > best_metrics["weighted_white_iou"]:
                best_metrics = metrics

    if best_metrics is None:
        return {
            "white_iou": 0.0,
            "weighted_white_iou": 0.0,
            "white_f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "offset_x": 0,
            "offset_y": 0,
        }
    return best_metrics


def _evaluate_position_template(row_image: np.ndarray, template_index: int, template: np.ndarray) -> Dict[str, float]:
    coefficient = _template_match_score(row_image, template)
    overlap_metrics = _best_template_overlap_metrics(row_image, template)
    best_coefficient = float(coefficient)
    best_overlap_metrics = overlap_metrics
    if int(template_index) == 12:
        inverse_template = cv2.bitwise_not(template)
        inverse_coefficient = _template_match_score(row_image, inverse_template)
        inverse_overlap_metrics = _best_template_overlap_metrics(row_image, inverse_template)
        if (
            float(inverse_coefficient) > best_coefficient
            or (
                float(inverse_coefficient) == best_coefficient
                and float(inverse_overlap_metrics["weighted_white_iou"]) > float(best_overlap_metrics["weighted_white_iou"])
            )
        ):
            best_coefficient = float(inverse_coefficient)
            best_overlap_metrics = inverse_overlap_metrics
    return {
        "template_index": int(template_index),
        "coefficient": float(best_coefficient),
        "white_iou": float(best_overlap_metrics["white_iou"]),
        "weighted_white_iou": float(best_overlap_metrics["weighted_white_iou"]),
        "white_f1": float(best_overlap_metrics["white_f1"]),
    }


def _position_template_candidate_score(template_score: Dict[str, float]) -> float:
    return (float(template_score["weighted_white_iou"]) * 10.0) + float(template_score["coefficient"])


def _build_position_metrics_from_template_scores(
    row_index: int,
    template_scores: List[Dict[str, float]],
    chosen_template_index: int | None = None,
) -> Dict[str, float]:
    coeff_sorted = sorted(template_scores, key=lambda item: item["coefficient"], reverse=True)
    coeff_best = coeff_sorted[0] if coeff_sorted else {"template_index": 0, "coefficient": 0.0, "white_iou": 0.0, "weighted_white_iou": 0.0}
    coeff_second = coeff_sorted[1] if len(coeff_sorted) > 1 else {"template_index": 0, "coefficient": 0.0, "white_iou": 0.0, "weighted_white_iou": 0.0}
    iou_sorted = sorted(template_scores, key=lambda item: item["weighted_white_iou"], reverse=True)
    iou_best = iou_sorted[0] if iou_sorted else {"template_index": 0, "coefficient": 0.0, "white_iou": 0.0, "weighted_white_iou": 0.0}

    shortlist = coeff_sorted[:3] if len(coeff_sorted) >= 3 else coeff_sorted
    hybrid_best = max(shortlist, key=lambda item: (item["weighted_white_iou"], item["coefficient"])) if shortlist else {"template_index": 0, "coefficient": 0.0, "white_iou": 0.0, "weighted_white_iou": 0.0}
    expected_score = next(
        (item for item in template_scores if int(item["template_index"]) == row_index + 1),
        {"coefficient": 0.0, "white_iou": 0.0, "weighted_white_iou": 0.0},
    )
    chosen_score = next(
        (item for item in template_scores if chosen_template_index is not None and int(item["template_index"]) == int(chosen_template_index)),
        hybrid_best,
    )
    all_coefficients = {f"PositionTemplate{template_index:02}_Coeff": None for template_index in range(1, 13)}
    for item in template_scores:
        all_coefficients[f"PositionTemplate{int(item['template_index']):02}_Coeff"] = round(float(item["coefficient"]), 3)
    return {
        "expected_position_score": round(float(expected_score["coefficient"]), 3),
        "expected_position_iou": round(float(expected_score["white_iou"]), 3),
        "expected_position_weighted_iou": round(float(expected_score["weighted_white_iou"]), 3),
        "best_position_score": round(float(coeff_best["coefficient"]), 3),
        "second_best_position_score": round(float(coeff_second["coefficient"]), 3),
        "position_margin": round(float(coeff_best["coefficient"]) - float(coeff_second["coefficient"]), 3),
        "any_position_score": round(float(coeff_best["coefficient"]), 3),
        "best_position_template": int(chosen_score["template_index"]),
        "best_position_coeff_template": int(coeff_best["template_index"]),
        "best_position_iou_template": int(iou_best["template_index"]),
        "coeff_ranked_templates": [int(item["template_index"]) for item in coeff_sorted],
        "best_position_iou": round(float(iou_best["white_iou"]), 3),
        "best_position_weighted_iou": round(float(iou_best["weighted_white_iou"]), 3),
        "best_position_template_score": round(float(chosen_score["coefficient"]), 3),
        "best_position_template_iou": round(float(chosen_score["white_iou"]), 3),
        "best_position_template_weighted_iou": round(float(chosen_score["weighted_white_iou"]), 3),
        "position_template_coefficients": all_coefficients,
    }


def _build_position_signal_metrics_fast(position_rows: List[np.ndarray], templates: List[np.ndarray]) -> List[Dict[str, float]]:
    beam = [{"score": 0.0, "prev_rank": 1, "chosen_ranks": [], "row_template_scores": []}]
    for row_index, row_image in enumerate(position_rows):
        row_number = row_index + 1
        next_beam = []
        for state in beam:
            previous_rank = int(state["prev_rank"])
            candidate_indices = {row_number} if row_index == 0 else {previous_rank, row_number}
            candidate_scores = []
            for candidate_index in sorted(candidate_indices):
                template = templates[candidate_index - 1]
                template_score = _evaluate_position_template(row_image, candidate_index, template)
                candidate_scores.append(template_score)
            for candidate_score in candidate_scores:
                next_beam.append(
                    {
                        "score": float(state["score"]) + _position_template_candidate_score(candidate_score),
                        "prev_rank": int(candidate_score["template_index"]),
                        "chosen_ranks": list(state["chosen_ranks"]) + [int(candidate_score["template_index"])],
                        "row_template_scores": list(state["row_template_scores"]) + [candidate_scores],
                    }
                )
        next_beam.sort(key=lambda item: item["score"], reverse=True)
        beam = next_beam[:POSITION_TEMPLATE_BEAM_WIDTH]
    best_state = beam[0] if beam else {"chosen_ranks": [], "row_template_scores": []}
    metrics = []
    for row_index, template_scores in enumerate(best_state.get("row_template_scores", [])):
        chosen_rank = None
        if row_index < len(best_state.get("chosen_ranks", [])):
            chosen_rank = int(best_state["chosen_ranks"][row_index])
        metrics.append(_build_position_metrics_from_template_scores(row_index, template_scores, chosen_rank))
    return metrics


def build_position_signal_metrics(processed_image: np.ndarray, score_layout_id: str | None = None) -> List[Dict[str, float]]:
    """Measure whether each row still shows a position number in the fixed left strip."""
    position_rows = extract_position_row_match_crops(processed_image, score_layout_id=score_layout_id)
    templates = load_position_row_templates()
    if POSITION_TEMPLATE_FAST_PATH_ENABLED:
        return _build_position_signal_metrics_fast(position_rows, templates)
    metrics = []
    for row_index, row_image in enumerate(position_rows):
        template_scores = [
            _evaluate_position_template(row_image, template_index, template)
            for template_index, template in enumerate(templates, start=1)
        ]
        metrics.append(_build_position_metrics_from_template_scores(row_index, template_scores))
    return metrics


def build_normalized_position_strip(processed_image: np.ndarray, score_layout_id: str | None = None) -> np.ndarray:
    """Return the position strip with the same binary polarity normalization used for matching."""
    (x1, y1), (x2, y2) = position_strip_roi(score_layout_id=score_layout_id)
    position_strip = processed_image[y1:y2, x1:x2]
    if len(position_strip.shape) == 3:
        position_strip = cv2.cvtColor(position_strip, cv2.COLOR_BGR2GRAY)
    _, position_strip = cv2.threshold(position_strip, 180, 255, cv2.THRESH_BINARY)
    return combine_position_rows(normalize_position_rows(position_strip), score_layout_id=score_layout_id)


def build_character_match_metrics(
    frame_image: np.ndarray,
    *,
    names: List[str] | None = None,
    name_confidences: List[int] | None = None,
    video_context: str | None = None,
    score_layout_id: str | None = None,
) -> List[Dict[str, object]]:
    """Template-match the full-color character icons for each scoreboard row."""
    templates = load_character_templates()
    if not templates:
        return [
            {
                "Character": "",
                "CharacterIndex": None,
                "CharacterMatchConfidence": 0.0,
                "CharacterMatchMethod": "missing_templates",
            }
            for _ in range(12)
        ]

    image_height, image_width = frame_image.shape[:2]
    metrics = []
    with CHARACTER_SHORTLIST_LOCK:
        shortlist_indices = set(CHARACTER_SHORTLIST_BY_VIDEO.get(str(video_context or ""), set()))
        prior_state_by_player = dict(PLAYER_CHARACTER_PRIORS.get(str(video_context or ""), {}))
    templates_by_index = {int(template["character_index"]): template for template in templates}
    for row_index in range(12):
        (x1, y1), (x2, y2) = character_row_roi(row_index, score_layout_id=score_layout_id)
        crop_x1 = max(0, min(image_width, x1))
        crop_x2 = max(crop_x1, min(image_width, x2))
        crop_y1 = max(0, min(image_height, y1))
        crop_y2 = max(crop_y1, min(image_height, y2))
        row_roi = frame_image[crop_y1:crop_y2, crop_x1:crop_x2]
        if row_roi.size == 0:
            metrics.append(
                {
                    "Character": "",
                    "CharacterIndex": None,
                    "CharacterMatchConfidence": 0.0,
                    "CharacterMatchMethod": "empty_roi",
                }
            )
            continue

        player_name = names[row_index] if names and row_index < len(names) else ""
        player_name_confidence = float(name_confidences[row_index]) if name_confidences and row_index < len(name_confidences) else 0.0
        player_key = player_identity_key(player_name)
        prior_state = prior_state_by_player.get(player_key)
        shortlist_templates = shortlist_character_templates(templates, shortlist_indices)
        if prior_state is not None:
            prior_index = int(prior_state["CharacterIndex"])
            if all(int(template["character_index"]) != prior_index for template in shortlist_templates):
                prior_template = templates_by_index.get(prior_index)
                if prior_template is not None:
                    shortlist_templates = shortlist_templates + [prior_template]
        shortlist_matches = best_character_matches(row_roi, shortlist_templates, limit=2)
        shortlist_best = shortlist_matches[0] if shortlist_matches else None
        shortlist_second = shortlist_matches[1] if len(shortlist_matches) > 1 else {"CharacterMatchConfidence": 0.0}
        shortlist_margin = (
            float(shortlist_best["CharacterMatchConfidence"]) - float(shortlist_second["CharacterMatchConfidence"])
            if shortlist_best is not None else 0.0
        )
        if (
            prior_state
            and player_name_confidence >= 80.0
            and int(prior_state.get("seen_count", 0)) >= CHARACTER_PRIOR_STABLE_MIN_SEEN
            and int(prior_state.get("fast_accepts_since_revalidation", 0)) < CHARACTER_PRIOR_MAX_FAST_ACCEPTS
        ):
            prior_character_name = str(prior_state.get("Character", ""))
            prior_is_best = (
                shortlist_best is not None
                and shortlist_best.get("CharacterIndex") is not None
                and int(shortlist_best["CharacterIndex"]) == int(prior_state["CharacterIndex"])
            )
            if (
                prior_is_best
                and float(shortlist_best["CharacterMatchConfidence"]) >= character_confidence_threshold(prior_character_name)
                and shortlist_margin >= character_margin_threshold(prior_character_name)
            ):
                CHARACTER_SHORTLIST_STATS["prior_accepts"] += 1
                best_match = dict(shortlist_best)
                best_match["CharacterMatchMethod"] = "character_prior_confirm"
                update_player_character_prior(video_context, player_name, best_match)
                metrics.append(best_match)
                continue
            CHARACTER_SHORTLIST_STATS["prior_fallbacks"] += 1

        use_shortlist = (
            shortlist_best is not None
            and len(shortlist_templates) < len(templates)
            and float(shortlist_best["CharacterMatchConfidence"]) >= character_confidence_threshold(str(shortlist_best.get("Character", "")))
            and shortlist_margin >= character_margin_threshold(str(shortlist_best.get("Character", "")))
        )

        if use_shortlist:
            CHARACTER_SHORTLIST_STATS["shortlist_accepts"] += 1
            best_match = dict(shortlist_best)
            best_match["CharacterMatchMethod"] = "character_shortlist_alpha_search"
        else:
            if len(shortlist_templates) < len(templates):
                CHARACTER_SHORTLIST_STATS["shortlist_fallbacks"] += 1
            CHARACTER_SHORTLIST_STATS["full_search_rows"] += 1
            full_matches = best_character_matches(row_roi, templates, limit=2)
            best_match = full_matches[0] if full_matches else None
            if best_match is not None and video_context:
                best_index = int(best_match["CharacterIndex"])
                with CHARACTER_SHORTLIST_LOCK:
                    shortlist = CHARACTER_SHORTLIST_BY_VIDEO[str(video_context)]
                    if best_index not in shortlist:
                        shortlist.add(best_index)
                        CHARACTER_SHORTLIST_STATS["shortlist_expansions"] += 1

        metrics.append(best_match or {
            "Character": "",
            "CharacterIndex": None,
            "CharacterMatchConfidence": 0.0,
            "CharacterMatchMethod": "no_match",
        })

        if metrics[-1].get("CharacterIndex") is not None:
            update_player_character_prior(video_context, player_name, metrics[-1])
    return metrics


def masked_character_match_score(source_image: np.ndarray, template_image: np.ndarray, template_alpha: np.ndarray) -> Dict[str, float]:
    """Match a character icon while ignoring fully transparent template pixels.

    The score is color-based and alpha-aware:
    - only pixels where the template alpha is visible contribute
    - transparent template background does not count toward confidence
    - the best local placement inside the padded ROI is returned
    """
    source_height, source_width = source_image.shape[:2]
    template_height, template_width = template_image.shape[:2]
    if source_height < template_height or source_width < template_width:
        source_image = cv2.resize(source_image, (template_width, template_height), interpolation=cv2.INTER_LINEAR)
        source_height, source_width = source_image.shape[:2]

    visible_mask = template_alpha > 16
    visible_pixels = int(np.count_nonzero(visible_mask))
    if visible_pixels <= 0:
        return {"score": 0.0, "offset_x": 0, "offset_y": 0}

    best_score = None
    best_offset = (0, 0)
    template_rgb = template_image.astype(np.float32)
    mask_3d = np.repeat(visible_mask[:, :, None], 3, axis=2)

    for offset_y in range(source_height - template_height + 1):
        for offset_x in range(source_width - template_width + 1):
            window = source_image[offset_y:offset_y + template_height, offset_x:offset_x + template_width].astype(np.float32)
            masked_template = template_rgb[mask_3d]
            masked_window = window[mask_3d]
            if masked_window.size == 0:
                continue
            mean_abs_diff = float(np.mean(np.abs(masked_template - masked_window)))
            score = max(0.0, 1.0 - (mean_abs_diff / 255.0))
            if best_score is None or score > best_score:
                best_score = score
                best_offset = (offset_x, offset_y)

    return {
        "score": float(best_score or 0.0),
        "offset_x": int(best_offset[0]),
        "offset_y": int(best_offset[1]),
    }


def build_row_presence_metrics(names: List[str], confidence_scores: List[int], race_points: List[str], total_points: List[str]) -> List[Dict[str, object]]:
    """Describe how strongly each of the 12 scoreboard rows looks occupied."""
    metrics = []
    for index, player_name in enumerate(names):
        stripped_name = re.sub(r"[^a-zA-Z0-9]", "", player_name or "")
        confidence = float(confidence_scores[index] if index < len(confidence_scores) else 0)
        has_strong_name = len(stripped_name) >= 3 and len(set(stripped_name)) >= 3
        has_any_name = bool(stripped_name)
        has_race_points = bool(race_points[index]) if index < len(race_points) else False
        has_total_points = bool(total_points[index]) if index < len(total_points) else False
        legacy_row_present = bool(
            has_strong_name
            or has_race_points
            or has_total_points
            or confidence >= 35
        )

        occupancy_score = 0.0
        if has_strong_name:
            occupancy_score += 1.5
        elif has_any_name:
            occupancy_score += 0.4
        if has_race_points:
            occupancy_score += 1.0
        if has_total_points:
            occupancy_score += 1.0
        if confidence >= 70:
            occupancy_score += 0.5
        elif confidence >= 35:
            occupancy_score += 0.2

        evidence_parts = []
        if has_strong_name:
            evidence_parts.append("strong_name")
        elif has_any_name:
            evidence_parts.append("weak_name")
        if has_race_points:
            evidence_parts.append("race_points")
        if has_total_points:
            evidence_parts.append("total_points")
        if confidence >= 35:
            evidence_parts.append(f"conf={int(round(confidence))}")

        metrics.append(
            {
                "row_number": index + 1,
                "legacy_row_present": legacy_row_present,
                "occupancy_score": round(occupancy_score, 2),
                "evidence": ",".join(evidence_parts) if evidence_parts else "empty",
            }
        )
    return metrics


def determine_position_guided_visible_rows(row_metrics: List[Dict[str, object]], occupancy_threshold: float = 1.0) -> int:
    """Count visible rows using occupancy plus position-strip presence.

    Player count should be based on whether a row visibly contains a plausible rank badge,
    not on whether the exact row-number template wins there. This avoids undercounting on
    ties or near-neighbour confusions such as row 12 visually preferring template 11.
    """
    for metric in reversed(row_metrics):
        row_number = int(metric.get("row_number", 0))
        best_position_score = float(metric.get("best_position_score", 0.0))
        occupancy_score = float(metric.get("occupancy_score", 0.0))

        threshold = POSITION_PRESENT_ROW1_COEFF_THRESHOLD if row_number == 1 else POSITION_PRESENT_COEFF_THRESHOLD
        row_supported = occupancy_score >= occupancy_threshold and best_position_score >= threshold
        if row_supported:
            return row_number
    return 0


def _observation_supports_low_res_row12_character_fallback(observation: Dict[str, object]) -> bool:
    row_metrics = observation.get("row_metrics") or []
    character_metrics = observation.get("character_metrics") or []
    if len(row_metrics) < 12 or len(character_metrics) < 12:
        return False

    row11 = row_metrics[10]
    row12 = row_metrics[11]
    row11_threshold = POSITION_PRESENT_COEFF_THRESHOLD
    row11_supported = float(row11.get("best_position_score", 0.0)) >= row11_threshold
    if not row11_supported:
        return False

    row12_character = character_metrics[11]
    if row12_character.get("CharacterIndex") is None:
        return False

    row12_character_confidence = float(row12_character.get("CharacterMatchConfidence", 0.0))
    if row12_character_confidence < LOW_RES_ROW12_CHARACTER_FALLBACK_MIN_CONFIDENCE:
        return False

    row12_position_score = max(
        float(row12.get("expected_position_score", 0.0)),
        float(row12.get("best_position_score", 0.0)),
        float(row12.get("any_position_score", 0.0)),
    )
    return row12_position_score >= LOW_RES_ROW12_CHARACTER_FALLBACK_MIN_POSITION_SCORE


def _frame_supports_low_res_row12_character_fallback(
    frame_image: np.ndarray,
    video_context: str | None = None,
    score_layout_id: str | None = None,
) -> bool:
    processed_image = process_image(frame_image, score_layout_id=score_layout_id)
    position_metrics = build_position_signal_metrics(processed_image, score_layout_id=score_layout_id)
    character_metrics = build_character_match_metrics(frame_image, video_context=video_context, score_layout_id=score_layout_id)
    observation = {
        "row_metrics": [
            {
                "best_position_score": float(metric.get("best_position_score", 0.0)),
                "expected_position_score": float(metric.get("expected_position_score", 0.0)),
                "any_position_score": float(metric.get("any_position_score", 0.0)),
            }
            for metric in position_metrics
        ],
        "character_metrics": character_metrics,
    }
    return _observation_supports_low_res_row12_character_fallback(observation)


def _frame_supports_ultra_low_res_row12_blob_fallback(frame_image: np.ndarray, score_layout_id: str | None = None) -> bool:
    image_height, image_width = frame_image.shape[:2]
    (x1, y1), (x2, y2) = ultra_low_res_combined_row_roi(11, score_layout_id=score_layout_id)
    crop_x1 = max(0, min(image_width, x1))
    crop_x2 = max(crop_x1, min(image_width, x2))
    crop_y1 = max(0, min(image_height, y1))
    crop_y2 = max(crop_y1, min(image_height, y2))
    row_roi = frame_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if row_roi.size == 0:
        return False

    gray = cv2.cvtColor(row_roi, cv2.COLOR_BGR2GRAY)
    stddev = float(gray.std())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size) if edges.size else 0.0
    return stddev >= ULTRA_LOW_RES_ROW_MIN_STDDEV and edge_density >= ULTRA_LOW_RES_ROW_MIN_EDGE_DENSITY


def summarize_row_metrics(row_metrics: List[Dict[str, object]]) -> str:
    parts = []
    for metric in row_metrics:
        parts.append(
            f"{int(metric['row_number']):02}:{float(metric['occupancy_score']):.2f}"
            f"[{metric['evidence']};pos={float(metric.get('expected_position_score', 0.0)):.2f}/{float(metric.get('best_position_score', 0.0)):.2f}"
            f"/{float(metric.get('second_best_position_score', 0.0)):.2f}"
            f"/iou{float(metric.get('best_position_iou', 0.0)):.2f}"
            f"/wiou{float(metric.get('best_position_weighted_iou', 0.0)):.2f}"
            f"/m{float(metric.get('position_margin', 0.0)):.2f}"
            f"#h{int(metric.get('best_position_template', 0))}"
            f"/c{int(metric.get('best_position_coeff_template', 0))}"
            f"/i{int(metric.get('best_position_iou_template', 0))}]"
        )
    return " | ".join(parts)


def summarize_count_votes(observations: List[Dict[str, object]], key: str) -> str:
    count_votes = Counter(int(observation.get(key, 0)) for observation in observations if int(observation.get(key, 0)) > 0)
    if not count_votes:
        return ""
    return ", ".join(f"{count}x{votes}" for count, votes in count_votes.most_common())


def apply_threshold(image: np.ndarray, threshold: int = 205) -> np.ndarray:
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image


def apply_inversion(image: np.ndarray) -> np.ndarray:
    return cv2.bitwise_not(image)


def preprocess_image_v_channel(image: np.ndarray, threshold_value: int = 205) -> np.ndarray:
    """Use the V channel because the scoreboard text survives there most consistently."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv_image)
    _, binary_v = cv2.threshold(v, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_v


def crop_and_process_image(frame: np.ndarray, coordinates: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                           image_type: str) -> List[np.ndarray]:
    """Prepare the scoreboard ROIs differently for names and digits."""
    cropped_images = []
    for (x1, y1), (x2, y2) in coordinates:
        section_img = frame[y1:y2, x1:x2]
        binary_section = preprocess_image_v_channel(section_img, 205)
        black_pixels = np.count_nonzero(binary_section == 0)
        white_pixels = np.count_nonzero(binary_section == 255)

        if white_pixels > black_pixels:
            section_img = Image.fromarray(cv2.bitwise_not(binary_section))
        else:
            section_img = Image.fromarray(binary_section)

        if image_type in ["race_points", "total_points"]:
            section_img_np = np.array(section_img)
            if len(section_img_np.shape) == 3 and section_img_np.shape[2] == 3:
                gray_image = cv2.cvtColor(section_img_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = section_img_np
            kernel = np.ones((2, 2), np.uint8)
            dilated_image = cv2.dilate(gray_image, kernel, iterations=1)
            section_img = Image.fromarray(cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2RGB))

        cropped_images.append(np.array(section_img))
    return cropped_images


def process_image(image_source, score_layout_id: str | None = None) -> np.ndarray:
    """Rewrite the scoreboard into OCR-friendly blocks before digit and name reading."""
    score_layout = get_score_layout(score_layout_id)
    coordinates = {
        "player_name": score_layout.player_name_coords,
        "race_points": score_layout.race_points_coords,
        "total_points": score_layout.total_points_coords,
    }

    if isinstance(image_source, str):
        image = cv2.imread(image_source, cv2.IMREAD_COLOR)
        image_path = image_source
    else:
        image = image_source
        image_path = "<array>"
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    processed_image = preprocess_image_v_channel(image)
    if len(processed_image.shape) == 2:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

    for region_type, coord_list in coordinates.items():
        rois = crop_and_process_image(processed_image, coord_list, region_type)
        for roi, ((x1, y1), (x2, y2)) in zip(rois, coord_list):
            if len(roi.shape) == 2:
                roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            processed_image[y1:y2, x1:x2] = roi
    return processed_image


def is_white_box(
    image: Image.Image,
    top_left: Tuple[int, int],
    box_size: Tuple[int, int] = (3, 2),
    *,
    neighborhood_radius: int = 1,
    white_threshold: int = 180,
    min_white_ratio: float = 0.35,
) -> bool:
    x, y = top_left
    width, height = box_size
    white_pixels = 0
    total_pixels = width * height
    for offset_x in range(width):
        for offset_y in range(height):
            sample_x = x + offset_x
            sample_y = y + offset_y
            found_white = False
            for delta_x in range(-neighborhood_radius, neighborhood_radius + 1):
                for delta_y in range(-neighborhood_radius, neighborhood_radius + 1):
                    pixel_x = sample_x + delta_x
                    pixel_y = sample_y + delta_y
                    if pixel_x < 0 or pixel_y < 0 or pixel_x >= image.width or pixel_y >= image.height:
                        continue
                    r, g, b = image.getpixel((pixel_x, pixel_y))
                    if r > white_threshold and g > white_threshold and b > white_threshold:
                        found_white = True
                        break
                if found_white:
                    break
            if found_white:
                white_pixels += 1
    return white_pixels >= max(1, int(total_pixels * min_white_ratio))


SEGMENT_LABELS_HORIZONTAL = {"top_middle", "center", "middle_bottom_edge"}
SEGMENT_LABELS_VERTICAL = {
    "left_middle",
    "right_middle",
    "left_bottom",
    "right_bottom",
    "middle_middle",
    "middle_bottom",
}
DIGIT_PATTERNS = [
    (8, {"top_middle", "left_middle", "right_middle", "center", "right_bottom", "left_bottom", "middle_bottom_edge"}),
    (0, {"top_middle", "left_middle", "right_middle", "left_bottom", "right_bottom", "middle_bottom_edge"}),
    (6, {"top_middle", "left_middle", "center", "right_bottom", "left_bottom", "middle_bottom_edge"}),
    (9, {"top_middle", "left_middle", "right_middle", "center", "right_bottom", "middle_bottom_edge"}),
    (2, {"top_middle", "right_middle", "center", "left_bottom", "middle_bottom_edge"}),
    (3, {"top_middle", "right_middle", "center", "right_bottom", "middle_bottom_edge"}),
    (5, {"top_middle", "left_middle", "center", "right_bottom", "middle_bottom_edge"}),
    (4, {"right_middle", "left_middle", "center", "right_bottom"}),
    (7, {"top_middle", "right_middle", "right_bottom"}),
    (1, {"middle_middle", "middle_bottom"}),
]


def segment_roi_bounds(
    box_top_left: Tuple[int, int],
    point_offset: Union[Tuple[int, int], Dict[str, int]],
    label: str,
) -> Tuple[int, int, int, int]:
    if isinstance(point_offset, dict):
        return (
            box_top_left[0] + point_offset["x"],
            box_top_left[1] + point_offset["y"],
            box_top_left[0] + point_offset["x"] + point_offset["width"],
            box_top_left[1] + point_offset["y"] + point_offset["height"],
        )

    anchor_x = box_top_left[0] + point_offset[0]
    anchor_y = box_top_left[1] + point_offset[1]
    if label in SEGMENT_LABELS_HORIZONTAL:
        half_width = 8
        half_height = 3
    elif label in SEGMENT_LABELS_VERTICAL:
        half_width = 2
        half_height = 8
    else:
        half_width = 3
        half_height = 3
    extra_right = 0
    if label in {"left_middle", "right_middle", "left_bottom", "right_bottom"}:
        extra_right = 12
    return (
        anchor_x - half_width,
        anchor_y - half_height,
        anchor_x + half_width + 1 + extra_right,
        anchor_y + half_height + 1,
    )


def segment_roi_stats(
    image: Image.Image,
    bounds: Tuple[int, int, int, int],
    *,
    white_threshold: int = 180,
) -> Dict[str, float]:
    x1, y1, x2, y2 = bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    white_pixels = 0
    black_pixels = 0
    total_pixels = 0
    for pixel_x in range(x1, x2):
        for pixel_y in range(y1, y2):
            total_pixels += 1
            r, g, b = image.getpixel((pixel_x, pixel_y))
            if r > white_threshold and g > white_threshold and b > white_threshold:
                white_pixels += 1
            else:
                black_pixels += 1
    if total_pixels == 0:
        return {"white_ratio": 0.0, "black_ratio": 1.0}
    return {
        "white_ratio": white_pixels / total_pixels,
        "black_ratio": black_pixels / total_pixels,
    }


def score_digit_candidate(
    segment_stats: Dict[str, Dict[str, float]],
    pattern: set[str],
    *,
    active_black_penalty: float,
) -> float:
    score = 0.0
    for label, stats in segment_stats.items():
        white_ratio = stats["white_ratio"]
        black_ratio = stats["black_ratio"]
        if label in pattern:
            score += white_ratio
            score -= black_ratio * active_black_penalty
        else:
            score += (1.0 - white_ratio) * 0.08
            score -= white_ratio * 0.45
    return score


def analyze_digit_segments(
    image: Image.Image,
    box_top_left: Tuple[int, int],
    red_pixels: Dict[str, Tuple[int, int]],
) -> Tuple[int, Dict[str, Dict[str, float]], set[str]]:
    """Identify a digit and return segment stats for debug rendering."""
    segment_stats = {
        label: segment_roi_stats(image, segment_roi_bounds(box_top_left, offset, label))
        for label, offset in red_pixels.items()
    }
    strong_active_labels = {
        label for label, stats in segment_stats.items() if stats["white_ratio"] >= 0.45
    }
    for digit, pattern in DIGIT_PATTERNS:
        if pattern.issubset(strong_active_labels):
            return digit, segment_stats, strong_active_labels
    if not strong_active_labels:
        return -1, segment_stats, strong_active_labels

    method_votes: list[int] = []
    aggregate_scores: Dict[int, float] = {}
    pattern_lookup = {digit: pattern for digit, pattern in DIGIT_PATTERNS}
    for active_black_penalty in (1.0, 2.0, 3.0):
        best_digit = -1
        best_score = float("-inf")
        for digit, pattern in DIGIT_PATTERNS:
            candidate_score = score_digit_candidate(
                segment_stats,
                pattern,
                active_black_penalty=active_black_penalty,
            )
            aggregate_scores[digit] = aggregate_scores.get(digit, 0.0) + candidate_score
            if candidate_score > best_score:
                best_score = candidate_score
                best_digit = digit
        if best_digit != -1:
            method_votes.append(best_digit)

    if method_votes:
        vote_counts = Counter(method_votes)
        best_digit, best_count = vote_counts.most_common(1)[0]
        if best_count >= 2:
            return best_digit, segment_stats, strong_active_labels

    best_digit = max(aggregate_scores.items(), key=lambda item: item[1])[0]
    best_pattern = pattern_lookup[best_digit]
    matched_ratio = len(best_pattern & strong_active_labels) / max(1, len(best_pattern))
    return (best_digit if matched_ratio >= 0.5 else -1), segment_stats, strong_active_labels


def identify_digit(image: Image.Image, box_top_left: Tuple[int, int], red_pixels: Dict[str, Tuple[int, int]]) -> int:
    """Identify a digit from segment ROIs instead of single-point hits."""
    digit, _segment_stats, _strong_active_labels = analyze_digit_segments(image, box_top_left, red_pixels)
    return digit


def ocr_digit_row_fallback(
    image: Image.Image,
    start_coords: List[Tuple[int, int]],
    row_offset: int,
    box_dims: Tuple[int, int],
    row_index: int,
    boxes_per_row: int,
    *,
    valid_min: int,
    valid_max: int,
    bundle_kind: str,
    field_name: str,
) -> str:
    y_offset = row_index * row_offset
    x1 = start_coords[0][0]
    y1 = start_coords[0][1] + y_offset
    x2 = start_coords[boxes_per_row - 1][0] + box_dims[0]
    y2 = y1 + box_dims[1]
    padding = 6
    crop = image.crop((x1 - padding, y1 - padding, x2 + padding, y2 + padding)).convert('L')
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.NEAREST)
    crop_np = np.array(crop)
    _, crop_np = cv2.threshold(crop_np, 180, 255, cv2.THRESH_BINARY)
    reader = _get_digit_easyocr_reader()
    if reader is None:
        return ''
    start_time = time.perf_counter()
    result = reader.readtext(crop_np, detail=0, paragraph=False, allowlist='0123456789')
    record_call_matrix(bundle_kind, field_name, "ocr_fallback", time.perf_counter() - start_time)
    text = ' '.join(str(item).strip() for item in result if str(item).strip())
    digits_only = re.sub(r'[^0-9]', '', text)
    if not digits_only:
        return ''
    try:
        value = int(digits_only)
    except ValueError:
        return ''
    if valid_min <= value <= valid_max:
        return digits_only
    return ''


def _get_digit_easyocr_reader():
    global _DIGIT_EASYOCR_READER
    if _DIGIT_EASYOCR_READER is not None:
        return _DIGIT_EASYOCR_READER
    if easyocr is None:
        return None
    with _DIGIT_EASYOCR_LOCK:
        if _DIGIT_EASYOCR_READER is None:
            _DIGIT_EASYOCR_READER = easyocr.Reader(['en'], gpu=easyocr_gpu_enabled(), verbose=False)
        return _DIGIT_EASYOCR_READER


def _digit_box_bounds(top_left: Tuple[int, int], box_dims: Tuple[int, int]) -> Tuple[int, int, int, int]:
    return (
        int(top_left[0]),
        int(top_left[1]),
        int(top_left[0] + box_dims[0]),
        int(top_left[1] + box_dims[1]),
    )


def _row_bounds(
    start_coords: List[Tuple[int, int]],
    row_offset: int,
    box_dims: Tuple[int, int],
    row_index: int,
    boxes_per_row: int,
) -> Tuple[int, int, int, int]:
    y_offset = row_index * row_offset
    x1 = int(start_coords[0][0])
    y1 = int(start_coords[0][1] + y_offset)
    x2 = int(start_coords[boxes_per_row - 1][0] + box_dims[0])
    y2 = int(y1 + box_dims[1])
    return (x1, y1, x2, y2)


def _rescale_bounds(
    bounds: Tuple[int, int, int, int],
    draw_scale_divisor: int,
) -> Tuple[int, int, int, int]:
    if int(draw_scale_divisor) <= 1:
        return bounds
    x1, y1, x2, y2 = bounds
    return (
        int(round(float(x1) / float(draw_scale_divisor))),
        int(round(float(y1) / float(draw_scale_divisor))),
        int(round(float(x2) / float(draw_scale_divisor))),
        int(round(float(y2) / float(draw_scale_divisor))),
    )


def detect_digits_in_image(image: Image.Image, start_coords: List[Tuple[int, int]], row_offset: int,
                           box_dims: Tuple[int, int], red_pixels: Dict[str, Tuple[int, int]],
                           num_rows: int, boxes_per_row: int, *, valid_min: int, valid_max: int,
                           annotation_prefix: str = "", bundle_kind: str = "", field_name: str = "",
                           draw_image: Image.Image | None = None, draw_scale_divisor: int = 1) -> Tuple[List[str], List[str]]:
    coordinate_set = []
    coordinate_sources = []
    draw = ImageDraw.Draw(draw_image if draw_image is not None else image)
    for row_index in range(num_rows):
        row_stage_start = time.perf_counter()
        y_offset = row_index * row_offset
        row_bounds = _row_bounds(start_coords, row_offset, box_dims, row_index, boxes_per_row)
        draw.rectangle(_rescale_bounds(row_bounds, draw_scale_divisor), outline="cyan", width=1)
        row_digits = []
        has_unknown_digit = False
        for box_index in range(boxes_per_row):
            start_x, start_y = start_coords[box_index]
            top_left = (start_x, start_y + y_offset)
            digit_bounds = _digit_box_bounds(top_left, box_dims)
            scaled_digit_bounds = _rescale_bounds(digit_bounds, draw_scale_divisor)
            draw.rectangle(scaled_digit_bounds, outline="yellow", width=1)
            digit, segment_stats, strong_active_labels = analyze_digit_segments(image, top_left, red_pixels)
            if digit != -1:
                row_digits.append(str(digit))
            else:
                row_digits.append('')
                has_unknown_digit = True
            digit_label = str(digit) if digit != -1 else "?"
            draw.text((scaled_digit_bounds[0] + 2, scaled_digit_bounds[1] + 2), digit_label, fill="cyan")
            for label, offset in red_pixels.items():
                roi_bounds = segment_roi_bounds(top_left, offset, label)
                segment_is_on = label in strong_active_labels
                outline_color = "lime" if segment_is_on else "red"
                draw.rectangle(_rescale_bounds(roi_bounds, draw_scale_divisor), outline=outline_color, width=1)
        row_number = ''.join(row_digits)
        numeric_value = parse_detected_int(row_number)
        recognized_indices = [index for index, digit_text in enumerate(row_digits) if digit_text]
        contiguous_digit_block = False
        if recognized_indices:
            first_recognized_index = recognized_indices[0]
            last_recognized_index = recognized_indices[-1]
            contiguous_digit_block = all(
                digit_text != '' for digit_text in row_digits[first_recognized_index:last_recognized_index + 1]
            )
        should_fallback = numeric_value is None or not (valid_min <= int(numeric_value) <= valid_max)
        if has_unknown_digit and not contiguous_digit_block:
            should_fallback = True
        row_source = "7-segment"
        record_call_matrix(bundle_kind, field_name, "seven_segment", time.perf_counter() - row_stage_start)
        if should_fallback and DIGIT_OCR_FALLBACK_ENABLED:
            fallback_value = ocr_digit_row_fallback(
                image,
                start_coords,
                row_offset,
                box_dims,
                row_index,
                boxes_per_row,
                valid_min=valid_min,
                valid_max=valid_max,
                bundle_kind=bundle_kind,
                field_name=field_name,
            )
            if fallback_value:
                row_number = fallback_value
                row_source = "ocr_fallback"
        coordinate_set.append(row_number)
        coordinate_sources.append(row_source)
    return coordinate_set, coordinate_sources


def scale_coords(coords, scale_factor):
    return [(x * scale_factor, y * scale_factor) for x, y in coords]


def scale_pixel_positions(pixels, scale_factor):
    scaled = {}
    for label, value in pixels.items():
        if isinstance(value, dict):
            scaled[label] = {
                "x": int(value["x"] * scale_factor),
                "y": int(value["y"] * scale_factor),
                "width": int(value["width"] * scale_factor),
                "height": int(value["height"] * scale_factor),
            }
        else:
            x, y = value
            scaled[label] = (x * scale_factor, y * scale_factor)
    return scaled


CANONICAL_SEVEN_SEGMENT_BOXES = {
    "top_middle": {"x": 28, "y": 8, "width": 17, "height": 9},
    "left_middle": {"x": 9, "y": 17, "width": 16, "height": 23},
    "middle_middle": {"x": 32, "y": 21, "width": 9, "height": 18},
    "right_middle": {"x": 51, "y": 17, "width": 15, "height": 23},
    "left_bottom": {"x": 9, "y": 57, "width": 16, "height": 23},
    "middle_bottom": {"x": 32, "y": 59, "width": 9, "height": 18},
    "right_bottom": {"x": 51, "y": 57, "width": 16, "height": 23},
    "middle_bottom_edge": {"x": 28, "y": 82, "width": 17, "height": 9},
    "center": {"x": 24, "y": 44, "width": 25, "height": 9},
}


def _scale_segment_boxes(
    segment_boxes: Dict[str, Dict[str, int]],
    *,
    source_box_width: int,
    source_box_height: int,
    target_box_width: int,
    target_box_height: int,
) -> Dict[str, Dict[str, int]]:
    scale_x = float(target_box_width) / float(source_box_width)
    scale_y = float(target_box_height) / float(source_box_height)
    scaled: Dict[str, Dict[str, int]] = {}
    for label, box in segment_boxes.items():
        scaled[label] = {
            "x": int(round(float(box["x"]) * scale_x)),
            "y": int(round(float(box["y"]) * scale_y)),
            "width": max(1, int(round(float(box["width"]) * scale_x))),
            "height": max(1, int(round(float(box["height"]) * scale_y))),
        }
    return scaled


def parse_detected_int(value: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return int(round(float(value)))

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        return int(float(text))

    stripped = re.sub(r"[^0-9]", "", text)
    if not stripped:
        return None
    return int(stripped)


def score_digit_layout(scale_factor: int = 5, score_layout_id: str | None = None):
    score_layout = get_score_layout(score_layout_id)
    start_coords_run1 = scale_coords(score_layout.race_digit_starts, scale_factor)
    race_digit_box = (13 * scale_factor, 19 * scale_factor)
    total_digit_box = (16 * scale_factor, 24 * scale_factor)
    red_pixels_run1 = {
        label: {key: int(value) for key, value in box.items()}
        for label, box in CANONICAL_SEVEN_SEGMENT_BOXES.items()
    }
    start_coords_run2 = scale_coords(score_layout.total_digit_starts, scale_factor)
    red_pixels_run2 = _scale_segment_boxes(
        CANONICAL_SEVEN_SEGMENT_BOXES,
        source_box_width=race_digit_box[0],
        source_box_height=race_digit_box[1],
        target_box_width=total_digit_box[0],
        target_box_height=total_digit_box[1],
    )
    return {
        "race_points": (start_coords_run1, 52 * scale_factor, race_digit_box, red_pixels_run1, 12, 2),
        "total_points": (start_coords_run2, 52 * scale_factor, total_digit_box, red_pixels_run2, 12, 3),
    }


def extract_scoreboard_observation(
    frame_image: np.ndarray,
    extract_player_names_batched,
    annotate_path: str | None = None,
    annotation_prefix: str = "",
    video_context: str | None = None,
    score_layout_id: str | None = None,
    bundle_kind: str = "",
    name_field_name: str = "",
    race_points_field_name: str = "",
    total_points_field_name: str = "",
) -> Dict[str, object]:
    """Read one score frame into names, race points, totals, and a visible-row estimate."""
    score_layout = get_score_layout(score_layout_id)
    stage_start = time.perf_counter()
    processed_img = process_image(frame_image, score_layout_id=score_layout.layout_id)
    record_observation_stage("process_image", time.perf_counter() - stage_start)

    (position_x1, position_y1), (position_x2, position_y2) = position_strip_roi(score_layout_id=score_layout.layout_id)

    stage_start = time.perf_counter()
    normalized_position_strip = build_normalized_position_strip(processed_img, score_layout_id=score_layout.layout_id)
    record_observation_stage("normalize_position_strip", time.perf_counter() - stage_start)

    processed_img[position_y1:position_y2, position_x1:position_x2] = cv2.cvtColor(normalized_position_strip, cv2.COLOR_GRAY2BGR)

    processed_img_pil = Image.fromarray(processed_img).convert("RGB")
    annotation_image = processed_img_pil.copy()
    scale_factor = 5
    stage_start = time.perf_counter()
    scaled_image = processed_img_pil.resize(
        (processed_img_pil.width * scale_factor, processed_img_pil.height * scale_factor),
        Image.NEAREST,
    )
    record_observation_stage("scale_image", time.perf_counter() - stage_start)
    layout = score_digit_layout(scale_factor, score_layout_id=score_layout.layout_id)

    race_points = [""] * layout["race_points"][4]
    race_point_sources = ["skipped"] * layout["race_points"][4]
    if race_points_field_name and (
        race_points_field_name != "RacePointsOnTotalScore" or TOTAL_SCORE_RACE_POINTS_ENABLED
    ):
        stage_start = time.perf_counter()
        race_points, race_point_sources = detect_digits_in_image(
            scaled_image,
            *layout["race_points"],
            valid_min=1,
            valid_max=15,
            annotation_prefix=f"{annotation_prefix}RP",
            bundle_kind=bundle_kind,
            field_name=race_points_field_name,
            draw_image=annotation_image,
            draw_scale_divisor=scale_factor,
        )
        record_observation_stage("detect_race_points", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    total_points, total_point_sources = detect_digits_in_image(
        scaled_image,
        *layout["total_points"],
        valid_min=0,
        valid_max=999,
        annotation_prefix=f"{annotation_prefix}TP",
        bundle_kind=bundle_kind,
        field_name=total_points_field_name,
        draw_image=annotation_image,
        draw_scale_divisor=scale_factor,
    )
    record_observation_stage("detect_total_points", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    annotated_image = cv2.cvtColor(np.array(annotation_image), cv2.COLOR_RGB2BGR)
    record_observation_stage("prepare_name_image", time.perf_counter() - stage_start)
    if annotate_path:
        Path(annotate_path).parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"format": "JPEG", "quality": 95} if EXPORT_IMAGE_FORMAT == "jpg" else {"format": "PNG"}
        annotation_image.save(annotate_path, **save_kwargs)

    stage_start = time.perf_counter()
    names, confidence_scores = extract_player_names_batched(
        annotated_image,
        score_layout.player_name_coords,
        bundle_kind=bundle_kind,
        field_name=name_field_name,
    )
    record_observation_stage("extract_player_names", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    character_metrics = build_character_match_metrics(
        frame_image,
        names=names,
        name_confidences=confidence_scores,
        video_context=video_context,
        score_layout_id=score_layout.layout_id,
    )
    record_observation_stage("character_metrics", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    row_metrics = build_row_presence_metrics(names, confidence_scores, race_points, total_points)
    record_observation_stage("build_row_presence_metrics", time.perf_counter() - stage_start)

    stage_start = time.perf_counter()
    position_metrics = build_position_signal_metrics(processed_img, score_layout_id=score_layout.layout_id)
    record_observation_stage("build_position_signal_metrics", time.perf_counter() - stage_start)
    for row_metric, position_metric in zip(row_metrics, position_metrics):
        row_metric.update(position_metric)
    valid_rows = [bool(metric["legacy_row_present"]) for metric in row_metrics]

    visible_rows = 0
    for index, row_present in enumerate(valid_rows, start=1):
        if row_present:
            visible_rows = index

    position_guided_visible_rows = determine_position_guided_visible_rows(row_metrics)

    template_row_confidence = max(0.0, min(1.0, visible_rows / 12.0))
    return {
        "names": names,
        "name_confidences": confidence_scores,
        "race_points": race_points,
        "race_point_sources": race_point_sources,
        "total_points": total_points,
        "total_point_sources": total_point_sources,
        "character_metrics": character_metrics,
        "row_metrics": row_metrics,
        "row_metrics_summary": summarize_row_metrics(row_metrics),
        "visible_rows": visible_rows,
        "position_guided_visible_rows": position_guided_visible_rows,
        "template_row_confidence": template_row_confidence,
        "score_layout_id": score_layout.layout_id,
    }


def normalize_name_for_vote(name: str) -> str:
    return collapse_name_whitespace(name)


def _normalize_name_key(name: str) -> str:
    return normalize_name_key(name)


def weighted_name_vote_details(values: List[Tuple[str, float]]) -> Tuple[str, float, dict[str, object]]:
    normalized_entries = []
    total_weight = 0.0
    for index, (value, weight) in enumerate(values):
        if value in (None, ""):
            continue
        numeric_weight = max(0.0, float(weight))
        if numeric_weight <= 0:
            continue
        raw_value = normalize_name_for_vote(str(value))
        key_value = _normalize_name_key(str(value))
        if not key_value:
            continue
        normalized_entries.append(
            {
                "raw": raw_value,
                "key": key_value,
                "weight": numeric_weight,
                "allowed_ratio": allowed_name_char_ratio(raw_value),
                "index": index,
            }
        )
        total_weight += numeric_weight
    if not normalized_entries:
        return "", 0.0, {
            "allowed_ratio": 0.0,
            "unknown_chars": "",
            "flags": "low_name_confidence|low_name_stability",
        }

    clusters: list[dict[str, object]] = []
    for entry in normalized_entries:
        assigned_cluster = None
        best_similarity = 0.0
        for cluster in clusters:
            cluster_key = str(cluster["representative_key"])
            similarity = difflib.SequenceMatcher(None, entry["key"], cluster_key).ratio()
            if similarity >= 0.72 and similarity > best_similarity:
                best_similarity = similarity
                assigned_cluster = cluster
        if assigned_cluster is None:
            assigned_cluster = {
                "members": [],
                "weight": 0.0,
                "representative_raw": entry["raw"],
                "representative_key": entry["key"],
                "representative_weight": entry["weight"],
                "representative_allowed_ratio": entry["allowed_ratio"],
                "representative_index": entry["index"],
            }
            clusters.append(assigned_cluster)
        assigned_cluster["members"].append(entry)
        assigned_cluster["weight"] = float(assigned_cluster["weight"]) + entry["weight"]
        current_rep_weight = float(assigned_cluster["representative_weight"])
        current_rep_raw = str(assigned_cluster["representative_raw"])
        if (
            entry["weight"] > current_rep_weight
            or (
                entry["weight"] == current_rep_weight
                and (
                    float(entry["allowed_ratio"]) > float(assigned_cluster["representative_allowed_ratio"])
                    or (
                        float(entry["allowed_ratio"]) == float(assigned_cluster["representative_allowed_ratio"])
                        and visible_name_length(str(entry["raw"])) >= visible_name_length(str(assigned_cluster["representative_raw"]))
                    )
                )
                and entry["raw"] != current_rep_raw
            )
        ):
            assigned_cluster["representative_raw"] = entry["raw"]
            assigned_cluster["representative_key"] = entry["key"]
            assigned_cluster["representative_weight"] = entry["weight"]
            assigned_cluster["representative_allowed_ratio"] = entry["allowed_ratio"]
            assigned_cluster["representative_index"] = entry["index"]

    best_cluster = max(
        clusters,
        key=lambda cluster: (
            float(cluster["weight"]),
            sum(float(member["allowed_ratio"]) for member in cluster["members"]) / max(len(cluster["members"]), 1),
            float(cluster["representative_weight"]),
            float(cluster["representative_allowed_ratio"]),
            -int(cluster["representative_index"]),
        ),
    )
    best_name = str(best_cluster["representative_raw"])
    best_weight = float(best_cluster["weight"])
    best_ratio = best_weight / total_weight if total_weight > 0 else 0.0
    mean_confidence = best_weight / max(len(best_cluster["members"]), 1)
    flags = []
    unknown_chars = unknown_name_chars(best_name)
    if unknown_chars:
        flags.append("unknown_chars")
    if mean_confidence < 50.0:
        flags.append("low_name_confidence")
    if best_ratio < 0.6:
        flags.append("low_name_stability")
    return best_name, best_ratio, {
        "allowed_ratio": round(allowed_name_char_ratio(best_name) * 100, 1),
        "unknown_chars": unknown_chars,
        "flags": "|".join(flags),
    }


def weighted_name_vote(values: List[Tuple[str, float]]) -> Tuple[str, float]:
    best_name, best_ratio, _details = weighted_name_vote_details(values)
    return best_name, best_ratio


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
    return best_value, (best_weight / total_weight if total_weight > 0 else 0.0)


def weighted_vote_with_source(values: List[Tuple[object, float, str]]) -> Tuple[object, float, str]:
    score_by_value = defaultdict(float)
    source_scores_by_value = defaultdict(lambda: defaultdict(float))
    total_weight = 0.0
    for value, weight, source in values:
        if value in (None, ""):
            continue
        numeric_weight = max(0.0, float(weight))
        score_by_value[value] += numeric_weight
        source_scores_by_value[value][str(source or "7-segment")] += numeric_weight
        total_weight += numeric_weight
    if not score_by_value:
        return None, 0.0, ""
    best_value, best_weight = max(score_by_value.items(), key=lambda item: item[1])
    best_source = max(source_scores_by_value[best_value].items(), key=lambda item: item[1])[0]
    return best_value, (best_weight / total_weight if total_weight > 0 else 0.0), best_source


TOTAL_SCORE_CONSENSUS_WINDOW_SIZE = 3
RACE_SCORE_POINT_WINDOW_SIZE = 5


def select_consensus_window(observations: List[Dict[str, object]], mode: str, size: int = 3) -> List[Dict[str, object]]:
    """Pick a stable observation subset for a specific OCR signal."""
    if not observations:
        return []
    if len(observations) <= size or mode == "all":
        return observations
    if mode == "early":
        return observations[:size]
    if mode == "late":
        return observations[-size:]
    if mode == "center":
        start_index = max(0, (len(observations) - size) // 2)
        return observations[start_index:start_index + size]
    return observations


def _parse_observation_value(observation: Dict[str, object], key: str, row_index: int) -> int | None:
    values = observation.get(key, [])
    if row_index >= len(values):
        return None
    return parse_detected_int(values[row_index])


def find_points_animation_transition_index(observations: List[Dict[str, object]]) -> int | None:
    """Find the first frame where both top rows switch from old totals/race points to animated values."""
    if len(observations) < 2:
        return None

    for index in range(1, len(observations)):
        previous = observations[index - 1]
        current = observations[index]
        changed_checks = []
        for row_index in (0, 1):
            previous_race = _parse_observation_value(previous, "race_points", row_index)
            current_race = _parse_observation_value(current, "race_points", row_index)
            previous_total = _parse_observation_value(previous, "total_points", row_index)
            current_total = _parse_observation_value(current, "total_points", row_index)
            changed_checks.extend(
                [
                    previous_race is not None and current_race is not None and previous_race != current_race,
                    previous_total is not None and current_total is not None and previous_total != current_total,
                ]
            )
        if len(changed_checks) == 4 and all(changed_checks):
            return index
    return None


def build_consensus_rows(
    *,
    name_observations: List[Dict[str, object]],
    point_observations: List[Dict[str, object]],
    visible_rows: int,
    points_key: str,
    points_source_key: str,
    secondary_points_key: str | None = None,
    secondary_points_source_key: str | None = None,
    character_observations: List[Dict[str, object]] | None = None,
    keep_all_visible_rows: bool = False,
) -> List[Dict[str, object]]:
    """Collapse multiple nearby frames into one best-effort row list."""
    character_observations = character_observations or point_observations
    rows = []
    for row_index in range(max(visible_rows, 1)):
        name_votes = []
        point_votes = []
        secondary_point_votes = []
        character_votes = []
        character_method_votes = defaultdict(float)
        for observation in name_observations:
            name = normalize_name_for_vote(observation["names"][row_index]) if row_index < len(observation["names"]) else ""
            name_conf = observation["name_confidences"][row_index] if row_index < len(observation["name_confidences"]) else 0
            name_votes.append((name, max(1.0, float(name_conf))))
        for observation in point_observations:
            point_value = parse_detected_int(observation[points_key][row_index])
            point_source_values = observation.get(points_source_key, [])
            point_source = point_source_values[row_index] if row_index < len(point_source_values) else "7-segment"
            point_votes.append((point_value, 1.0, point_source))
            if secondary_points_key and secondary_points_source_key:
                secondary_point_value = parse_detected_int(observation[secondary_points_key][row_index])
                secondary_source_values = observation.get(secondary_points_source_key, [])
                secondary_point_source = secondary_source_values[row_index] if row_index < len(secondary_source_values) else "7-segment"
                secondary_point_votes.append((secondary_point_value, 1.0, secondary_point_source))
        for observation in character_observations:
            if row_index < len(observation.get("character_metrics", [])):
                character_match = observation["character_metrics"][row_index]
                if character_match.get("CharacterIndex") is not None:
                    character_weight = max(1.0, float(character_match.get("CharacterMatchConfidence", 0.0)))
                    character_key = (
                        int(character_match["CharacterIndex"]),
                        str(character_match.get("Character", "")),
                    )
                    character_votes.append(
                        (
                            character_key,
                            character_weight,
                        )
                    )
                    character_method_votes[(character_key, str(character_match.get("CharacterMatchMethod", "")))] += character_weight

        player_name, name_confidence, name_vote_details = weighted_name_vote_details(name_votes)
        detected_value, point_confidence, detected_value_source = weighted_vote_with_source(point_votes)
        secondary_detected_value, _secondary_point_confidence, secondary_detected_value_source = weighted_vote_with_source(secondary_point_votes)
        character_vote, character_vote_confidence = weighted_vote(character_votes)
        visible_name = str(player_name or "")
        if visible_name_length(visible_name) < 3 or distinct_visible_name_count(visible_name) < 3:
            if detected_value is None:
                if not keep_all_visible_rows:
                    continue
                if not player_name:
                    player_name = f"PlayerNameMissing_{row_index + 1}"

        character_index = None
        character_name = ""
        character_method = ""
        if character_vote is not None:
            character_index, character_name = character_vote
            matching_methods = {
                method: weight
                for (method_key, method), weight in character_method_votes.items()
                if method_key == character_vote
            }
            if matching_methods:
                character_method = max(matching_methods.items(), key=lambda item: item[1])[0]

        rows.append(
            {
                "RowIndex": row_index,
                "PlayerName": player_name or "",
                "NameConfidence": round(name_confidence * 100, 1),
                "NameAllowedCharRatio": name_vote_details["allowed_ratio"],
                "NameUnknownChars": name_vote_details["unknown_chars"],
                "NameValidationFlags": name_vote_details["flags"],
                "DetectedValue": detected_value,
                "DetectedValueSource": detected_value_source,
                "DetectedSecondaryValue": secondary_detected_value,
                "DetectedSecondaryValueSource": secondary_detected_value_source,
                "DigitConfidence": round(point_confidence * 100, 1),
                "Character": character_name,
                "CharacterIndex": character_index,
                "CharacterMatchConfidence": round(character_vote_confidence * 100, 1),
                "CharacterMatchMethod": character_method or "",
            }
        )
    return rows


def map_total_rows_to_race_rows(
    score_rows: List[Dict[str, object]],
    total_rows: List[Dict[str, object]],
    preprocess_name,
    weighted_similarity,
    total_row_metrics: List[Dict[str, object]] | None = None,
) -> List[Dict[str, object]]:
    """Attach total-score rows to race-score rows, preferring same-name matches over row order."""
    mapped_rows = []
    total_row_metrics = total_row_metrics or []
    if not score_rows:
        return mapped_rows
    if not total_rows:
        for score_row in score_rows:
            mapped_rows.append(
                {
                    "RacePosition": len(mapped_rows) + 1,
                    "PositionAfterRace": None,
                    "PlayerName": score_row["PlayerName"],
                    "Character": score_row.get("Character", ""),
                    "CharacterIndex": score_row.get("CharacterIndex"),
                    "CharacterMatchConfidence": score_row.get("CharacterMatchConfidence", 0.0),
                    "CharacterMatchMethod": score_row.get("CharacterMatchMethod", ""),
                    "DetectedRacePoints": score_row["DetectedValue"],
                    "DetectedRacePointsSource": score_row.get("DetectedValueSource", ""),
                    "DetectedOldTotalScore": score_row.get("DetectedSecondaryValue"),
                    "DetectedOldTotalScoreSource": score_row.get("DetectedSecondaryValueSource", ""),
                    "DetectedTotalScore": None,
                    "DetectedNewTotalScore": None,
                    "DetectedTotalScoreSource": "",
                    "DetectedNewTotalScoreSource": "",
                    "NameConfidence": score_row["NameConfidence"],
                    "NameAllowedCharRatio": score_row.get("NameAllowedCharRatio", 0.0),
                    "NameUnknownChars": score_row.get("NameUnknownChars", ""),
                    "NameValidationFlags": score_row.get("NameValidationFlags", ""),
                    "DigitConsensus": score_row["DigitConfidence"],
                    "TotalScoreMappingMethod": "missing_total_rows",
                    "TotalScoreMappingScore": None,
                    "TotalScoreMappingMargin": None,
                    "TotalScoreNameSimilarity": None,
                    **{f"PositionTemplate{template_index:02}_Coeff": None for template_index in range(1, 13)},
                }
            )
        return mapped_rows

    def _expected_total_similarity(score_row: Dict[str, object], total_row: Dict[str, object]) -> float:
        detected_old_total = parse_detected_int(score_row.get("DetectedSecondaryValue"))
        detected_race_points = parse_detected_int(score_row.get("DetectedValue"))
        detected_total = parse_detected_int(total_row.get("DetectedValue"))
        if detected_old_total is None or detected_race_points is None or detected_total is None:
            return 0.0
        expected_total = int(detected_old_total) + int(detected_race_points)
        gap = abs(int(detected_total) - expected_total)
        if gap == 0:
            return 1.0
        if gap == 1:
            return 0.75
        if gap == 2:
            return 0.45
        if gap <= 4:
            return 0.15
        return -0.35

    score_name_counts = Counter(
        preprocess_name(str(score_row["PlayerName"] or ""))
        for score_row in score_rows
        if preprocess_name(str(score_row["PlayerName"] or ""))
    )
    total_name_counts = Counter(
        preprocess_name(str(total_row["PlayerName"] or ""))
        for total_row in total_rows
        if preprocess_name(str(total_row["PlayerName"] or ""))
    )

    candidate_matches = []
    candidate_matches_by_score_index = defaultdict(list)
    for score_index, score_row in enumerate(score_rows):
        normalized_score_name = preprocess_name(str(score_row["PlayerName"] or ""))
        for total_index, total_row in enumerate(total_rows):
            normalized_total_name = preprocess_name(str(total_row["PlayerName"] or ""))
            if not normalized_score_name or not normalized_total_name:
                continue
            similarity = 1.0 if normalized_score_name == normalized_total_name else weighted_similarity(score_row["PlayerName"], total_row["PlayerName"])
            confidence_floor = min(float(score_row["NameConfidence"]), float(total_row["NameConfidence"])) / 100.0
            score_character = score_row.get("CharacterIndex")
            total_character = total_row.get("CharacterIndex")
            if score_character is not None and total_character is not None:
                character_score = 1.0 if int(score_character) == int(total_character) else -0.25
            elif score_character is None and total_character is None:
                character_score = 0.0
            else:
                character_score = 0.1
            duplicate_name = max(score_name_counts[normalized_score_name], total_name_counts[normalized_total_name]) > 1
            if duplicate_name:
                expected_total_score = _expected_total_similarity(score_row, total_row)
                effective_name_score = 0.25 if normalized_score_name == normalized_total_name else similarity * 0.25
                combined_score = (
                    (effective_name_score * 0.20)
                    + (confidence_floor * 0.10)
                    + (character_score * 0.25)
                    + (expected_total_score * 0.45)
                )
            else:
                combined_score = (similarity * 0.65) + (confidence_floor * 0.15) + (character_score * 0.20)
            if similarity >= 0.72 or normalized_score_name == normalized_total_name:
                candidate_matches.append((combined_score, similarity, duplicate_name, score_index, total_index))
                candidate_matches_by_score_index[score_index].append((combined_score, similarity, total_index))

    assigned_score_indices = set()
    assigned_total_indices = set()
    matched_totals_by_score_index = {}
    for _combined_score, similarity, duplicate_name, score_index, total_index in sorted(candidate_matches, reverse=True):
        if score_index in assigned_score_indices or total_index in assigned_total_indices:
            continue
        if duplicate_name:
            mapping_method = "duplicate_name_expected_total"
        else:
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

        mapping_score = None
        mapping_margin = None
        mapping_similarity = None
        scored_candidates = sorted(candidate_matches_by_score_index.get(score_index, []), reverse=True)
        if matched_total_index is not None and scored_candidates:
            matched_candidate = next((item for item in scored_candidates if int(item[2]) == int(matched_total_index)), None)
            if matched_candidate is not None:
                mapping_score = float(matched_candidate[0])
                mapping_similarity = float(matched_candidate[1])
                best_other_score = next((float(item[0]) for item in scored_candidates if int(item[2]) != int(matched_total_index)), None)
                mapping_margin = mapping_score - best_other_score if best_other_score is not None else mapping_score

        total_score = None
        total_digit_confidence = 0.0
        mapped_character = score_row.get("Character", "")
        mapped_character_index = score_row.get("CharacterIndex")
        mapped_character_confidence = score_row.get("CharacterMatchConfidence", 0.0)
        mapped_character_method = score_row.get("CharacterMatchMethod", "")
        total_row_position_coefficients = {f"PositionTemplate{template_index:02}_Coeff": None for template_index in range(1, 13)}
        if matched_total_index is not None:
            total_row = total_rows[matched_total_index]
            total_score = total_row["DetectedValue"]
            total_digit_confidence = float(total_row["DigitConfidence"])
            if total_row.get("CharacterIndex") is not None:
                mapped_character = total_row.get("Character", mapped_character)
                mapped_character_index = total_row.get("CharacterIndex")
                mapped_character_confidence = total_row.get("CharacterMatchConfidence", mapped_character_confidence)
                mapped_character_method = total_row.get("CharacterMatchMethod", mapped_character_method)
            total_row_index = int(total_row.get("RowIndex", -1))
            if 0 <= total_row_index < len(total_row_metrics):
                total_row_position_coefficients.update(
                    total_row_metrics[total_row_index].get("position_template_coefficients", {})
                )

        mapped_rows.append(
            {
                "RacePosition": len(mapped_rows) + 1,
                "PositionAfterRace": (matched_total_index + 1) if matched_total_index is not None else None,
                "PlayerName": score_row["PlayerName"],
                "Character": mapped_character,
                "CharacterIndex": mapped_character_index,
                "CharacterMatchConfidence": mapped_character_confidence,
                "CharacterMatchMethod": mapped_character_method,
                "DetectedRacePoints": score_row["DetectedValue"],
                "DetectedRacePointsSource": score_row.get("DetectedValueSource", ""),
                "DetectedOldTotalScore": score_row.get("DetectedSecondaryValue"),
                "DetectedOldTotalScoreSource": score_row.get("DetectedSecondaryValueSource", ""),
                "DetectedTotalScore": total_score,
                "DetectedNewTotalScore": total_score,
                "DetectedTotalScoreSource": total_row.get("DetectedValueSource", "") if matched_total_index is not None else "",
                "DetectedNewTotalScoreSource": total_row.get("DetectedValueSource", "") if matched_total_index is not None else "",
                "NameConfidence": score_row["NameConfidence"],
                "NameAllowedCharRatio": score_row.get("NameAllowedCharRatio", 0.0),
                "NameUnknownChars": score_row.get("NameUnknownChars", ""),
                "NameValidationFlags": score_row.get("NameValidationFlags", ""),
                "DigitConsensus": round((float(score_row["DigitConfidence"]) + total_digit_confidence) / 2.0, 1),
                "TotalScoreMappingMethod": mapping_method,
                "TotalScoreMappingScore": round(mapping_score, 4) if mapping_score is not None else None,
                "TotalScoreMappingMargin": round(mapping_margin, 4) if mapping_margin is not None else None,
                "TotalScoreNameSimilarity": round(mapping_similarity, 4) if mapping_similarity is not None else None,
                **total_row_position_coefficients,
            }
        )
    return mapped_rows


def reconcile_race_points_with_total_delta(
    rows: List[Dict[str, object]],
    score_observations: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Prefer the old->new total delta when it is at least as well-supported as the selected point vote."""
    if not rows or not score_observations:
        return rows

    reconciled_rows = []
    for row_index, row in enumerate(rows):
        updated_row = dict(row)
        detected_race_points = parse_detected_int(updated_row.get("DetectedRacePoints"))
        detected_old_total = parse_detected_int(updated_row.get("DetectedOldTotalScore"))
        detected_new_total = parse_detected_int(updated_row.get("DetectedTotalScore"))
        if detected_old_total is None or detected_new_total is None:
            reconciled_rows.append(updated_row)
            continue

        implied_race_points = int(detected_new_total) - int(detected_old_total)
        if implied_race_points < 1 or implied_race_points > 15:
            reconciled_rows.append(updated_row)
            continue
        if detected_race_points == implied_race_points:
            reconciled_rows.append(updated_row)
            continue

        vote_counter = Counter()
        for observation in score_observations:
            observation_race_points = observation.get("race_points", [])
            if row_index >= len(observation_race_points):
                continue
            voted_value = parse_detected_int(observation_race_points[row_index])
            if voted_value is None or not (1 <= int(voted_value) <= 15):
                continue
            vote_counter[int(voted_value)] += 1

        implied_support = int(vote_counter.get(int(implied_race_points), 0))
        current_support = int(vote_counter.get(int(detected_race_points), 0)) if detected_race_points is not None else 0
        if implied_support >= max(1, current_support):
            updated_row["DetectedRacePoints"] = int(implied_race_points)
            updated_row["DetectedRacePointsSource"] = "total_delta_consensus"
        reconciled_rows.append(updated_row)

    return reconciled_rows


def select_race_score_recovery(
    score_observations: List[Dict[str, object]],
    *,
    current_score_rows: int,
    current_confidence: float,
    total_score_rows: int,
) -> Dict[str, object]:
    """Try a slightly later RaceScore frame when the chosen frame is obscured by the black results bar."""
    if not score_observations:
        return {"used": False, "source": "", "count": current_score_rows}

    suspicious = (
        current_confidence < 60.0
        or (total_score_rows and current_score_rows < total_score_rows)
    )
    if not suspicious:
        return {"used": False, "source": "", "count": current_score_rows}

    center_index = len(score_observations) // 2
    candidate_indices = [index for index in range(center_index + 3, len(score_observations))]
    if not candidate_indices:
        candidate_indices = [index for index in range(center_index + 1, len(score_observations))]

    best_candidate = None
    for candidate_index in candidate_indices:
        candidate_observation = score_observations[candidate_index]
        candidate_count = int(candidate_observation.get("position_guided_visible_rows", 0))
        candidate_strength = float(candidate_observation.get("template_row_confidence", 0.0))
        if candidate_count <= 0:
            continue
        if best_candidate is None or (candidate_count, candidate_strength) > (best_candidate["count"], best_candidate["strength"]):
            best_candidate = {
                "used": candidate_count > current_score_rows or (
                    candidate_count == current_score_rows and candidate_strength > (current_confidence / 100.0)
                ),
                "source": f"late_frame_plus_{candidate_index - center_index}",
                "count": candidate_count,
                "strength": candidate_strength,
                "index": candidate_index,
            }

    if best_candidate and best_candidate["used"]:
        return best_candidate
    return {"used": False, "source": "", "count": current_score_rows}


def build_consensus_observation(frames: List[np.ndarray], total_frames: List[np.ndarray], extract_player_names_batched,
                                preprocess_name, weighted_similarity, annotate_path: str | None = None,
                                frame_numbers: List[int] | None = None,
                                total_annotate_path: str | None = None,
                                annotate_paths: List[str | None] | None = None,
                                total_annotate_paths: List[str | None] | None = None,
                                video_context: str | None = None, is_low_res: bool = False,
                                score_layout_id: str | None = None) -> Dict[str, object]:
    """Combine several neighbouring score frames into one stable observation."""
    if not frames:
        return {"rows": [], "visible_rows": 0, "row_count_confidence": 0.0, "name_confidence": 0.0, "digit_consensus": 0.0}

    score_observations = []
    total_observations = []
    for index, frame in enumerate(frames):
        current_annotate_path = None
        if annotate_paths and index < len(annotate_paths):
            current_annotate_path = annotate_paths[index]
        elif index == len(frames) // 2:
            current_annotate_path = annotate_path
        score_observations.append(
            extract_scoreboard_observation(
                frame,
                extract_player_names_batched,
                current_annotate_path,
                annotation_prefix="RS-",
                video_context=video_context,
                score_layout_id=score_layout_id,
                bundle_kind="2RaceScore",
                name_field_name="RacePlayerName",
                race_points_field_name="RacePoints",
                total_points_field_name="OldTotalScore",
            )
        )
    score_core_observations = score_observations[-APP_CONFIG.ocr_consensus_frames:] or score_observations
    total_frames_for_observation = select_consensus_window(
        total_frames,
        "center",
        size=TOTAL_SCORE_CONSENSUS_WINDOW_SIZE,
    )
    for frame in total_frames_for_observation:
        total_index = len(total_observations)
        current_total_annotate_path = None
        if total_annotate_paths and total_index < len(total_annotate_paths):
            current_total_annotate_path = total_annotate_paths[total_index]
        elif total_index == len(total_frames_for_observation) // 2:
            current_total_annotate_path = total_annotate_path
        total_observations.append(
            extract_scoreboard_observation(
                frame,
                extract_player_names_batched,
                current_total_annotate_path,
                annotation_prefix="TS-",
                video_context=video_context,
                score_layout_id=score_layout_id,
                bundle_kind="3TotalScore",
                name_field_name="TotalPlayerName",
                race_points_field_name="RacePointsOnTotalScore",
                total_points_field_name="NewTotalScore",
            )
        )
    if not total_observations:
        total_observations = score_observations

    frame_numbers = frame_numbers or list(range(len(frames)))
    race_point_anchor_frame = frame_numbers[-1] if frame_numbers else None
    score_name_observations = select_consensus_window(score_core_observations, "all")
    score_point_source_observations = score_observations
    if len(score_observations) > APP_CONFIG.ocr_consensus_frames:
        transition_index = find_points_animation_transition_index(score_observations)
        if transition_index is not None:
            score_point_source_observations = score_observations[max(0, transition_index - RACE_SCORE_POINT_WINDOW_SIZE):transition_index]
            if transition_index - 1 < len(frame_numbers):
                race_point_anchor_frame = frame_numbers[max(0, transition_index - 1)]
        else:
            score_radius = max(0, APP_CONFIG.ocr_consensus_frames // 2)
            anchor_index = max(0, len(score_observations) - score_radius - 1)
            point_window_start = max(0, anchor_index - (RACE_SCORE_POINT_WINDOW_SIZE - 1))
            score_point_source_observations = score_observations[point_window_start:anchor_index + 1]
            if anchor_index < len(frame_numbers):
                race_point_anchor_frame = frame_numbers[anchor_index]
    score_point_observations = select_consensus_window(score_point_source_observations, "early")
    score_character_observations = select_consensus_window(score_core_observations, "late")
    score_position_observations = select_consensus_window(score_core_observations, "late")
    score_count_observations = select_consensus_window(score_core_observations, "early")
    total_consensus_observations = total_observations
    visible_votes = Counter(observation["visible_rows"] for observation in score_count_observations if observation["visible_rows"] > 0)
    visible_rows = visible_votes.most_common(1)[0][0] if visible_votes else 0
    row_count_confidence = (visible_votes[visible_rows] / len(score_count_observations)) if visible_rows and score_count_observations else 0.0
    position_guided_visible_votes = Counter(
        observation["position_guided_visible_rows"] for observation in score_count_observations if observation["position_guided_visible_rows"] > 0
    )
    position_guided_visible_rows = position_guided_visible_votes.most_common(1)[0][0] if position_guided_visible_votes else 0
    position_guided_row_count_confidence = (
        position_guided_visible_votes[position_guided_visible_rows] / len(score_count_observations)
        if position_guided_visible_rows and score_count_observations else 0.0
    )
    total_visible_votes = Counter(
        observation["visible_rows"] for observation in total_consensus_observations if observation["visible_rows"] > 0
    )
    total_visible_rows = total_visible_votes.most_common(1)[0][0] if total_visible_votes else visible_rows
    total_position_guided_visible_votes = Counter(
        observation["position_guided_visible_rows"]
        for observation in total_consensus_observations
        if observation["position_guided_visible_rows"] > 0
    )
    total_position_guided_visible_rows = (
        total_position_guided_visible_votes.most_common(1)[0][0] if total_position_guided_visible_votes else position_guided_visible_rows
    )

    low_res_row12_character_support_votes = 0
    low_res_center_frame_row12_support = False
    ultra_low_res_center_frame_row12_support = False
    if (
        is_low_res
        and position_guided_visible_rows == 11
        and total_position_guided_visible_rows >= 12
    ):
        if frames:
            center_frame = frames[len(frames) // 2]
            low_res_center_frame_row12_support = _frame_supports_low_res_row12_character_fallback(
                center_frame,
                video_context=video_context,
                score_layout_id=score_layout_id,
            )
            ultra_low_res_center_frame_row12_support = _frame_supports_ultra_low_res_row12_blob_fallback(
                center_frame,
                score_layout_id=score_layout_id,
            )
        if score_observations:
            low_res_row12_character_support_votes = sum(
                1
                for observation in score_observations
                if _observation_supports_low_res_row12_character_fallback(observation)
            )
        if (
            low_res_center_frame_row12_support
            or low_res_row12_character_support_votes >= 1
            or ultra_low_res_center_frame_row12_support
        ):
            position_guided_visible_rows = 12
            position_guided_row_count_confidence = max(
                position_guided_row_count_confidence,
                max(
                    (low_res_row12_character_support_votes / len(score_observations)) if score_observations else 0.0,
                    1.0 if low_res_center_frame_row12_support else 0.0,
                    1.0 if ultra_low_res_center_frame_row12_support else 0.0,
                ),
            )

    recovery = select_race_score_recovery(
        score_position_observations,
        current_score_rows=position_guided_visible_rows,
        current_confidence=round(position_guided_row_count_confidence * 100, 1),
        total_score_rows=total_position_guided_visible_rows,
    )
    if recovery["used"]:
        position_guided_visible_rows = int(recovery["count"])
        position_guided_row_count_confidence = max(position_guided_row_count_confidence, 0.85)

    representative_score_observation = score_position_observations[len(score_position_observations) // 2] if score_position_observations else {}
    representative_total_observation = (
        total_consensus_observations[len(total_consensus_observations) // 2] if total_consensus_observations else {}
    )

    # Use the position-guided count as the official row count. The older OCR-only count
    # is still returned in debug data as a legacy reference.
    score_rows = build_consensus_rows(
        name_observations=score_name_observations,
        point_observations=score_point_observations,
        visible_rows=position_guided_visible_rows,
        points_key="race_points",
        points_source_key="race_point_sources",
        secondary_points_key="total_points",
        secondary_points_source_key="total_point_sources",
        character_observations=score_character_observations,
        keep_all_visible_rows=is_low_res,
    )
    total_rows = build_consensus_rows(
        name_observations=total_consensus_observations,
        point_observations=total_consensus_observations,
        visible_rows=total_position_guided_visible_rows,
        points_key="total_points",
        points_source_key="total_point_sources",
        character_observations=total_consensus_observations,
        keep_all_visible_rows=is_low_res,
    )
    rows = map_total_rows_to_race_rows(
        score_rows,
        total_rows,
        preprocess_name,
        weighted_similarity,
        representative_total_observation.get("row_metrics", []),
    )
    rows = reconcile_race_points_with_total_delta(rows, score_observations)
    name_confidences = [float(row["NameConfidence"]) / 100.0 for row in rows if row.get("NameConfidence") is not None]
    digit_confidences = [float(row["DigitConsensus"]) / 100.0 for row in rows if row.get("DigitConsensus") is not None]

    return {
        "rows": rows,
        "visible_rows": len(rows),
        "score_visible_rows": position_guided_visible_rows,
        "total_visible_rows": total_position_guided_visible_rows,
        "legacy_score_visible_rows": visible_rows,
        "legacy_total_visible_rows": total_visible_rows,
        "row_count_confidence": round(position_guided_row_count_confidence * 100, 1),
        "legacy_row_count_confidence": round(row_count_confidence * 100, 1),
        "score_count_votes": summarize_count_votes(score_count_observations, "position_guided_visible_rows"),
        "total_count_votes": summarize_count_votes(total_consensus_observations, "position_guided_visible_rows"),
        "legacy_score_count_votes": summarize_count_votes(score_count_observations, "visible_rows"),
        "legacy_total_count_votes": summarize_count_votes(total_consensus_observations, "visible_rows"),
        "score_row_metrics_summary": representative_score_observation.get("row_metrics_summary", ""),
        "total_row_metrics_summary": representative_total_observation.get("row_metrics_summary", ""),
        "representative_score_observation": representative_score_observation,
        "representative_total_observation": representative_total_observation,
        "race_point_anchor_frame": race_point_anchor_frame,
        "race_score_recovery_used": bool(recovery["used"]),
        "race_score_recovery_source": str(recovery.get("source", "")) or (
            (
                "ultra_low_res_row12_blob_fallback_center_frame"
                if ultra_low_res_center_frame_row12_support
                else
                "low_res_row12_character_fallback_center_frame"
                if low_res_center_frame_row12_support
                else f"low_res_row12_character_fallback_{low_res_row12_character_support_votes}of{len(score_observations)}"
            )
            if (
                ultra_low_res_center_frame_row12_support
                or low_res_center_frame_row12_support
                or low_res_row12_character_support_votes
            )
            and position_guided_visible_rows == 12 and not recovery["used"]
            else ""
        ),
        "race_score_recovery_count": int(recovery.get("count", position_guided_visible_rows)),
        "name_confidence": round((sum(name_confidences) / len(name_confidences)) * 100, 1) if name_confidences else 0.0,
        "digit_consensus": round((sum(digit_confidences) / len(digit_confidences)) * 100, 1) if digit_confidences else 0.0,
        "score_layout_id": get_score_layout(score_layout_id).layout_id,
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
    return {row["index"]: int(row["session_new_total"]) for row in prepared_rows}
