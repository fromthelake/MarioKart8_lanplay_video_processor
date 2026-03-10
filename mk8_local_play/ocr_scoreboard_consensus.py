import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .data_paths import resolve_asset_file
from .game_catalog import load_game_catalog


PLAYER_NAME_COORDS = [
    ((428, 52), (620, 96)), ((428, 104), (620, 148)),
    ((428, 156), (620, 200)), ((428, 208), (620, 252)),
    ((428, 260), (620, 304)), ((428, 312), (620, 356)),
    ((428, 364), (620, 408)), ((428, 416), (620, 460)),
    ((428, 468), (620, 512)), ((428, 520), (620, 564)),
    ((428, 572), (620, 617)), ((428, 624), (620, 669)),
]
BASE_POSITION_STRIP_ROI = ((315, 57), (367, 667))
POSITION_STRIP_OFFSET_X = -2
POSITION_STRIP_OFFSET_Y = -2
POSITION_STRIP_PADDING_X = 2
POSITION_STRIP_PADDING_Y = 2
POSITION_ROW_PADDING_X = 1
POSITION_ROW_PADDING_TOP = 1
POSITION_ROW_PADDING_BOTTOM = 4
POSITION_TEMPLATE_FILENAME = "Score_template_fix.png"
POSITION_TEMPLATE_WIDTH = 56
POSITION_TEMPLATE_HEIGHT = 36
POSITION_TEMPLATE_ROW_STARTS = [0, 50, 102, 154, 206, 258, 310, 362, 414, 466, 518, 570]
POSITION_FALSE_NEGATIVE_WEIGHT = 2.0
POSITION_PRESENT_COEFF_THRESHOLD = 0.60
CHARACTER_ROI_LEFT = 377
CHARACTER_TEMPLATE_SIZE = 48
CHARACTER_ROW_START = 49
CHARACTER_ROW_STEP = 52
CHARACTER_ROW_PADDING_TOP = 2
CHARACTER_ROW_PADDING_BOTTOM = 2


def position_strip_roi() -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Return the adjusted position-strip ROI after applying the current offset and padding."""
    (base_x1, base_y1), (base_x2, base_y2) = BASE_POSITION_STRIP_ROI
    shifted_x1 = base_x1 + POSITION_STRIP_OFFSET_X
    shifted_y1 = base_y1 + POSITION_STRIP_OFFSET_Y
    shifted_x2 = base_x2 + POSITION_STRIP_OFFSET_X
    shifted_y2 = base_y2 + POSITION_STRIP_OFFSET_Y
    return (
        (shifted_x1 - POSITION_STRIP_PADDING_X, shifted_y1 - POSITION_STRIP_PADDING_Y),
        (shifted_x2 + POSITION_STRIP_PADDING_X, shifted_y2 + POSITION_STRIP_PADDING_Y),
    )


def character_row_roi(row_index: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    row_top = CHARACTER_ROW_START + (row_index * CHARACTER_ROW_STEP)
    return (
        (CHARACTER_ROI_LEFT, row_top - CHARACTER_ROW_PADDING_TOP),
        (CHARACTER_ROI_LEFT + CHARACTER_TEMPLATE_SIZE, row_top + CHARACTER_TEMPLATE_SIZE + CHARACTER_ROW_PADDING_BOTTOM),
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
    Score_template_fix.png: 0, 50, 102, 154, ... with 52 px spacing after row 1.
    """
    template_rows = []
    image_height, image_width = template_binary.shape[:2]
    crop_width = min(POSITION_TEMPLATE_WIDTH, image_width)
    for start_y in POSITION_TEMPLATE_ROW_STARTS:
        end_y = min(image_height, start_y + POSITION_TEMPLATE_HEIGHT)
        row_crop = template_binary[start_y:end_y, :crop_width]
        template_rows.append(normalize_binary_foreground(row_crop))
    return template_rows


def combine_position_rows(row_images: List[np.ndarray]) -> np.ndarray:
    """Rebuild the full strip after per-row normalization using the template row windows."""
    if not row_images:
        return np.zeros((0, 0), dtype=np.uint8)
    strip_width = row_images[0].shape[1]
    strip_height = position_strip_roi()[1][1] - position_strip_roi()[0][1]
    combined = np.zeros((strip_height, strip_width), dtype=np.uint8)
    for row_image, (start_y, end_y) in zip(row_images, position_template_row_windows()):
        clipped_start = max(0, min(strip_height, start_y))
        clipped_end = max(clipped_start, min(strip_height, end_y))
        height = min(row_image.shape[0], clipped_end - clipped_start)
        combined[clipped_start:clipped_start + height, :] = row_image[:height, :]
    return combined


def extract_position_row_match_crops(processed_image: np.ndarray) -> List[np.ndarray]:
    """Extract per-row position ROIs based on the template row windows.

    The core template area is 56x36. We keep just 1 px padding around it so the
    matcher can adjust slightly inside a tight 58x38 ROI.
    """
    (x1, y1), (x2, y2) = position_strip_roi()
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


def build_position_signal_metrics(processed_image: np.ndarray) -> List[Dict[str, float]]:
    """Measure whether each row still shows a position number in the fixed left strip."""
    position_rows = extract_position_row_match_crops(processed_image)
    templates = load_position_row_templates()
    metrics = []
    for row_index, row_image in enumerate(position_rows):
        template_scores = []
        for template_index, template in enumerate(templates, start=1):
            coefficient = _template_match_score(row_image, template)
            overlap_metrics = _best_template_overlap_metrics(row_image, template)
            template_scores.append(
                {
                    "template_index": template_index,
                    "coefficient": float(coefficient),
                    "white_iou": float(overlap_metrics["white_iou"]),
                    "weighted_white_iou": float(overlap_metrics["weighted_white_iou"]),
                    "white_f1": float(overlap_metrics["white_f1"]),
                }
            )

        coeff_sorted = sorted(template_scores, key=lambda item: item["coefficient"], reverse=True)
        coeff_best = coeff_sorted[0] if coeff_sorted else {"template_index": 0, "coefficient": 0.0, "white_iou": 0.0}
        coeff_second = coeff_sorted[1] if len(coeff_sorted) > 1 else {"template_index": 0, "coefficient": 0.0, "white_iou": 0.0}
        iou_sorted = sorted(template_scores, key=lambda item: item["weighted_white_iou"], reverse=True)
        iou_best = iou_sorted[0] if iou_sorted else {"template_index": 0, "coefficient": 0.0, "white_iou": 0.0}

        shortlist = coeff_sorted[:3] if len(coeff_sorted) >= 3 else coeff_sorted
        hybrid_best = max(shortlist, key=lambda item: (item["weighted_white_iou"], item["coefficient"])) if shortlist else {"template_index": 0, "coefficient": 0.0, "white_iou": 0.0}
        expected_score = template_scores[row_index] if row_index < len(template_scores) else {"coefficient": 0.0, "white_iou": 0.0, "weighted_white_iou": 0.0}
        metrics.append(
            {
                "expected_position_score": round(float(expected_score["coefficient"]), 3),
                "expected_position_iou": round(float(expected_score["white_iou"]), 3),
                "expected_position_weighted_iou": round(float(expected_score["weighted_white_iou"]), 3),
                "best_position_score": round(float(coeff_best["coefficient"]), 3),
                "second_best_position_score": round(float(coeff_second["coefficient"]), 3),
                "position_margin": round(float(coeff_best["coefficient"]) - float(coeff_second["coefficient"]), 3),
                "any_position_score": round(float(coeff_best["coefficient"]), 3),
                "best_position_template": int(hybrid_best["template_index"]),
                "best_position_coeff_template": int(coeff_best["template_index"]),
                "best_position_iou_template": int(iou_best["template_index"]),
                "coeff_ranked_templates": [int(item["template_index"]) for item in coeff_sorted],
                "best_position_iou": round(float(iou_best["white_iou"]), 3),
                "best_position_weighted_iou": round(float(iou_best["weighted_white_iou"]), 3),
                "best_position_template_score": round(float(hybrid_best["coefficient"]), 3),
                "best_position_template_iou": round(float(hybrid_best["white_iou"]), 3),
                "best_position_template_weighted_iou": round(float(hybrid_best["weighted_white_iou"]), 3),
                "position_template_coefficients": {
                    f"PositionTemplate{item['template_index']:02}_Coeff": round(float(item["coefficient"]), 3)
                    for item in template_scores
                },
            }
        )
    return metrics


def build_normalized_position_strip(processed_image: np.ndarray) -> np.ndarray:
    """Return the position strip with the same binary polarity normalization used for matching."""
    (x1, y1), (x2, y2) = position_strip_roi()
    position_strip = processed_image[y1:y2, x1:x2]
    if len(position_strip.shape) == 3:
        position_strip = cv2.cvtColor(position_strip, cv2.COLOR_BGR2GRAY)
    _, position_strip = cv2.threshold(position_strip, 180, 255, cv2.THRESH_BINARY)
    return combine_position_rows(normalize_position_rows(position_strip))


def build_character_match_metrics(frame_image: np.ndarray) -> List[Dict[str, object]]:
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
    for row_index in range(12):
        (x1, y1), (x2, y2) = character_row_roi(row_index)
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

        best_match = None
        for template in templates:
            match_result = masked_character_match_score(
                row_roi,
                template["template_image"],
                template["template_alpha"],
            )
            candidate = {
                "Character": template["character_name"],
                "CharacterIndex": template["character_index"],
                "CharacterMatchConfidence": round(float(match_result["score"]) * 100.0, 1),
                "CharacterMatchMethod": "alpha_aware_color_template_local_search",
            }
            if best_match is None or candidate["CharacterMatchConfidence"] > best_match["CharacterMatchConfidence"]:
                best_match = candidate
        metrics.append(best_match or {
            "Character": "",
            "CharacterIndex": None,
            "CharacterMatchConfidence": 0.0,
            "CharacterMatchMethod": "no_match",
        })
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
    """Count visible rows using occupancy plus the position-number templates.

    This is now the preferred player-count method:
    - OCR/name/points evidence says whether a row looks occupied
    - position templates confirm that a real rank number is visible
    - the chosen rank sequence may stay equal or increase as the table goes down

    The position presence gate is intentionally hard now:
    - `Coeff >= 0.60` means the row is visually present
    - below that threshold the row is treated as empty

    This keeps noisy lower rows from surviving on weak OCR evidence alone.
    """
    visible_rows = 0
    last_confirmed_rank = None
    for metric in row_metrics:
        best_position_score = float(metric.get("best_position_score", 0.0))
        best_rank = int(metric.get("best_position_template", 0))
        coeff_ranked_templates = [int(value) for value in metric.get("coeff_ranked_templates", [])]

        row_supported = best_position_score >= POSITION_PRESENT_COEFF_THRESHOLD

        # Ranking rows should never decrease as the table goes downward. If the preferred
        # template drops below the previous row, try the next logical high-coefficient
        # candidate first instead of stopping immediately. This helps with near-neighbours
        # such as 3 vs 8 while still enforcing a non-decreasing rank sequence.
        if row_supported and last_confirmed_rank is not None and best_rank < last_confirmed_rank:
            fallback_rank = next((template_rank for template_rank in coeff_ranked_templates if template_rank >= last_confirmed_rank), 0)
            if fallback_rank > 0:
                best_rank = fallback_rank
            else:
                row_supported = False

        if not row_supported:
            break
        last_confirmed_rank = best_rank
        visible_rows = int(metric["row_number"])
    return visible_rows


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


def process_image(image_source) -> np.ndarray:
    """Rewrite the scoreboard into OCR-friendly blocks before digit and name reading."""
    coordinates = {
        "player_name": PLAYER_NAME_COORDS,
        "race_points": [
            ((825, 52), (861, 96)), ((825, 104), (861, 148)),
            ((825, 156), (861, 200)), ((825, 208), (861, 252)),
            ((825, 260), (861, 304)), ((825, 312), (861, 356)),
            ((825, 364), (861, 408)), ((825, 416), (861, 460)),
            ((825, 468), (861, 512)), ((825, 520), (861, 564)),
            ((825, 572), (861, 617)), ((825, 624), (861, 669)),
        ],
        "total_points": [
            ((910, 52), (973, 96)), ((910, 104), (973, 148)),
            ((910, 156), (973, 200)), ((910, 208), (973, 252)),
            ((910, 260), (973, 304)), ((910, 312), (973, 356)),
            ((910, 364), (973, 408)), ((910, 416), (973, 460)),
            ((910, 468), (973, 512)), ((910, 520), (973, 564)),
            ((910, 572), (973, 617)), ((910, 624), (973, 669)),
        ],
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


def is_white_box(image: Image.Image, top_left: Tuple[int, int], box_size: Tuple[int, int] = (3, 2)) -> bool:
    x, y = top_left
    width, height = box_size
    white_pixels = 0
    total_pixels = width * height
    for offset_x in range(width):
        for offset_y in range(height):
            r, g, b = image.getpixel((x + offset_x, y + offset_y))
            if r > 180 and g > 180 and b > 180:
                white_pixels += 1
    return white_pixels >= total_pixels / 2


def identify_digit(image: Image.Image, box_top_left: Tuple[int, int], red_pixels: Dict[str, Tuple[int, int]]) -> int:
    """Use a fixed pixel-signature because MK8 digits are visually stable after preprocessing."""
    white_pixels = {
        label: is_white_box(image, (box_top_left[0] + x, box_top_left[1] + y))
        for label, (x, y) in red_pixels.items()
    }
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
        (1, {"middle_middle", "middle_bottom"}),
    ]
    for digit, pattern in digit_patterns:
        if all(white_pixels.get(label, False) for label in pattern):
            return digit
    return -1


def detect_digits_in_image(image: Image.Image, start_coords: List[Tuple[int, int]], row_offset: int,
                           box_dims: Tuple[int, int], red_pixels: Dict[str, Tuple[int, int]],
                           num_rows: int, boxes_per_row: int) -> List[str]:
    coordinate_set = []
    draw = ImageDraw.Draw(image)
    for row_index in range(num_rows):
        y_offset = row_index * row_offset
        row_number = ""
        for box_index in range(boxes_per_row):
            start_x, start_y = start_coords[box_index]
            top_left = (start_x, start_y + y_offset)
            digit = identify_digit(image, top_left, red_pixels)
            if digit != -1:
                row_number += str(digit)
            for _label, (x, y) in red_pixels.items():
                rect_top_left = (top_left[0] + x, top_left[1] + y)
                rect_bottom_right = (rect_top_left[0] + 3, rect_top_left[1] + 2)
                draw.rectangle([rect_top_left, rect_bottom_right], outline="red", fill="red")
        coordinate_set.append(row_number)
    return coordinate_set


def scale_coords(coords, scale_factor):
    return [(x * scale_factor, y * scale_factor) for x, y in coords]


def scale_pixel_positions(pixels, scale_factor):
    return {label: (x * scale_factor, y * scale_factor) for label, (x, y) in pixels.items()}


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


def score_digit_layout(scale_factor: int = 5):
    start_coords_run1 = scale_coords([(830, 71), (843, 71)], scale_factor)
    red_pixels_run1 = scale_pixel_positions(
        {
            "top_middle": (7, 2), "left_middle": (2, 5), "middle_middle": (7, 5),
            "right_middle": (11, 5), "left_bottom": (2, 13), "middle_bottom": (7, 13),
            "right_bottom": (11, 13), "middle_bottom_edge": (7, 17), "center": (7, 9),
        },
        scale_factor,
    )
    start_coords_run2 = scale_coords([(916, 66), (933, 66), (950, 66)], scale_factor)
    red_pixels_run2 = scale_pixel_positions(
        {
            "top_middle": (8, 2), "left_middle": (2, 7), "middle_middle": (8, 7),
            "right_middle": (13, 7), "left_bottom": (2, 16), "middle_bottom": (8, 16),
            "right_bottom": (13, 16), "middle_bottom_edge": (8, 21), "center": (8, 11),
        },
        scale_factor,
    )
    return {
        "race_points": (start_coords_run1, 52 * scale_factor, (13 * scale_factor, 19 * scale_factor), red_pixels_run1, 12, 2),
        "total_points": (start_coords_run2, 52 * scale_factor, (16 * scale_factor, 24 * scale_factor), red_pixels_run2, 12, 3),
    }


def extract_scoreboard_observation(frame_image: np.ndarray, extract_player_names_batched, annotate_path: str | None = None) -> Dict[str, object]:
    """Read one score frame into names, race points, totals, and a visible-row estimate."""
    processed_img = process_image(frame_image)
    character_metrics = build_character_match_metrics(frame_image)
    (position_x1, position_y1), (position_x2, position_y2) = position_strip_roi()
    raw_position_strip = frame_image[position_y1:position_y2, position_x1:position_x2].copy()
    normalized_position_strip = build_normalized_position_strip(processed_img)
    processed_img[position_y1:position_y2, position_x1:position_x2] = cv2.cvtColor(normalized_position_strip, cv2.COLOR_GRAY2BGR)
    match_row_crops = extract_position_row_match_crops(processed_img)
    processed_img_pil = Image.fromarray(processed_img).convert("RGB")
    scale_factor = 5
    scaled_image = processed_img_pil.resize(
        (processed_img_pil.width * scale_factor, processed_img_pil.height * scale_factor),
        Image.NEAREST,
    )
    layout = score_digit_layout(scale_factor)
    race_points = detect_digits_in_image(scaled_image, *layout["race_points"])
    total_points = detect_digits_in_image(scaled_image, *layout["total_points"])

    scaled_image_resized = scaled_image.resize((processed_img_pil.width, processed_img_pil.height), Image.NEAREST)
    annotated_image = cv2.cvtColor(np.array(scaled_image_resized), cv2.COLOR_RGB2BGR)
    if annotate_path:
        scaled_image_resized.save(annotate_path)

    names, confidence_scores = extract_player_names_batched(annotated_image, PLAYER_NAME_COORDS)
    row_metrics = build_row_presence_metrics(names, confidence_scores, race_points, total_points)
    position_metrics = build_position_signal_metrics(processed_img)
    for row_metric, position_metric in zip(row_metrics, position_metrics):
        row_metric.update(position_metric)
    processed_position_strip = processed_img[position_y1:position_y2, position_x1:position_x2].copy()
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
        "total_points": total_points,
        "character_metrics": character_metrics,
        "raw_position_strip": raw_position_strip,
        "processed_position_strip": processed_position_strip,
        "position_match_row_crops": match_row_crops,
        "row_metrics": row_metrics,
        "row_metrics_summary": summarize_row_metrics(row_metrics),
        "visible_rows": visible_rows,
        "position_guided_visible_rows": position_guided_visible_rows,
        "template_row_confidence": template_row_confidence,
    }


def normalize_name_for_vote(name: str) -> str:
    text = "" if name is None else str(name)
    return re.sub(r"\s+", " ", text.strip())


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


def build_consensus_rows(observations: List[Dict[str, object]], visible_rows: int, points_key: str) -> List[Dict[str, object]]:
    """Collapse multiple nearby frames into one best-effort row list."""
    rows = []
    for row_index in range(max(visible_rows, 1)):
        name_votes = []
        point_votes = []
        character_votes = []
        for observation in observations:
            name = normalize_name_for_vote(observation["names"][row_index]) if row_index < len(observation["names"]) else ""
            name_conf = observation["name_confidences"][row_index] if row_index < len(observation["name_confidences"]) else 0
            name_votes.append((name, max(1.0, float(name_conf))))
            point_votes.append((parse_detected_int(observation[points_key][row_index]), 1.0))
            if row_index < len(observation.get("character_metrics", [])):
                character_match = observation["character_metrics"][row_index]
                if character_match.get("CharacterIndex") is not None:
                    character_votes.append(
                        (
                            (
                                int(character_match["CharacterIndex"]),
                                str(character_match.get("Character", "")),
                                str(character_match.get("CharacterMatchMethod", "")),
                            ),
                            max(1.0, float(character_match.get("CharacterMatchConfidence", 0.0))),
                        )
                    )

        player_name, name_confidence = weighted_vote(name_votes)
        detected_value, point_confidence = weighted_vote(point_votes)
        character_vote, character_vote_confidence = weighted_vote(character_votes)
        stripped_name = re.sub(r"[^a-zA-Z0-9]", "", str(player_name or ""))
        if len(stripped_name) < 3 or len(set(stripped_name)) < 3:
            if detected_value is None:
                continue

        character_index = None
        character_name = ""
        character_method = ""
        if character_vote is not None:
            character_index, character_name, character_method = character_vote

        rows.append(
            {
                "RowIndex": row_index,
                "PlayerName": player_name or "",
                "NameConfidence": round(name_confidence * 100, 1),
                "DetectedValue": detected_value,
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
                    "DetectedTotalScore": None,
                    "NameConfidence": score_row["NameConfidence"],
                    "DigitConsensus": score_row["DigitConfidence"],
                    "TotalScoreMappingMethod": "missing_total_rows",
                    **{f"PositionTemplate{template_index:02}_Coeff": None for template_index in range(1, 13)},
                }
            )
        return mapped_rows

    candidate_matches = []
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
            combined_score = (similarity * 0.65) + (confidence_floor * 0.15) + (character_score * 0.20)
            if similarity >= 0.72 or normalized_score_name == normalized_total_name:
                candidate_matches.append((combined_score, similarity, score_index, total_index))

    assigned_score_indices = set()
    assigned_total_indices = set()
    matched_totals_by_score_index = {}
    for _combined_score, similarity, score_index, total_index in sorted(candidate_matches, reverse=True):
        if score_index in assigned_score_indices or total_index in assigned_total_indices:
            continue
        matched_totals_by_score_index[score_index] = (total_index, "name_exact" if similarity >= 0.999 else "name_fuzzy")
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
                "DetectedTotalScore": total_score,
                "NameConfidence": score_row["NameConfidence"],
                "DigitConsensus": round((float(score_row["DigitConfidence"]) + total_digit_confidence) / 2.0, 1),
                "TotalScoreMappingMethod": mapping_method,
                **total_row_position_coefficients,
            }
        )
    return mapped_rows


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
                                preprocess_name, weighted_similarity, annotate_path: str | None = None) -> Dict[str, object]:
    """Combine several neighbouring score frames into one stable observation."""
    if not frames:
        return {"rows": [], "visible_rows": 0, "row_count_confidence": 0.0, "name_confidence": 0.0, "digit_consensus": 0.0}

    score_observations = []
    total_observations = []
    for index, frame in enumerate(frames):
        score_observations.append(
            extract_scoreboard_observation(frame, extract_player_names_batched, annotate_path if index == len(frames) // 2 else None)
        )
    for frame in total_frames:
        total_observations.append(extract_scoreboard_observation(frame, extract_player_names_batched))
    if not total_observations:
        total_observations = score_observations

    visible_votes = Counter(observation["visible_rows"] for observation in score_observations if observation["visible_rows"] > 0)
    visible_rows = visible_votes.most_common(1)[0][0] if visible_votes else 0
    row_count_confidence = (visible_votes[visible_rows] / len(score_observations)) if visible_rows and score_observations else 0.0
    position_guided_visible_votes = Counter(
        observation["position_guided_visible_rows"] for observation in score_observations if observation["position_guided_visible_rows"] > 0
    )
    position_guided_visible_rows = position_guided_visible_votes.most_common(1)[0][0] if position_guided_visible_votes else 0
    position_guided_row_count_confidence = (
        position_guided_visible_votes[position_guided_visible_rows] / len(score_observations)
        if position_guided_visible_rows and score_observations else 0.0
    )
    total_visible_votes = Counter(observation["visible_rows"] for observation in total_observations if observation["visible_rows"] > 0)
    total_visible_rows = total_visible_votes.most_common(1)[0][0] if total_visible_votes else visible_rows
    total_position_guided_visible_votes = Counter(
        observation["position_guided_visible_rows"] for observation in total_observations if observation["position_guided_visible_rows"] > 0
    )
    total_position_guided_visible_rows = (
        total_position_guided_visible_votes.most_common(1)[0][0] if total_position_guided_visible_votes else position_guided_visible_rows
    )

    recovery = select_race_score_recovery(
        score_observations,
        current_score_rows=position_guided_visible_rows,
        current_confidence=round(position_guided_row_count_confidence * 100, 1),
        total_score_rows=total_position_guided_visible_rows,
    )
    if recovery["used"]:
        position_guided_visible_rows = int(recovery["count"])
        recovery_start_index = int(recovery["index"])
        recovery_observations = score_observations[recovery_start_index:]
        position_guided_row_count_confidence = max(position_guided_row_count_confidence, 0.85)
    else:
        recovery_observations = score_observations

    representative_score_observation = score_observations[len(score_observations) // 2] if score_observations else {}
    representative_total_observation = total_observations[len(total_observations) // 2] if total_observations else {}

    # Use the position-guided count as the official row count. The older OCR-only count
    # is still returned in debug data as a legacy reference.
    score_rows = build_consensus_rows(recovery_observations, position_guided_visible_rows, "race_points")
    total_rows = build_consensus_rows(total_observations, total_position_guided_visible_rows, "total_points")
    rows = map_total_rows_to_race_rows(
        score_rows,
        total_rows,
        preprocess_name,
        weighted_similarity,
        representative_total_observation.get("row_metrics", []),
    )
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
        "score_count_votes": summarize_count_votes(score_observations, "position_guided_visible_rows"),
        "total_count_votes": summarize_count_votes(total_observations, "position_guided_visible_rows"),
        "legacy_score_count_votes": summarize_count_votes(score_observations, "visible_rows"),
        "legacy_total_count_votes": summarize_count_votes(total_observations, "visible_rows"),
        "score_row_metrics_summary": representative_score_observation.get("row_metrics_summary", ""),
        "total_row_metrics_summary": representative_total_observation.get("row_metrics_summary", ""),
        "representative_score_observation": representative_score_observation,
        "representative_total_observation": representative_total_observation,
        "race_score_recovery_used": bool(recovery["used"]),
        "race_score_recovery_source": str(recovery.get("source", "")),
        "race_score_recovery_count": int(recovery.get("count", position_guided_visible_rows)),
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
    return {row["index"]: int(row["session_new_total"]) for row in prepared_rows}
