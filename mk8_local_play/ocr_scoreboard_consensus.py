import json
import os
import re
import hashlib
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .data_paths import resolve_asset_file


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


def position_debug_enabled() -> bool:
    return str(os.environ.get("MK8_WRITE_DEBUG_POSITION_ROIS", "")).strip().lower() in {"1", "true", "yes", "on"}


def position_debug_rows() -> List[int]:
    raw_value = str(os.environ.get("MK8_DEBUG_POSITION_ROWS", "10,11,12"))
    rows = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            row_number = int(part)
        except ValueError:
            continue
        if 1 <= row_number <= 12:
            rows.append(row_number)
    return rows or [10, 11, 12]


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


def export_position_row_templates_once(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_file = output_dir / "_templates_written.txt"
    template_path = resolve_asset_file("templates", POSITION_TEMPLATE_FILENAME)
    template_hash = hashlib.sha256(template_path.read_bytes()).hexdigest()
    marker_text = (
        f"{POSITION_TEMPLATE_FILENAME}\n"
        f"{template_hash}\n"
        f"width={POSITION_TEMPLATE_WIDTH}\n"
        f"height={POSITION_TEMPLATE_HEIGHT}\n"
        f"starts={','.join(str(value) for value in POSITION_TEMPLATE_ROW_STARTS)}\n"
    )
    if marker_file.exists() and marker_file.read_text(encoding="utf-8") == marker_text:
        return
    for row_index, template in enumerate(load_position_row_templates(), start=1):
        cv2.imwrite(str(output_dir / f"template_row_{row_index:02}.png"), template)
    marker_file.write_text(marker_text, encoding="utf-8")


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
                "best_position_iou": round(float(iou_best["white_iou"]), 3),
                "best_position_weighted_iou": round(float(iou_best["weighted_white_iou"]), 3),
                "best_position_template_score": round(float(hybrid_best["coefficient"]), 3),
                "best_position_template_iou": round(float(hybrid_best["white_iou"]), 3),
                "best_position_template_weighted_iou": round(float(hybrid_best["weighted_white_iou"]), 3),
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


def write_position_roi_debug_bundle(output_dir: Path, base_name: str, raw_strip: np.ndarray, processed_strip: np.ndarray,
                                    row_metrics: List[Dict[str, object]], match_row_crops: List[np.ndarray] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_position_row_templates_once(output_dir / "templates")

    cv2.imwrite(str(output_dir / f"{base_name}_position_strip_raw.png"), raw_strip)
    cv2.imwrite(str(output_dir / f"{base_name}_position_strip_processed.png"), processed_strip)

    row_windows = position_template_row_windows()
    for row_number in position_debug_rows():
        row_index = row_number - 1
        start_y, end_y = row_windows[row_index]
        raw_crop = raw_strip[start_y:end_y, :]
        processed_crop = processed_strip[start_y:end_y, :]
        cv2.imwrite(str(output_dir / f"{base_name}_row_{row_number:02}_raw.png"), raw_crop)
        cv2.imwrite(str(output_dir / f"{base_name}_row_{row_number:02}_processed.png"), processed_crop)
        if match_row_crops and row_index < len(match_row_crops):
            cv2.imwrite(
                str(output_dir / f"{base_name}_row_{row_number:02}_match_processed.png"),
                match_row_crops[row_index],
            )

        metadata = {
            "base_name": base_name,
            "roi": {
                "x1": int(position_strip_roi()[0][0]),
                "y1": int(position_strip_roi()[0][1]),
                "x2": int(position_strip_roi()[1][0]),
                "y2": int(position_strip_roi()[1][1]),
                "offset_x": int(POSITION_STRIP_OFFSET_X),
                "offset_y": int(POSITION_STRIP_OFFSET_Y),
                "padding_x": int(POSITION_STRIP_PADDING_X),
                "padding_y": int(POSITION_STRIP_PADDING_Y),
                "row_padding_x": int(POSITION_ROW_PADDING_X),
                "row_padding_top": int(POSITION_ROW_PADDING_TOP),
                "row_padding_bottom": int(POSITION_ROW_PADDING_BOTTOM),
            },
            "row_number": row_number,
            "row_y_start": start_y,
            "row_y_end": end_y,
            "metrics": row_metrics[row_index],
        }
        (output_dir / f"{base_name}_row_{row_number:02}_meta.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )


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

        row_supported = best_position_score >= POSITION_PRESENT_COEFF_THRESHOLD

        # Ranking rows should never decrease as the table goes downward. Ties remain valid.
        if row_supported and last_confirmed_rank is not None and best_rank < last_confirmed_rank:
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
    stripped = re.sub(r"[^0-9]", "", str(value))
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
        for observation in observations:
            name = normalize_name_for_vote(observation["names"][row_index]) if row_index < len(observation["names"]) else ""
            name_conf = observation["name_confidences"][row_index] if row_index < len(observation["name_confidences"]) else 0
            name_votes.append((name, max(1.0, float(name_conf))))
            point_votes.append((parse_detected_int(observation[points_key][row_index]), 1.0))

        player_name, name_confidence = weighted_vote(name_votes)
        detected_value, point_confidence = weighted_vote(point_votes)
        stripped_name = re.sub(r"[^a-zA-Z0-9]", "", str(player_name or ""))
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


def map_total_rows_to_race_rows(score_rows: List[Dict[str, object]], total_rows: List[Dict[str, object]], preprocess_name, weighted_similarity) -> List[Dict[str, object]]:
    """Attach total-score rows to race-score rows, preferring same-name matches over row order."""
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
        normalized_score_name = preprocess_name(str(score_row["PlayerName"] or ""))
        for total_index, total_row in enumerate(total_rows):
            normalized_total_name = preprocess_name(str(total_row["PlayerName"] or ""))
            if not normalized_score_name or not normalized_total_name:
                continue
            similarity = 1.0 if normalized_score_name == normalized_total_name else weighted_similarity(score_row["PlayerName"], total_row["PlayerName"])
            confidence_floor = min(float(score_row["NameConfidence"]), float(total_row["NameConfidence"])) / 100.0
            combined_score = (similarity * 0.8) + (confidence_floor * 0.2)
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

    # Use the position-guided count as the official row count. The older OCR-only count
    # is still returned in debug data as a legacy reference.
    score_rows = build_consensus_rows(score_observations, position_guided_visible_rows, "race_points")
    total_rows = build_consensus_rows(total_observations, total_position_guided_visible_rows, "total_points")
    rows = map_total_rows_to_race_rows(score_rows, total_rows, preprocess_name, weighted_similarity)
    name_confidences = [float(row["NameConfidence"]) / 100.0 for row in rows if row.get("NameConfidence") is not None]
    digit_confidences = [float(row["DigitConsensus"]) / 100.0 for row in rows if row.get("DigitConsensus") is not None]
    representative_score_observation = score_observations[len(score_observations) // 2] if score_observations else {}
    representative_total_observation = total_observations[len(total_observations) // 2] if total_observations else {}

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
