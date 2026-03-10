import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


PLAYER_NAME_COORDS = [
    ((428, 52), (620, 96)), ((428, 104), (620, 148)),
    ((428, 156), (620, 200)), ((428, 208), (620, 252)),
    ((428, 260), (620, 304)), ((428, 312), (620, 356)),
    ((428, 364), (620, 408)), ((428, 416), (620, 460)),
    ((428, 468), (620, 512)), ((428, 520), (620, 564)),
    ((428, 572), (620, 617)), ((428, 624), (620, 669)),
]


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
    valid_rows = []
    for index, player_name in enumerate(names):
        stripped_name = re.sub(r"[^a-zA-Z0-9]", "", player_name)
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
    total_visible_votes = Counter(observation["visible_rows"] for observation in total_observations if observation["visible_rows"] > 0)
    total_visible_rows = total_visible_votes.most_common(1)[0][0] if total_visible_votes else visible_rows

    score_rows = build_consensus_rows(score_observations, visible_rows, "race_points")
    total_rows = build_consensus_rows(total_observations, total_visible_rows, "total_points")
    rows = map_total_rows_to_race_rows(score_rows, total_rows, preprocess_name, weighted_similarity)
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
    return {row["index"]: int(row["session_new_total"]) for row in prepared_rows}
