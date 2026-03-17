from __future__ import annotations

import csv
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from .app_runtime import load_app_config
from .ocr_name_matching import choose_canonical_name, normalize_name_for_vote
from .ocr_scoreboard_consensus import (
    character_row_roi,
    load_character_templates,
    ultra_low_res_combined_row_roi,
)
from .score_layouts import get_score_layout, score_layout_id_from_filename

LOW_RES_PLACEHOLDER_PREFIX = "PlayerNameMissing_"
LOW_RES_NAME_ROW_HEIGHT = 45
LOW_RES_NAME_WEIGHT = 0.75
LOW_RES_CHARACTER_WEIGHT = 0.25
LOW_RES_UNKNOWN_CHARACTER_SCORE = 0.5
LOW_RES_MISMATCH_CHARACTER_SCORE = 0.0
LOW_RES_MATCH_CHARACTER_SCORE = 1.0
APP_CONFIG = load_app_config()
LOW_RES_CHARACTER_ROI_PAD_X = APP_CONFIG.low_res_character_roi_pad_x
LOW_RES_CHARACTER_ROI_PAD_Y = APP_CONFIG.low_res_character_roi_pad_y
LOW_RES_CHARACTER_TEMPLATE_WIDTH = APP_CONFIG.low_res_character_template_width
LOW_RES_CHARACTER_TEMPLATE_HEIGHT = APP_CONFIG.low_res_character_template_height
LOW_RES_CHARACTER_OFFSET_X = APP_CONFIG.low_res_character_offset_x
LOW_RES_CHARACTER_OFFSET_Y = APP_CONFIG.low_res_character_offset_y
LOW_RES_CHARACTER_ROI_LEFT_SHIFT = LOW_RES_CHARACTER_OFFSET_X - LOW_RES_CHARACTER_ROI_PAD_X
LOW_RES_CHARACTER_ROI_TOP_SHIFT = LOW_RES_CHARACTER_OFFSET_Y - LOW_RES_CHARACTER_ROI_PAD_Y
ULTRA_LOW_RES_BLOB_MATCH_MIN_SCORE = APP_CONFIG.ultra_low_res_blob_match_min_score
ULTRA_LOW_RES_BLOB_MATCH_MIN_MARGIN = APP_CONFIG.ultra_low_res_blob_match_min_margin


def is_low_res_height(source_height: int | None, max_source_height: int) -> bool:
    if source_height is None:
        return False
    return int(source_height) <= int(max_source_height)


def race_score_image_path(frames_folder: str | Path, race_class: str, race_id: int) -> Path:
    frames_root = Path(frames_folder)
    exact_path = frames_root / f"{race_class}+Race_{race_id:03}+2RaceScore.png"
    if exact_path.exists():
        return exact_path
    candidates = sorted(frames_root.glob(f"{race_class}+Race_{race_id:03}+2RaceScore+*.png"))
    return candidates[0] if candidates else exact_path


def _placeholder_name(index: int) -> str:
    return f"{LOW_RES_PLACEHOLDER_PREFIX}{index}"


def _extract_name_roi(frame: np.ndarray, position: int, score_layout_id: str | None = None) -> np.ndarray:
    player_name_coords = get_score_layout(score_layout_id).player_name_coords
    (x1, y1), (x2, _y2) = player_name_coords[position - 1]
    return frame[y1:y1 + LOW_RES_NAME_ROW_HEIGHT, x1:x2].copy()


def _extract_low_res_character_roi(frame: np.ndarray, position: int, score_layout_id: str | None = None) -> np.ndarray:
    (x1, y1), (_x2, _y2) = character_row_roi(position - 1, score_layout_id=score_layout_id)
    crop_x1 = max(0, x1 + LOW_RES_CHARACTER_ROI_LEFT_SHIFT)
    crop_y1 = max(0, y1 + LOW_RES_CHARACTER_ROI_TOP_SHIFT)
    crop_x2 = min(frame.shape[1], crop_x1 + LOW_RES_CHARACTER_TEMPLATE_WIDTH)
    crop_y2 = min(frame.shape[0], crop_y1 + LOW_RES_CHARACTER_TEMPLATE_HEIGHT)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    if crop.shape[1] != LOW_RES_CHARACTER_TEMPLATE_WIDTH or crop.shape[0] != LOW_RES_CHARACTER_TEMPLATE_HEIGHT:
        crop = cv2.resize(
            crop,
            (LOW_RES_CHARACTER_TEMPLATE_WIDTH, LOW_RES_CHARACTER_TEMPLATE_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
    return crop


def _extract_ultra_low_res_combined_roi(frame: np.ndarray, position: int, score_layout_id: str | None = None) -> np.ndarray:
    (x1, y1), (x2, y2) = ultra_low_res_combined_row_roi(position - 1, score_layout_id=score_layout_id)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return crop
    target_width = max(1, x2 - x1)
    target_height = max(1, y2 - y1)
    if crop.shape[1] != target_width or crop.shape[0] != target_height:
        crop = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return crop


def _compare_ultra_low_res_blob(reference_roi: np.ndarray, query_roi: np.ndarray) -> float:
    if reference_roi.size == 0 or query_roi.size == 0:
        return 0.0
    if reference_roi.shape[:2] != query_roi.shape[:2]:
        query_roi = cv2.resize(query_roi, (reference_roi.shape[1], reference_roi.shape[0]), interpolation=cv2.INTER_LINEAR)
    reference_gray = cv2.cvtColor(reference_roi, cv2.COLOR_BGR2GRAY)
    query_gray = cv2.cvtColor(query_roi, cv2.COLOR_BGR2GRAY)
    reference_vec = reference_gray.astype(np.float32).reshape(-1)
    query_vec = query_gray.astype(np.float32).reshape(-1)
    if reference_vec.std() > 1e-6 and query_vec.std() > 1e-6:
        corr = float(np.corrcoef(reference_vec, query_vec)[0, 1])
    else:
        corr = 0.0
    reference_edges = cv2.Canny(reference_gray, 80, 160)
    query_edges = cv2.Canny(query_gray, 80, 160)
    edge_union = np.logical_or(reference_edges > 0, query_edges > 0).sum()
    edge_intersection = np.logical_and(reference_edges > 0, query_edges > 0).sum()
    edge_iou = float(edge_intersection / edge_union) if edge_union else 0.0
    mad = float(np.mean(np.abs(reference_roi.astype(np.float32) - query_roi.astype(np.float32))))
    mad_score = max(0.0, 1.0 - (mad / 255.0))
    return float((0.45 * ((corr + 1.0) / 2.0)) + (0.30 * edge_iou) + (0.25 * mad_score))


def _restore_missing_rows_via_ultra_low_res_blob_fallback(group: pd.DataFrame, frames_folder: str | Path, race_class: str) -> pd.DataFrame:
    group = group.copy()
    race_ids = sorted(int(race_id) for race_id in group['RaceIDNumber'].unique())
    if not race_ids:
        return group
    expected_players = int(group['TotalScorePlayerCount'].fillna(0).max() or 0)
    if expected_players < 12:
        return group
    visible_rows_by_race = {
        race_id: int(group.loc[group['RaceIDNumber'] == race_id].shape[0])
        for race_id in race_ids
    }
    seed_race_id = min(race_ids, key=lambda race_id: (-visible_rows_by_race[race_id], race_id))
    if visible_rows_by_race.get(seed_race_id, 0) < expected_players:
        return group
    seed_frame_path = race_score_image_path(frames_folder, race_class, seed_race_id)
    seed_frame = cv2.imread(str(seed_frame_path), cv2.IMREAD_COLOR)
    if seed_frame is None:
        return group
    seed_layout_id = score_layout_id_from_filename(seed_frame_path)
    reference_blobs = [
        _extract_ultra_low_res_combined_roi(seed_frame, position, score_layout_id=seed_layout_id)
        for position in range(1, expected_players + 1)
    ]

    synthetic_rows = []
    for race_id in race_ids:
        race_mask = group['RaceIDNumber'] == race_id
        race_rows = group.loc[race_mask].sort_values('RacePosition', kind='stable')
        visible_rows = int(race_rows.shape[0])
        if visible_rows != expected_players - 1:
            continue
        race_score_players = int(race_rows['RaceScorePlayerCount'].iloc[0] or 0)
        total_score_players = int(race_rows['TotalScorePlayerCount'].iloc[0] or 0)
        if race_score_players >= expected_players or total_score_players < expected_players:
            continue
        frame_path = race_score_image_path(frames_folder, race_class, race_id)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        query_blob = _extract_ultra_low_res_combined_roi(
            frame,
            expected_players,
            score_layout_id=score_layout_id_from_filename(frame_path),
        )
        scores = [_compare_ultra_low_res_blob(reference_blob, query_blob) for reference_blob in reference_blobs]
        if not scores:
            continue
        ranked_scores = sorted(scores, reverse=True)
        best_score = float(ranked_scores[0])
        second_score = float(ranked_scores[1]) if len(ranked_scores) > 1 else 0.0
        if best_score < ULTRA_LOW_RES_BLOB_MATCH_MIN_SCORE or (best_score - second_score) < ULTRA_LOW_RES_BLOB_MATCH_MIN_MARGIN:
            continue

        base_row = race_rows.iloc[-1].copy()
        base_row['RacePosition'] = expected_players
        base_row['PlayerName'] = f'PlayerNameMissing_{expected_players}'
        base_row['Character'] = ''
        base_row['CharacterIndex'] = pd.NA
        base_row['CharacterMatchConfidence'] = 0.0
        base_row['CharacterMatchMethod'] = 'ultra_low_res_blob_row12_restore'
        base_row['RacePoints'] = low_res_race_points(expected_players, expected_players)
        base_row['DetectedRacePoints'] = pd.NA
        base_row['DetectedRacePointsSource'] = ''
        base_row['DetectedOldTotalScore'] = pd.NA
        base_row['DetectedOldTotalScoreSource'] = ''
        base_row['DetectedTotalScore'] = pd.NA
        base_row['DetectedTotalScoreSource'] = ''
        base_row['DetectedNewTotalScore'] = pd.NA
        base_row['DetectedNewTotalScoreSource'] = ''
        base_row['PositionAfterRace'] = pd.NA
        base_row['NameConfidence'] = 0.0
        base_row['DigitConsensus'] = 0.0
        base_row['RaceScorePlayerCount'] = expected_players
        base_row['RowCountConfidence'] = max(float(race_rows['RowCountConfidence'].iloc[0] or 0.0), 85.0)
        base_row['RaceScoreRecoveryUsed'] = True
        base_row['RaceScoreRecoverySource'] = f'ultra_low_res_blob_row12_restore_{best_score:.3f}'
        base_row['RaceScoreRecoveryCount'] = expected_players
        review_reason = str(base_row.get('ReviewReason') or '').strip(';')
        extra_reason = 'ultra_low_res_blob_row12_restore'
        base_row['ReviewReason'] = f'{review_reason};{extra_reason}'.strip(';') if review_reason else extra_reason
        synthetic_rows.append(base_row)

        group.loc[race_mask, 'RaceScorePlayerCount'] = expected_players
        group.loc[race_mask, 'RowCountConfidence'] = group.loc[race_mask, 'RowCountConfidence'].clip(lower=85.0)
        group.loc[race_mask, 'RaceScoreRecoveryUsed'] = True
        group.loc[race_mask, 'RaceScoreRecoverySource'] = f'ultra_low_res_blob_row12_restore_{best_score:.3f}'
        group.loc[race_mask, 'RaceScoreRecoveryCount'] = expected_players

    if synthetic_rows:
        for synthetic_row in synthetic_rows:
            row_values = []
            for column in group.columns:
                value = synthetic_row.get(column, np.nan)
                if value is pd.NA:
                    value = np.nan
                row_values.append(value)
            group.loc[len(group)] = row_values
        group = group.sort_values(['RaceIDNumber', 'RacePosition'], kind='stable').reset_index(drop=True)
    return group


def _resize_template_for_low_res(template: dict) -> dict:
    rgba = np.dstack((template["template_image"], template["template_alpha"]))
    resized_rgba = cv2.resize(
        rgba,
        (LOW_RES_CHARACTER_TEMPLATE_WIDTH, LOW_RES_CHARACTER_TEMPLATE_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    return {
        "character_index": int(template["character_index"]),
        "character_name": str(template["character_name"]),
        "template_image": resized_rgba[:, :, :3],
        "template_alpha": resized_rgba[:, :, 3],
    }


def _fixed_offset_character_match_score(source_image: np.ndarray, template_image: np.ndarray, template_alpha: np.ndarray) -> float:
    visible_mask = template_alpha > 16
    if int(np.count_nonzero(visible_mask)) <= 0:
        return 0.0
    mask_3d = np.repeat(visible_mask[:, :, None], 3, axis=2)
    if source_image.shape[1] != template_image.shape[1] or source_image.shape[0] != template_image.shape[0]:
        source_image = cv2.resize(
            source_image,
            (template_image.shape[1], template_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    window = source_image.astype(np.float32)
    template_rgb = template_image.astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(template_rgb[mask_3d] - window[mask_3d])))
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def _gamma_correct(gray: np.ndarray, gamma: float) -> np.ndarray:
    normalized = gray.astype(np.float32) / 255.0
    adjusted = np.power(normalized, gamma)
    return np.clip(adjusted * 255.0, 0, 255).astype(np.uint8)


def preprocess_low_res_name_roi(row_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)
    gray = _gamma_correct(gray, 0.8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white = int(np.count_nonzero(binary == 255))
    black = int(np.count_nonzero(binary == 0))
    if white < black:
        binary = cv2.bitwise_not(binary)
    return binary


def _corr_score(a: np.ndarray, b: np.ndarray) -> float:
    a_vec = a.astype(np.float32).reshape(-1)
    b_vec = b.astype(np.float32).reshape(-1)
    if a_vec.std() < 1e-6 or b_vec.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a_vec, b_vec)[0, 1])


def _profile_score(a: np.ndarray, b: np.ndarray) -> float:
    a_col = a.mean(axis=0)
    b_col = b.mean(axis=0)
    a_row = a.mean(axis=1)
    b_row = b.mean(axis=1)
    col = _corr_score(a_col[:, None], b_col[:, None])
    row = _corr_score(a_row[:, None], b_row[:, None])
    return (col + row) / 2.0


def _iou_score(a: np.ndarray, b: np.ndarray) -> float:
    a_mask = a > 0
    b_mask = b > 0
    union = np.logical_or(a_mask, b_mask).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(a_mask, b_mask).sum()
    return float(inter / union)


def compare_low_res_name_features(query_binary: np.ndarray, reference_binary: np.ndarray) -> float:
    score = 0.45 * _corr_score(query_binary, reference_binary)
    score += 0.35 * _profile_score(query_binary, reference_binary)
    score += 0.20 * _iou_score(query_binary, reference_binary)
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def compare_character_indices(identity_character_index, row_character_index) -> float:
    if pd.isna(identity_character_index) and pd.isna(row_character_index):
        return LOW_RES_UNKNOWN_CHARACTER_SCORE
    if pd.isna(identity_character_index) or pd.isna(row_character_index):
        return LOW_RES_UNKNOWN_CHARACTER_SCORE
    return LOW_RES_MATCH_CHARACTER_SCORE if int(identity_character_index) == int(row_character_index) else LOW_RES_MISMATCH_CHARACTER_SCORE


def _dominant_character_index(character_votes: List[int | float | None]):
    candidates = []
    for value in character_votes:
        if pd.isna(value):
            continue
        try:
            candidates.append(int(value))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return pd.NA
    return Counter(candidates).most_common(1)[0][0]


def _placeholder_sort_key(placeholder: str) -> int:
    suffix = str(placeholder).replace(LOW_RES_PLACEHOLDER_PREFIX, "")
    try:
        return int(suffix)
    except ValueError:
        return 9999


def _solve_max_assignment(score_matrix: np.ndarray) -> List[int]:
    row_count, column_count = score_matrix.shape
    if row_count == 0 or column_count == 0:
        return []

    @lru_cache(maxsize=None)
    def best(row_index: int, used_mask: int) -> float:
        if row_index >= row_count:
            return 0.0
        best_score = float('-inf')
        for column_index in range(column_count):
            if used_mask & (1 << column_index):
                continue
            candidate = float(score_matrix[row_index, column_index]) + best(row_index + 1, used_mask | (1 << column_index))
            if candidate > best_score:
                best_score = candidate
        return best_score

    assignments: List[int] = []
    used_mask = 0
    for row_index in range(row_count):
        chosen_column = 0
        chosen_score = float('-inf')
        for column_index in range(column_count):
            if used_mask & (1 << column_index):
                continue
            candidate = float(score_matrix[row_index, column_index]) + best(row_index + 1, used_mask | (1 << column_index))
            if candidate > chosen_score:
                chosen_score = candidate
                chosen_column = column_index
        assignments.append(chosen_column)
        used_mask |= 1 << chosen_column
    return assignments


def _full_search_low_res_characters(frame: np.ndarray, visible_rows: int, score_layout_id: str | None = None) -> List[dict]:
    templates = [_resize_template_for_low_res(template) for template in load_character_templates()]
    matches: List[dict] = []
    for position in range(1, visible_rows + 1):
        row_roi = _extract_low_res_character_roi(frame, position, score_layout_id=score_layout_id)
        scored_matches = []
        for template in templates:
            confidence = round(
                _fixed_offset_character_match_score(row_roi, template["template_image"], template["template_alpha"]) * 100.0,
                1,
            )
            scored_matches.append(
                {
                    "Character": template["character_name"],
                    "CharacterIndex": template["character_index"],
                    "CharacterMatchConfidence": confidence,
                    "CharacterMatchMethod": "low_res_full_search_fixed_roi",
                }
            )
        scored_matches.sort(key=lambda item: float(item["CharacterMatchConfidence"]), reverse=True)
        matches.append(scored_matches[0] if scored_matches else {
            "Character": "",
            "CharacterIndex": pd.NA,
            "CharacterMatchConfidence": 0.0,
            "CharacterMatchMethod": "low_res_full_search_missing",
        })
    return matches


def _previous_race_shortlist_characters(
    frame: np.ndarray,
    previous_matches: List[dict],
    visible_rows: int,
    score_layout_id: str | None = None,
) -> List[dict]:
    templates = [_resize_template_for_low_res(template) for template in load_character_templates()]
    templates_by_index = {int(template["character_index"]): template for template in templates}
    previous_slots = []
    for match in previous_matches:
        character_index = match.get("CharacterIndex")
        if pd.isna(character_index):
            continue
        template = templates_by_index.get(int(character_index))
        if template is None:
            continue
        previous_slots.append(template)

    if not previous_slots or visible_rows > len(previous_slots):
        return _full_search_low_res_characters(frame, visible_rows, score_layout_id=score_layout_id)

    score_matrix = np.zeros((visible_rows, len(previous_slots)), dtype=np.float32)
    detail_scores: List[List[dict]] = [[{} for _ in range(len(previous_slots))] for _ in range(visible_rows)]

    for row_pos in range(visible_rows):
        position = row_pos + 1
        row_roi = _extract_low_res_character_roi(frame, position, score_layout_id=score_layout_id)
        for slot_pos, template in enumerate(previous_slots):
            confidence = round(
                _fixed_offset_character_match_score(
                    row_roi,
                    template["template_image"],
                    template["template_alpha"],
                ) * 100.0,
                1,
            )
            score_matrix[row_pos, slot_pos] = float(confidence)
            detail_scores[row_pos][slot_pos] = {
                "Character": template["character_name"],
                "CharacterIndex": int(template["character_index"]),
                "CharacterMatchConfidence": confidence,
                "CharacterMatchMethod": "low_res_prev_race_shortlist",
            }

    assignments = _solve_max_assignment(score_matrix)
    matches: List[dict] = []
    for row_pos, slot_pos in enumerate(assignments):
        matches.append(detail_scores[row_pos][slot_pos])
    return matches


def _recompute_low_res_characters(group: pd.DataFrame, frames_folder: str | Path, race_class: str) -> pd.DataFrame:
    group = group.copy()
    race_ids = sorted(int(race_id) for race_id in group['RaceIDNumber'].unique())
    previous_matches: List[dict] | None = None

    for race_id in race_ids:
        race_rows = group[group['RaceIDNumber'] == race_id].sort_values('RacePosition', kind='stable')
        visible_rows = int(race_rows.shape[0])
        frame_path = race_score_image_path(frames_folder, race_class, race_id)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        score_layout_id = score_layout_id_from_filename(frame_path)

        if previous_matches is None:
            current_matches = _full_search_low_res_characters(frame, visible_rows, score_layout_id=score_layout_id)
        else:
            current_matches = _previous_race_shortlist_characters(
                frame,
                previous_matches,
                visible_rows,
                score_layout_id=score_layout_id,
            )

        for row_index, match in zip(race_rows.index, current_matches):
            group.at[row_index, 'Character'] = match.get('Character', '')
            group.at[row_index, 'CharacterIndex'] = match.get('CharacterIndex', pd.NA)
            group.at[row_index, 'CharacterMatchConfidence'] = match.get('CharacterMatchConfidence', 0.0)
            group.at[row_index, 'CharacterMatchMethod'] = match.get('CharacterMatchMethod', '')

        previous_matches = current_matches

    return group




def low_res_race_points(position: int, num_players: int) -> int:
    points_table = {
        12: [15, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        11: [13, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        10: [12, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        9: [11, 9, 8, 6, 5, 4, 3, 2, 1, 0],
        8: [10, 8, 6, 5, 4, 3, 2, 1, 0],
        7: [9, 7, 5, 4, 3, 2, 1, 0],
    }
    if num_players not in points_table:
        num_players = 12
    if 1 <= position <= len(points_table[num_players]):
        return int(points_table[num_players][position - 1])
    return 0


def _clean_review_reason_codes(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    kept = []
    for code in str(value).split(';'):
        normalized = code.strip()
        if not normalized or normalized.lower() == 'nan':
            continue
        if normalized in {'low_name_confidence', 'low_digit_consensus'}:
            continue
        kept.append(normalized)
    return ';'.join(dict.fromkeys(kept))


def _resolve_placeholder_names(identity_state: Dict[str, dict]) -> Dict[str, dict]:
    evidence = {}
    resolved = {}
    used_names = set()
    for placeholder, state in identity_state.items():
        candidate_scores = defaultdict(float)
        candidate_counts = defaultdict(int)
        for item in state['ocr_history']:
            name = normalize_name_for_vote(item['ocr_name'])
            if not name:
                continue
            if str(name).startswith(LOW_RES_PLACEHOLDER_PREFIX):
                continue
            candidate_scores[name] += float(item['assignment_score'])
            candidate_counts[name] += 1
        ranked = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], -candidate_counts[kv[0]], kv[0]))
        top_name = ranked[0][0] if ranked else ''
        top_score = ranked[0][1] if ranked else 0.0
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        evidence_count = sum(candidate_counts.values())
        evidence[placeholder] = {
            'TopCandidate': top_name,
            'TopScore': round(top_score, 3),
            'SecondScore': round(second_score, 3),
            'EvidenceCount': evidence_count,
            'ResolvedName': placeholder,
            'Reason': 'unresolved',
        }

    for pass_index in (1, 2):
        for placeholder in sorted(identity_state.keys(), key=_placeholder_sort_key):
            current = evidence[placeholder]
            if current['ResolvedName'] != placeholder:
                continue
            ranked = []
            state = identity_state[placeholder]
            candidate_scores = defaultdict(float)
            candidate_counts = defaultdict(int)
            for item in state['ocr_history']:
                name = normalize_name_for_vote(item['ocr_name'])
                if not name or name in used_names:
                    continue
                if str(name).startswith(LOW_RES_PLACEHOLDER_PREFIX):
                    continue
                candidate_scores[name] += float(item['assignment_score'])
                candidate_counts[name] += 1
            ranked = sorted(candidate_scores.items(), key=lambda kv: (-kv[1], -candidate_counts[kv[0]], kv[0]))
            if not ranked:
                continue
            top_name, top_score = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else 0.0
            ratio = top_score / max(second_score, 0.001)
            evidence_count = sum(candidate_counts.values())
            if pass_index == 1:
                should_resolve = evidence_count >= 3 and top_score >= 2.0 and (top_score - second_score >= 1.0 or ratio >= 1.18)
                reason = 'resolved_pass1'
            else:
                should_resolve = evidence_count >= 2 and top_score >= 1.2 and (top_score - second_score >= 0.35 or ratio >= 1.05)
                reason = 'resolved_pass2'
            if should_resolve:
                current['TopCandidate'] = top_name
                current['TopScore'] = round(top_score, 3)
                current['SecondScore'] = round(second_score, 3)
                current['EvidenceCount'] = evidence_count
                current['ResolvedName'] = top_name
                current['Reason'] = reason
                used_names.add(top_name)
    return evidence


def _write_debug_outputs(debug_dir: Path, race_class: str, assignment_rows: List[dict], resolution_rows: List[dict]) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = debug_dir / f'low_res_identity_assignment_{race_class}.csv'
    with assignment_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignment_rows[0].keys()) if assignment_rows else ['Race'])
        writer.writeheader()
        writer.writerows(assignment_rows)
    resolution_path = debug_dir / f'low_res_identity_resolution_{race_class}.csv'
    with resolution_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Placeholder', 'TopCandidate', 'TopScore', 'SecondScore', 'EvidenceCount', 'ResolvedName', 'Reason'])
        writer.writeheader()
        writer.writerows(resolution_rows)


def apply_low_res_identity_pipeline(race_class_df: pd.DataFrame, frames_folder: str | Path, race_class: str,
                                    *, write_debug_outputs: bool = False, debug_dir: str | Path | None = None) -> pd.DataFrame:
    group = race_class_df.sort_values(['RaceIDNumber', 'RacePosition'], kind='stable').copy()
    if group.empty:
        return group
    group = _restore_missing_rows_via_ultra_low_res_blob_fallback(group, frames_folder, race_class)
    group = _recompute_low_res_characters(group, frames_folder, race_class)

    race_ids = sorted(int(race_id) for race_id in group['RaceIDNumber'].unique())
    visible_rows_by_race = {
        race_id: int(group.loc[group['RaceIDNumber'] == race_id].shape[0])
        for race_id in race_ids
    }
    seed_race_id = min(race_ids, key=lambda race_id: (-visible_rows_by_race[race_id], race_id))
    max_players = int(visible_rows_by_race[seed_race_id])
    placeholders = [_placeholder_name(index) for index in range(1, max_players + 1)]

    identity_state: Dict[str, dict] = {
        placeholder: {
            'placeholder': placeholder,
            'name_refs': [],
            'character_votes': [],
            'ocr_history': [],
        }
        for placeholder in placeholders
    }
    row_assignments: Dict[int, dict] = {}
    debug_assignment_rows: List[dict] = []

    for race_id in race_ids:
        race_rows = group[group['RaceIDNumber'] == race_id].sort_values('RacePosition', kind='stable')
        visible_rows = int(race_rows.shape[0])
        frame_path = race_score_image_path(frames_folder, race_class, race_id)
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        score_layout_id = score_layout_id_from_filename(frame_path)

        row_features = []
        row_indices = []
        for row_index, row in race_rows.iterrows():
            position = int(row['RacePosition'])
            name_roi = _extract_name_roi(frame, position, score_layout_id=score_layout_id)
            row_features.append({
                'position': position,
                'name_binary': preprocess_low_res_name_roi(name_roi),
                'character_index': row.get('CharacterIndex'),
                'ocr_name': str(row.get('PlayerName') or ''),
            })
            row_indices.append(row_index)

        if race_id == seed_race_id:
            for row_index, row_feature, placeholder in zip(row_indices, row_features, placeholders[:visible_rows]):
                state = identity_state[placeholder]
                state['name_refs'].append(row_feature['name_binary'])
                state['character_votes'].append(row_feature['character_index'])
                state['ocr_history'].append({'ocr_name': row_feature['ocr_name'], 'assignment_score': 1.0})
                row_assignments[row_index] = {
                    'placeholder': placeholder,
                    'assigned_name': placeholder,
                    'combined_score': 1.0,
                    'name_score': 1.0,
                    'character_score': 1.0,
                }
                debug_assignment_rows.append({
                    'Race': race_id,
                    'Position': row_feature['position'],
                    'Placeholder': placeholder,
                    'AssignedName': placeholder,
                    'CombinedScore': 1.0,
                    'NameScore': 1.0,
                    'CharacterScore': 1.0,
                })
            continue

        score_matrix = np.zeros((visible_rows, max_players), dtype=np.float32)
        detail_scores = [[None for _ in range(max_players)] for _ in range(visible_rows)]
        for row_pos, row_feature in enumerate(row_features):
            for col_pos, placeholder in enumerate(placeholders):
                state = identity_state[placeholder]
                name_score = 0.0
                if state['name_refs']:
                    name_score = max(compare_low_res_name_features(row_feature['name_binary'], ref) for ref in state['name_refs'][-5:])
                character_score = compare_character_indices(_dominant_character_index(state['character_votes']), row_feature['character_index'])
                combined_score = (LOW_RES_NAME_WEIGHT * name_score) + (LOW_RES_CHARACTER_WEIGHT * character_score)
                score_matrix[row_pos, col_pos] = combined_score
                detail_scores[row_pos][col_pos] = (name_score, character_score, combined_score)

        assignments = _solve_max_assignment(score_matrix)
        for row_pos, col_pos in enumerate(assignments):
            row_index = row_indices[row_pos]
            row_feature = row_features[row_pos]
            placeholder = placeholders[col_pos]
            name_score, character_score, combined_score = detail_scores[row_pos][col_pos]
            state = identity_state[placeholder]
            state['name_refs'].append(row_feature['name_binary'])
            state['character_votes'].append(row_feature['character_index'])
            state['ocr_history'].append({'ocr_name': row_feature['ocr_name'], 'assignment_score': combined_score})
            row_assignments[row_index] = {
                'placeholder': placeholder,
                'assigned_name': placeholder,
                'combined_score': round(float(combined_score), 3),
                'name_score': round(float(name_score), 3),
                'character_score': round(float(character_score), 3),
            }
            debug_assignment_rows.append({
                'Race': race_id,
                'Position': row_feature['position'],
                'Placeholder': placeholder,
                'AssignedName': placeholder,
                'CombinedScore': round(float(combined_score), 3),
                'NameScore': round(float(name_score), 3),
                'CharacterScore': round(float(character_score), 3),
            })

    resolution_by_placeholder = _resolve_placeholder_names(identity_state)
    group['IdentityLabel'] = group.index.map(lambda idx: row_assignments[idx]['placeholder'])
    group['IdentityResolutionMethod'] = group.index.map(lambda idx: 'low_res_name_character_assignment')
    group['FixPlayerName'] = group.index.map(
        lambda idx: resolution_by_placeholder[row_assignments[idx]['placeholder']]['ResolvedName']
    )
    group['PlayerName'] = group['PlayerName'].fillna('')
    group['NameConfidence'] = group.index.map(lambda idx: int(round(row_assignments[idx]['combined_score'] * 100.0)))
    group['ReviewReason'] = group['ReviewReason'].apply(_clean_review_reason_codes)

    for race_id, race_rows in group.groupby('RaceIDNumber', sort=True):
        player_count = int(race_rows.shape[0])
        for row_index, row in race_rows.iterrows():
            position = int(row['RacePosition'])
            group.at[row_index, 'RacePoints'] = low_res_race_points(position, player_count)
            group.at[row_index, 'DetectedRacePoints'] = pd.NA
            group.at[row_index, 'DetectedRacePointsSource'] = 'low_res_disabled'
            group.at[row_index, 'DetectedOldTotalScore'] = pd.NA
            group.at[row_index, 'DetectedOldTotalScoreSource'] = 'low_res_disabled'
            group.at[row_index, 'DetectedTotalScore'] = pd.NA
            group.at[row_index, 'DetectedTotalScoreSource'] = 'low_res_disabled'
            group.at[row_index, 'DetectedNewTotalScore'] = pd.NA
            group.at[row_index, 'DetectedNewTotalScoreSource'] = 'low_res_disabled'
            group.at[row_index, 'TotalScoreMappingMethod'] = 'low_res_computed'
            group.at[row_index, 'IdentityResolutionMethod'] = resolution_by_placeholder[row_assignments[row_index]['placeholder']]['Reason']

    if write_debug_outputs and debug_dir is not None:
        resolution_rows = [
            {'Placeholder': placeholder, **resolution_by_placeholder[placeholder]}
            for placeholder in sorted(resolution_by_placeholder.keys(), key=_placeholder_sort_key)
        ]
        _write_debug_outputs(Path(debug_dir), race_class, debug_assignment_rows, resolution_rows)

    return group
