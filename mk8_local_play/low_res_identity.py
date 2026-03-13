from __future__ import annotations

import csv
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from .ocr_name_matching import choose_canonical_name, normalize_name_for_vote
from .ocr_scoreboard_consensus import PLAYER_NAME_COORDS, character_row_roi

LOW_RES_PLACEHOLDER_PREFIX = "PlayerNameMissing_"
LOW_RES_NAME_ROW_HEIGHT = 45
LOW_RES_NAME_WEIGHT = 0.75
LOW_RES_CHARACTER_WEIGHT = 0.25
LOW_RES_UNKNOWN_CHARACTER_SCORE = 0.5
LOW_RES_MISMATCH_CHARACTER_SCORE = 0.0
LOW_RES_MATCH_CHARACTER_SCORE = 1.0


def is_low_res_height(source_height: int | None, max_source_height: int) -> bool:
    if source_height is None:
        return False
    return int(source_height) <= int(max_source_height)


def race_score_image_path(frames_folder: str | Path, race_class: str, race_id: int) -> Path:
    return Path(frames_folder) / f"{race_class}+Race_{race_id:03}+2RaceScore.png"


def _placeholder_name(index: int) -> str:
    return f"{LOW_RES_PLACEHOLDER_PREFIX}{index}"


def _extract_name_roi(frame: np.ndarray, position: int) -> np.ndarray:
    (x1, y1), (x2, _y2) = PLAYER_NAME_COORDS[position - 1]
    return frame[y1:y1 + LOW_RES_NAME_ROW_HEIGHT, x1:x2].copy()


def _extract_character_roi(frame: np.ndarray, position: int) -> np.ndarray:
    (x1, y1), (x2, y2) = character_row_roi(position - 1)
    return frame[y1:y2, x1:x2].copy()


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

        row_features = []
        row_indices = []
        for row_index, row in race_rows.iterrows():
            position = int(row['RacePosition'])
            name_roi = _extract_name_roi(frame, position)
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
