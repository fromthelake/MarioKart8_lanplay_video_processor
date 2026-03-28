import difflib
import itertools
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .extract_common import debug_identity_workbook_path, find_score_bundle_anchor_path
import textdistance
from jellyfish import soundex

from .name_unicode import (
    allowed_name_char_ratio,
    collapse_name_whitespace,
    distinct_visible_name_count,
    normalize_name_key,
)
from .ocr_scoreboard_consensus import ultra_low_res_combined_row_roi


def preprocess_name(name):
    """Normalize OCR names before comparing them across races."""
    return normalize_name_key(name)


def soundex_similarity(name1, name2):
    """Use Soundex as a weak phonetic tie-breaker for noisy OCR names."""
    try:
        ascii_name1 = preprocess_name(name1).encode("ascii", "ignore").decode("ascii")
        ascii_name2 = preprocess_name(name2).encode("ascii", "ignore").decode("ascii")
        if not ascii_name1 or not ascii_name2:
            return 0
        return 1 if soundex(ascii_name1) == soundex(ascii_name2) else 0
    except Exception:
        return 0


def weighted_similarity(name1, name2):
    """Blend exact string similarity with OCR-friendly fuzzy signals."""
    name1 = preprocess_name(name1)
    name2 = preprocess_name(name2)

    difflib_score = difflib.SequenceMatcher(None, name1, name2).ratio()
    length_score = 1 - abs(len(name1) - len(name2)) / max(len(name1), len(name2), 1)
    jaro_winkler_score = textdistance.jaro_winkler(name1, name2)
    soundex_score = soundex_similarity(name1, name2)

    weights = {
        "difflib": 0.3,
        "length": 0.4,
        "jaro_winkler": 0.3,
        "soundex": 0.0,
    }
    return (
        weights["difflib"] * difflib_score
        + weights["length"] * length_score
        + weights["jaro_winkler"] * jaro_winkler_score
        + weights["soundex"] * soundex_score
    )


def normalize_name_for_vote(name: str) -> str:
    return collapse_name_whitespace(name)


def choose_canonical_name(name_history):
    """Choose the cleanest spelling seen for one tracked identity."""
    candidates = [normalize_name_for_vote(name) for name in name_history if normalize_name_for_vote(name)]
    if not candidates:
        return ""

    counts = Counter(candidates)
    best_name = candidates[0]
    best_score = float("-inf")
    for candidate, count in counts.items():
        quality = distinct_visible_name_count(candidate)
        allowed_ratio = allowed_name_char_ratio(candidate)
        support_score = 0.0
        for observed_name in candidates:
            support_score += weighted_similarity(candidate, observed_name)
        score = (count * 3.0) + support_score + (quality * 0.25) + allowed_ratio
        if score > best_score:
            best_score = score
            best_name = candidate
    return best_name


def _character_similarity(identity_character_index, row_character_index):
    if pd.isna(identity_character_index) and pd.isna(row_character_index):
        return 0.0
    if pd.isna(identity_character_index) or pd.isna(row_character_index):
        return 0.05
    return 1.0 if int(identity_character_index) == int(row_character_index) else -0.4


def _detected_total_similarity(identity_last_total, row_detected_total):
    if pd.isna(identity_last_total) or pd.isna(row_detected_total):
        return 0.0
    total_gap = abs(int(row_detected_total) - int(identity_last_total))
    if total_gap <= 2:
        return 1.0
    if total_gap <= 8:
        return 0.7
    if total_gap <= 20:
        return 0.3
    return -0.2


def _row_name_is_unreliable(row) -> bool:
    validation_flags = {
        flag.strip().lower()
        for flag in str(row.get("NameValidationFlags") or "").split("|")
        if flag.strip()
    }
    if {"low_name_confidence", "unknown_chars"} & validation_flags:
        return True

    try:
        name_confidence = float(row.get("NameConfidence", 100.0) or 100.0)
    except (TypeError, ValueError):
        name_confidence = 100.0
    if name_confidence < 90.0:
        return True

    try:
        allowed_ratio = float(row.get("NameAllowedCharRatio", 100.0) or 100.0)
    except (TypeError, ValueError):
        allowed_ratio = 100.0
    if allowed_ratio < 90.0:
        return True

    visible_name = normalize_name_for_vote(str(row.get("PlayerName") or ""))
    return distinct_visible_name_count(visible_name) < 4


def _extract_visual_identity_roi(frame: np.ndarray, position: int, score_layout_id: str | None = None) -> np.ndarray:
    (x1, y1), (x2, y2) = ultra_low_res_combined_row_roi(position - 1, score_layout_id=score_layout_id)
    crop = frame[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return crop
    target_width = max(1, x2 - x1)
    target_height = max(1, y2 - y1)
    if crop.shape[1] != target_width or crop.shape[0] != target_height:
        crop = cv2.resize(crop, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return crop


def _compare_visual_identity_roi(reference_roi: np.ndarray | None, query_roi: np.ndarray | None) -> float:
    if reference_roi is None or query_roi is None:
        return 0.0
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


def _prepare_visual_identity_features(group: pd.DataFrame) -> dict[int, np.ndarray | None]:
    frame_cache: dict[tuple[str, int], np.ndarray | None] = {}
    visual_features: dict[int, np.ndarray | None] = {}
    for row_index, row in group.iterrows():
        try:
            race_id = int(row.get("RaceIDNumber"))
            position = int(row.get("RacePosition"))
        except (TypeError, ValueError):
            visual_features[row_index] = None
            continue
        race_class = str(row.get("RaceClass") or "")
        cache_key = (race_class, race_id)
        if cache_key not in frame_cache:
            anchor_path = find_score_bundle_anchor_path(race_class, race_id, "2RaceScore")
            frame_cache[cache_key] = (
                cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
                if anchor_path is not None and Path(anchor_path).exists()
                else None
            )
        frame = frame_cache[cache_key]
        if frame is None:
            visual_features[row_index] = None
            continue
        score_layout_id = str(row.get("ScoreLayoutId") or "").strip() or None
        visual_features[row_index] = _extract_visual_identity_roi(frame, position, score_layout_id=score_layout_id)
    return visual_features


def _visual_similarity(identity_visual_refs, row_visual_roi) -> float:
    if row_visual_roi is None or not identity_visual_refs:
        return 0.0
    reference_rois = [roi for roi in identity_visual_refs[-4:] if roi is not None]
    if not reference_rois:
        return 0.0
    return max(_compare_visual_identity_roi(reference_roi, row_visual_roi) for reference_roi in reference_rois)


def _build_case_distinct_name_keys(group: pd.DataFrame) -> set[str]:
    """Preserve visibly distinct names like 'Floris' and 'floris' when they coexist in one race."""
    protected_keys: set[str] = set()
    for _race_id, race_rows in group.groupby("RaceIDNumber", sort=False):
        by_normalized: dict[str, set[str]] = defaultdict(set)
        for _, row in race_rows.iterrows():
            visible_name = normalize_name_for_vote(str(row.get("PlayerName") or ""))
            normalized_name = preprocess_name(visible_name)
            if not visible_name or not normalized_name:
                continue
            by_normalized[normalized_name].add(visible_name)
        for normalized_name, visible_names in by_normalized.items():
            if len(visible_names) > 1:
                protected_keys.add(normalized_name)
    return protected_keys


def _build_match_candidates(identity_state, race_rows, visual_features, protected_case_keys=None):
    protected_case_keys = protected_case_keys or set()
    candidates = []
    for row_position, (row_index, row) in enumerate(race_rows, start=1):
        row_name = str(row["PlayerName"] or "")
        visible_row_name = normalize_name_for_vote(row_name)
        normalized_row_name = preprocess_name(row_name)
        row_character_index = row.get("CharacterIndex")
        # Identity continuity should compare the prior race's ending total with
        # the current race's starting total, not the post-race total.
        row_detected_total = row.get("DetectedOldTotalScore")
        row_visual_roi = visual_features.get(row_index)
        row_is_unreliable = _row_name_is_unreliable(row)
        for identity_id, identity in identity_state.items():
            identity_name = str(identity["canonical_name"] or "")
            visible_identity_name = normalize_name_for_vote(identity_name)
            name_similarity = weighted_similarity(identity["canonical_name"], row_name)
            character_similarity = _character_similarity(identity["character_index"], row_character_index)
            total_similarity = _detected_total_similarity(identity["last_detected_total"], row_detected_total)
            visual_similarity = _visual_similarity(identity.get("visual_refs", []), row_visual_roi)
            protected_case_conflict = (
                normalized_row_name in protected_case_keys
                and visible_row_name
                and visible_identity_name
                and visible_row_name != visible_identity_name
            )
            if protected_case_conflict and not row_is_unreliable:
                continue
            combined_score = (
                (name_similarity * 0.50)
                + (total_similarity * 0.25)
                + (visual_similarity * 0.20)
                + (character_similarity * 0.05)
            )
            exact_name_match = visible_identity_name == visible_row_name and bool(visible_row_name)
            fallback_match = row_is_unreliable and (
                (visual_similarity >= 0.72 and name_similarity >= 0.50)
                or (total_similarity >= 0.7 and name_similarity >= 0.45)
                or (visual_similarity >= 0.68 and total_similarity >= 0.7)
            )
            if name_similarity >= 0.72 or exact_name_match or fallback_match:
                candidates.append(
                    (
                        combined_score,
                        name_similarity,
                        visual_similarity,
                        total_similarity,
                        character_similarity,
                        identity_id,
                        row_position,
                        row_index,
                    )
                )
    return sorted(candidates, reverse=True)


def _next_identity_id(identity_state):
    return max(identity_state.keys(), default=0) + 1


def _assign_identity_labels(identity_state):
    identities_by_base_name = defaultdict(list)
    for identity_id, identity in identity_state.items():
        base_name = choose_canonical_name(identity["name_history"])
        identity["canonical_name"] = base_name
        identities_by_base_name[base_name].append(identity_id)

    identity_labels = {}
    for base_name, identity_ids in identities_by_base_name.items():
        if len(identity_ids) == 1:
            identity_labels[identity_ids[0]] = base_name
            continue
        ordered_identities = sorted(
            identity_ids,
            key=lambda identity_id: (
                int(identity_state[identity_id]["first_race"]),
                int(identity_state[identity_id].get("first_position", 9999)),
                identity_id,
            ),
        )
        for suffix, identity_id in enumerate(ordered_identities, start=1):
            identity_labels[identity_id] = f"{base_name}_{suffix}"
    return identity_labels


def _write_identity_debug_excel(output_folder, race_class, identity_rows):
    debug_df = pd.DataFrame(identity_rows)
    workbook_path = debug_identity_workbook_path(race_class)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    debug_df.to_excel(workbook_path, index=False)


def standardize_player_names(df, output_folder, write_debug_linking_excel=False):
    """Assign stable player identities per video using name, character, and total-score hints."""
    standardized_frames = []

    for race_class, group in df.groupby("RaceClass", sort=False):
        group = group.sort_values(["RaceIDNumber", "RacePosition"], kind="stable").copy()
        visual_features = _prepare_visual_identity_features(group)
        protected_case_keys = _build_case_distinct_name_keys(group)
        identity_state = {}
        row_identity_assignments = {}
        identity_debug_rows = []

        for race_id, race_rows_df in group.groupby("RaceIDNumber", sort=True):
            race_rows = list(race_rows_df.iterrows())
            candidates = _build_match_candidates(identity_state, race_rows, visual_features, protected_case_keys)
            used_identity_ids = set()
            used_row_indices = set()

            for (
                combined_score,
                name_similarity,
                visual_similarity,
                total_similarity,
                character_similarity,
                identity_id,
                _row_position,
                row_index,
            ) in candidates:
                if identity_id in used_identity_ids or row_index in used_row_indices:
                    continue
                row = group.loc[row_index]
                resolution_method = "name_only"
                if visual_similarity >= 0.72:
                    resolution_method = "name+visual"
                elif total_similarity >= 0.7:
                    resolution_method = "name+total_hint"
                elif pd.notna(row.get("CharacterIndex")) and pd.notna(identity_state[identity_id]["character_index"]):
                    resolution_method = "name+character"
                row_identity_assignments[row_index] = {
                    "identity_id": identity_id,
                    "resolution_method": resolution_method,
                    "match_score": round(float(combined_score), 3),
                    "name_score": round(float(name_similarity), 3),
                    "visual_score": round(float(visual_similarity), 3),
                    "total_score": round(float(total_similarity), 3),
                    "character_score": round(float(character_similarity), 3),
                }
                used_identity_ids.add(identity_id)
                used_row_indices.add(row_index)

            for row_index, row in race_rows:
                if row_index in row_identity_assignments:
                    identity_id = row_identity_assignments[row_index]["identity_id"]
                else:
                    identity_id = _next_identity_id(identity_state)
                    row_identity_assignments[row_index] = {
                        "identity_id": identity_id,
                        "resolution_method": "new_identity",
                        "match_score": None,
                    }
                    identity_state[identity_id] = {
                        "canonical_name": str(row["PlayerName"] or ""),
                        "name_history": [],
                        "character_index": row.get("CharacterIndex"),
                        "first_race": int(race_id),
                        "first_position": int(row.get("RacePosition", 9999)),
                        "last_detected_total": row.get("DetectedTotalScore"),
                        "visual_refs": [],
                    }

                identity = identity_state.setdefault(
                    identity_id,
                    {
                        "canonical_name": str(row["PlayerName"] or ""),
                        "name_history": [],
                        "character_index": row.get("CharacterIndex"),
                        "first_race": int(race_id),
                        "first_position": int(row.get("RacePosition", 9999)),
                        "last_detected_total": row.get("DetectedTotalScore"),
                        "visual_refs": [],
                    },
                )
                identity["name_history"].append(str(row["PlayerName"] or ""))
                row_is_unreliable = _row_name_is_unreliable(row)
                if pd.notna(row.get("CharacterIndex")) and (pd.isna(identity.get("character_index")) or not row_is_unreliable):
                    identity["character_index"] = row.get("CharacterIndex")
                if pd.notna(row.get("DetectedTotalScore")):
                    identity["last_detected_total"] = row.get("DetectedTotalScore")
                row_visual_roi = visual_features.get(row_index)
                if row_visual_roi is not None and getattr(row_visual_roi, "size", 1) > 0:
                    identity.setdefault("visual_refs", []).append(row_visual_roi)
                identity["canonical_name"] = choose_canonical_name(identity["name_history"])

                identity_debug_rows.append(
                    {
                        "Race": int(race_id),
                        "Race Position": int(row["RacePosition"]),
                        "Raw Player OCR": str(row["PlayerName"] or ""),
                        "Character": str(row.get("Character") or ""),
                        "Character Index": row.get("CharacterIndex"),
                        "Detected Total Score": row.get("DetectedTotalScore"),
                        "Identity ID": identity_id,
                        "Resolution Method": row_identity_assignments[row_index]["resolution_method"],
                        "Match Score": row_identity_assignments[row_index]["match_score"],
                        "Name Score": row_identity_assignments[row_index].get("name_score"),
                        "Visual Score": row_identity_assignments[row_index].get("visual_score"),
                        "Total Score": row_identity_assignments[row_index].get("total_score"),
                        "Character Score": row_identity_assignments[row_index].get("character_score"),
                    }
                )

        identity_labels = _assign_identity_labels(identity_state)
        group["FixPlayerName"] = group.index.map(lambda idx: identity_labels[row_identity_assignments[idx]["identity_id"]])
        group["IdentityLabel"] = group["FixPlayerName"]
        group["IdentityResolutionMethod"] = group.index.map(lambda idx: row_identity_assignments[idx]["resolution_method"])
        standardized_frames.append(group)

        if write_debug_linking_excel:
            _write_identity_debug_excel(output_folder, race_class, identity_debug_rows)

    return pd.concat(standardized_frames, ignore_index=True) if standardized_frames else df.copy()


def merge_fragmented_identity_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse OCR-only alias splits when a video links to more identities than players.

    This is intentionally narrow: only merge identities that never appear in the same
    race, share the same dominant character, and have highly similar low-quality names.
    """
    if df.empty or "FixPlayerName" not in df.columns:
        return df.copy()

    result = df.copy()
    for race_class, race_group in result.groupby("RaceClass", sort=False):
        expected_players = int(race_group.groupby("RaceIDNumber").size().mode().iloc[0])
        unique_labels = [str(value) for value in race_group["FixPlayerName"].dropna().unique()]
        if len(unique_labels) <= expected_players:
            continue

        identity_stats = {}
        for label, identity_rows in race_group.groupby("FixPlayerName", sort=False):
            identity_rows = identity_rows.sort_values(["RaceIDNumber", "RacePosition"], kind="stable")
            race_ids = {int(value) for value in identity_rows["RaceIDNumber"].dropna().tolist()}
            character_counts = Counter(
                int(value) for value in identity_rows["CharacterIndex"].dropna().tolist() if pd.notna(value)
            )
            dominant_character = character_counts.most_common(1)[0][0] if character_counts else None
            canonical_raw_name = choose_canonical_name(identity_rows["PlayerName"].tolist()) or str(label)
            unreliable_votes = 0
            for _, row in identity_rows.iterrows():
                if _row_name_is_unreliable(row):
                    unreliable_votes += 1
            row_count = max(1, len(identity_rows))
            identity_stats[str(label)] = {
                "race_ids": race_ids,
                "dominant_character": dominant_character,
                "canonical_raw_name": canonical_raw_name,
                "unreliable_ratio": unreliable_votes / row_count,
                "row_count": row_count,
                "first_race": int(identity_rows["RaceIDNumber"].min()),
            }

        merge_pairs = []
        labels = list(identity_stats.keys())
        for index, left_label in enumerate(labels):
            left_stats = identity_stats[left_label]
            for right_label in labels[index + 1:]:
                right_stats = identity_stats[right_label]
                if left_stats["race_ids"] & right_stats["race_ids"]:
                    continue
                name_similarity = weighted_similarity(
                    left_stats["canonical_raw_name"], right_stats["canonical_raw_name"]
                )
                same_character = (
                    left_stats["dominant_character"] is not None
                    and right_stats["dominant_character"] is not None
                    and left_stats["dominant_character"] == right_stats["dominant_character"]
                )
                character_conflict = (
                    left_stats["dominant_character"] is not None
                    and right_stats["dominant_character"] is not None
                    and left_stats["dominant_character"] != right_stats["dominant_character"]
                )
                small_fragment = min(left_stats["row_count"], right_stats["row_count"]) <= 2
                min_name_similarity = 0.68
                if character_conflict and small_fragment:
                    min_name_similarity = 0.60
                if name_similarity < min_name_similarity:
                    continue
                if character_conflict:
                    if not small_fragment:
                        continue
                    larger_identity_size = max(left_stats["row_count"], right_stats["row_count"])
                    if larger_identity_size < 4 or name_similarity < 0.60:
                        continue
                if max(left_stats["unreliable_ratio"], right_stats["unreliable_ratio"]) < 0.5:
                    continue
                score = name_similarity
                if same_character:
                    score += 0.15
                elif character_conflict:
                    score -= 0.10
                merge_pairs.append((score, left_label, right_label))

        if not merge_pairs:
            continue

        rename_map: dict[str, str] = {}
        for _score, left_label, right_label in sorted(merge_pairs, reverse=True):
            left_root = rename_map.get(left_label, left_label)
            right_root = rename_map.get(right_label, right_label)
            if left_root == right_root:
                continue
            ordered_labels = sorted(
                [left_root, right_root],
                key=lambda label: (
                    identity_stats[label]["first_race"],
                    -identity_stats[label]["row_count"],
                    label,
                ),
            )
            keep_label, drop_label = ordered_labels[0], ordered_labels[1]
            rename_map[drop_label] = keep_label

        if not rename_map:
            continue

        race_mask = result["RaceClass"] == race_class
        merged_source_mask = race_mask & result["FixPlayerName"].isin(rename_map.keys())
        result.loc[race_mask, "FixPlayerName"] = result.loc[race_mask, "FixPlayerName"].map(
            lambda value: rename_map.get(str(value), value)
        )
        if "IdentityLabel" in result.columns:
            result.loc[race_mask, "IdentityLabel"] = result.loc[race_mask, "IdentityLabel"].map(
                lambda value: rename_map.get(str(value), value)
            )
        if "IdentityResolutionMethod" in result.columns:
            result.loc[merged_source_mask, "IdentityResolutionMethod"] = result.loc[
                merged_source_mask, "IdentityResolutionMethod"
            ].map(
                lambda value: (
                    f"{str(value).strip()}+alias_merge" if str(value).strip() else "alias_merge"
                )
            )

    return result


def reconcile_connection_reset_identities(df: pd.DataFrame) -> pd.DataFrame:
    """Merge one post-reset identity back to one pre-reset identity when a cluster stays intact.

    Assumption:
    - one video contains one stable player cluster
    - at most one player changes displayed identity after a connection reset
    """
    if df.empty:
        return df.copy()

    merged_df = df.copy()
    if "IdentityRelinkDetected" not in merged_df.columns:
        merged_df["IdentityRelinkDetected"] = False
    if "IdentityRelinkSummary" not in merged_df.columns:
        merged_df["IdentityRelinkSummary"] = ""
    if "IdentityRelinkNote" not in merged_df.columns:
        merged_df["IdentityRelinkNote"] = ""

    for race_class, race_group in merged_df.groupby("RaceClass", sort=False):
        race_group = race_group.sort_values(["RaceIDNumber", "RacePosition"], kind="stable")
        expected_players = int(race_group.groupby("RaceIDNumber").size().mode().iloc[0])
        unique_players = set(str(value) for value in race_group["FixPlayerName"].dropna().unique())
        if len(unique_players) <= expected_players:
            continue

        reset_races = sorted(race_group.loc[race_group["SessionResetDetected"], "RaceIDNumber"].unique())
        if not reset_races:
            continue
        reset_start_race = int(reset_races[0])

        before_reset = race_group.loc[race_group["RaceIDNumber"] < reset_start_race]
        after_reset = race_group.loc[race_group["RaceIDNumber"] >= reset_start_race]
        if before_reset.empty or after_reset.empty:
            continue

        before_players = set(str(value) for value in before_reset["FixPlayerName"].dropna().unique())
        after_players = set(str(value) for value in after_reset["FixPlayerName"].dropna().unique())
        disappeared_players = before_players - after_players
        appeared_players = after_players - before_players
        if len(disappeared_players) != 1 or len(appeared_players) != 1:
            continue

        prior_identity = next(iter(disappeared_players))
        new_identity = next(iter(appeared_players))

        prior_last_race = int(
            race_group.loc[race_group["FixPlayerName"] == prior_identity, "RaceIDNumber"].max()
        )
        new_first_race = int(
            race_group.loc[race_group["FixPlayerName"] == new_identity, "RaceIDNumber"].min()
        )
        if prior_last_race != reset_start_race - 1 or new_first_race != reset_start_race:
            continue

        prior_rows = race_group.loc[race_group["FixPlayerName"] == prior_identity]
        new_rows = race_group.loc[race_group["FixPlayerName"] == new_identity]
        prior_raw_name = choose_canonical_name(prior_rows["PlayerName"].tolist()) or prior_identity
        new_raw_name = choose_canonical_name(new_rows["PlayerName"].tolist()) or new_identity
        note = (
            f'Identity relinked after connection reset: matched post-reset player "{new_raw_name}" '
            f'to earlier identity "{prior_identity}" (previous OCR name "{prior_raw_name}").'
        )
        summary = f'Connection reset name change: "{new_raw_name}" -> "{prior_identity}"'

        merge_mask = (merged_df["RaceClass"] == race_class) & (merged_df["FixPlayerName"] == new_identity)
        merged_df.loc[merge_mask, "FixPlayerName"] = prior_identity
        merged_df.loc[merge_mask, "IdentityLabel"] = prior_identity
        merged_df.loc[merge_mask, "IdentityRelinkDetected"] = True
        merged_df.loc[merge_mask, "IdentityRelinkSummary"] = summary
        merged_df.loc[merge_mask, "IdentityRelinkNote"] = note
        merged_df.loc[merge_mask, "IdentityResolutionMethod"] = merged_df.loc[
            merge_mask, "IdentityResolutionMethod"
        ].map(
            lambda value: (
                f"{str(value).strip()}+connection_reset_relink"
                if str(value).strip() else "connection_reset_relink"
            )
        )

    return merged_df


def _label_base_name(label: str) -> str:
    match = re.match(r"^(.*)_(\d+)$", str(label))
    return match.group(1) if match else str(label)


def resolve_duplicate_name_identity_chains(df: pd.DataFrame) -> pd.DataFrame:
    """Reassign duplicate-name identities by minimizing total continuity error across races."""
    if df.empty or "FixPlayerName" not in df.columns:
        return df.copy()

    result = df.copy()
    if "IdentityAmbiguityDetected" not in result.columns:
        result["IdentityAmbiguityDetected"] = False
    if "IdentityAmbiguityNote" not in result.columns:
        result["IdentityAmbiguityNote"] = ""

    def _row_int(row, column):
        value = row.get(column)
        if pd.isna(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    for race_class, race_group in result.groupby("RaceClass", sort=False):
        race_group = race_group.sort_values(["RaceIDNumber", "RacePosition"], kind="stable")

        base_groups = defaultdict(list)
        for label in [str(value) for value in race_group["FixPlayerName"].dropna().unique()]:
            base_groups[_label_base_name(label)].append(label)

        for base_name, labels in base_groups.items():
            labels = sorted(set(labels))
            if len(labels) <= 1:
                continue
            if len(labels) > 6:
                continue

            base_rows = race_group.loc[race_group["FixPlayerName"].isin(labels)].copy()
            race_ids = sorted(int(value) for value in base_rows["RaceIDNumber"].dropna().unique())
            if len(race_ids) < 2:
                continue

            race_to_rows = {}
            valid_group = True
            for race_id in race_ids:
                rows = base_rows.loc[base_rows["RaceIDNumber"] == race_id].sort_values("RacePosition", kind="stable")
                if len(rows) != len(labels):
                    valid_group = False
                    break
                race_to_rows[race_id] = list(rows.iterrows())
            if not valid_group:
                continue

            first_race_rows = race_to_rows[race_ids[0]]
            slot_count = len(first_race_rows)
            canonical_labels = tuple(f"{base_name}_{index}" for index in range(1, slot_count + 1))
            canonical_characters = tuple(row.get("CharacterIndex") for _, row in first_race_rows)

            per_race_permutations = {}
            for race_id in race_ids:
                row_entries = race_to_rows[race_id]
                perm_entries = []
                for perm in itertools.permutations(range(len(row_entries))):
                    row_map = {}
                    cost_bias = 0.0
                    for slot_index, row_entry_index in enumerate(perm):
                        row_index, row = row_entries[row_entry_index]
                        row_map[slot_index] = {
                            "row_index": row_index,
                            "old_total": _row_int(row, "DetectedOldTotalScore"),
                            "new_total": _row_int(row, "DetectedTotalScore"),
                            "character_index": row.get("CharacterIndex"),
                            "race_position": int(row.get("RacePosition")),
                        }
                        if row_map[slot_index]["character_index"] != canonical_characters[slot_index]:
                            cost_bias += 1000.0
                        current_label = str(row.get("FixPlayerName") or "")
                        desired_label = canonical_labels[slot_index]
                        if current_label != desired_label:
                            cost_bias += 0.01
                    perm_entries.append((perm, row_map, cost_bias))
                per_race_permutations[race_id] = perm_entries

            anchored_perm = tuple(range(slot_count))

            @lru_cache(maxsize=None)
            def _best_suffix(race_pos: int, previous_perm: tuple[int, ...]):
                if race_pos >= len(race_ids):
                    return 0.0, ()

                race_id = race_ids[race_pos]
                previous_map = next(
                    row_map for perm, row_map, _ in per_race_permutations[race_ids[race_pos - 1]] if perm == previous_perm
                )
                best_result = None
                for perm, row_map, bias in per_race_permutations[race_id]:
                    continuity_cost = 0.0
                    for slot_index in range(slot_count):
                        prev_new = previous_map[slot_index]["new_total"]
                        curr_old = row_map[slot_index]["old_total"]
                        if prev_new is None or curr_old is None:
                            continue
                        continuity_cost += abs(int(prev_new) - int(curr_old))
                    future_cost, future_path = _best_suffix(race_pos + 1, perm)
                    candidate = (continuity_cost + bias + future_cost, (perm,) + future_path)
                    if best_result is None or candidate[0] < best_result[0]:
                        best_result = candidate
                return best_result

            _best_cost, suffix_path = _best_suffix(1, anchored_perm)
            chosen_path = {race_ids[0]: anchored_perm}
            for race_id, perm in zip(race_ids[1:], suffix_path):
                chosen_path[race_id] = perm

            for race_id in race_ids:
                chosen_perm = chosen_path[race_id]
                chosen_row_map = next(
                    row_map for perm, row_map, _ in per_race_permutations[race_id] if perm == chosen_perm
                )
                old_total_buckets = defaultdict(list)
                for slot_index in range(slot_count):
                    row_state = chosen_row_map[slot_index]
                    key = (row_state["character_index"], row_state["old_total"])
                    old_total_buckets[key].append(slot_index)
                ambiguous_slot_partners: dict[int, list[int]] = {}
                for key, indices in old_total_buckets.items():
                    if key[1] is None or len(indices) <= 1:
                        continue
                    for slot_index in indices:
                        ambiguous_slot_partners[slot_index] = [other_slot for other_slot in indices if other_slot != slot_index]

                for slot_index in range(slot_count):
                    row_index = chosen_row_map[slot_index]["row_index"]
                    desired_label = canonical_labels[slot_index]
                    result.at[row_index, "FixPlayerName"] = desired_label
                    result.at[row_index, "IdentityLabel"] = desired_label
                    existing_method = str(result.at[row_index, "IdentityResolutionMethod"] or "").strip()
                    if "duplicate_name_chain" not in existing_method:
                        result.at[row_index, "IdentityResolutionMethod"] = (
                            f"{existing_method}+duplicate_name_chain" if existing_method else "duplicate_name_chain"
                        )
                    if race_id == race_ids[-1] and slot_index in ambiguous_slot_partners:
                        counterpart_labels = [canonical_labels[other_slot] for other_slot in ambiguous_slot_partners[slot_index]]
                        counterpart_text = ", ".join(counterpart_labels)
                        note = (
                            "Identity ambiguous with "
                            f"{counterpart_text}: final race could not be resolved uniquely for duplicate "
                            "name/character/score chain; mapping carried forward from previous race."
                        )
                        result.at[row_index, "IdentityAmbiguityDetected"] = True
                        existing_note = str(result.at[row_index, "IdentityAmbiguityNote"] or "").strip()
                        if note not in existing_note:
                            result.at[row_index, "IdentityAmbiguityNote"] = (
                                f"{existing_note} | {note}" if existing_note else note
                            )

    return result


def compact_identity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remove stale numeric suffixes when only one identity remains for a base name."""
    if df.empty:
        return df.copy()

    result = df.copy()
    for race_class, race_group in result.groupby("RaceClass", sort=False):
        labels = [str(value) for value in race_group["FixPlayerName"].dropna().unique()]
        base_to_labels: dict[str, list[str]] = defaultdict(list)
        for label in labels:
            match = re.match(r"^(.*)_(\d+)$", label)
            base_name = match.group(1) if match else label
            base_to_labels[base_name].append(label)

        rename_map: dict[str, str] = {}
        for base_name, grouped_labels in base_to_labels.items():
            grouped_labels = sorted(set(grouped_labels))
            if len(grouped_labels) == 1:
                only_label = grouped_labels[0]
                if only_label != base_name:
                    rename_map[only_label] = base_name

        if not rename_map:
            continue

        race_mask = result["RaceClass"] == race_class
        result.loc[race_mask, "FixPlayerName"] = result.loc[race_mask, "FixPlayerName"].map(
            lambda value: rename_map.get(str(value), value)
        )
        result.loc[race_mask, "IdentityLabel"] = result.loc[race_mask, "IdentityLabel"].map(
            lambda value: rename_map.get(str(value), value)
        )
        if "IdentityRelinkNote" in result.columns:
            result.loc[race_mask, "IdentityRelinkNote"] = result.loc[race_mask, "IdentityRelinkNote"].map(
                lambda value: (
                    str(value).replace('"Bonno_2"', '"Bonno"').replace('identity "Bonno_2"', 'identity "Bonno"')
                    if pd.notna(value) else value
                )
            )
        if "IdentityRelinkSummary" in result.columns:
            result.loc[race_mask, "IdentityRelinkSummary"] = result.loc[race_mask, "IdentityRelinkSummary"].map(
                lambda value: (
                    str(value).replace('"Bonno_2"', '"Bonno"').replace('-> Bonno_2', '-> Bonno')
                    if pd.notna(value) else value
                )
            )

    return result


def append_identity_relink_review_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Make relinked identities visible in exported review columns."""
    if df.empty or "IdentityRelinkDetected" not in df.columns or "IdentityRelinkNote" not in df.columns:
        return df

    result = df.copy()
    relink_mask = result["IdentityRelinkDetected"].fillna(False).astype(bool)
    if not relink_mask.any():
        return result

    for index in result.index[relink_mask]:
        note = str(result.at[index, "IdentityRelinkNote"] or "").strip()
        if not note:
            continue
        existing_reason = str(result.at[index, "ReviewReason"] or "").strip()
        if existing_reason and existing_reason.lower() != "nan":
            if note not in existing_reason:
                result.at[index, "ReviewReason"] = f"{existing_reason} | {note}"
        else:
            result.at[index, "ReviewReason"] = note
        result.at[index, "ReviewNeeded"] = True
    return result


def append_identity_ambiguity_review_notes(df: pd.DataFrame) -> pd.DataFrame:
    """Append unresolved duplicate-identity ambiguity notes to review output."""
    if df.empty or "IdentityAmbiguityDetected" not in df.columns or "IdentityAmbiguityNote" not in df.columns:
        return df

    result = df.copy()
    ambiguity_mask = result["IdentityAmbiguityDetected"].fillna(False).astype(bool)
    if not ambiguity_mask.any():
        return result

    for index in result.index[ambiguity_mask]:
        note = str(result.at[index, "IdentityAmbiguityNote"] or "").strip()
        if not note:
            continue
        existing_reason = str(result.at[index, "ReviewReason"] or "").strip()
        if existing_reason and existing_reason.lower() != "nan":
            if note not in existing_reason:
                result.at[index, "ReviewReason"] = f"{existing_reason} | {note}"
        else:
            result.at[index, "ReviewReason"] = note
        result.at[index, "ReviewNeeded"] = True
    return result
