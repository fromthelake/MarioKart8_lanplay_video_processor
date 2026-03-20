import difflib
import re
from collections import Counter, defaultdict

import pandas as pd

from .extract_common import debug_identity_workbook_path
import textdistance
from jellyfish import soundex

from .name_unicode import (
    allowed_name_char_ratio,
    collapse_name_whitespace,
    distinct_visible_name_count,
    normalize_name_key,
)


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


def _build_match_candidates(identity_state, race_rows):
    candidates = []
    for row_position, (row_index, row) in enumerate(race_rows, start=1):
        row_name = str(row["PlayerName"] or "")
        row_character_index = row.get("CharacterIndex")
        row_detected_total = row.get("DetectedTotalScore")
        for identity_id, identity in identity_state.items():
            name_similarity = weighted_similarity(identity["canonical_name"], row_name)
            character_similarity = _character_similarity(identity["character_index"], row_character_index)
            total_similarity = _detected_total_similarity(identity["last_detected_total"], row_detected_total)
            combined_score = (name_similarity * 0.55) + (character_similarity * 0.30) + (total_similarity * 0.15)
            if name_similarity >= 0.72 or preprocess_name(identity["canonical_name"]) == preprocess_name(row_name):
                candidates.append((combined_score, name_similarity, identity_id, row_position, row_index))
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
                int(identity_state[identity_id]["character_index"])
                if pd.notna(identity_state[identity_id]["character_index"]) else 9999,
                int(identity_state[identity_id]["first_race"]),
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
        identity_state = {}
        row_identity_assignments = {}
        identity_debug_rows = []

        for race_id, race_rows_df in group.groupby("RaceIDNumber", sort=True):
            race_rows = list(race_rows_df.iterrows())
            candidates = _build_match_candidates(identity_state, race_rows)
            used_identity_ids = set()
            used_row_indices = set()

            for combined_score, name_similarity, identity_id, _row_position, row_index in candidates:
                if identity_id in used_identity_ids or row_index in used_row_indices:
                    continue
                row = group.loc[row_index]
                resolution_method = "name_only"
                if pd.notna(row.get("CharacterIndex")) and pd.notna(identity_state[identity_id]["character_index"]):
                    resolution_method = "name+character"
                elif pd.notna(row.get("DetectedTotalScore")) and pd.notna(identity_state[identity_id]["last_detected_total"]):
                    resolution_method = "name+total_hint"
                row_identity_assignments[row_index] = {
                    "identity_id": identity_id,
                    "resolution_method": resolution_method,
                    "match_score": round(float(combined_score), 3),
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
                        "last_detected_total": row.get("DetectedTotalScore"),
                    }

                identity = identity_state.setdefault(
                    identity_id,
                    {
                        "canonical_name": str(row["PlayerName"] or ""),
                        "name_history": [],
                        "character_index": row.get("CharacterIndex"),
                        "first_race": int(race_id),
                        "last_detected_total": row.get("DetectedTotalScore"),
                    },
                )
                identity["name_history"].append(str(row["PlayerName"] or ""))
                if pd.notna(row.get("CharacterIndex")):
                    identity["character_index"] = row.get("CharacterIndex")
                if pd.notna(row.get("DetectedTotalScore")):
                    identity["last_detected_total"] = row.get("DetectedTotalScore")
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
