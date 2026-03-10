import difflib
import re
from collections import Counter, defaultdict

import pandas as pd
import textdistance
from jellyfish import soundex


def preprocess_name(name):
    """Normalize OCR names before comparing them across races."""
    text = "" if name is None else str(name)
    text = re.sub(r"\W+", " ", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def soundex_similarity(name1, name2):
    """Use Soundex as a weak phonetic tie-breaker for noisy OCR names."""
    return 1 if soundex(name1) == soundex(name2) else 0


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
    text = "" if name is None else str(name)
    return re.sub(r"\s+", " ", text.strip())


def choose_canonical_name(name_history):
    """Choose the cleanest spelling seen for one tracked identity."""
    candidates = [normalize_name_for_vote(name) for name in name_history if normalize_name_for_vote(name)]
    if not candidates:
        return ""

    counts = Counter(candidates)
    best_name = candidates[0]
    best_score = float("-inf")
    for candidate, count in counts.items():
        quality = len(set(re.sub(r"[^a-zA-Z0-9]", "", candidate)))
        support_score = 0.0
        for observed_name in candidates:
            support_score += weighted_similarity(candidate, observed_name)
        score = (count * 3.0) + support_score + (quality * 0.25)
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
    debug_df.to_excel(f"{output_folder}/identity_linking_{race_class}.xlsx", index=False)


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
