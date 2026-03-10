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


def match_names(previous_names_with_indices, current_names):
    """Map the current race names onto prior rows so the same player stays on one row."""
    matches = {}
    used_previous_names = set()
    used_current_names = set()
    exact_match_rows = set()

    previous_names = [name for name, _ in previous_names_with_indices]

    for curr_name in current_names:
        if curr_name in previous_names:
            idx = previous_names.index(curr_name)
            row_idx = previous_names_with_indices[idx][1]
            matches[curr_name] = (curr_name, row_idx)
            used_previous_names.add(curr_name)
            used_current_names.add(curr_name)
            exact_match_rows.add(row_idx)

    similarity_scores = []
    for curr_name in current_names:
        if curr_name in used_current_names:
            continue
        for prev_name, prev_idx in previous_names_with_indices:
            if prev_name in used_previous_names or prev_idx in exact_match_rows:
                continue
            score = weighted_similarity(prev_name, curr_name)
            similarity_scores.append((score, prev_name, curr_name, prev_idx))

    similarity_scores.sort(reverse=True, key=lambda item: item[0])
    for _score, prev_name, curr_name, prev_idx in similarity_scores:
        if curr_name not in used_current_names and prev_name not in used_previous_names:
            matches[curr_name] = (prev_name, prev_idx)
            used_current_names.add(curr_name)
            used_previous_names.add(prev_name)

    remaining_current_names = [name for name in current_names if name not in used_current_names]
    remaining_previous_names = [name for name, _idx in previous_names_with_indices if name not in used_previous_names]

    for curr_name, prev_name in zip(remaining_current_names, remaining_previous_names):
        prev_idx = previous_names.index(prev_name)
        matches[curr_name] = (prev_name, prev_idx)

    for curr_name in remaining_current_names[len(remaining_previous_names):]:
        matches[curr_name] = (curr_name, None)

    return matches


def build_name_links(df, output_folder, write_debug_linking_excel=False):
    """Build per-race-class row histories so later races can inherit stable player rows."""
    name_links = defaultdict(list)
    all_player_names_df = {}

    for race_class, group in df.groupby("RaceClass"):
        races = sorted(group["RaceIDNumber"].unique())
        num_races = len(races)
        max_players = group.groupby("RaceIDNumber").size().max()
        player_names_df = pd.DataFrame(index=range(max_players), columns=range(num_races))

        initial_race_id = races[0]
        initial_names = group[group["RaceIDNumber"] == initial_race_id]["PlayerName"].tolist()
        player_names_df[0] = initial_names + [None] * (len(player_names_df) - len(initial_names))

        if num_races < 2:
            single_race_df = group[["PlayerName", "RaceIDNumber"]].copy()
            single_race_df.columns = ["PlayerName", 0]
            single_race_df = single_race_df.set_index("PlayerName").T
            all_player_names_df[race_class] = single_race_df
            continue

        for col_idx, race_id in enumerate(races[1:], start=1):
            current_race_names = group[group["RaceIDNumber"] == race_id]["PlayerName"].tolist()
            all_previous_names_with_indices = [
                (player_names_df.at[row_idx, prev_col], row_idx)
                for prev_col in range(col_idx)
                for row_idx in range(max_players)
                if pd.notna(player_names_df.at[row_idx, prev_col])
            ]
            matches = match_names(all_previous_names_with_indices, current_race_names)

            used_rows_exact = set()
            used_rows_similarity = set()

            for curr_name in current_race_names:
                if curr_name not in matches:
                    continue
                _matched_name, row_idx = matches[curr_name]
                if row_idx is not None and row_idx not in used_rows_exact:
                    player_names_df.at[row_idx, col_idx] = curr_name
                    used_rows_exact.add(row_idx)

            for curr_name in current_race_names:
                if curr_name not in matches or curr_name in player_names_df.iloc[:, col_idx].dropna().values:
                    continue
                _matched_name, row_idx = matches[curr_name]
                if row_idx is not None and row_idx not in used_rows_exact and row_idx not in used_rows_similarity:
                    for candidate_row in range(max_players):
                        if candidate_row not in used_rows_exact and candidate_row not in used_rows_similarity and pd.isna(player_names_df.at[candidate_row, col_idx]):
                            player_names_df.at[candidate_row, col_idx] = curr_name
                            used_rows_similarity.add(candidate_row)
                            break

            used_names = set(player_names_df.iloc[:, col_idx].dropna().values)
            for name in current_race_names:
                if name in used_names:
                    continue
                for candidate_row in range(max_players):
                    if candidate_row not in used_rows_exact and candidate_row not in used_rows_similarity and pd.isna(player_names_df.at[candidate_row, col_idx]):
                        player_names_df.at[candidate_row, col_idx] = name
                        used_names.add(name)
                        used_rows_similarity.add(candidate_row)
                        break

        for idx in range(max_players):
            name_link = player_names_df.loc[idx].dropna().tolist()
            if name_link:
                most_common_name = Counter(name_link).most_common(1)[0][0]
                for name in name_link:
                    name_links[name].append(most_common_name)

        if write_debug_linking_excel:
            output_path = f"{output_folder}/linking_{race_class}.xlsx"
            player_names_df.to_excel(output_path, index=False)

        all_player_names_df[race_class] = player_names_df

    return name_links, all_player_names_df


def normalize_name_for_vote(name: str) -> str:
    text = "" if name is None else str(name)
    return re.sub(r"\s+", " ", text.strip())


def choose_canonical_name(name_link, group):
    """Choose the final spelling for one linked player row using all OCR evidence."""
    candidates = [normalize_name_for_vote(name) for name in name_link if normalize_name_for_vote(name)]
    if not candidates:
        return ""

    candidate_counts = Counter(candidates)
    confidence_lookup = defaultdict(list)
    for candidate in candidates:
        matching_rows = group[group["PlayerName"] == candidate]
        if not matching_rows.empty:
            confidence_lookup[candidate].extend(matching_rows["NameConfidence"].tolist())

    best_name = candidates[0]
    best_score = float("-inf")
    for candidate in sorted(candidate_counts):
        support_score = 0.0
        for observed_name in candidates:
            observed_confidences = confidence_lookup.get(observed_name, [])
            average_confidence = sum(observed_confidences) / max(1, len(observed_confidences))
            weight = 1.0 + average_confidence / 100.0
            support_score += weighted_similarity(candidate, observed_name) * weight
        quality = len(set(re.sub(r"[^a-zA-Z0-9]", "", candidate)))
        confidence_bonus = sum(confidence_lookup.get(candidate, [0])) / max(1, len(confidence_lookup.get(candidate, [])))
        score = support_score + candidate_counts[candidate] * 2.5 + quality * 0.3 + confidence_bonus / 100.0
        if score > best_score:
            best_score = score
            best_name = candidate
    return best_name


def standardize_names(player_names_df, group):
    """Pick one canonical spelling per linked row."""
    name_mapping = {}
    standardized_names = {}

    for idx in range(len(player_names_df)):
        name_link = player_names_df.loc[idx].dropna().tolist()
        if not name_link:
            continue
        canonical_name = choose_canonical_name(name_link, group)
        for name in name_link:
            name_mapping[name] = (canonical_name, idx)
            standardized_names[idx] = canonical_name

    return name_mapping, standardized_names


def standardize_player_names(df, output_folder, write_debug_linking_excel=False):
    """Rewrite noisy OCR player names into stable names within each video/race class."""
    standardized_names = pd.DataFrame()
    standardized_names_dict = {}
    _name_links, all_player_names_df = build_name_links(df, output_folder, write_debug_linking_excel)

    for race_class, player_names_df in all_player_names_df.items():
        if player_names_df.shape[1] < 2:
            group = df[df["RaceClass"] == race_class].copy()
            group.loc[:, "FixPlayerName"] = group["PlayerName"]
            standardized_names = pd.concat([standardized_names, group], ignore_index=True)
            continue

        group = df[df["RaceClass"] == race_class].copy()
        local_name_mapping, local_standardized_names_dict = standardize_names(player_names_df, group)
        standardized_names_dict.update(local_standardized_names_dict)

        def get_standardized_name(row):
            player_name = row["PlayerName"]
            if player_name in local_name_mapping:
                return standardized_names_dict[local_name_mapping[player_name][1]]
            return player_name

        group.loc[:, "FixPlayerName"] = group.apply(get_standardized_name, axis=1)
        standardized_names = pd.concat([standardized_names, group], ignore_index=True)

    return standardized_names
