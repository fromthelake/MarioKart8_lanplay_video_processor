from collections import defaultdict

import numpy as np
import pandas as pd


VALIDATION_STATUS_LABELS = {
    "computed_only": "Computed only",
    "race_points_match": "Race points matched OCR",
    "validated": "Validated against OCR total score",
    "race_points_mismatch": "Race points mismatch",
    "total_score_mismatch": "Total score mismatch",
}


def format_validation_status(status_code: str) -> str:
    return VALIDATION_STATUS_LABELS.get(str(status_code), str(status_code).replace("_", " ").capitalize())


def build_review_reason_messages(
    review_reason_codes,
    *,
    race_points,
    detected_race,
    expected_session_total,
    detected_total,
    name_confidence,
    digit_consensus,
    row_count_confidence,
    expected_players,
    race_score_players,
    total_score_players,
):
    messages = []
    for code in review_reason_codes:
        if code == "race_points_mismatch":
            messages.append(
                f"Race points should be {race_points}, but OCR read {detected_race if detected_race is not None else 'nothing'}."
            )
        elif code == "total_score_mismatch":
            messages.append(
                f"Expected total after race is {expected_session_total}, but OCR total score read {detected_total if detected_total is not None else 'nothing'}."
            )
        elif code == "low_name_confidence":
            messages.append(f"Player name confidence is low ({name_confidence:.0f}%).")
        elif code == "low_digit_consensus":
            messages.append(f"Digit confidence is low ({digit_consensus:.0f}%).")
        elif code == "unstable_row_count":
            messages.append(f"Player count confidence is low ({row_count_confidence:.0f}%).")
        elif code == "player_count_mismatch":
            messages.append(
                f"Race score screen shows {race_score_players} players, expected {expected_players}."
            )
        elif code == "race_total_player_count_mismatch":
            messages.append(
                f"Race score screen shows {race_score_players} players, but total score screen shows {total_score_players}."
            )
        else:
            messages.append(str(code).replace("_", " ").capitalize())
    return messages


def should_start_new_session(session_totals, detected_totals) -> bool:
    """Detect when a recording likely restarted a fresh tournament session."""
    parsed_detected_totals = [int(value) for value in detected_totals if pd.notna(value)]
    if not parsed_detected_totals:
        return False

    previous_totals = [int(value) for value in session_totals.values() if int(value) > 0]
    if len(previous_totals) < max(4, len(parsed_detected_totals) // 3):
        return False

    previous_max = max(previous_totals)
    previous_median = float(np.median(previous_totals))
    current_max = max(parsed_detected_totals)
    current_median = float(np.median(parsed_detected_totals))
    low_total_count = sum(1 for value in parsed_detected_totals if value <= 20)

    enough_history = previous_max >= 30 and previous_median >= 20
    broad_drop = low_total_count >= max(3, len(parsed_detected_totals) // 2)
    lower_than_previous = current_max <= previous_max * 0.55 and current_median <= previous_median * 0.6
    return enough_history and broad_drop and lower_than_previous


def apply_session_validation(df, parse_detected_int, exact_total_score_fallback):
    """Compute running totals, session boundaries, and review flags for OCR rows."""
    df = df.copy()
    df["ReviewNeeded"] = False
    df["SessionIndex"] = 1
    df["SessionOldTotalScore"] = 0
    df["SessionNewTotalScore"] = 0
    df["OldTotalScore"] = 0
    df["NewTotalScore"] = 0
    df["ScoreValidationStatus"] = "computed_only"

    for race_class, race_group in df.groupby("RaceClass", sort=False):
        # One source recording can contain several back-to-back sessions, so we keep
        # both a whole-video total and a resettable per-session total.
        tournament_totals = defaultdict(int)
        session_totals = defaultdict(int)
        session_index = 1
        race_ids = sorted(race_group["RaceIDNumber"].unique())
        expected_players = int(race_group.groupby("RaceIDNumber").size().mode().iloc[0])

        for race_id in race_ids:
            race_mask = (df["RaceClass"] == race_class) & (df["RaceIDNumber"] == race_id)
            race_rows = df[race_mask].sort_values("RacePosition")
            detected_totals = [value for value in race_rows["DetectedTotalScore"].tolist() if pd.notna(value)]
            if race_id != race_ids[0] and should_start_new_session(session_totals, detected_totals):
                session_index += 1
                session_totals = defaultdict(int)

            prepared_rows = []
            for index, row in race_rows.iterrows():
                player_key = row["FixPlayerName"]
                old_total = tournament_totals[player_key]
                session_old_total = session_totals[player_key]
                race_points = int(row["RacePoints"])
                prepared_rows.append(
                    {
                        "index": index,
                        "player_key": player_key,
                        "old_total": old_total,
                        "session_old_total": session_old_total,
                        "race_points": race_points,
                        "new_total": old_total + race_points,
                        "session_new_total": session_old_total + race_points,
                        "detected_race": parse_detected_int(row["DetectedRacePoints"]),
                        "detected_total": parse_detected_int(row["DetectedTotalScore"]),
                    }
                )

            remapped_totals_by_index = exact_total_score_fallback(prepared_rows)

            for prepared_row, (_, row) in zip(prepared_rows, race_rows.iterrows()):
                index = prepared_row["index"]
                player_key = prepared_row["player_key"]
                old_total = prepared_row["old_total"]
                session_old_total = prepared_row["session_old_total"]
                race_points = prepared_row["race_points"]
                new_total = prepared_row["new_total"]
                session_new_total = prepared_row["session_new_total"]
                detected_race = prepared_row["detected_race"]
                detected_total = remapped_totals_by_index.get(index, prepared_row["detected_total"])
                review_reasons = [reason for reason in str(row["ReviewReason"]).split(";") if reason]
                score_status = "computed_only"

                if detected_race is not None:
                    if detected_race == race_points:
                        score_status = "race_points_match"
                    else:
                        review_reasons.append("race_points_mismatch")
                        score_status = "race_points_mismatch"

                if detected_total is not None:
                    if detected_total == session_new_total:
                        score_status = "validated"
                    else:
                        review_reasons.append("total_score_mismatch")
                        score_status = "total_score_mismatch"

                if row["NameConfidence"] < 45:
                    review_reasons.append("low_name_confidence")
                if row["DigitConsensus"] < 55:
                    review_reasons.append("low_digit_consensus")
                if row["RowCountConfidence"] < 60:
                    review_reasons.append("unstable_row_count")
                if int(row["RaceScorePlayerCount"]) != expected_players:
                    review_reasons.append("player_count_mismatch")
                if int(row["TotalScorePlayerCount"]) != int(row["RaceScorePlayerCount"]):
                    review_reasons.append("race_total_player_count_mismatch")

                review_reasons = sorted(set(review_reasons))
                review_reason_messages = build_review_reason_messages(
                    review_reasons,
                    race_points=race_points,
                    detected_race=detected_race,
                    expected_session_total=session_new_total,
                    detected_total=detected_total,
                    name_confidence=float(row["NameConfidence"]),
                    digit_consensus=float(row["DigitConsensus"]),
                    row_count_confidence=float(row["RowCountConfidence"]),
                    expected_players=expected_players,
                    race_score_players=int(row["RaceScorePlayerCount"]),
                    total_score_players=int(row["TotalScorePlayerCount"]),
                )
                df.at[index, "SessionIndex"] = session_index
                df.at[index, "SessionOldTotalScore"] = session_old_total
                df.at[index, "SessionNewTotalScore"] = session_new_total
                df.at[index, "OldTotalScore"] = old_total
                df.at[index, "NewTotalScore"] = new_total
                if index in remapped_totals_by_index:
                    df.at[index, "DetectedTotalScore"] = detected_total
                    existing_mapping_method = str(df.at[index, "TotalScoreMappingMethod"]).strip()
                    df.at[index, "TotalScoreMappingMethod"] = (
                        f"{existing_mapping_method}+score_fallback" if existing_mapping_method else "score_fallback"
                    )
                df.at[index, "ReviewReason"] = " | ".join(review_reason_messages)
                df.at[index, "ReviewNeeded"] = bool(review_reasons)
                df.at[index, "ScoreValidationStatus"] = format_validation_status(score_status)

                tournament_totals[player_key] = new_total
                session_totals[player_key] = session_new_total

    return df
