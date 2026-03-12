from collections import defaultdict

import numpy as np
import pandas as pd


VALIDATION_STATUS_LABELS = {
    "computed_only": "Computed only",
    "race_points_match": "Race points matched OCR",
    "validated": "Validated against OCR total score",
    "race_points_mismatch": "Race points mismatch",
    "race_points_out_of_range": "Race points out of range",
    "total_score_mismatch": "Total score mismatch",
    "rebased": "Session rebased from OCR total score",
    "connection_reset": "Connection reset detected",
    "total_score_order_violation": "Total score order violation",
    "total_score_out_of_range": "Total score out of range",
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
    authoritative_total_before_race,
    authoritative_total_after_race,
    name_confidence,
    digit_consensus,
    row_count_confidence,
    expected_players,
    race_score_players,
    total_score_players,
    session_rebased,
    session_rebase_reason,
    session_reset_detected,
    session_reset_reason,
):
    messages = []
    for code in review_reason_codes:
        if code == "race_points_mismatch":
            messages.append(
                f"Race points should be {race_points}, but OCR read {detected_race if detected_race is not None else 'nothing'}."
            )
        elif code == "race_points_out_of_range":
            messages.append(
                f"OCR race points {detected_race if detected_race is not None else 'nothing'} are outside the expected range 1..15."
            )
        elif code == "total_score_mismatch":
            messages.append(
                f"Expected total after race is {expected_session_total}, but OCR total score read {detected_total if detected_total is not None else 'nothing'}."
            )
        elif code == "session_rebased":
            messages.append(str(session_rebase_reason or "Session rebased from this race because earlier footage is missing."))
        elif code == "connection_reset":
            messages.append(
                str(
                    session_reset_reason
                    or (
                        f"Connection reset detected. OCR total score read "
                        f"{detected_total if detected_total is not None else 'nothing'}, but tournament totals continue "
                        f"from {authoritative_total_before_race} to {authoritative_total_after_race}."
                    )
                )
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
        elif code == "total_score_order_violation":
            messages.append(
                "Total score order on the scoreboard is not monotonic from top to bottom."
            )
        elif code == "total_score_out_of_range":
            messages.append(
                f"OCR total score {detected_total if detected_total is not None else 'nothing'} is outside the expected range 1..999."
            )
        else:
            messages.append(str(code).replace("_", " ").capitalize())
    return messages


def detect_rebase_candidate(prepared_rows, race_index: int) -> bool:
    """Detect when the first visible race is already mid-session and must become the new base."""
    if race_index != 0:
        return False

    rows_with_detected_totals = [row for row in prepared_rows if row["detected_total"] is not None]
    if len(rows_with_detected_totals) < max(4, len(prepared_rows) // 2):
        return False

    mismatching_rows = [
        row
        for row in rows_with_detected_totals
        if int(row["detected_total"]) != int(row["session_new_total"])
    ]
    if len(mismatching_rows) < max(4, int(len(rows_with_detected_totals) * 0.8)):
        return False

    positive_offsets = [
        int(row["detected_total"]) - int(row["session_new_total"])
        for row in mismatching_rows
    ]
    median_offset = float(np.median(positive_offsets)) if positive_offsets else 0.0
    materially_higher_rows = [offset for offset in positive_offsets if offset >= 4]
    return median_offset >= 5.0 and len(materially_higher_rows) >= max(4, int(len(rows_with_detected_totals) * 0.6))


def detect_connection_reset(previous_validated_totals, prepared_rows) -> bool:
    """Detect a scoreboard reset where OCR totals drop, but tournament totals should keep running."""
    if not previous_validated_totals:
        return False

    rows_with_detected_totals = [row for row in prepared_rows if row["detected_total"] is not None]
    if len(rows_with_detected_totals) < max(4, len(prepared_rows) // 2):
        return False

    broad_drops = 0
    for row in rows_with_detected_totals:
        player_key = row["player_key"]
        previous_total = previous_validated_totals.get(player_key)
        if previous_total is None:
            continue
        if int(row["detected_total"]) + max(2, int(row["race_points"])) < int(previous_total):
            broad_drops += 1
    return broad_drops >= max(4, int(len(rows_with_detected_totals) * 0.7))


def detect_total_score_order_violations(race_rows: pd.DataFrame, remapped_totals_by_index: dict[int, int]) -> set[int]:
    """Flag OCR total-score rows that break the expected descending scoreboard order."""
    ordered_rows = []
    for index, row in race_rows.iterrows():
        scoreboard_position = row.get("PositionAfterRace")
        if pd.isna(scoreboard_position):
            continue
        try:
            scoreboard_position = int(scoreboard_position)
        except (TypeError, ValueError):
            continue
        detected_total = remapped_totals_by_index.get(index, row.get("DetectedTotalScore"))
        if pd.isna(detected_total):
            continue
        try:
            detected_total = int(detected_total)
        except (TypeError, ValueError):
            continue
        ordered_rows.append((scoreboard_position, index, detected_total))

    ordered_rows.sort(key=lambda item: item[0])
    violating_indices: set[int] = set()
    previous_entry = None
    for current_entry in ordered_rows:
        if previous_entry is not None:
            previous_total = int(previous_entry[2])
            current_total = int(current_entry[2])
            if current_total > previous_total:
                violating_indices.add(int(previous_entry[1]))
                violating_indices.add(int(current_entry[1]))
        previous_entry = current_entry
    return violating_indices


def assign_shared_positions_after_race(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute post-race positions from authoritative totals, allowing ties.

    We use competition ranking ("1224" style):
    - equal totals share the same position
    - the next lower total skips ahead by the number of tied rows above

    This keeps Position After Race aligned with the final validated tournament totals
    instead of the earlier OCR-only row mapping guess.
    """
    df = df.copy()
    ranked = (
        df.groupby(["RaceClass", "RaceIDNumber"], sort=False)["NewTotalScore"]
        .rank(method="min", ascending=False)
    )
    df["PositionAfterRace"] = ranked.astype("Int64")
    return df


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
    df["SessionRebased"] = False
    df["SessionRebaseReason"] = ""
    df["SessionResetDetected"] = False
    df["SessionResetReason"] = ""

    for race_class, race_group in df.groupby("RaceClass", sort=False):
        # One source recording can contain several back-to-back sessions, so we keep
        # both a whole-video total and a resettable per-session total.
        tournament_totals = defaultdict(int)
        session_totals = defaultdict(int)
        session_index = 1
        race_ids = sorted(race_group["RaceIDNumber"].unique())
        expected_players = int(race_group.groupby("RaceIDNumber").size().mode().iloc[0])
        previous_validated_totals = {}

        for race_index, race_id in enumerate(race_ids):
            race_mask = (df["RaceClass"] == race_class) & (df["RaceIDNumber"] == race_id)
            race_rows = df[race_mask].sort_values("RacePosition")

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

            session_rebased = detect_rebase_candidate(prepared_rows, race_index)
            session_rebase_reason = (
                "Session rebased from this race because earlier race footage is missing."
                if session_rebased else ""
            )
            session_reset_detected = detect_connection_reset(previous_validated_totals, prepared_rows)
            session_reset_reason = (
                "Connection reset detected. Tournament totals continue from the standings before the reset."
                if session_reset_detected else ""
            )

            if session_rebased:
                for prepared_row in prepared_rows:
                    if prepared_row["detected_total"] is None:
                        continue
                    rebased_old_total = max(0, int(prepared_row["detected_total"]) - int(prepared_row["race_points"]))
                    prepared_row["old_total"] = rebased_old_total
                    prepared_row["session_old_total"] = rebased_old_total
                    prepared_row["new_total"] = int(prepared_row["detected_total"])
                    prepared_row["session_new_total"] = int(prepared_row["detected_total"])

            remapped_totals_by_index = exact_total_score_fallback(prepared_rows)
            total_score_order_violations = detect_total_score_order_violations(race_rows, remapped_totals_by_index)

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
                existing_review_reason = row["ReviewReason"]
                if pd.isna(existing_review_reason):
                    review_reasons = []
                else:
                    review_reasons = [reason for reason in str(existing_review_reason).split(";") if reason and reason.lower() != "nan"]
                score_status = "computed_only"

                if session_rebased:
                    review_reasons.append("session_rebased")
                    score_status = "rebased"
                if detected_race is not None:
                    if not (1 <= int(detected_race) <= 15):
                        review_reasons.append("race_points_out_of_range")
                        score_status = "race_points_mismatch"
                    elif detected_race == race_points:
                        score_status = "race_points_match" if score_status == "computed_only" else score_status
                    else:
                        review_reasons.append("race_points_mismatch")
                        score_status = "race_points_mismatch"

                if detected_total is not None:
                    if not (1 <= int(detected_total) <= 999):
                        review_reasons.append("total_score_out_of_range")
                        score_status = "total_score_mismatch"
                    elif detected_total == session_new_total:
                        score_status = "validated" if score_status in {"computed_only", "race_points_match"} else score_status
                    else:
                        review_reasons.append("total_score_mismatch")
                        score_status = "total_score_mismatch"

                if index in total_score_order_violations:
                    review_reasons.append("total_score_order_violation")
                    if score_status in {"computed_only", "race_points_match", "validated", "rebased"}:
                        score_status = "total_score_order_violation"

                if session_reset_detected:
                    review_reasons.append("connection_reset")
                    score_status = "connection_reset"

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
                    authoritative_total_before_race=old_total,
                    authoritative_total_after_race=new_total,
                    name_confidence=float(row["NameConfidence"]),
                    digit_consensus=float(row["DigitConsensus"]),
                    row_count_confidence=float(row["RowCountConfidence"]),
                    expected_players=expected_players,
                    race_score_players=int(row["RaceScorePlayerCount"]),
                    total_score_players=int(row["TotalScorePlayerCount"]),
                    session_rebased=session_rebased,
                    session_rebase_reason=session_rebase_reason,
                    session_reset_detected=session_reset_detected,
                    session_reset_reason=session_reset_reason,
                )
                df.at[index, "SessionIndex"] = session_index
                df.at[index, "SessionOldTotalScore"] = session_old_total
                df.at[index, "SessionNewTotalScore"] = session_new_total
                df.at[index, "OldTotalScore"] = old_total
                df.at[index, "NewTotalScore"] = new_total
                df.at[index, "SessionRebased"] = bool(session_rebased)
                df.at[index, "SessionRebaseReason"] = session_rebase_reason
                df.at[index, "SessionResetDetected"] = bool(session_reset_detected)
                df.at[index, "SessionResetReason"] = session_reset_reason
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
                previous_validated_totals[player_key] = new_total

    return assign_shared_positions_after_race(df)
