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


def _parse_review_reason_codes(value) -> list[str]:
    if isinstance(value, list):
        return _parse_review_reason_codes(";".join(str(item) for item in value))
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    parsed_codes = []
    for token in str(value).split(";"):
        normalized = token.strip()
        if not normalized or normalized.lower() == "nan":
            continue
        if "|" in normalized or " " in normalized or "." in normalized:
            continue
        parsed_codes.append(normalized)
    return list(dict.fromkeys(parsed_codes))


def build_review_reason_messages(
    review_reason_codes,
    *,
    race_points,
    detected_race,
    expected_session_total,
    detected_total,
    detected_race_source,
    detected_total_source,
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
                f"Race points mismatch: expected {race_points}, OCR {detected_race if detected_race is not None else 'none'}."
            )
        elif code == "race_points_out_of_range":
            messages.append(
                f"Race points out of range: OCR {detected_race if detected_race is not None else 'none'}."
            )
        elif code == "total_score_mismatch":
            messages.append(
                f"Total mismatch: expected {expected_session_total}, OCR {detected_total if detected_total is not None else 'none'}."
            )
        elif code == "session_rebased":
            messages.append("Session rebased from this race.")
        elif code == "connection_reset":
            messages.append(
                str(
                    session_reset_reason
                    or "Connection reset detected; tournament totals continue from prior standings."
                )
            )
        elif code == "low_name_confidence":
            messages.append(f"Low name confidence ({name_confidence:.0f}%).")
        elif code == "low_digit_consensus":
            messages.append(f"Low digit confidence ({digit_consensus:.0f}%).")
        elif code == "unstable_row_count":
            messages.append(f"Low player-count confidence ({row_count_confidence:.0f}%).")
        elif code == "player_count_mismatch":
            messages.append(
                f"Race screen player count mismatch: {race_score_players} shown, expected {expected_players}."
            )
        elif code == "race_total_player_count_mismatch":
            messages.append(
                f"Race/total player counts differ: race {race_score_players}, total {total_score_players}."
            )
        elif code == "total_score_order_violation":
            messages.append("Scoreboard total order is not descending.")
        elif code == "total_score_out_of_range":
            messages.append(
                f"Total score out of range: OCR {detected_total if detected_total is not None else 'none'}."
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


def detect_initial_old_total_baseline(prepared_rows, race_index: int) -> bool:
    """Detect when the first visible race already has prior totals on-screen.

    This covers recordings that start after race 1's track intro, where the first
    exported race still has valid RaceScore/TotalScore screens but should not start
    tournament totals from zero.
    """
    if race_index != 0:
        return False

    rows_with_old_totals = [
        row for row in prepared_rows
        if row.get("detected_old_total") is not None and int(row.get("detected_old_total", 0)) > 0
    ]
    if len(rows_with_old_totals) < max(4, len(prepared_rows) // 2):
        return False

    consistent_rows = []
    materially_positive_rows = []
    for row in rows_with_old_totals:
        detected_old_total = int(row["detected_old_total"])
        detected_total = row.get("detected_total")
        race_points = int(row["race_points"])
        if detected_total is not None and int(detected_total) == detected_old_total + race_points:
            consistent_rows.append(row)
        if detected_old_total >= 1:
            materially_positive_rows.append(row)

    required_rows = max(4, int(len(rows_with_old_totals) * 0.7))
    return (
        len(consistent_rows) >= required_rows
        and len(materially_positive_rows) >= required_rows
    )


def detect_connection_reset(previous_displayed_totals, prepared_rows) -> bool:
    """Detect a scoreboard reset where displayed totals drop across races."""
    if not previous_displayed_totals:
        return False

    rows_with_detected_totals = [row for row in prepared_rows if row["detected_total"] is not None]
    if len(rows_with_detected_totals) < max(4, len(prepared_rows) // 2):
        return False

    broad_drops = 0
    for row in rows_with_detected_totals:
        player_key = row["player_key"]
        previous_total = previous_displayed_totals.get(player_key)
        if previous_total is None:
            continue
        if int(row["detected_total"]) + max(2, int(row["race_points"])) < int(previous_total):
            broad_drops += 1
    return broad_drops >= max(4, int(len(rows_with_detected_totals) * 0.7))


def detect_obvious_total_score_reset(prepared_rows) -> bool:
    """Detect reset-like total-score screens from whole-race OCR patterns."""
    rows_with_detected_totals = [row for row in prepared_rows if row["detected_total"] is not None]
    if len(rows_with_detected_totals) < max(4, len(prepared_rows) // 2):
        return False

    total_rows = len(rows_with_detected_totals)
    mismatching_expected = 0
    race_points_matches = 0
    materially_higher_expected = 0

    for row in rows_with_detected_totals:
        detected_total = int(row["detected_total"])
        expected_total = int(row["session_new_total"])
        race_points = int(row["race_points"])
        if detected_total != expected_total:
            mismatching_expected += 1
        if detected_total == race_points:
            race_points_matches += 1
        if expected_total >= detected_total + max(3, race_points):
            materially_higher_expected += 1

    # A total-score OCR result that mostly equals the race-points table while
    # also disagreeing with the expected continuation totals is a strong reset
    # signature, even when the broad-drop heuristic misses a few low-total rows.
    return (
        mismatching_expected >= max(4, int(total_rows * 0.8))
        and race_points_matches >= max(4, int(total_rows * 0.6))
        and materially_higher_expected >= max(4, int(total_rows * 0.6))
    )


def detect_total_score_order_violations(race_rows: pd.DataFrame, remapped_totals_by_index: dict[int, int]) -> set[int]:
    """Flag OCR total-score rows that break the expected descending scoreboard order."""
    ordered_rows = []
    for index, row in race_rows.iterrows():
        scoreboard_position = row.get("DetectedPositionAfterRace", row.get("PositionAfterRace"))
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
    """Recompute post-race positions from authoritative totals.

    We assign a full ordered standing per race after validation:
    - higher ``NewTotalScore`` ranks earlier
    - equal totals share the same rank
    - on-screen race position and incoming row order only break display order within ties

    This keeps ``PositionAfterRace`` aligned with the final validated tournament
    totals and avoids stale OCR-only tie positions surviving after relinking.
    """
    df = df.copy()
    df["_position_rank_order"] = range(len(df))
    if "RacePosition" not in df.columns:
        df["RacePosition"] = pd.NA

    ranked_frames: list[pd.DataFrame] = []
    for (_race_class, _race_id), race_rows in df.groupby(["RaceClass", "RaceIDNumber"], sort=False):
        ordered_rows = race_rows.sort_values(
            ["NewTotalScore", "RacePosition", "_position_rank_order"],
            ascending=[False, True, True],
            kind="stable",
            na_position="last",
        ).copy()
        ordered_rows["PositionAfterRace"] = (
            ordered_rows["NewTotalScore"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
        ranked_frames.append(ordered_rows)

    ranked_df = pd.concat(ranked_frames, axis=0).sort_values("_position_rank_order", kind="stable")
    ranked_df = ranked_df.drop(columns=["_position_rank_order"])
    return ranked_df


def apply_session_validation(df, parse_detected_int, exact_total_score_fallback):
    """Compute running totals, session boundaries, and review flags for OCR rows."""
    df = df.copy()
    if "DetectedPositionAfterRace" not in df.columns:
        df["DetectedPositionAfterRace"] = df.get("PositionAfterRace")
    if "ReviewReasonCodes" not in df.columns:
        df["ReviewReasonCodes"] = df.get("ReviewReason", "").apply(_parse_review_reason_codes)
    else:
        df["ReviewReasonCodes"] = df["ReviewReasonCodes"].apply(_parse_review_reason_codes)
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
        previous_displayed_totals = {}

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
                        "detected_old_total": parse_detected_int(row.get("DetectedOldTotalScore")),
                        "detected_total": parse_detected_int(row["DetectedTotalScore"]),
                        "detected_race_source": str(row.get("DetectedRacePointsSource", "")),
                        "detected_old_total_source": str(row.get("DetectedOldTotalScoreSource", "")),
                        "detected_total_source": str(row.get("DetectedTotalScoreSource", "")),
                    }
                )

            session_rebased = (
                detect_initial_old_total_baseline(prepared_rows, race_index)
                or detect_rebase_candidate(prepared_rows, race_index)
            )
            session_rebase_reason = (
                "Session rebased from this race."
                if session_rebased else ""
            )
            session_reset_detected = (
                detect_connection_reset(previous_displayed_totals, prepared_rows)
                or detect_obvious_total_score_reset(prepared_rows)
            )
            session_reset_reason = (
                "Connection reset detected; tournament totals continue from prior standings."
                if session_reset_detected else ""
            )
            announce_connection_reset = bool(session_reset_detected)
            current_session_index = session_index + 1 if announce_connection_reset else session_index

            if session_rebased:
                for prepared_row in prepared_rows:
                    detected_old_total = prepared_row.get("detected_old_total")
                    detected_total = prepared_row.get("detected_total")
                    if detected_old_total is not None:
                        rebased_old_total = max(0, int(detected_old_total))
                        rebased_new_total = rebased_old_total + int(prepared_row["race_points"])
                    elif detected_total is not None:
                        rebased_old_total = max(0, int(detected_total) - int(prepared_row["race_points"]))
                        rebased_new_total = int(detected_total)
                    else:
                        continue
                    prepared_row["old_total"] = rebased_old_total
                    prepared_row["session_old_total"] = rebased_old_total
                    prepared_row["new_total"] = rebased_new_total
                    prepared_row["session_new_total"] = rebased_new_total

            if announce_connection_reset:
                for prepared_row in prepared_rows:
                    prepared_row["session_old_total"] = 0
                    prepared_row["session_new_total"] = int(prepared_row["race_points"])

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
                review_reasons = _parse_review_reason_codes(row.get("ReviewReasonCodes"))
                score_status = "computed_only"

                if session_rebased:
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

                if index in total_score_order_violations and detected_total is not None and detected_total != session_new_total:
                    review_reasons.append("total_score_order_violation")
                    if score_status in {"computed_only", "race_points_match", "validated", "rebased"}:
                        score_status = "total_score_order_violation"

                if announce_connection_reset:
                    review_reasons.append("connection_reset")
                    score_status = "connection_reset"

                is_low_res = bool(row.get("IsLowRes", False))
                if not is_low_res and row["NameConfidence"] < 45:
                    review_reasons.append("low_name_confidence")
                if not is_low_res and row["DigitConsensus"] < 55:
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
                    detected_race_source=prepared_row.get("detected_race_source", ""),
                    detected_total_source=prepared_row.get("detected_total_source", ""),
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
                df.at[index, "SessionIndex"] = current_session_index
                df.at[index, "SessionOldTotalScore"] = session_old_total
                df.at[index, "SessionNewTotalScore"] = session_new_total
                df.at[index, "OldTotalScore"] = old_total
                df.at[index, "NewTotalScore"] = new_total
                df.at[index, "SessionRebased"] = bool(session_rebased)
                df.at[index, "SessionRebaseReason"] = session_rebase_reason
                df.at[index, "SessionResetDetected"] = bool(announce_connection_reset)
                df.at[index, "SessionResetReason"] = session_reset_reason if announce_connection_reset else ""
                if index in remapped_totals_by_index:
                    df.at[index, "DetectedTotalScore"] = detected_total
                    existing_mapping_method = str(df.at[index, "TotalScoreMappingMethod"]).strip()
                    df.at[index, "TotalScoreMappingMethod"] = (
                        f"{existing_mapping_method}+score_fallback" if existing_mapping_method else "score_fallback"
                    )
                df.at[index, "ReviewReasonCodes"] = ";".join(review_reasons)
                df.at[index, "ReviewReason"] = " | ".join(review_reason_messages)
                df.at[index, "ReviewNeeded"] = bool(review_reasons)
                df.at[index, "ScoreValidationStatus"] = format_validation_status(score_status)

                tournament_totals[player_key] = new_total
                session_totals[player_key] = session_new_total
                previous_validated_totals[player_key] = new_total
                if detected_total is not None:
                    previous_displayed_totals[player_key] = int(detected_total)

            session_index = current_session_index

    return assign_shared_positions_after_race(df)
