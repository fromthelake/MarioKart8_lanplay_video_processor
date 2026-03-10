import time
from datetime import datetime
from pathlib import Path

import pandas as pd


POSITION_TEMPLATE_COEFF_COLUMN_MAP = {
    f"PositionTemplate{template_index:02}_Coeff": f"Position Template {template_index:02} Coeff"
    for template_index in range(1, 13)
}


USER_EXPORT_COLUMN_MAP = {
    "RaceClass": "Video",
    "RaceIDNumber": "Race",
    "TrackName": "Track",
    "CupName": "Cup",
    "RacePosition": "Position",
    "FixPlayerName": "Player",
    "Character": "Character",
    "RacePoints": "Race Points",
    "OldTotalScore": "Total Before Race",
    "NewTotalScore": "Total After Race",
    "PositionAfterRace": "Position After Race",
    "ReviewNeeded": "Needs Review",
    "ReviewReason": "Review Reason",
}


DEBUG_EXPORT_COLUMN_MAP = {
    "RaceClass": "Video",
    "RaceIDNumber": "Race",
    "TrackName": "Track",
    "TrackID": "Track ID",
    "CupName": "Cup",
    "RacePosition": "Position",
    "PlayerName": "Raw Player OCR",
    "FixPlayerName": "Standardized Player",
    "IdentityLabel": "Identity Label",
    "IdentityResolutionMethod": "Identity Resolution Method",
    "Character": "Character",
    "CharacterIndex": "Character Index",
    "CharacterMatchConfidence": "Character Match Confidence",
    "CharacterMatchMethod": "Character Match Method",
    "RacePoints": "Race Points",
    "DetectedRacePoints": "OCR Race Points",
    "DetectedTotalScore": "OCR Total Score",
    "SessionOldTotalScore": "Session Total Before Race",
    "SessionNewTotalScore": "Expected Total After Race",
    "OldTotalScore": "Tournament Total Before Race",
    "NewTotalScore": "Tournament Total After Race",
    "PositionAfterRace": "Position After Race",
    **POSITION_TEMPLATE_COEFF_COLUMN_MAP,
    "SessionIndex": "Session",
    "SessionRebased": "Session Rebased",
    "SessionRebaseReason": "Session Rebase Reason",
    "SessionResetDetected": "Session Reset Detected",
    "SessionResetReason": "Session Reset Reason",
    "NameConfidence": "Name Confidence",
    "DigitConsensus": "Digit Confidence",
    "RowCountConfidence": "Player Count Confidence",
    "RaceScorePlayerCount": "Players On Race Score Screen",
    "TotalScorePlayerCount": "Players On Total Score Screen",
    "LegacyRaceScorePlayerCount": "Legacy Players On Race Score Screen",
    "LegacyTotalScorePlayerCount": "Legacy Players On Total Score Screen",
    "LegacyRowCountConfidence": "Legacy Player Count Confidence",
    "RaceScoreCountVotes": "Race Score Count Votes",
    "TotalScoreCountVotes": "Total Score Count Votes",
    "LegacyRaceScoreCountVotes": "Legacy Race Score Count Votes",
    "LegacyTotalScoreCountVotes": "Legacy Total Score Count Votes",
    "RaceScoreRowSignals": "Race Score Row Signals",
    "TotalScoreRowSignals": "Total Score Row Signals",
    "RaceScoreRecoveryUsed": "RaceScore Recovery Used",
    "RaceScoreRecoverySource": "RaceScore Recovery Source",
    "RaceScoreRecoveryCount": "RaceScore Recovery Count",
    "TotalScoreMappingMethod": "Total Score Match Method",
    "ScoreValidationStatus": "Validation Status",
    "ReviewNeeded": "Needs Review",
    "ReviewReason": "Review Reason",
}


def build_user_export_df(df):
    ordered_df = df[list(USER_EXPORT_COLUMN_MAP.keys())].copy()
    return ordered_df.rename(columns=USER_EXPORT_COLUMN_MAP)


def build_debug_export_df(df):
    ordered_df = df[list(DEBUG_EXPORT_COLUMN_MAP.keys())].copy()
    return ordered_df.rename(columns=DEBUG_EXPORT_COLUMN_MAP)


def write_results_workbooks(df, folder_path):
    """Write a clean user workbook and a full debug workbook, both timestamped."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(folder_path).resolve().parent
    debug_output_dir = output_dir / "Debug"
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    user_df = build_user_export_df(df)
    debug_df = build_debug_export_df(df)

    output_excel_path = output_dir / f"{timestamp}_Tournament_Results.xlsx"
    debug_output_excel_path = debug_output_dir / f"{timestamp}_Tournament_Results_Debug.xlsx"

    with pd.ExcelWriter(output_excel_path) as writer:
        user_df.to_excel(writer, index=False, sheet_name="Results")
    with pd.ExcelWriter(debug_output_excel_path) as writer:
        debug_df.to_excel(writer, index=False, sheet_name="Debug Results")
    return {
        "user_df": user_df,
        "debug_df": debug_df,
        "output_excel_path": output_excel_path,
        "debug_output_excel_path": debug_output_excel_path,
    }


def build_player_count_summary_lines(df, build_race_warning_messages, pluralize):
    """Build the human-readable OCR completion summary shown in the console."""
    per_video_summary = {}
    lines = ["", "Per-video player count summary"]
    for race_class, race_group in df.groupby("RaceClass", sort=False):
        race_count_for_class = int(race_group["RaceIDNumber"].nunique())
        player_count_distribution = race_group.groupby("RaceIDNumber").size().value_counts().sort_index(ascending=False)
        dominant_players = int(race_group.groupby("RaceIDNumber").size().mode().iloc[0])
        review_row_count = int(race_group["ReviewNeeded"].sum())
        review_race_count = int(race_group.loc[race_group["ReviewNeeded"], "RaceIDNumber"].nunique())
        inconsistent_races = []
        for race_id, race_rows in race_group.groupby("RaceIDNumber", sort=True):
            race_score_players = int(race_rows["RaceScorePlayerCount"].iloc[0])
            total_score_players = int(race_rows["TotalScorePlayerCount"].iloc[0])
            track_name = str(race_rows["TrackName"].iloc[0])
            messages = build_race_warning_messages(
                dominant_players,
                race_score_players,
                total_score_players,
                float(race_rows["RowCountConfidence"].iloc[0]),
            )
            if messages:
                inconsistent_races.append((int(race_id), track_name, messages))

        per_video_summary[race_class] = {
            "race_count": race_count_for_class,
            "dominant_players": dominant_players,
            "review_row_count": review_row_count,
            "review_race_count": review_race_count,
            "player_count_distribution": {int(player_count): int(count) for player_count, count in player_count_distribution.items()},
        }

        if not inconsistent_races:
            lines.append(
                f"- {race_class}: {race_count_for_class} {pluralize(race_count_for_class, 'race')} | "
                f"Player count was consistent ({dominant_players} players)"
            )
            continue

        distribution_text = ", ".join(f"{player_count} players x {count}" for player_count, count in player_count_distribution.items())
        lines.append(f"- {race_class}: {race_count_for_class} {pluralize(race_count_for_class, 'race')} | Player count was not consistent")
        lines.append(f"  Most races showed {dominant_players} players")
        lines.append(f"  Summary: {distribution_text}")
        lines.append("  Please review these races:")
        for race_id, track_name, messages in inconsistent_races:
            for message in messages:
                lines.append(f"  - Race {race_id:03} | Track: {track_name} | {message}")

    return lines, per_video_summary


def build_completion_payload(df, folder_path, phase_start_time, progress_peak_lines, ocr_profiler_lines,
                             per_video_durations, build_race_warning_messages, pluralize, format_duration):
    """Prepare workbook output and the final OCR summary payload in one place."""
    workbook_payload = write_results_workbooks(df, folder_path)
    race_count = int(df[["RaceClass", "RaceIDNumber"]].drop_duplicates().shape[0])

    lines = [f"Duration: {format_duration(time.time() - phase_start_time)}", f"Races processed: {race_count}"]
    lines.extend(progress_peak_lines)
    lines.extend(["", "OCR call profile"])
    lines.extend(ocr_profiler_lines)
    summary_lines, per_video_summary = build_player_count_summary_lines(df, build_race_warning_messages, pluralize)
    lines.extend(summary_lines)
    lines.extend(
        [
            "",
            "Output workbooks:",
            str(workbook_payload["output_excel_path"]),
            str(workbook_payload["debug_output_excel_path"]),
        ]
    )

    return {
        "user_df": workbook_payload["user_df"],
        "debug_df": workbook_payload["debug_df"],
        "lines": lines,
        "output_excel_path": str(workbook_payload["output_excel_path"]),
        "debug_output_excel_path": str(workbook_payload["debug_output_excel_path"]),
        "race_count": race_count,
        "per_video_summary": per_video_summary,
        "per_video_durations": dict(per_video_durations),
        "duration_s": time.time() - phase_start_time,
    }
