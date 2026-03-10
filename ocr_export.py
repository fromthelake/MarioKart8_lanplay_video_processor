import shutil
import time
from datetime import datetime
from pathlib import Path


DESIRED_EXPORT_COLUMNS = [
    "RaceClass", "RaceIDNumber", "TrackName", "TrackID", "CupName", "RacePosition", "PlayerName",
    "FixPlayerName", "RacePoints", "DetectedRacePoints", "DetectedTotalScore",
    "RaceScorePlayerCount", "TotalScorePlayerCount", "TotalScoreMappingMethod",
    "SessionIndex", "SessionOldTotalScore", "SessionNewTotalScore",
    "OldTotalScore", "NewTotalScore", "NameConfidence", "DigitConsensus", "RowCountConfidence",
    "ScoreValidationStatus", "ReviewNeeded", "ReviewReason",
]


def reorder_export_columns(df):
    """Keep the workbook column order stable so downstream review stays predictable."""
    return df[DESIRED_EXPORT_COLUMNS]


def write_results_workbooks(df, folder_path):
    """Write both a timestamped workbook and a stable latest-results workbook."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(folder_path).resolve().parent
    output_excel_path = output_dir / f"{timestamp}_Tournament_Results.xlsx"
    stable_output_excel_path = output_dir / "Tournament_Results.xlsx"
    df.to_excel(output_excel_path, index=False)
    shutil.copy2(output_excel_path, stable_output_excel_path)
    return output_excel_path, stable_output_excel_path


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
    df = reorder_export_columns(df)
    output_excel_path, stable_output_excel_path = write_results_workbooks(df, folder_path)
    race_count = int(df[["RaceClass", "RaceIDNumber"]].drop_duplicates().shape[0])

    lines = [f"Duration: {format_duration(time.time() - phase_start_time)}", f"Races processed: {race_count}"]
    lines.extend(progress_peak_lines)
    lines.extend(["", "OCR call profile"])
    lines.extend(ocr_profiler_lines)
    summary_lines, per_video_summary = build_player_count_summary_lines(df, build_race_warning_messages, pluralize)
    lines.extend(summary_lines)
    lines.extend(["", "Output workbooks:", str(output_excel_path), str(stable_output_excel_path)])

    return {
        "df": df,
        "lines": lines,
        "output_excel_path": str(output_excel_path),
        "stable_output_excel_path": str(stable_output_excel_path),
        "race_count": race_count,
        "per_video_summary": per_video_summary,
        "per_video_durations": dict(per_video_durations),
        "duration_s": time.time() - phase_start_time,
    }
