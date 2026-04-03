import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
from openpyxl.utils import get_column_letter

from .game_catalog import load_game_catalog

USER_REVIEW_REASON_MAX_LENGTH = 160
DEBUG_REVIEW_REASON_MAX_LENGTH = 240


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
    "CharacterRosterName": "Character Roster",
    "RacePoints": "Race Points",
    "OldTotalScore": "Total Before Race",
    "NewTotalScore": "Total After Race",
    "PositionAfterRace": "Position After Race",
    "ReviewNeeded": "Needs Review",
    "ReviewReason": "Review Reason",
    "CountsTowardTotals": "Counts Toward Totals",
    "ExcludedFromTotalsReason": "Scoring Note",
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
    "IdentityRelinkDetected": "Identity Relink Detected",
    "IsLowRes": "Is Low Res",
    "Character": "Character",
    "CharacterRosterIndex": "Character Roster Index",
    "CharacterRosterName": "Character Roster Name",
    "CharacterIndex": "Character Index",
    "CharacterMatchConfidence": "Character Match Confidence",
    "CharacterMatchMethod": "Character Match Method",
    "RacePoints": "Race Points",
    "DetectedRacePoints": "OCR Race Points",
    "DetectedRacePointsSource": "OCR Race Points Source",
    "DetectedOldTotalScore": "OCR Old Total Score",
    "DetectedOldTotalScoreSource": "OCR Old Total Score Source",
    "DetectedTotalScore": "OCR Total Score",
    "DetectedTotalScoreSource": "OCR Total Score Source",
    "DetectedNewTotalScore": "OCR New Total Score",
    "DetectedNewTotalScoreSource": "OCR New Total Score Source",
    "DetectedPositionAfterRace": "OCR Position After Race",
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
    "NameAllowedCharRatio": "Name Allowed Char Ratio",
    "NameUnknownChars": "Name Unknown Chars",
    "NameValidationFlags": "Name Validation Flags",
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
    "RacePointsAnchorFrame": "RacePoints Anchor Frame",
    "TotalScoreMappingMethod": "Total Score Match Method",
    "TotalScoreMappingScore": "Total Score Match Score",
    "TotalScoreMappingMargin": "Total Score Match Margin",
    "TotalScoreNameSimilarity": "Total Score Name Similarity",
    "ScoreValidationStatus": "Validation Status",
    "ReviewNeeded": "Needs Review",
    "ReviewReason": "Review Reason",
    "CountsTowardTotals": "Counts Toward Totals",
    "ExcludedFromTotalsReason": "Scoring Note",
    "ScoringPlayerCount": "Scoring Player Count",
}


def _dedupe_review_reason_parts(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []

    parts = []
    seen = set()
    for raw_part in str(value).split("|"):
        part = raw_part.strip()
        if not part or part.lower() == "nan":
            continue
        normalized = " ".join(part.casefold().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(part)
    return parts


def _truncate_review_reason(parts: list[str], max_length: int) -> str:
    if not parts:
        return ""

    joined = " | ".join(parts)
    if len(joined) <= max_length:
        return joined

    kept_parts: list[str] = []
    for index, part in enumerate(parts):
        remaining = len(parts) - index - 1
        candidate_parts = kept_parts + [part]
        candidate = " | ".join(candidate_parts)
        suffix = f" ... (+{remaining} more)" if remaining > 0 else ""
        if len(candidate) + len(suffix) <= max_length:
            kept_parts.append(part)
            continue
        break

    omitted = len(parts) - len(kept_parts)
    if kept_parts:
        truncated = " | ".join(kept_parts)
        if omitted > 0:
            return f"{truncated} ... (+{omitted} more)"
        return truncated

    first_part = parts[0]
    if max_length <= 3:
        return first_part[:max_length]
    return first_part[: max_length - 3].rstrip() + "..."


def format_review_reason_for_export(value: object, max_length: int) -> str:
    return _truncate_review_reason(_dedupe_review_reason_parts(value), max_length)


def build_user_export_df(df):
    ordered_df = enrich_character_roster_columns(df)[list(USER_EXPORT_COLUMN_MAP.keys())].copy()
    ordered_df["ReviewReason"] = ordered_df["ReviewReason"].apply(
        lambda value: format_review_reason_for_export(value, USER_REVIEW_REASON_MAX_LENGTH)
    )
    return ordered_df.rename(columns=USER_EXPORT_COLUMN_MAP)


def build_debug_export_df(df):
    ordered_df = enrich_character_roster_columns(df)
    for column_name in DEBUG_EXPORT_COLUMN_MAP.keys():
        if column_name not in ordered_df.columns:
            ordered_df[column_name] = ""
    ordered_df = ordered_df[list(DEBUG_EXPORT_COLUMN_MAP.keys())].copy()
    ordered_df["ReviewReason"] = ordered_df["ReviewReason"].apply(
        lambda value: format_review_reason_for_export(value, DEBUG_REVIEW_REASON_MAX_LENGTH)
    )
    return ordered_df.rename(columns=DEBUG_EXPORT_COLUMN_MAP)


def _normalize_character_value(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


@lru_cache(maxsize=1)
def _character_roster_lookup() -> dict[str, tuple[int | None, str]]:
    catalog = load_game_catalog()
    roster_name_overrides = {
        7: "Birdo",
        8: "Yoshi",
        11: "Shy Guy",
        21: "Metal/Gold Mario",
        40: "Inkling",
        41: "Villager",
        43: "Link",
        47: "Mii",
    }

    lookup: dict[str, tuple[int | None, str]] = {}
    for character in catalog.characters:
        roster_index = int(character.roster_index)
        roster_name = roster_name_overrides.get(roster_index, str(character.name_uk))
        lookup[str(character.name_uk).strip()] = (roster_index, roster_name)
    return lookup


def _character_roster_fields(character_name: object) -> tuple[object, str]:
    normalized = _normalize_character_value(character_name)
    if not normalized:
        return pd.NA, ""
    return _character_roster_lookup().get(normalized, (pd.NA, normalized))


def enrich_character_roster_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    enriched = df.copy()
    roster_fields = enriched["Character"].apply(_character_roster_fields)
    enriched["CharacterRosterIndex"] = roster_fields.apply(lambda item: item[0])
    enriched["CharacterRosterName"] = roster_fields.apply(lambda item: item[1])
    return enriched


def _select_most_used_character(player_rows: pd.DataFrame) -> str:
    character_counts: dict[str, int] = {}
    last_seen_race: dict[str, int] = {}
    for _, row in player_rows.iterrows():
        character_name = _normalize_character_value(row.get("Character"))
        if not character_name:
            continue
        character_counts[character_name] = character_counts.get(character_name, 0) + 1
        try:
            race_id = int(row.get("RaceIDNumber", 0))
        except (TypeError, ValueError):
            race_id = 0
        last_seen_race[character_name] = max(last_seen_race.get(character_name, 0), race_id)

    if not character_counts:
        return ""

    return min(
        character_counts,
        key=lambda name: (-character_counts[name], -last_seen_race.get(name, 0), name.lower(), name),
    )


def _build_reset_segment_columns(df: pd.DataFrame, final_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "SessionIndex" not in df.columns or "SessionNewTotalScore" not in df.columns:
        return pd.DataFrame(index=final_rows.index), []

    session_counts_by_video = (
        df.groupby("RaceClass", sort=False)["SessionIndex"]
        .max()
        .fillna(1)
        .astype(int)
        .to_dict()
    )
    max_session_count = max(session_counts_by_video.values(), default=1)
    if max_session_count <= 1:
        return pd.DataFrame(index=final_rows.index), []

    valid_rows = df.loc[df["FixPlayerName"].notna()].copy()
    if valid_rows.empty:
        return pd.DataFrame(index=final_rows.index), []

    valid_rows["SessionIndex"] = pd.to_numeric(valid_rows["SessionIndex"], errors="coerce").astype("Int64")
    valid_rows["SessionNewTotalScore"] = pd.to_numeric(valid_rows["SessionNewTotalScore"], errors="coerce")
    session_totals = (
        valid_rows.dropna(subset=["SessionIndex", "SessionNewTotalScore"])
        .groupby(["RaceClass", "FixPlayerName", "SessionIndex"], sort=False)["SessionNewTotalScore"]
        .max()
        .to_dict()
    )

    segment_column_names = [f"Points Segment {segment_index}" for segment_index in range(1, max_session_count + 1)]
    reset_segments = pd.DataFrame(index=final_rows.index, columns=segment_column_names, dtype="Int64")

    for row_index, row in final_rows.iterrows():
        race_class = row["RaceClass"]
        player_name = row["FixPlayerName"]
        session_count = int(session_counts_by_video.get(race_class, 1))
        if session_count <= 1:
            continue
        for session_index in range(1, session_count + 1):
            value = session_totals.get((race_class, player_name, session_index))
            if value is None or pd.isna(value):
                continue
            reset_segments.at[row_index, f"Points Segment {session_index}"] = int(value)

    return reset_segments, segment_column_names


def build_final_standings_df(df):
    race_counts_by_video = (
        df.groupby("RaceClass", sort=True)["RaceIDNumber"]
        .nunique()
        .to_dict()
    )

    final_rows = (
        df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable")
        .groupby(["RaceClass", "FixPlayerName"], sort=False, as_index=False)
        .tail(1)
        .copy()
        .reset_index(drop=True)
    )

    final_rows["Races"] = final_rows["RaceClass"].map(lambda value: int(race_counts_by_video.get(value, 0)))
    final_rows["Character"] = final_rows.apply(
        lambda row: _select_most_used_character(
            df.loc[
                (df["RaceClass"] == row["RaceClass"])
                & (df["FixPlayerName"] == row["FixPlayerName"])
            ]
        ),
        axis=1,
    )
    roster_fields = final_rows["Character"].apply(_character_roster_fields)
    final_rows["CharacterRosterName"] = roster_fields.apply(lambda item: item[1])

    standings_df = pd.DataFrame(
        {
            "VideoName": final_rows["RaceClass"],
            "Races": final_rows["Races"],
            "PlayerName": final_rows["FixPlayerName"],
            "TotalPoints": final_rows["NewTotalScore"],
            "Character": final_rows["Character"],
            "CharacterRosterName": final_rows["CharacterRosterName"],
        }
    )
    reset_segments_df, reset_segment_column_names = _build_reset_segment_columns(df, final_rows)
    for column_name in reset_segment_column_names:
        standings_df[column_name] = reset_segments_df[column_name].to_numpy()

    standings_df["Position"] = (
        standings_df.groupby("VideoName", sort=False)["TotalPoints"]
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    ordered_columns = ["VideoName", "Races", "Position", "PlayerName", "TotalPoints", "Character", "CharacterRosterName"]
    ordered_columns.extend(reset_segment_column_names)
    standings_df = standings_df[ordered_columns]

    numeric_columns = ["Races", "Position", "TotalPoints", *reset_segment_column_names]
    for column_name in numeric_columns:
        standings_df[column_name] = pd.to_numeric(standings_df[column_name], errors="coerce").astype("Int64")

    return standings_df.sort_values(["VideoName", "Position", "PlayerName"], kind="stable").reset_index(drop=True)


def autosize_worksheet_columns(worksheet, dataframe, padding: int = 2, max_width: int = 60):
    for column_index, column_name in enumerate(dataframe.columns, start=1):
        values = [column_name]
        values.extend("" if value is None else str(value) for value in dataframe.iloc[:, column_index - 1])
        width = min(max_width, max(len(value) for value in values) + padding)
        worksheet.column_dimensions[get_column_letter(column_index)].width = max(8, width)


def write_results_workbooks(df, folder_path):
    """Write clean workbook and CSV exports for both user and debug output."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(folder_path).resolve().parent
    debug_output_dir = output_dir / "Debug"
    debug_output_dir.mkdir(parents=True, exist_ok=True)

    user_df = build_user_export_df(df)
    final_standings_df = build_final_standings_df(df)
    debug_df = build_debug_export_df(df)

    output_excel_path = output_dir / f"{timestamp}_Tournament_Results.xlsx"
    debug_output_excel_path = debug_output_dir / f"{timestamp}_Tournament_Results_Debug.xlsx"
    output_csv_path = output_dir / f"{timestamp}_Tournament_Results.csv"
    final_standings_csv_path = output_dir / f"{timestamp}_Final_Standings.csv"
    debug_output_csv_path = debug_output_dir / f"{timestamp}_Tournament_Results_Debug.csv"

    with pd.ExcelWriter(output_excel_path) as writer:
        user_df.to_excel(writer, index=False, sheet_name="Results")
        autosize_worksheet_columns(writer.sheets["Results"], user_df)
        final_standings_df.to_excel(writer, index=False, sheet_name="Final Standings")
        autosize_worksheet_columns(writer.sheets["Final Standings"], final_standings_df)
    with pd.ExcelWriter(debug_output_excel_path) as writer:
        debug_df.to_excel(writer, index=False, sheet_name="Debug Results")
        autosize_worksheet_columns(writer.sheets["Debug Results"], debug_df)
    user_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    final_standings_df.to_csv(final_standings_csv_path, index=False, encoding="utf-8-sig")
    debug_df.to_csv(debug_output_csv_path, index=False, encoding="utf-8-sig")
    return {
        "user_df": user_df,
        "debug_df": debug_df,
        "output_excel_path": output_excel_path,
        "debug_output_excel_path": debug_output_excel_path,
        "output_csv_path": output_csv_path,
        "final_standings_csv_path": final_standings_csv_path,
        "debug_output_csv_path": debug_output_csv_path,
    }


def build_player_count_summary_lines(df, build_race_warning_messages, pluralize):
    """Build the human-readable OCR completion summary shown in the console."""
    per_video_summary = {}
    lines = ["", "Player count check"]
    for race_class, race_group in df.groupby("RaceClass", sort=False):
        race_count_for_class = int(race_group["RaceIDNumber"].nunique())
        player_count_distribution = race_group.groupby("RaceIDNumber").size().value_counts().sort_index(ascending=False)
        dominant_players = int(race_group.groupby("RaceIDNumber").size().mode().iloc[0])
        distribution_text = ", ".join(f"{player_count} players x {count}" for player_count, count in player_count_distribution.items())
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
            "player_count_summary": (
                f"consistent ({dominant_players} players)"
                if not inconsistent_races else
                f"mixed ({distribution_text})"
            ),
        }

        if not inconsistent_races:
            lines.append(
                f"- {race_class}: {race_count_for_class} {pluralize(race_count_for_class, 'race')} | "
                f"consistent at {dominant_players} players"
            )
            continue

        lines.append(f"- {race_class}: {race_count_for_class} {pluralize(race_count_for_class, 'race')} | mixed player counts")
        lines.append(f"  Most common: {dominant_players} players")
        lines.append(f"  Breakdown: {distribution_text}")
        lines.append("  Review these races:")
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
    lines.extend(["", "OCR mode"])
    lines.extend(ocr_profiler_lines)
    summary_lines, per_video_summary = build_player_count_summary_lines(df, build_race_warning_messages, pluralize)
    lines.extend(summary_lines)
    lines.extend(
        [
            "",
            "Saved files",
            f"- Results workbook: {workbook_payload['output_excel_path']}",
            f"- Debug workbook: {workbook_payload['debug_output_excel_path']}",
            f"- Results CSV: {workbook_payload['output_csv_path']}",
            f"- Final standings CSV: {workbook_payload['final_standings_csv_path']}",
            f"- Debug CSV: {workbook_payload['debug_output_csv_path']}",
        ]
    )

    return {
        "user_df": workbook_payload["user_df"],
        "debug_df": workbook_payload["debug_df"],
        "lines": lines,
        "output_excel_path": str(workbook_payload["output_excel_path"]),
        "debug_output_excel_path": str(workbook_payload["debug_output_excel_path"]),
        "output_csv_path": str(workbook_payload["output_csv_path"]),
        "final_standings_csv_path": str(workbook_payload["final_standings_csv_path"]),
        "debug_output_csv_path": str(workbook_payload["debug_output_csv_path"]),
        "race_count": race_count,
        "per_video_summary": per_video_summary,
        "per_video_durations": dict(per_video_durations),
        "duration_s": time.time() - phase_start_time,
    }
