from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .ocr_session_validation import assign_shared_positions_after_race


PLAYER_DROP_EXCLUSION_CODE = "temporary_player_drop_excluded"


def _append_reason_message(existing_value: object, message: str) -> str:
    existing_parts = [
        part.strip()
        for part in str(existing_value or "").split("|")
        if part.strip()
    ]
    if message not in existing_parts:
        existing_parts.append(message)
    return " | ".join(existing_parts)


def _append_reason_code(existing_value: object, code: str) -> str:
    existing_parts = [
        part.strip()
        for part in str(existing_value or "").split(";")
        if part.strip()
    ]
    if code not in existing_parts:
        existing_parts.append(code)
    return ";".join(existing_parts)


def _race_level_player_count(race_rows: pd.DataFrame) -> int:
    visible_rows = int(race_rows.shape[0])
    race_score_count = int(pd.to_numeric(race_rows.get("RaceScorePlayerCount"), errors="coerce").fillna(0).max() or 0)
    if race_score_count > 0:
        return race_score_count
    return visible_rows


def _scoring_player_key(row: pd.Series) -> str:
    is_low_res = bool(row.get("IsLowRes", False))
    identity_label = str(row.get("IdentityLabel") or "").strip()
    if is_low_res and identity_label:
        return identity_label
    return str(row.get("FixPlayerName") or "").strip()


def _seed_session_baseline_from_first_race(
    race_rows: pd.DataFrame,
    tournament_totals: dict[str, int],
    session_totals: dict[str, int],
) -> None:
    """Preserve a validated first-race old-total baseline when all present rows have one.

    This keeps the scoring-policy recompute aligned with the validation rule that a
    video/session may start mid-tournament when the first RaceScore frame already
    shows prior totals for the players actually present.
    """
    if race_rows.empty:
        return

    present_players = max(0, _race_level_player_count(race_rows))
    if present_players <= 0:
        return

    candidate_rows = race_rows.sort_values("RacePosition", kind="stable").head(present_players).copy()
    if candidate_rows.empty or len(candidate_rows) < present_players:
        return

    old_totals = pd.to_numeric(candidate_rows.get("OldTotalScore"), errors="coerce")
    session_old_totals = pd.to_numeric(candidate_rows.get("SessionOldTotalScore"), errors="coerce")

    if old_totals.isna().any() or (old_totals <= 0).any():
        return

    for row_index, row in candidate_rows.iterrows():
        player_key = _scoring_player_key(row)
        if not player_key:
            continue
        tournament_totals[player_key] = int(old_totals.loc[row_index])
        if row_index in session_old_totals.index and not pd.isna(session_old_totals.loc[row_index]):
            session_totals[player_key] = int(session_old_totals.loc[row_index])
        else:
            session_totals[player_key] = int(old_totals.loc[row_index])


def _mark_temporary_player_drop_races(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "CountsTowardTotals" not in df.columns:
        df["CountsTowardTotals"] = True
    else:
        df["CountsTowardTotals"] = df["CountsTowardTotals"].fillna(True).astype(bool)
    if "ExcludedFromTotalsReason" not in df.columns:
        df["ExcludedFromTotalsReason"] = ""
    else:
        df["ExcludedFromTotalsReason"] = df["ExcludedFromTotalsReason"].fillna("")
    df["ScoringPlayerCount"] = pd.NA

    grouping_columns = ["RaceClass"]
    if "SessionIndex" in df.columns:
        grouping_columns.append("SessionIndex")

    for group_key, group in df.groupby(grouping_columns, sort=False):
        race_counts: list[tuple[int, int]] = []
        for race_id, race_rows in group.groupby("RaceIDNumber", sort=True):
            race_counts.append((int(race_id), _race_level_player_count(race_rows)))
        if not race_counts:
            continue

        suffix_max_by_race: dict[int, int] = {}
        suffix_max = 0
        for race_id, player_count in reversed(race_counts):
            suffix_max = max(suffix_max, int(player_count))
            suffix_max_by_race[int(race_id)] = int(suffix_max)

        race_class = group["RaceClass"].iloc[0]
        session_text = ""
        if "SessionIndex" in group.columns:
            try:
                session_text = f" in session {int(group['SessionIndex'].iloc[0])}"
            except (TypeError, ValueError):
                session_text = ""

        for race_id, player_count in race_counts:
            recovered_count = int(suffix_max_by_race[int(race_id)])
            race_mask = (
                (df["RaceClass"] == race_class)
                & (df["RaceIDNumber"] == int(race_id))
            )
            if "SessionIndex" in group.columns:
                race_mask &= (df["SessionIndex"] == group["SessionIndex"].iloc[0])
            df.loc[race_mask, "ScoringPlayerCount"] = int(player_count)
            if int(player_count) >= recovered_count:
                continue
            reason_message = (
                "Race excluded from totals due to temporary player drop"
                f"{session_text}; later races recovered to {recovered_count} players."
            )
            df.loc[race_mask, "CountsTowardTotals"] = False
            df.loc[race_mask, "ExcludedFromTotalsReason"] = reason_message
            df.loc[race_mask, "ReviewReason"] = df.loc[race_mask, "ReviewReason"].apply(
                lambda value, msg=reason_message: _append_reason_message(value, msg)
            )
            df.loc[race_mask, "ReviewReasonCodes"] = df.loc[race_mask, "ReviewReasonCodes"].apply(
                lambda value, code=PLAYER_DROP_EXCLUSION_CODE: _append_reason_code(value, code)
            )
            df.loc[race_mask, "ReviewNeeded"] = True
    return df


def _recompute_scoring_totals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for race_class, race_group in df.groupby("RaceClass", sort=False):
        tournament_totals: dict[str, int] = defaultdict(int)
        session_totals: dict[str, int] = defaultdict(int)
        current_session_index = None
        for race_id in sorted(race_group["RaceIDNumber"].unique()):
            race_mask = (df["RaceClass"] == race_class) & (df["RaceIDNumber"] == race_id)
            race_rows = df[race_mask].sort_values("RacePosition", kind="stable")
            if race_rows.empty:
                continue
            race_session_index = race_rows["SessionIndex"].iloc[0] if "SessionIndex" in race_rows.columns else 1
            if current_session_index is None or race_session_index != current_session_index:
                session_totals = defaultdict(int)
                current_session_index = race_session_index
                _seed_session_baseline_from_first_race(race_rows, tournament_totals, session_totals)
            for index, row in race_rows.iterrows():
                player_key = _scoring_player_key(row)
                old_total = int(tournament_totals[player_key])
                old_session_total = int(session_totals[player_key])
                counts_toward_totals = bool(row.get("CountsTowardTotals", True))
                race_points = int(row.get("RacePoints", 0) or 0) if counts_toward_totals else 0
                new_total = old_total + race_points
                new_session_total = old_session_total + race_points
                df.at[index, "OldTotalScore"] = old_total
                df.at[index, "NewTotalScore"] = new_total
                if "SessionOldTotalScore" in df.columns:
                    df.at[index, "SessionOldTotalScore"] = old_session_total
                if "SessionNewTotalScore" in df.columns:
                    df.at[index, "SessionNewTotalScore"] = new_session_total
                tournament_totals[player_key] = new_total
                session_totals[player_key] = new_session_total

    return assign_shared_positions_after_race(df)


def apply_temporary_player_drop_scoring_policy(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude lower-player races when later races recover to a higher player count."""
    if df.empty:
        return df
    adjusted = _mark_temporary_player_drop_races(df)
    return _recompute_scoring_totals(adjusted)
