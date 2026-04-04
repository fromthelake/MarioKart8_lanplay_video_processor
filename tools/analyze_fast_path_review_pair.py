from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mk8_local_play.project_paths import PROJECT_ROOT


TOURNAMENT_KEY = ["Video", "Race", "Track", "Position"]
FINAL_KEY = ["VideoName", "Position", "PlayerName"]
TRACE_KEY = ["RaceClass", "RaceIDNumber", "RacePosition"]
TRACE_STAGES = [
    "raw_ocr_input",
    "after_standardize",
    "after_duplicate_name_chain_resolution",
    "after_alias_merge",
    "after_connection_reset_relink",
    "final_identity_export_input",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze baseline vs transition-only fast-path review pair output.")
    parser.add_argument(
        "--root",
        required=True,
        help="Review pair root under .codex_tmp (contains baseline/ and transition_only/).",
    )
    return parser.parse_args()


def _latest_matching_file(folder: Path, suffix: str) -> Path:
    matches = sorted(folder.glob(f"*_{suffix}"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No '*_{suffix}' file found in {folder}")
    return matches[0]


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, sep=None, engine="python")
    df.columns = [str(column).lstrip("\ufeff") for column in df.columns]
    return df


def _normalize_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def _compare_frame_bundle_strings(left: str, right: str) -> bool:
    return _normalize_value(left) == _normalize_value(right)


def _bundle_semantics(review_root: Path) -> pd.DataFrame:
    baseline_trace = _load_csv(review_root / "baseline" / "total_score_frame_trace.csv")
    baseline_trace = baseline_trace.rename(columns={"Video": "VideoRelativePath"})
    records: list[dict] = []
    for row in baseline_trace.to_dict(orient="records"):
        video_label = str(row["VideoLabel"])
        race = int(row["Race"])
        baseline_race_dir = review_root / "baseline" / "Frames" / video_label / f"Race_{race:03d}" / "2RaceScore"
        new_race_dir = review_root / "transition_only" / "Frames" / video_label / f"Race_{race:03d}" / "2RaceScore"
        baseline_total_dir = review_root / "baseline" / "Frames" / video_label / f"Race_{race:03d}" / "3TotalScore"
        new_total_dir = review_root / "transition_only" / "Frames" / video_label / f"Race_{race:03d}" / "3TotalScore"
        records.append(
            {
                "VideoLabel": video_label,
                "VideoRelativePath": str(row["VideoRelativePath"]),
                "Race": race,
                "Race Anchor Same": _single_named_file_same(baseline_race_dir, new_race_dir, "anchor_"),
                "Race Consensus Same": _consensus_bundle_same(baseline_race_dir, new_race_dir),
                "Total Anchor Same": _single_named_file_same(baseline_total_dir, new_total_dir, "anchor_"),
                "Total Consensus Same": _consensus_bundle_same(baseline_total_dir, new_total_dir),
            }
        )
    return pd.DataFrame(records)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _single_named_file_same(left_dir: Path, right_dir: Path, prefix: str) -> bool:
    left_files = sorted(left_dir.glob(f"{prefix}*")) if left_dir.exists() else []
    right_files = sorted(right_dir.glob(f"{prefix}*")) if right_dir.exists() else []
    if len(left_files) != 1 or len(right_files) != 1:
        return False
    return _read_bytes(left_files[0]) == _read_bytes(right_files[0])


def _consensus_bundle_same(left_dir: Path, right_dir: Path) -> bool:
    left_files = sorted([path for path in left_dir.iterdir() if path.is_file() and not path.name.startswith("anchor_")]) if left_dir.exists() else []
    right_files = sorted([path for path in right_dir.iterdir() if path.is_file() and not path.name.startswith("anchor_")]) if right_dir.exists() else []
    if len(left_files) != len(right_files):
        return False
    return all(left.name == right.name and _read_bytes(left) == _read_bytes(right) for left, right in zip(left_files, right_files))


def _first_diverging_stage(review_root: Path) -> pd.DataFrame:
    base_stage_root = review_root / "baseline" / "OCR_Tracing" / "identity_stages"
    new_stage_root = review_root / "transition_only" / "OCR_Tracing" / "identity_stages"
    records: list[dict] = []
    video_dirs = sorted({path.name for path in base_stage_root.iterdir() if path.is_dir()} | {path.name for path in new_stage_root.iterdir() if path.is_dir()})
    for video_dir in video_dirs:
        stage_frames: dict[str, pd.DataFrame] = {}
        race_keys = set()
        for stage_name in TRACE_STAGES:
            base_path = base_stage_root / video_dir / f"{stage_name}.csv"
            new_path = new_stage_root / video_dir / f"{stage_name}.csv"
            base_df = _load_csv(base_path) if base_path.exists() else pd.DataFrame(columns=TRACE_KEY)
            new_df = _load_csv(new_path) if new_path.exists() else pd.DataFrame(columns=TRACE_KEY)
            merged = base_df.merge(new_df, on=TRACE_KEY, how="outer", suffixes=("_BASE", "_NEW"))
            stage_frames[stage_name] = merged
            for row in merged[TRACE_KEY].itertuples(index=False):
                race_keys.add(tuple(row))
        for race_class, race_id, race_position in sorted(race_keys):
            first_stage = "none"
            changed_fields = ""
            for stage_name in TRACE_STAGES:
                stage_df = stage_frames[stage_name]
                row = stage_df[
                    (stage_df["RaceClass"] == race_class)
                    & (stage_df["RaceIDNumber"] == race_id)
                    & (stage_df["RacePosition"] == race_position)
                ]
                if row.empty:
                    continue
                row_data = row.iloc[0].to_dict()
                differing = []
                for key, value in row_data.items():
                    if key in TRACE_KEY or key.endswith("_BASE") is False and key.endswith("_NEW") is False:
                        continue
                compared_columns = sorted({col[:-5] for col in row_data if col.endswith("_BASE")})
                for col_name in compared_columns:
                    left = _normalize_value(row_data.get(f"{col_name}_BASE", ""))
                    right = _normalize_value(row_data.get(f"{col_name}_NEW", ""))
                    if left != right:
                        differing.append(col_name)
                if differing:
                    first_stage = stage_name
                    changed_fields = ", ".join(differing)
                    break
            records.append(
                {
                    "VideoLabel": video_dir,
                    "Race": int(race_id),
                    "Position": int(race_position),
                    "First Diverging Stage": first_stage,
                    "Stage Fields": changed_fields,
                }
            )
    return pd.DataFrame(records)


def _compare_rows(base_df: pd.DataFrame, new_df: pd.DataFrame, key_cols: list[str], value_name: str) -> pd.DataFrame:
    merged = base_df.merge(new_df, on=key_cols, how="outer", suffixes=("_BASE", "_NEW"))
    compare_cols = sorted({col[:-5] for col in merged.columns if col.endswith("_BASE")})
    records: list[dict] = []
    for row in merged.to_dict(orient="records"):
        differing = []
        changed_samples = []
        for col in compare_cols:
            left = _normalize_value(row.get(f"{col}_BASE", ""))
            right = _normalize_value(row.get(f"{col}_NEW", ""))
            if left != right:
                differing.append(col)
                if len(changed_samples) < 3:
                    changed_samples.append(f"{col}: {left} -> {right}")
        record = {key: row.get(key, "") for key in key_cols}
        record[f"{value_name} Changed"] = bool(differing)
        record[f"{value_name} Changed Fields"] = ", ".join(differing)
        record[f"{value_name} Changed Samples"] = " | ".join(changed_samples)
        records.append(record)
    return pd.DataFrame(records)


def _build_race_level_compare(review_root: Path) -> pd.DataFrame:
    baseline_dir = review_root / "baseline"
    new_dir = review_root / "transition_only"
    base_tournament = _load_csv(_latest_matching_file(baseline_dir, "Tournament_Results.csv"))
    new_tournament = _load_csv(_latest_matching_file(new_dir, "Tournament_Results.csv"))
    base_final = _load_csv(_latest_matching_file(baseline_dir, "Final_Standings.csv"))
    new_final = _load_csv(_latest_matching_file(new_dir, "Final_Standings.csv"))

    tournament_compare = _compare_rows(base_tournament, new_tournament, TOURNAMENT_KEY, "Tournament")
    final_compare = _compare_rows(base_final, new_final, FINAL_KEY, "Final")
    final_by_video = (
        final_compare.groupby("VideoName", dropna=False, as_index=False)
        .agg(
            **{
                "Final Changed Rows": ("Final Changed", lambda s: int(sum(bool(v) for v in s))),
                "Final Changed Fields": ("Final Changed Fields", lambda s: ", ".join(sorted({part for value in s for part in str(value).split(", ") if part}))),
            }
        )
    )

    tournament_race = (
        tournament_compare.groupby(["Video", "Race"], dropna=False, as_index=False)
        .agg(
            **{
                "Tournament Changed Rows": ("Tournament Changed", lambda s: int(sum(bool(v) for v in s))),
                "Tournament Changed Fields": ("Tournament Changed Fields", lambda s: ", ".join(sorted({part for value in s for part in str(value).split(", ") if part}))),
                "Tournament Changed Samples": ("Tournament Changed Samples", lambda s: " | ".join([str(v) for v in s if str(v)])[:2000]),
            }
        )
    )
    tournament_race = tournament_race.rename(columns={"Video": "VideoLabel"})
    final_by_video = final_by_video.rename(columns={"VideoName": "VideoLabel"})

    bundle_df = _bundle_semantics(review_root)
    stage_df = _first_diverging_stage(review_root)
    stage_race = (
        stage_df.groupby(["VideoLabel", "Race"], dropna=False, as_index=False)
        .agg(
            **{
                "First Diverging Stage": (
                    "First Diverging Stage",
                    lambda s: next((value for value in s if value != "none"), "none"),
                ),
                "Stage Fields": ("Stage Fields", lambda s: ", ".join(sorted({part for value in s for part in str(value).split(", ") if part}))),
            }
        )
    )

    race_df = tournament_race.merge(bundle_df, on=["VideoLabel", "Race"], how="left")
    race_df = race_df.merge(stage_race, on=["VideoLabel", "Race"], how="left")
    race_df = race_df.merge(final_by_video, on="VideoLabel", how="left")
    race_df["Tournament Changed Rows"] = race_df["Tournament Changed Rows"].fillna(0).astype(int)
    race_df["Final Changed Rows"] = race_df["Final Changed Rows"].fillna(0).astype(int)
    race_df["Output Changed"] = (race_df["Tournament Changed Rows"] > 0) | (race_df["Final Changed Rows"] > 0)
    race_df["Drift Type"] = race_df.apply(_classify_drift, axis=1)
    return race_df.sort_values(["VideoLabel", "Race"], kind="stable").reset_index(drop=True)


def _classify_drift(row: pd.Series) -> str:
    if not bool(row.get("Output Changed", False)):
        return "none"
    if row.get("Race Consensus Same") is False:
        return "upstream race consensus drift"
    first_stage = str(row.get("First Diverging Stage") or "none")
    if first_stage == "none":
        return "export-only drift"
    if first_stage == "raw_ocr_input":
        return "ocr input drift"
    return f"downstream {first_stage} drift"


def _build_summary(race_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        race_df.groupby(["Drift Type"], dropna=False, as_index=False)
        .agg(
            Races=("Race", "count"),
            ChangedRaces=("Output Changed", lambda s: int(sum(bool(v) for v in s))),
        )
        .sort_values(["ChangedRaces", "Races"], ascending=[False, False], kind="stable")
    )
    return summary


def _write_markdown(review_root: Path, race_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = ["# Fast Path Review Pair Analysis", ""]
    lines.append("## Summary")
    lines.append("")
    lines.extend(_simple_table_lines(summary_df))
    lines.append("")
    lines.append("## Changed Races")
    lines.append("")
    changed = race_df[race_df["Output Changed"]].copy()
    if changed.empty:
        lines.append("No changed races.")
    else:
        display_cols = [
            "VideoLabel",
            "Race",
            "Tournament Changed Rows",
            "Final Changed Rows",
            "Race Consensus Same",
            "First Diverging Stage",
            "Drift Type",
            "Tournament Changed Fields",
            "Tournament Changed Samples",
        ]
        lines.extend(_simple_table_lines(changed[display_cols]))
    (review_root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def _simple_table_lines(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["(none)"]
    rows = [[_normalize_value(value) for value in row] for row in df.to_numpy().tolist()]
    headers = [str(col) for col in df.columns]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    lines = [
        "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |",
        "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    return lines


def _write_excel(review_root: Path, race_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    target = review_root / "analysis_review.xlsx"
    with pd.ExcelWriter(target, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        race_df.to_excel(writer, sheet_name="Race Analysis", index=False)


def main() -> None:
    args = parse_args()
    review_root = Path(args.root)
    if not review_root.is_absolute():
        review_root = PROJECT_ROOT / review_root
    race_df = _build_race_level_compare(review_root)
    summary_df = _build_summary(race_df)
    race_df.to_csv(review_root / "analysis_race_level.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(review_root / "analysis_summary.csv", index=False, encoding="utf-8-sig")
    _write_markdown(review_root, race_df, summary_df)
    _write_excel(review_root, race_df, summary_df)
    print(review_root / "analysis_report.md")
    print(review_root / "analysis_review.xlsx")


if __name__ == "__main__":
    main()
