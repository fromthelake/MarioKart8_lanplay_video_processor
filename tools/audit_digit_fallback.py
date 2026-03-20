import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from mk8_local_play.ocr_common import (
    find_metadata_entry,
    load_consensus_frame_entries,
    load_exported_frame_metadata,
)
from mk8_local_play.extract_common import (
    find_score_bundle_anchor_path,
    find_score_bundle_consensus_paths,
)
from mk8_local_play.ocr_scoreboard_consensus import (
    TOTAL_SCORE_CONSENSUS_WINDOW_SIZE,
    analyze_digit_segments,
    build_normalized_position_strip,
    ocr_digit_row_fallback,
    parse_detected_int,
    position_strip_roi,
    process_image,
    score_digit_layout,
    select_consensus_window,
)
from mk8_local_play.project_paths import PROJECT_ROOT
from mk8_local_play.score_layouts import get_score_layout, score_layout_id_from_filename


def build_expected_maps(results_csv: Path, race_class: str):
    df = pd.read_csv(results_csv)
    df = df[df["Video"] == race_class].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for video '{race_class}' in {results_csv}")

    race_points_expected = {}
    old_total_expected = {}
    new_total_expected = {}
    for row in df.to_dict(orient="records"):
        race = int(row.get("Race", 0) or 0)
        position = int(row.get("Position", 0) or 0)
        pos_after = row.get("Position After Race")
        if race > 0 and position > 0:
            race_points_expected[(race, position)] = parse_detected_int(row.get("Race Points"))
            old_total_expected[(race, position)] = parse_detected_int(row.get("Total Before Race"))
        if race > 0 and pd.notna(pos_after):
            new_total_expected[(race, int(pos_after))] = parse_detected_int(row.get("Total After Race"))
    return race_points_expected, old_total_expected, new_total_expected


def prepare_digit_image(frame_image: np.ndarray, score_layout_id: str):
    processed_img = process_image(frame_image, score_layout_id=score_layout_id)
    (position_x1, position_y1), (position_x2, position_y2) = position_strip_roi(score_layout_id=score_layout_id)
    normalized_position_strip = build_normalized_position_strip(processed_img, score_layout_id=score_layout_id)
    processed_img[position_y1:position_y2, position_x1:position_x2] = cv2.cvtColor(normalized_position_strip, cv2.COLOR_GRAY2BGR)
    processed_img_pil = Image.fromarray(processed_img).convert("RGB")
    return processed_img_pil.resize(
        (processed_img_pil.width * 5, processed_img_pil.height * 5),
        Image.NEAREST,
    )


def detect_digits_without_fallback(
    image: Image.Image,
    start_coords,
    row_offset,
    box_dims,
    red_pixels,
    num_rows,
    boxes_per_row,
):
    row_values = []
    row_should_fallback = []
    for row_index in range(num_rows):
        y_offset = row_index * row_offset
        row_digits = []
        has_unknown_digit = False
        for box_index in range(boxes_per_row):
            start_x, start_y = start_coords[box_index]
            top_left = (start_x, start_y + y_offset)
            digit, _segment_stats, _strong_active_labels = analyze_digit_segments(image, top_left, red_pixels)
            if digit != -1:
                row_digits.append(str(digit))
            else:
                row_digits.append("")
                has_unknown_digit = True
        row_number = "".join(row_digits)
        numeric_value = parse_detected_int(row_number)
        recognized_indices = [index for index, digit_text in enumerate(row_digits) if digit_text]
        contiguous_digit_block = False
        if recognized_indices:
            first_recognized_index = recognized_indices[0]
            last_recognized_index = recognized_indices[-1]
            contiguous_digit_block = all(
                digit_text != "" for digit_text in row_digits[first_recognized_index:last_recognized_index + 1]
            )
        row_values.append(row_number)
        row_should_fallback.append((numeric_value is None, has_unknown_digit, contiguous_digit_block))
    return row_values, row_should_fallback


def classify_outcome(seven_value, fallback_value, expected_value):
    seven_int = parse_detected_int(seven_value)
    fallback_int = parse_detected_int(fallback_value)
    seven_correct = seven_int == expected_value and expected_value is not None
    fallback_correct = fallback_int == expected_value and expected_value is not None
    if seven_correct and fallback_correct:
        return "not_needed_same"
    if seven_correct and not fallback_value:
        return "not_needed_blank"
    if seven_correct and fallback_value and not fallback_correct:
        return "would_hurt"
    if (not seven_correct) and fallback_correct:
        return "helped"
    if (not seven_correct) and (not fallback_value):
        return "no_help_blank"
    return "no_help_wrong"


def format_table(title: str, rows: list[dict[str, object]]) -> str:
    if not rows:
        return f"{title}\n(no rows)"
    columns = list(rows[0].keys())
    widths = {column: len(column) for column in columns}
    for row in rows:
        for column in columns:
            widths[column] = max(widths[column], len(str(row[column])))
    lines = [title]
    lines.append("  ".join(f"{column:<{widths[column]}}" for column in columns))
    lines.append("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        rendered = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                rendered.append(f"{value:>{widths[column]}.4f}")
            elif isinstance(value, int):
                rendered.append(f"{value:>{widths[column]}}")
            else:
                safe_text = str(value).encode("cp1252", errors="replace").decode("cp1252")
                rendered.append(f"{safe_text:<{widths[column]}}")
        lines.append("  ".join(rendered))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit seven-segment vs OCR digit fallback.")
    parser.add_argument("--race-class", required=True)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--show-rows", type=int, default=25)
    args = parser.parse_args()

    project_root = Path(PROJECT_ROOT)
    results_csv = Path(args.results_csv)
    if not results_csv.is_absolute():
        results_csv = project_root / results_csv

    metadata_index = load_exported_frame_metadata(project_root)
    input_videos_folder = project_root / "Input_Videos"
    race_points_expected, old_total_expected, new_total_expected = build_expected_maps(results_csv, args.race_class)
    race_numbers = sorted({race for race, _position in race_points_expected.keys()})

    summary = defaultdict(Counter)
    example_rows = []

    for race_number in race_numbers:
        race_metadata = find_metadata_entry(metadata_index, args.race_class, race_number, "RaceScore")
        race_score_entries = []
        race_score_layout_id = ""
        if race_metadata is not None:
            race_score_entries = load_consensus_frame_entries(
                str(race_metadata.get("anchor_path", "")),
                race_metadata,
                input_videos_folder,
                7,
            )
            race_score_layout_id = str(race_metadata.get("score_layout_id", "")).strip()
        else:
            anchor_path = find_score_bundle_anchor_path(args.race_class, race_number, "2RaceScore")
            consensus_paths = find_score_bundle_consensus_paths(args.race_class, race_number, "2RaceScore")
            race_score_entries = []
            for path in consensus_paths:
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is not None:
                    race_score_entries.append((0, frame))
            if not race_score_entries and anchor_path is not None:
                anchor_frame = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
                if anchor_frame is not None:
                    race_score_entries.append((0, anchor_frame))
            race_score_layout_id = score_layout_id_from_filename(anchor_path)
        if race_score_entries:
            scaled_image_layout = score_digit_layout(5, score_layout_id=race_score_layout_id)
            for frame_number, frame_image in race_score_entries:
                digit_image = prepare_digit_image(frame_image, race_score_layout_id)
                for field_name, expected_map in (("RacePoints", race_points_expected), ("OldTotalScore", old_total_expected)):
                    layout_key = "race_points" if field_name == "RacePoints" else "total_points"
                    args_tuple = scaled_image_layout[layout_key]
                    row_values, fallback_flags = detect_digits_without_fallback(digit_image, *args_tuple)
                    for row_index, (row_value, fallback_flag) in enumerate(zip(row_values, fallback_flags), start=1):
                        numeric_missing, has_unknown_digit, contiguous_digit_block = fallback_flag
                        expected_value = expected_map.get((race_number, row_index))
                        if expected_value is None:
                            continue
                        should_fallback = parse_detected_int(row_value) is None
                        if parse_detected_int(row_value) is not None:
                            if field_name == "RacePoints":
                                should_fallback = not (1 <= int(parse_detected_int(row_value)) <= 15)
                            else:
                                should_fallback = not (0 <= int(parse_detected_int(row_value)) <= 999)
                        if has_unknown_digit and not contiguous_digit_block:
                            should_fallback = True
                        summary[field_name]["rows"] += 1
                        if should_fallback:
                            summary[field_name]["fallback_rows"] += 1
                            fallback_value = ocr_digit_row_fallback(
                                digit_image,
                                args_tuple[0],
                                args_tuple[1],
                                args_tuple[2],
                                row_index - 1,
                                args_tuple[5],
                                valid_min=1 if field_name == "RacePoints" else 0,
                                valid_max=15 if field_name == "RacePoints" else 999,
                                bundle_kind="audit",
                                field_name=field_name,
                            )
                            outcome = classify_outcome(row_value, fallback_value, expected_value)
                            summary[field_name][outcome] += 1
                            example_rows.append(
                                {
                                    "Field": field_name,
                                    "Race": race_number,
                                    "Frame": frame_number,
                                    "Row": row_index,
                                    "Expected": expected_value,
                                    "SevenSeg": row_value,
                                    "Fallback": fallback_value,
                                    "Outcome": outcome,
                                }
                            )
                        else:
                            if parse_detected_int(row_value) == expected_value:
                                summary[field_name]["seven_correct_no_fallback"] += 1
                            else:
                                summary[field_name]["seven_wrong_no_fallback"] += 1

        total_metadata = find_metadata_entry(metadata_index, args.race_class, race_number, "TotalScore")
        total_score_entries = []
        total_score_layout_id = ""
        if total_metadata is not None:
            total_score_entries = load_consensus_frame_entries(
                str(total_metadata.get("anchor_path", "")),
                total_metadata,
                input_videos_folder,
                3,
            )
            total_score_entries = select_consensus_window(total_score_entries, "center", size=TOTAL_SCORE_CONSENSUS_WINDOW_SIZE)
            total_score_layout_id = str(total_metadata.get("score_layout_id", "")).strip()
        else:
            anchor_path = find_score_bundle_anchor_path(args.race_class, race_number, "3TotalScore")
            consensus_paths = find_score_bundle_consensus_paths(args.race_class, race_number, "3TotalScore")
            total_score_entries = []
            for path in consensus_paths:
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is not None:
                    total_score_entries.append((0, frame))
            total_score_entries = select_consensus_window(total_score_entries, "center", size=TOTAL_SCORE_CONSENSUS_WINDOW_SIZE)
            if not total_score_entries and anchor_path is not None:
                anchor_frame = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
                if anchor_frame is not None:
                    total_score_entries.append((0, anchor_frame))
            total_score_layout_id = score_layout_id_from_filename(anchor_path)
        if total_score_entries:
            scaled_image_layout = score_digit_layout(5, score_layout_id=total_score_layout_id)
            for frame_number, frame_image in total_score_entries:
                digit_image = prepare_digit_image(frame_image, total_score_layout_id)
                args_tuple = scaled_image_layout["total_points"]
                row_values, fallback_flags = detect_digits_without_fallback(digit_image, *args_tuple)
                for row_index, (row_value, fallback_flag) in enumerate(zip(row_values, fallback_flags), start=1):
                    expected_value = new_total_expected.get((race_number, row_index))
                    if expected_value is None:
                        continue
                    _numeric_missing, has_unknown_digit, contiguous_digit_block = fallback_flag
                    should_fallback = parse_detected_int(row_value) is None or not (0 <= int(parse_detected_int(row_value) or -1) <= 999)
                    if has_unknown_digit and not contiguous_digit_block:
                        should_fallback = True
                    summary["NewTotalScore"]["rows"] += 1
                    if should_fallback:
                        summary["NewTotalScore"]["fallback_rows"] += 1
                        fallback_value = ocr_digit_row_fallback(
                            digit_image,
                            args_tuple[0],
                            args_tuple[1],
                            args_tuple[2],
                            row_index - 1,
                            args_tuple[5],
                            valid_min=0,
                            valid_max=999,
                            bundle_kind="audit",
                            field_name="NewTotalScore",
                        )
                        outcome = classify_outcome(row_value, fallback_value, expected_value)
                        summary["NewTotalScore"][outcome] += 1
                        example_rows.append(
                            {
                                "Field": "NewTotalScore",
                                "Race": race_number,
                                "Frame": frame_number,
                                "Row": row_index,
                                "Expected": expected_value,
                                "SevenSeg": row_value,
                                "Fallback": fallback_value,
                                "Outcome": outcome,
                            }
                        )
                    else:
                        if parse_detected_int(row_value) == expected_value:
                            summary["NewTotalScore"]["seven_correct_no_fallback"] += 1
                        else:
                            summary["NewTotalScore"]["seven_wrong_no_fallback"] += 1

    summary_rows = []
    for field_name in ("RacePoints", "OldTotalScore", "NewTotalScore"):
        counter = summary[field_name]
        summary_rows.append(
            {
                "Field": field_name,
                "Rows": counter["rows"],
                "FallbackRows": counter["fallback_rows"],
                "Helped": counter["helped"],
                "WouldHurt": counter["would_hurt"],
                "NoHelpWrong": counter["no_help_wrong"],
                "NoHelpBlank": counter["no_help_blank"],
                "NotNeeded": counter["not_needed_same"] + counter["not_needed_blank"],
                "SevenCorrectNoFallback": counter["seven_correct_no_fallback"],
                "SevenWrongNoFallback": counter["seven_wrong_no_fallback"],
            }
        )

    print(format_table("Digit Fallback Summary", summary_rows))
    print()
    ordered_examples = sorted(
        example_rows,
        key=lambda row: (
            {"helped": 0, "would_hurt": 1, "no_help_wrong": 2, "no_help_blank": 3, "not_needed_same": 4, "not_needed_blank": 5}.get(str(row["Outcome"]), 9),
            str(row["Field"]),
            int(row["Race"]),
            int(row["Frame"]),
            int(row["Row"]),
        ),
    )
    print(format_table("Digit Fallback Examples", ordered_examples[: args.show_rows]))


if __name__ == "__main__":
    main()
