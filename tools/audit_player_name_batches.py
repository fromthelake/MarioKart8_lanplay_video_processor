import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from mk8_local_play.extract_text import (
    OCR_CONSENSUS_FRAMES,
    PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE,
    _player_name_candidate_score,
    _run_easyocr_player_name_for_context,
    _run_easyocr_player_names_batched_variant,
)
from mk8_local_play.ocr_common import (
    find_metadata_entry,
    load_consensus_frame_entries,
    load_exported_frame_metadata,
)
from mk8_local_play.ocr_name_matching import preprocess_name, weighted_similarity
from mk8_local_play.ocr_scoreboard_consensus import process_image
from mk8_local_play.ocr_scoreboard_consensus import TOTAL_SCORE_CONSENSUS_WINDOW_SIZE, select_consensus_window
from mk8_local_play.project_paths import PROJECT_ROOT
from mk8_local_play.score_layouts import get_score_layout


def prepare_name_image(frame_image, score_layout_id: str):
    processed_img = process_image(frame_image, score_layout_id=score_layout_id)
    processed_img_pil = Image.fromarray(processed_img).convert("RGB")
    scaled_image = processed_img_pil.resize(
        (processed_img_pil.width * 5, processed_img_pil.height * 5),
        Image.NEAREST,
    )
    scaled_image_resized = scaled_image.resize((processed_img_pil.width, processed_img_pil.height), Image.NEAREST)
    return cv2.cvtColor(np.array(scaled_image_resized), cv2.COLOR_RGB2BGR)


def build_expected_maps(results_csv: Path, race_class: str):
    df = pd.read_csv(results_csv)
    df = df[df["Video"] == race_class].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for video '{race_class}' in {results_csv}")

    race_expected = {}
    total_expected = {}
    for row in df.to_dict(orient="records"):
        video = str(row.get("Video", ""))
        race = int(row.get("Race", 0) or 0)
        position = int(row.get("Position", 0) or 0)
        player = str(row.get("Player", ""))
        if video and race > 0 and position > 0 and player:
            race_expected[(video, race, position)] = player
        position_after = row.get("Position After Race")
        try:
            if pd.notna(position_after):
                total_expected[(video, race, int(position_after))] = player
        except Exception:
            pass
    return race_expected, total_expected


def evaluate_candidate(candidate_text: str, confidence: int, expected_name: str):
    normalized_candidate = preprocess_name(candidate_text)
    normalized_expected = preprocess_name(expected_name)
    exact = int(bool(normalized_candidate and normalized_candidate == normalized_expected))
    similarity = weighted_similarity(candidate_text, expected_name) if candidate_text else 0.0
    return {
        "text": candidate_text,
        "confidence": int(confidence),
        "exact": exact,
        "similarity": similarity,
        "score": _player_name_candidate_score(candidate_text, int(confidence)) if candidate_text else float("-inf"),
    }


def variant_winner(raw_eval, inv_eval):
    if raw_eval["exact"] != inv_eval["exact"]:
        return "raw" if raw_eval["exact"] > inv_eval["exact"] else "inv_otsu"
    if abs(raw_eval["similarity"] - inv_eval["similarity"]) > 1e-9:
        return "raw" if raw_eval["similarity"] > inv_eval["similarity"] else "inv_otsu"
    if raw_eval["confidence"] != inv_eval["confidence"]:
        return "raw" if raw_eval["confidence"] > inv_eval["confidence"] else "inv_otsu"
    return "tie"


def format_summary_table(title: str, rows: list[dict[str, object]]) -> str:
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
            if isinstance(value, (int, float)) and column not in {"Bundle", "Metric", "Winner"}:
                if isinstance(value, int):
                    rendered.append(f"{value:>{widths[column]}}")
                else:
                    rendered.append(f"{value:>{widths[column]}.4f}")
            else:
                safe_text = str(value).encode("cp1252", errors="replace").decode("cp1252")
                rendered.append(f"{safe_text:<{widths[column]}}")
        lines.append("  ".join(rendered))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Audit raw vs inv_otsu player-name batch OCR.")
    parser.add_argument("--race-class", required=True)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--show-rows", type=int, default=20)
    args = parser.parse_args()

    project_root = Path(PROJECT_ROOT)
    results_csv = Path(args.results_csv)
    if not results_csv.is_absolute():
        results_csv = project_root / results_csv

    metadata_index = load_exported_frame_metadata(project_root)
    input_videos_folder = project_root / "Input_Videos"
    race_expected, total_expected = build_expected_maps(results_csv, args.race_class)
    race_numbers = sorted({key[1] for key in race_expected.keys()})

    per_bundle_counter = defaultdict(Counter)
    disagreement_rows = []
    fallback_rows = []

    for race_number in race_numbers:
        for kind, bundle_kind, field_name, expected_map in (
            ("RaceScore", "2RaceScore", "RacePlayerName", race_expected),
            ("TotalScore", "3TotalScore", "TotalPlayerName", total_expected),
        ):
            metadata_entry = find_metadata_entry(metadata_index, args.race_class, race_number, kind)
            if metadata_entry is None:
                continue
            anchor_path = str(metadata_entry.get("anchor_path", "") or "")
            entries = load_consensus_frame_entries(anchor_path, metadata_entry, input_videos_folder, OCR_CONSENSUS_FRAMES)
            if bundle_kind == "3TotalScore":
                entries = select_consensus_window(entries, "center", size=TOTAL_SCORE_CONSENSUS_WINDOW_SIZE)
            score_layout_id = str(metadata_entry.get("score_layout_id", "")).strip()
            layout = get_score_layout(score_layout_id)
            coord_list = layout.player_name_coords

            for frame_number, frame_image in entries:
                prepared = prepare_name_image(frame_image, score_layout_id)
                raw_names, raw_confidences = _run_easyocr_player_names_batched_variant(
                    prepared,
                    coord_list,
                    preprocess="raw",
                )
                inv_names, inv_confidences = _run_easyocr_player_names_batched_variant(
                    prepared,
                    coord_list,
                    preprocess="inv_otsu3",
                )

                for row_index, ((x1, y1), (x2, y2)) in enumerate(coord_list, start=1):
                    expected_name = expected_map.get((args.race_class, race_number, row_index), "")
                    if not expected_name:
                        continue

                    raw_eval = evaluate_candidate(raw_names[row_index - 1], raw_confidences[row_index - 1], expected_name)
                    inv_eval = evaluate_candidate(inv_names[row_index - 1], inv_confidences[row_index - 1], expected_name)
                    winner = variant_winner(raw_eval, inv_eval)

                    counter = per_bundle_counter[bundle_kind]
                    counter["rows"] += 1
                    counter[f"winner_{winner}"] += 1
                    counter["raw_exact"] += raw_eval["exact"]
                    counter["inv_exact"] += inv_eval["exact"]
                    counter["raw_similarity_sum"] += raw_eval["similarity"]
                    counter["inv_similarity_sum"] += inv_eval["similarity"]
                    counter["raw_confidence_sum"] += raw_eval["confidence"]
                    counter["inv_confidence_sum"] += inv_eval["confidence"]

                    if winner != "tie":
                        disagreement_rows.append(
                            {
                                "Bundle": bundle_kind,
                                "Race": race_number,
                                "Frame": frame_number,
                                "Row": row_index,
                                "Expected": expected_name,
                                "Winner": winner,
                                "RawText": raw_eval["text"],
                                "RawConf": raw_eval["confidence"],
                                "RawSim": round(raw_eval["similarity"], 4),
                                "InvText": inv_eval["text"],
                                "InvConf": inv_eval["confidence"],
                                "InvSim": round(inv_eval["similarity"], 4),
                            }
                        )

                    inv_weak = (
                        inv_eval["confidence"] < PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE
                        or len(preprocess_name(inv_eval["text"])) < 3
                    )
                    if inv_weak:
                        fallback_text, fallback_conf = _run_easyocr_player_name_for_context(
                            prepared,
                            x1,
                            y1,
                            x2,
                            y2,
                            bundle_kind=bundle_kind,
                            field_name=field_name,
                            method_prefix="audit_row_fallback",
                        )
                        fallback_eval = evaluate_candidate(fallback_text, fallback_conf, expected_name)
                        counter["inv_weak_rows"] += 1
                        if (
                            fallback_eval["exact"] > inv_eval["exact"]
                            or fallback_eval["similarity"] > inv_eval["similarity"]
                        ):
                            counter["fallback_helped"] += 1
                            fallback_outcome = "helped"
                        else:
                            counter["fallback_not_needed"] += 1
                            fallback_outcome = "not_needed"
                        fallback_rows.append(
                            {
                                "Bundle": bundle_kind,
                                "Race": race_number,
                                "Frame": frame_number,
                                "Row": row_index,
                                "Expected": expected_name,
                                "Outcome": fallback_outcome,
                                "InvText": inv_eval["text"],
                                "InvConf": inv_eval["confidence"],
                                "InvSim": round(inv_eval["similarity"], 4),
                                "FallbackText": fallback_eval["text"],
                                "FallbackConf": fallback_eval["confidence"],
                                "FallbackSim": round(fallback_eval["similarity"], 4),
                            }
                        )

    summary_rows = []
    for bundle_kind in ("2RaceScore", "3TotalScore"):
        counter = per_bundle_counter[bundle_kind]
        rows = max(1, counter["rows"])
        summary_rows.append(
            {
                "Bundle": bundle_kind,
                "Rows": counter["rows"],
                "RawExact%": round((counter["raw_exact"] / rows) * 100.0, 1),
                "InvExact%": round((counter["inv_exact"] / rows) * 100.0, 1),
                "RawSim": round(counter["raw_similarity_sum"] / rows, 4),
                "InvSim": round(counter["inv_similarity_sum"] / rows, 4),
                "RawConf": round(counter["raw_confidence_sum"] / rows, 1),
                "InvConf": round(counter["inv_confidence_sum"] / rows, 1),
                "RawWins": counter["winner_raw"],
                "InvWins": counter["winner_inv_otsu"],
                "Ties": counter["winner_tie"],
                "InvWeak": counter["inv_weak_rows"],
                "FallbackHelped": counter["fallback_helped"],
                "FallbackNotNeeded": counter["fallback_not_needed"],
            }
        )

    overall = Counter()
    for counter in per_bundle_counter.values():
        overall.update(counter)
    total_rows = max(1, overall["rows"])
    overall_rows = [
        {
            "Bundle": "Overall",
            "Rows": overall["rows"],
            "RawExact%": round((overall["raw_exact"] / total_rows) * 100.0, 1),
            "InvExact%": round((overall["inv_exact"] / total_rows) * 100.0, 1),
            "RawSim": round(overall["raw_similarity_sum"] / total_rows, 4),
            "InvSim": round(overall["inv_similarity_sum"] / total_rows, 4),
            "RawConf": round(overall["raw_confidence_sum"] / total_rows, 1),
            "InvConf": round(overall["inv_confidence_sum"] / total_rows, 1),
            "RawWins": overall["winner_raw"],
            "InvWins": overall["winner_inv_otsu"],
            "Ties": overall["winner_tie"],
            "InvWeak": overall["inv_weak_rows"],
            "FallbackHelped": overall["fallback_helped"],
            "FallbackNotNeeded": overall["fallback_not_needed"],
        }
    ]

    print(format_summary_table("Batch Comparison Summary", summary_rows + overall_rows))
    print()

    disagreement_rows = sorted(
        disagreement_rows,
        key=lambda row: (
            0 if row["Winner"] == "raw" else 1,
            abs(float(row["RawSim"]) - float(row["InvSim"])),
        ),
        reverse=True,
    )
    print(format_summary_table("Top Batch Disagreements", disagreement_rows[: args.show_rows]))
    print()

    fallback_rows = sorted(
        fallback_rows,
        key=lambda row: (
            0 if row["Outcome"] == "helped" else 1,
            abs(float(row["FallbackSim"]) - float(row["InvSim"])),
        ),
        reverse=False,
    )
    print(format_summary_table("Fallback Audit", fallback_rows[: args.show_rows]))


if __name__ == "__main__":
    main()
