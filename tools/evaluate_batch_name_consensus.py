#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

from mk8_local_play.app_runtime import configure_tesseract, load_app_config
from mk8_local_play.extract_text import (
    PLAYER_NAME_BATCH_CONFIG,
    PLAYER_NAME_BATCH_HORIZONTAL_PADDING,
    PLAYER_NAME_BATCH_SEPARATOR_HEIGHT,
    PLAYER_NAME_BATCH_VERTICAL_PADDING,
    run_tesseract_image_to_data,
)
from mk8_local_play.ocr_common import find_metadata_entry, load_consensus_frames, load_exported_frame_metadata
from mk8_local_play.ocr_name_matching import preprocess_name, weighted_similarity
from mk8_local_play.ocr_scoreboard_consensus import (
    PLAYER_NAME_COORDS,
    process_image,
    build_normalized_position_strip,
    position_strip_roi,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare batch-only 7-frame name consensus against current OCR output.")
    parser.add_argument("--video", required=True, help="Video stem, e.g. Divisie_1")
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("Output_Results") / "Debug" / "20260311_153835_Tournament_Results_Debug.xlsx",
        help="Current debug workbook to compare against.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(".codex_tmp") / "batch_name_consensus_eval.json",
        help="Write detailed JSON output here.",
    )
    return parser.parse_args()


def build_batch_canvas(image: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rois = []
    widths = []
    heights = []
    for (x1, y1), (x2, y2) in PLAYER_NAME_COORDS:
        roi = image[y1:y2, x1:x2]
        rois.append(roi)
        heights.append(max(1, roi.shape[0]))
        widths.append(max(1, roi.shape[1]))
    canvas_width = max(widths) + PLAYER_NAME_BATCH_HORIZONTAL_PADDING * 2
    canvas_height = (
        sum(height + PLAYER_NAME_BATCH_VERTICAL_PADDING * 2 for height in heights)
        + PLAYER_NAME_BATCH_SEPARATOR_HEIGHT * (len(rois) - 1)
    )
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    row_ranges: list[tuple[int, int]] = []
    cursor = 0
    for roi in rois:
        height, width = roi.shape[:2]
        start_y = cursor + PLAYER_NAME_BATCH_VERTICAL_PADDING
        end_y = start_y + height
        start_x = PLAYER_NAME_BATCH_HORIZONTAL_PADDING
        end_x = start_x + width
        canvas[start_y:end_y, start_x:end_x] = roi
        row_ranges.append((start_y, end_y))
        cursor = end_y + PLAYER_NAME_BATCH_VERTICAL_PADDING + PLAYER_NAME_BATCH_SEPARATOR_HEIGHT
    return canvas, row_ranges


def extract_batch_names_only(frame_image: np.ndarray) -> tuple[list[str], list[int]]:
    processed_img = process_image(frame_image)
    (position_x1, position_y1), (position_x2, position_y2) = position_strip_roi()
    normalized_position_strip = build_normalized_position_strip(processed_img)
    processed_img[position_y1:position_y2, position_x1:position_x2] = cv2.cvtColor(
        normalized_position_strip,
        cv2.COLOR_GRAY2BGR,
    )

    processed_img_pil = Image.fromarray(processed_img).convert("RGB")
    scaled_image = processed_img_pil.resize(
        (processed_img_pil.width * 5, processed_img_pil.height * 5),
        Image.Resampling.NEAREST,
    )
    scaled_image_resized = scaled_image.resize(
        (processed_img_pil.width, processed_img_pil.height),
        Image.Resampling.NEAREST,
    )
    annotated_image = cv2.cvtColor(np.array(scaled_image_resized), cv2.COLOR_RGB2BGR)

    canvas, row_ranges = build_batch_canvas(annotated_image)
    data = run_tesseract_image_to_data(canvas, "eng", PLAYER_NAME_BATCH_CONFIG, "eval_player_name_batch")
    texts_by_row = [[] for _ in PLAYER_NAME_COORDS]
    confidences_by_row = [[] for _ in PLAYER_NAME_COORDS]
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        conf = int(data["conf"][index])
        if conf < 0 or not text:
            continue
        y = int(data["top"][index])
        height = int(data["height"][index])
        center_y = y + max(1, height // 2)
        for row_index, (start_y, end_y) in enumerate(row_ranges):
            if start_y <= center_y < end_y:
                texts_by_row[row_index].append(text)
                confidences_by_row[row_index].append(conf)
                break

    names = []
    confidences = []
    for row_index in range(len(PLAYER_NAME_COORDS)):
        combined_text = " ".join(texts_by_row[row_index]).strip()
        average_confidence = int(sum(confidences_by_row[row_index]) // len(confidences_by_row[row_index])) if confidences_by_row[row_index] else 0
        names.append(combined_text)
        confidences.append(average_confidence)
    return names, confidences


def weighted_name_vote(values: list[tuple[str, float]]) -> tuple[str, float]:
    votes: dict[str, float] = defaultdict(float)
    for name, weight in values:
        normalized = re.sub(r"\s+", " ", str(name or "").strip())
        if not normalized:
            continue
        votes[normalized] += max(1.0, float(weight))
    if not votes:
        return "", 0.0
    best_name, best_weight = max(votes.items(), key=lambda item: item[1])
    total_weight = sum(votes.values())
    confidence = (best_weight / total_weight) if total_weight else 0.0
    return best_name, confidence


def roster_consensus(values: list[tuple[str, float]], roster: list[str]) -> tuple[str, float]:
    candidate_scores: dict[str, float] = defaultdict(float)
    for observed_name, observed_weight in values:
        normalized_observed = preprocess_name(observed_name)
        if not normalized_observed:
            continue
        for roster_name in roster:
            similarity = weighted_similarity(observed_name, roster_name)
            if similarity < 0.55:
                continue
            candidate_scores[roster_name] += similarity * max(1.0, float(observed_weight))
    if not candidate_scores:
        return "", 0.0
    winner, winner_score = max(candidate_scores.items(), key=lambda item: item[1])
    total_score = sum(candidate_scores.values())
    confidence = (winner_score / total_score) if total_score else 0.0
    return winner, confidence


def main() -> int:
    args = parse_args()
    configure_tesseract(pytesseract, load_app_config())

    workbook_df = pd.read_excel(args.workbook)
    video_df = workbook_df[workbook_df["Video"].astype(str) == args.video].copy()
    if video_df.empty:
        raise SystemExit(f"No rows for video {args.video} in {args.workbook}")

    roster = sorted({str(value).strip() for value in video_df["Standardized Player"].dropna() if str(value).strip()})

    base_dir = Path.cwd()
    metadata = load_exported_frame_metadata(base_dir)
    frames_dir = base_dir / "Output_Results" / "Frames"
    input_dir = base_dir / "Input_Videos"

    race_summaries = []
    exact_raw_matches = 0
    exact_standardized_matches = 0
    total_rows = 0

    for race_number in sorted(video_df["Race"].astype(int).unique()):
        race_rows = video_df[video_df["Race"].astype(int) == race_number].sort_values("Position")
        race_metadata = find_metadata_entry(metadata, args.video, int(race_number), "RaceScore")
        race_image = str(frames_dir / f"{args.video}+Race_{int(race_number):03d}+2RaceScore.png")
        race_frames = load_consensus_frames(race_image, race_metadata, input_dir, 7)
        frame_outputs = [extract_batch_names_only(frame) for frame in race_frames]

        per_row_details = []
        for row_position, (_, workbook_row) in enumerate(race_rows.iterrows(), start=1):
            votes = []
            for names, confidences in frame_outputs:
                name = names[row_position - 1] if row_position - 1 < len(names) else ""
                confidence = confidences[row_position - 1] if row_position - 1 < len(confidences) else 0
                votes.append((name, confidence))

            raw_consensus, raw_consensus_confidence = weighted_name_vote(votes)
            roster_name, roster_confidence = roster_consensus(votes, roster)
            current_raw = str(workbook_row["Raw Player OCR"] or "").strip()
            current_standardized = str(workbook_row["Standardized Player"] or "").strip()

            raw_match = preprocess_name(raw_consensus) == preprocess_name(current_raw)
            standardized_match = preprocess_name(roster_name) == preprocess_name(current_standardized)
            exact_raw_matches += int(raw_match)
            exact_standardized_matches += int(standardized_match)
            total_rows += 1

            per_row_details.append(
                {
                    "position": row_position,
                    "batch_votes": [{"text": name, "confidence": confidence} for name, confidence in votes],
                    "raw_consensus": raw_consensus,
                    "raw_consensus_confidence": round(raw_consensus_confidence * 100, 1),
                    "roster_consensus": roster_name,
                    "roster_consensus_confidence": round(roster_confidence * 100, 1),
                    "current_raw_player_ocr": current_raw,
                    "current_standardized_player": current_standardized,
                    "raw_match": raw_match,
                    "standardized_match": standardized_match,
                }
            )

        race_summaries.append(
            {
                "race": int(race_number),
                "track": str(race_rows["Track"].iloc[0]),
                "rows": per_row_details,
            }
        )

    summary = {
        "video": args.video,
        "workbook": str(args.workbook),
        "roster_size": len(roster),
        "total_rows": total_rows,
        "raw_consensus_exact_match_rows": exact_raw_matches,
        "raw_consensus_exact_match_pct": round((exact_raw_matches / total_rows) * 100, 1) if total_rows else 0.0,
        "roster_consensus_exact_match_rows": exact_standardized_matches,
        "roster_consensus_exact_match_pct": round((exact_standardized_matches / total_rows) * 100, 1) if total_rows else 0.0,
        "races": race_summaries,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Video: {args.video}")
    print(f"Rows compared: {total_rows}")
    print(f"Batch-only raw consensus exact match: {exact_raw_matches}/{total_rows} ({summary['raw_consensus_exact_match_pct']}%)")
    print(f"Batch-only roster consensus exact match: {exact_standardized_matches}/{total_rows} ({summary['roster_consensus_exact_match_pct']}%)")
    print(f"Detailed JSON: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
