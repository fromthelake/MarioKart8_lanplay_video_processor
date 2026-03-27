from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from mk8_local_play.data_paths import resolve_asset_file
from mk8_local_play.extract_common import TARGET_HEIGHT, TARGET_WIDTH, crop_and_upscale_image, frame_to_timecode
from mk8_local_play.extract_initial_scan import INITIAL_SCAN_TARGETS, _initial_scan_score_gate, _match_score_target_layouts


TEMPLATE_GRID_X = 313
TEMPLATE_GRID_Y = 46
TEMPLATE_GRID_SIZE = 52


@dataclass
class ScoreTemplate:
    name: str
    gray: object
    alpha_mask: object | None


def _load_alpha_tile(template_name: str, pos_number: int) -> ScoreTemplate:
    template_path = resolve_asset_file("templates", template_name)
    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
    if template is None:
        raise FileNotFoundError(f"Could not load template: {template_path}")
    y = (int(pos_number) - 1) * TEMPLATE_GRID_SIZE
    tile = template[y:y + TEMPLATE_GRID_SIZE, 0:TEMPLATE_GRID_SIZE]
    if tile.shape[0] != TEMPLATE_GRID_SIZE or tile.shape[1] != TEMPLATE_GRID_SIZE:
        raise ValueError(f"Unexpected tile size for {template_name} pos {pos_number}: {tile.shape}")
    if len(tile.shape) == 3 and tile.shape[2] == 4:
        template_gray = cv2.cvtColor(tile[:, :, :3], cv2.COLOR_BGR2GRAY)
        _, alpha_mask = cv2.threshold(tile[:, :, 3], 0, 255, cv2.THRESH_BINARY)
    elif len(tile.shape) == 3 and tile.shape[2] == 3:
        template_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        alpha_mask = None
    else:
        template_gray = tile
        alpha_mask = None
    return ScoreTemplate(template_name, template_gray, alpha_mask)


def _grid_roi(upscaled_image, pos_number: int):
    x1 = int(TEMPLATE_GRID_X)
    y1 = int(TEMPLATE_GRID_Y) + ((int(pos_number) - 1) * int(TEMPLATE_GRID_SIZE))
    x2 = x1 + int(TEMPLATE_GRID_SIZE)
    y2 = y1 + int(TEMPLATE_GRID_SIZE)
    return upscaled_image[y1:y2, x1:x2]


def _match_masked_template(search_gray, template_gray, alpha_mask) -> float:
    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=alpha_mask)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def _best_inverse_tile_score(search_gray, pos_number: int, templates_by_variant: dict[str, dict[int, ScoreTemplate]]) -> tuple[float, str]:
    threshold = float(os.environ.get("MK8_INITIAL_SCAN_GATE_MIN_COEFF", "0.4"))
    best_score = 0.0
    best_name = ""
    for variant_name in ("white", "black"):
        templates_for_positions = templates_by_variant[variant_name]
        template = templates_for_positions[int(pos_number)]
        score = _match_masked_template(search_gray, template.gray, template.alpha_mask)
        if score > best_score:
            best_score = score
            best_name = variant_name
        if score >= threshold:
            return best_score, best_name
    return best_score, best_name


def _experimental_score_value(upscaled, templates_by_variant: dict[str, dict[int, ScoreTemplate]]) -> tuple[float, str]:
    score5_roi = _grid_roi(upscaled, 5)
    score6_roi = _grid_roi(upscaled, 6)
    score5_gray = cv2.cvtColor(score5_roi, cv2.COLOR_BGR2GRAY)
    score6_gray = cv2.cvtColor(score6_roi, cv2.COLOR_BGR2GRAY)
    best_5, variant_5 = _best_inverse_tile_score(score5_gray, 5, templates_by_variant)
    best_6, variant_6 = _best_inverse_tile_score(score6_gray, 6, templates_by_variant)
    combined_score = min(float(best_5), float(best_6))
    return combined_score, f"5={variant_5}:{best_5:.3f}|6={variant_6}:{best_6:.3f}"


def run_probe(video_path: Path, threshold: float, output_csv: Path | None) -> None:
    templates_by_variant = {
        "black": {pos: _load_alpha_tile("Score_template_black.png", pos) for pos in (5, 6)},
        "white": {pos: _load_alpha_tile("Score_template_white.png", pos) for pos in (5, 6)},
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    base_step = max(1, int(round(3.0 * fps)))
    score_skip = int(round(next(target for target in INITIAL_SCAN_TARGETS if target["kind"] == "score")["skip_seconds"] * fps))

    current_frame = 0
    visited = 0
    current_hits = []
    experimental_hits = []
    csv_rows: list[list[object]] = []

    while current_frame < frame_count:
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = capture.read()
        if not ret:
            break
        actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        upscaled = crop_and_upscale_image(frame, 0, 0, frame.shape[1], frame.shape[0], TARGET_WIDTH, TARGET_HEIGHT)

        stats = defaultdict(float)
        gate_result = _initial_scan_score_gate(upscaled, stats)
        if gate_result["passed"]:
            current_score, rejected_as_blank, score_layout_id = _match_score_target_layouts(
                upscaled,
                [],
                stats,
                stats_scope="probe",
            )
            current_pass = bool((not rejected_as_blank) and current_score > float(next(target for target in INITIAL_SCAN_TARGETS if target["kind"] == "score")["match_threshold"]))
        else:
            current_score = 0.0
            current_pass = False
            score_layout_id = str(gate_result["layout_id"])

        experimental_score, experimental_template = _experimental_score_value(upscaled, templates_by_variant)
        experimental_pass = experimental_score >= float(threshold)

        timecode = frame_to_timecode(actual_frame, fps)
        csv_rows.append([
            video_path.name,
            actual_frame,
            timecode,
            current_pass,
            float(current_score),
            str(score_layout_id),
            bool(gate_result["passed"]),
            float(gate_result["max_val"]),
            experimental_pass,
            float(experimental_score),
            experimental_template,
        ])

        if current_pass:
            current_hits.append((actual_frame, timecode, float(current_score), str(score_layout_id)))
        if experimental_pass:
            experimental_hits.append((actual_frame, timecode, float(experimental_score), experimental_template))

        visited += 1
        advance = base_step
        if current_pass or experimental_pass:
            advance = max(advance, score_skip)
        current_frame = actual_frame + advance

    capture.release()

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow([
                "Video",
                "Frame",
                "Timecode",
                "CurrentPass",
                "CurrentScore",
                "CurrentLayout",
                "CurrentGatePass",
                "CurrentGateScore",
                "ExperimentalPass",
                "ExperimentalScore",
                "ExperimentalTemplate",
            ])
            writer.writerows(csv_rows)

    print(f"Video: {video_path.name}")
    print(f"Visited scan frames: {visited}")
    print(f"Base step: {base_step} frames")
    print(f"Current score hits: {len(current_hits)}")
    print(f"Experimental score hits: {len(experimental_hits)} | threshold {threshold:.3f}")
    print("Current hits:")
    for frame_number, timecode, score, layout_id in current_hits[:40]:
        print(f"  {frame_number:>7} | {timecode} | {score:.3f} | {layout_id}")
    print("Experimental hits:")
    for frame_number, timecode, score, template_name in experimental_hits[:40]:
        print(f"  {frame_number:>7} | {timecode} | {score:.3f} | {template_name}")
    if output_csv is not None:
        print(f"CSV: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental first-scan score probe using black/white score templates.")
    parser.add_argument("--video", required=True, help="Video filename or path")
    parser.add_argument("--threshold", type=float, default=0.60, help="Experimental score threshold")
    parser.add_argument("--csv", help="Optional CSV output path")
    args = parser.parse_args()

    video_arg = Path(args.video)
    if not video_arg.exists():
        candidate = Path("Input_Videos") / args.video
        if candidate.exists():
            video_arg = candidate
    output_csv = Path(args.csv) if args.csv else Path("Output_Results") / "Debug" / f"{video_arg.stem}_score_template_probe.csv"
    run_probe(video_arg, threshold=float(args.threshold), output_csv=output_csv)


if __name__ == "__main__":
    main()
