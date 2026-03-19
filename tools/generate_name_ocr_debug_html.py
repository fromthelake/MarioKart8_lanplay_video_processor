#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import html
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from mk8_local_play.extract_common import find_score_bundle_anchor_path, find_score_bundle_consensus_paths
from mk8_local_play.extract_text import (
    PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE,
    _run_easyocr_player_name,
    _run_easyocr_player_names_batched,
    extract_player_names_batched,
)
from mk8_local_play.name_unicode import distinct_visible_name_count, visible_name_length
from mk8_local_play.ocr_common import find_metadata_entry, load_consensus_frames, load_exported_frame_metadata
from mk8_local_play.ocr_name_matching import preprocess_name, weighted_similarity
from mk8_local_play.ocr_scoreboard_consensus import (
    PLAYER_NAME_COORDS,
    build_consensus_observation,
    build_normalized_position_strip,
    position_strip_roi,
    process_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML debug view for one race name-OCR bundle.")
    parser.add_argument("--video", required=True, help="Race class / video stem, e.g. Divisie_1")
    parser.add_argument("--race", required=True, type=int, help="Race number, e.g. 1")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".codex_tmp") / "name_fallback_examples" / "race_name_debug.html",
        help="Output HTML path.",
    )
    return parser.parse_args()


def image_to_data_uri(image: np.ndarray, scale: int = 1, draw_boxes: list[tuple[int, int, int, int]] | None = None) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if scale > 1:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    pil_image = Image.fromarray(rgb)
    if draw_boxes:
        draw = ImageDraw.Draw(pil_image)
        for x, y, w, h in draw_boxes:
            sx = x * scale
            sy = y * scale
            sw = w * scale
            sh = h * scale
            draw.rectangle([sx, sy, sx + sw, sy + sh], outline=(0, 255, 0), width=2)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def full_frame_to_data_uri(
    image: np.ndarray,
    highlight_row: int | None = None,
    ocr_boxes: list[tuple[int, int, int, int]] | None = None,
    fallback_boxes: list[tuple[int, int, int, int]] | None = None,
) -> str:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)
    if highlight_row is not None:
        (x1, y1), (x2, y2) = PLAYER_NAME_COORDS[highlight_row]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    if ocr_boxes:
        for x, y, w, h in ocr_boxes:
            draw.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=2)
    if fallback_boxes:
        for x, y, w, h in fallback_boxes:
            draw.rectangle([x, y, x + w, y + h], outline=(0, 128, 255), width=2)
    pil_image = pil_image.resize((320, 180), Image.Resampling.NEAREST)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
def analyze_frame(image: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray, list[tuple[int, int, int, int]]]:
    # Actual runtime-faithful name-OCR input path.
    processed_img = process_image(image)
    (position_x1, position_y1), (position_x2, position_y2) = position_strip_roi()
    normalized_position_strip = build_normalized_position_strip(processed_img)
    processed_img[position_y1:position_y2, position_x1:position_x2] = cv2.cvtColor(normalized_position_strip, cv2.COLOR_GRAY2BGR)
    processed_img_pil = Image.fromarray(processed_img).convert("RGB")
    scaled_image = processed_img_pil.resize(
        (processed_img_pil.width * 5, processed_img_pil.height * 5),
        Image.Resampling.NEAREST,
    )
    scaled_image_resized = scaled_image.resize((processed_img_pil.width, processed_img_pil.height), Image.Resampling.NEAREST)
    annotated_image = cv2.cvtColor(np.array(scaled_image_resized), cv2.COLOR_RGB2BGR)
    batch_names, batch_confidences = _run_easyocr_player_names_batched(annotated_image, PLAYER_NAME_COORDS)

    details: list[dict[str, Any]] = []
    for row_index, ((x1, y1), (x2, y2)) in enumerate(PLAYER_NAME_COORDS):
        batch_text = batch_names[row_index] if row_index < len(batch_names) else ""
        batch_conf = batch_confidences[row_index] if row_index < len(batch_confidences) else 0
        low_conf = batch_conf < PLAYER_NAME_BATCH_FALLBACK_CONFIDENCE
        short_text = visible_name_length(batch_text) < 3
        low_div = distinct_visible_name_count(batch_text) < 3
        fallback_needed = low_conf or short_text or low_div

        row_roi = annotated_image[y1:y2, x1:x2]
        fallback_text = batch_text
        fallback_conf = batch_conf
        fallback_boxes: list[tuple[int, int, int, int]] = []
        if fallback_needed:
            fallback_text, fallback_conf = _run_easyocr_player_name(annotated_image, x1, y1, x2, y2)

        details.append(
            {
                "row": row_index + 1,
                "batch_text": batch_text,
                "batch_conf": batch_conf,
                "fallback_needed": fallback_needed,
                "fallback_text": fallback_text,
                "fallback_conf": fallback_conf,
                "delta_conf": fallback_conf - batch_conf,
                "reasons": [
                    reason
                    for reason, active in (
                        ("low_confidence", low_conf),
                        ("short_text", short_text),
                        ("low_diversity", low_div),
                    )
                    if active
                ],
                "preprocessed_frame_uri": full_frame_to_data_uri(
                    annotated_image,
                    row_index,
                    ocr_boxes=[],
                    fallback_boxes=[],
                ),
                "preprocessed_frame_batch_uri": full_frame_to_data_uri(
                    annotated_image,
                    row_index,
                ),
                "preprocessed_frame_combined_uri": full_frame_to_data_uri(
                    annotated_image,
                    row_index,
                    fallback_boxes=[],
                ),
                "row_roi_uri": image_to_data_uri(row_roi, scale=6, draw_boxes=fallback_boxes),
                "preprocessed_row_roi_uri": image_to_data_uri(annotated_image[y1:y2, x1:x2], scale=6),
                "batch_row_uri": image_to_data_uri(annotated_image[y1:y2, x1:x2], scale=6),
            }
        )
    return details, annotated_image, []


def render_html(
    *,
    video: str,
    race: int,
    source_frame_numbers: list[int],
    frame_details: list[list[dict[str, Any]]],
    consensus_rows: list[dict[str, Any]],
) -> str:
    consensus_by_row = {int(row["RacePosition"]): row for row in consensus_rows}
    frame_headers = []
    for index, source_frame in enumerate(source_frame_numbers, start=1):
        frame_headers.append(f"<th>Frame {index}<br><small>src {source_frame}</small></th>")

    rows_html = []
    for row_number in range(1, 13):
        consensus = consensus_by_row.get(row_number, {})
        cells = []
        for frame_index, details in enumerate(frame_details, start=1):
            item = details[row_number - 1]
            batch_line = f"Batch: <code>{html.escape(item['batch_text'])}</code> ({item['batch_conf']})"
            fallback_line = ""
            if item["fallback_needed"]:
                fallback_line = (
                    f"<br>Fallback: <code>{html.escape(item['fallback_text'])}</code> "
                    f"({item['fallback_conf']}, delta {item['delta_conf']:+d})"
                    f"<br><small>{', '.join(item['reasons'])}</small>"
                )
            cells.append(
                "<td>"
                f"{batch_line}{fallback_line}"
                "<br>"
                f"<img src=\"{item['row_roi_uri']}\" alt=\"row roi raw\">"
                "<br><small>Preprocessed row ROI</small><br>"
                f"<img src=\"{item['preprocessed_row_roi_uri']}\" alt=\"row roi preprocessed\">"
                "<br>"
                f"<details><summary>Batch row / frame</summary>"
                f"<img src=\"{item['batch_row_uri']}\" alt=\"batch row\">"
                "<br><small>Preprocessed frame with ROI + batch + fallback boxes</small><br>"
                f"<img src=\"{item['preprocessed_frame_combined_uri']}\" alt=\"frame overlay preprocessed combined\">"
                "</details>"
                "</td>"
            )
        consensus_line = (
            f"<strong>{html.escape(str(consensus.get('PlayerName', '')))}</strong>"
            f"<br>Confidence: {consensus.get('NameConfidence', '')}"
        )
        rows_html.append(
            "<tr>"
            f"<td><strong>{row_number}</strong></td>"
            f"{''.join(cells)}"
            f"<td>{consensus_line}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Name OCR Debug - {html.escape(video)} race {race}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f6f3ec; color: #1f1b16; }}
    table {{ border-collapse: collapse; width: 100%; table-layout: fixed; }}
    th, td {{ border: 1px solid #c8bba7; vertical-align: top; padding: 8px; background: #fffaf3; }}
    th {{ background: #e9dcc9; }}
    td img {{ max-width: 100%; height: auto; border: 1px solid #b7a78e; background: #111; }}
    code {{ background: #efe5d6; padding: 1px 4px; }}
    .intro {{ max-width: 1200px; line-height: 1.4; }}
  </style>
</head>
<body>
  <h1>Name OCR Debug</h1>
  <div class="intro">
    <p><strong>Video:</strong> {html.escape(video)} | <strong>Race:</strong> {race}</p>
    <p>This table shows one race with all 12 rows. Each frame column is one of the 7 nearby RaceScore frames used for OCR voting. For each frame, the code first runs the batched EasyOCR path used by runtime. If that row looks weak, it OCRs the single row again as a fallback. The last column shows the final consensus result used for the race output.</p>
    <p><strong>How to read it:</strong> “Batch” is what the batched EasyOCR path saw for that row in that frame. “Fallback” is the second OCR pass on just that row, if triggered. All visuals on this page come from the same preprocessed <code>annotated_image</code> that the runtime passes into name OCR.</p>
    <p><strong>Legend:</strong> red = selected row ROI.</p>
  </div>
  <table>
    <thead>
      <tr>
        <th>Row</th>
        {''.join(frame_headers)}
        <th>Consensus</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""


def main() -> int:
    args = parse_args()
    base_dir = Path.cwd()
    metadata = load_exported_frame_metadata(base_dir)
    frames_dir = base_dir / "Output_Results" / "Frames"
    input_dir = base_dir / "Input_Videos"

    race_metadata = find_metadata_entry(metadata, args.video, args.race, "RaceScore")
    total_metadata = find_metadata_entry(metadata, args.video, args.race, "TotalScore")
    race_anchor = find_score_bundle_anchor_path(args.video, args.race, "2RaceScore")
    total_anchor = find_score_bundle_anchor_path(args.video, args.race, "3TotalScore")
    race_image = str(race_anchor) if race_anchor is not None else ""
    total_image = str(total_anchor) if total_anchor is not None else ""
    race_frames = load_consensus_frames(race_image, race_metadata, input_dir, 7)
    total_frames = load_consensus_frames(total_image, total_metadata, input_dir, 7)

    consensus_paths = find_score_bundle_consensus_paths(args.video, args.race, "2RaceScore")
    if consensus_paths:
        source_frame_numbers = [
            int(path.stem.split("_", 1)[1])
            for path in consensus_paths
            if "_" in path.stem
        ]
    else:
        actual_frame = int(race_metadata["actual_frame"]) if race_metadata else 0
        source_frame_numbers = list(range(actual_frame - 3, actual_frame + 4))
    frame_details = []
    for frame in race_frames:
        details, _canvas, _ranges = analyze_frame(frame)
        frame_details.append(details)

    consensus = build_consensus_observation(
        race_frames,
        total_frames,
        extract_player_names_batched,
        preprocess_name,
        weighted_similarity,
        None,
        video_context=args.video,
    )

    html_content = render_html(
        video=args.video,
        race=args.race,
        source_frame_numbers=source_frame_numbers,
        frame_details=frame_details,
        consensus_rows=consensus["rows"],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_content, encoding="utf-8")

    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "video": args.video,
                "race": args.race,
                "source_frame_numbers": source_frame_numbers,
                "consensus_rows": consensus["rows"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
