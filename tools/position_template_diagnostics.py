from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from mk8_local_play.ocr_scoreboard_consensus import POSITION_TEMPLATE_FILENAME, load_position_row_templates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = PROJECT_ROOT / "Output_Results" / "Debug" / "Position_ROI_Checks"
REPORT_DIR = DEBUG_DIR / "reports"
FALSE_NEGATIVE_WEIGHT = 2.0


EXPECTED_SEQUENCES: Dict[Tuple[str, str], List[int | None]] = {
    ("006", "RaceScore"): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, None],
    ("006", "TotalScore"): [1, 1, 1, 4, 5, 6, 7, 8, 9, 10, None, None],
    ("008", "RaceScore"): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, None],
    ("008", "TotalScore"): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None, None],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an HTML report with template coefficient scores and overlay images.",
    )
    parser.add_argument(
        "--video-prefix",
        default="Stolk Staal Mario Kart toernooi - Oktober 2025 - Kwalificatie poule B",
        help="The filename prefix used in Position_ROI_Checks.",
    )
    return parser.parse_args()


def load_row_crop(video_prefix: str, race_id: str, screen: str, row_number: int) -> np.ndarray:
    crop_path = DEBUG_DIR / f"{video_prefix}_Race_{race_id}_{screen}_row_{row_number:02}_match_processed.png"
    image = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing processed row crop: {crop_path}")
    return image


def template_match_with_location(source_image: np.ndarray, template_image: np.ndarray) -> Tuple[float, Tuple[int, int]]:
    source_height, source_width = source_image.shape[:2]
    template_height, template_width = template_image.shape[:2]
    if source_height < template_height or source_width < template_width:
        source_image = cv2.resize(source_image, (template_width, template_height), interpolation=cv2.INTER_NEAREST)
        source_height, source_width = source_image.shape[:2]
    result = cv2.matchTemplate(source_image, template_image, cv2.TM_CCOEFF_NORMED)
    _, max_value, _, max_loc = cv2.minMaxLoc(result)
    return float(max_value), (int(max_loc[0]), int(max_loc[1]))


def best_white_overlap_metrics(source_image: np.ndarray, template_image: np.ndarray) -> Dict[str, object]:
    source_height, source_width = source_image.shape[:2]
    template_height, template_width = template_image.shape[:2]
    if source_height < template_height or source_width < template_width:
        source_image = cv2.resize(source_image, (template_width, template_height), interpolation=cv2.INTER_NEAREST)
        source_height, source_width = source_image.shape[:2]

    template_white = template_image > 200
    best_metrics = None
    for offset_y in range(source_height - template_height + 1):
        for offset_x in range(source_width - template_width + 1):
            window = source_image[offset_y:offset_y + template_height, offset_x:offset_x + template_width]
            row_white = window > 200
            true_positive = int(np.count_nonzero(template_white & row_white))
            false_positive = int(np.count_nonzero((~template_white) & row_white))
            false_negative = int(np.count_nonzero(template_white & (~row_white)))
            union = true_positive + false_positive + false_negative
            weighted_union = true_positive + false_positive + (FALSE_NEGATIVE_WEIGHT * false_negative)
            white_iou = true_positive / union if union else 0.0
            weighted_white_iou = true_positive / weighted_union if weighted_union else 0.0
            white_f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative) if (2 * true_positive + false_positive + false_negative) else 0.0
            metrics = {
                "white_iou": float(white_iou),
                "weighted_white_iou": float(weighted_white_iou),
                "white_f1": float(white_f1),
                "tp": int(true_positive),
                "fp": int(false_positive),
                "fn": int(false_negative),
                "location": (int(offset_x), int(offset_y)),
            }
            if best_metrics is None or metrics["weighted_white_iou"] > best_metrics["weighted_white_iou"]:
                best_metrics = metrics
    return best_metrics or {
        "white_iou": 0.0,
        "weighted_white_iou": 0.0,
        "white_f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "location": (0, 0),
    }


def score_templates(row_image: np.ndarray, templates: List[np.ndarray]) -> List[Dict[str, object]]:
    scores = []
    for template_index, template_image in enumerate(templates, start=1):
        coefficient, location = template_match_with_location(row_image, template_image)
        overlap_metrics = best_white_overlap_metrics(row_image, template_image)
        scores.append(
            {
                "template_index": template_index,
                "coefficient": coefficient,
                "location": location,
                "white_iou": float(overlap_metrics["white_iou"]),
                "weighted_white_iou": float(overlap_metrics["weighted_white_iou"]),
                "white_f1": float(overlap_metrics["white_f1"]),
                "iou_location": overlap_metrics["location"],
                "template_image": template_image,
            }
        )
    return scores


def make_overlay(row_image: np.ndarray, template_image: np.ndarray, location: Tuple[int, int], color_bgr: Tuple[int, int, int]) -> np.ndarray:
    if len(row_image.shape) == 2:
        base = cv2.cvtColor(row_image, cv2.COLOR_GRAY2BGR)
    else:
        base = row_image.copy()
    overlay = base.copy()
    template_height, template_width = template_image.shape[:2]
    x, y = location
    x2 = min(base.shape[1], x + template_width)
    y2 = min(base.shape[0], y + template_height)
    template_crop = template_image[: y2 - y, : x2 - x]
    mask = template_crop == 255
    region = overlay[y:y2, x:x2]
    region[mask] = color_bgr
    cv2.rectangle(overlay, (x, y), (max(x, x2 - 1), max(y, y2 - 1)), color_bgr, 1)
    return cv2.addWeighted(overlay, 0.55, base, 0.45, 0)


def write_overlay_set(
    video_prefix: str,
    race_id: str,
    screen: str,
    row_number: int,
    row_image: np.ndarray,
    best_score: Dict[str, object],
    expected_score: Dict[str, object] | None,
) -> Dict[str, str]:
    overlay_dir = REPORT_DIR / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    best_overlay = make_overlay(
        row_image,
        best_score["template_image"],
        best_score["location"],
        (57, 255, 20),
    )
    best_name = f"{video_prefix}_Race_{race_id}_{screen}_row_{row_number:02}_best_overlay.png"
    cv2.imwrite(str(overlay_dir / best_name), best_overlay)

    result = {"best": f"overlays/{best_name}"}
    if expected_score is not None:
        expected_overlay = make_overlay(
            row_image,
            expected_score["template_image"],
            expected_score["location"],
            (255, 255, 0),
        )
        expected_name = f"{video_prefix}_Race_{race_id}_{screen}_row_{row_number:02}_expected_overlay.png"
        cv2.imwrite(str(overlay_dir / expected_name), expected_overlay)
        result["expected"] = f"overlays/{expected_name}"
    return result


def row_key(race_id: str, screen: str, row_number: int) -> str:
    return f"{race_id}_{screen}_{row_number:02}"


def manual_fit_html(race_id: str, screen: str, row_number: int, expected_template: int | None, expected_score: Dict[str, object] | None) -> str:
    if expected_template is None or expected_score is None:
        return ""
    options = []
    for template_index in range(1, 13):
        selected = " selected" if template_index == expected_template else ""
        options.append(f"<option value=\"{template_index}\"{selected}>T{template_index:02}</option>")
    return (
        f"<div class=\"manual-fit\" "
        f"data-row-key=\"{html.escape(row_key(race_id, screen, row_number))}\" "
        f"data-default-x=\"{int(expected_score['location'][0])}\" "
        f"data-default-y=\"{int(expected_score['location'][1])}\" "
        f"data-expected-template=\"{int(expected_template)}\">"
        f"<div class=\"manual-fit-controls\">"
        f"<label>template <select class=\"fit-template\">{''.join(options)}</select></label>"
        f"</div>"
        f"<div class=\"manual-fit-controls\">"
        f"<label>x <input type=\"number\" value=\"{int(expected_score['location'][0])}\" class=\"fit-x\"></label>"
        f"<label>y <input type=\"number\" value=\"{int(expected_score['location'][1])}\" class=\"fit-y\"></label>"
        f"</div>"
        f"<div class=\"manual-fit-range\">Allowed x: <span class=\"fit-range-x\">?</span> | Allowed y: <span class=\"fit-range-y\">?</span></div>"
        f"<div class=\"manual-fit-range\">Delta x: <span class=\"fit-delta-x\">0</span> | Delta y: <span class=\"fit-delta-y\">0</span></div>"
        f"<div class=\"manual-fit-score\">Coeff: <span class=\"fit-coeff\">loading</span></div>"
        f"<div class=\"manual-fit-score\">White IoU: <span class=\"fit-white-iou\">loading</span></div>"
        f"<div class=\"manual-fit-score\">Weighted White IoU: <span class=\"fit-weighted-white-iou\">loading</span></div>"
        f"<div class=\"manual-fit-score\">Green: <span class=\"fit-green\">0</span> | Red: <span class=\"fit-red\">0</span> | Yellow: <span class=\"fit-yellow\">0</span></div>"
        f"<div class=\"manual-fit-warning\"></div>"
        f"<canvas class=\"manual-fit-canvas\" width=\"124\" height=\"114\"></canvas>"
        f"<canvas class=\"manual-overlap-canvas\" width=\"124\" height=\"114\"></canvas>"
        f"</div>"
    )


def render_row_table(row_entries: List[Dict[str, object]]) -> str:
    rows = []
    for entry in row_entries:
        cells = []
        best_index = int(entry["best_template"])
        best_iou_index = int(entry["best_iou_template"])
        best_weighted_iou_index = int(entry["best_weighted_iou_template"])
        expected_index = entry["expected_template"]
        for score in entry["scores"]:
            template_index = int(score["template_index"])
            classes = []
            if expected_index is not None and template_index == int(expected_index):
                classes.append("expected")
            class_attr = f' class="{" ".join(classes)}"' if classes else ""
            cells.append(
                f"<td{class_attr}"
                f' data-template-index="{template_index}"'
                f' data-coeff="{score["coefficient"]:.6f}"'
                f' data-white-iou="{score["white_iou"]:.6f}"'
                f' data-weighted-white-iou="{score["weighted_white_iou"]:.6f}">'
                f"T{template_index}<br>c {score['coefficient']:.4f}<br>iou {score['white_iou']:.4f}<br>wiou {score['weighted_white_iou']:.4f}</td>"
            )
        expected_label = "-" if expected_index is None else f"T{expected_index}"
        rows.append(
            f"<tr data-best-coeff-template=\"{best_index}\""
            f" data-best-white-iou-template=\"{best_iou_index}\""
            f" data-best-weighted-white-iou-template=\"{best_weighted_iou_index}\">"
            f"<th>Row {int(entry['row_number']):02}</th>"
            f"<td>{expected_label}</td>"
            f"<td>T{best_index}<br>c {float(entry['best_coefficient']):.4f}<br>iou {float(entry['best_white_iou']):.4f}<br>wiou {float(entry['best_weighted_white_iou']):.4f}</td>"
            f"<td>T{int(entry['best_iou_template'])}<br>iou {float(entry['best_iou_score']):.4f}<br>wiou {float(entry['best_iou_weighted_white_iou']):.4f}<br>c {float(entry['best_iou_coefficient']):.4f}</td>"
            f"<td>{float(entry['second_coefficient']):.4f}</td>"
            f"<td>{float(entry['margin']):.4f}</td>"
            f"<td><img src=\"{html.escape(entry['best_overlay'])}\" alt=\"best overlay\"></td>"
            f"<td><img src=\"{html.escape(entry.get('expected_overlay', ''))}\" alt=\"expected overlay\"></td>"
            f"<td>{entry['manual_fit_html']}</td>"
            + "".join(cells)
            + "</tr>"
        )
    return "\n".join(rows)


def build_report(video_prefix: str) -> str:
    templates = load_position_row_templates()
    sections = []
    for race_id, screen in [("006", "RaceScore"), ("006", "TotalScore"), ("008", "RaceScore"), ("008", "TotalScore")]:
        expected_sequence = EXPECTED_SEQUENCES[(race_id, screen)]
        row_entries = []
        for row_number in range(1, 13):
            row_image = load_row_crop(video_prefix, race_id, screen, row_number)
            scores = score_templates(row_image, templates)
            sorted_scores = sorted(scores, key=lambda item: float(item["coefficient"]), reverse=True)
            best_score = sorted_scores[0]
            second_score = sorted_scores[1]
            best_iou_score = max(scores, key=lambda item: float(item["white_iou"]))
            best_weighted_iou_score = max(scores, key=lambda item: float(item["weighted_white_iou"]))
            expected_template = expected_sequence[row_number - 1]
            expected_score = None
            if expected_template is not None:
                expected_score = scores[expected_template - 1]
            overlay_paths = write_overlay_set(video_prefix, race_id, screen, row_number, row_image, best_score, expected_score)
            row_entries.append(
                {
                    "row_number": row_number,
                    "expected_template": expected_template,
                    "best_template": int(best_score["template_index"]),
                    "best_coefficient": float(best_score["coefficient"]),
                    "best_white_iou": float(best_score["white_iou"]),
                    "best_weighted_white_iou": float(best_score["weighted_white_iou"]),
                    "best_iou_template": int(best_iou_score["template_index"]),
                    "best_iou_score": float(best_iou_score["white_iou"]),
                    "best_iou_coefficient": float(best_iou_score["coefficient"]),
                    "best_iou_weighted_white_iou": float(best_iou_score["weighted_white_iou"]),
                    "best_weighted_iou_template": int(best_weighted_iou_score["template_index"]),
                    "second_coefficient": float(second_score["coefficient"]),
                    "margin": float(best_score["coefficient"]) - float(second_score["coefficient"]),
                    "best_overlay": overlay_paths["best"],
                    "expected_overlay": overlay_paths.get("expected", ""),
                    "manual_fit_html": manual_fit_html(race_id, screen, row_number, expected_template, expected_score),
                    "scores": sorted(scores, key=lambda item: int(item["template_index"])),
                }
            )

        sections.append(
            f"""
            <section>
              <h2>Race {race_id} {screen}</h2>
              <table>
                <thead>
                  <tr>
                    <th>Row</th>
                    <th>Expected</th>
                    <th>Best Coeff</th>
                    <th>Best White IoU</th>
                    <th>Second</th>
                    <th>Margin</th>
                    <th>Best Overlay</th>
                    <th>Expected Overlay</th>
                    <th>Manual Fit</th>
                    {''.join(f'<th>T{i:02}</th>' for i in range(1, 13))}
                  </tr>
                </thead>
                <tbody>
                  {render_row_table(row_entries)}
                </tbody>
              </table>
            </section>
            """
        )

    row_image_map = {}
    for race_id, screen in [("006", "RaceScore"), ("006", "TotalScore"), ("008", "RaceScore"), ("008", "TotalScore")]:
        for row_number in range(1, 13):
            row_image = load_row_crop(video_prefix, race_id, screen, row_number)
            row_image_map[row_key(race_id, screen, row_number)] = {
                "width": int(row_image.shape[1]),
                "height": int(row_image.shape[0]),
                "pixels": row_image.flatten().tolist(),
            }
    template_map = {
        str(index): {
            "width": int(template.shape[1]),
            "height": int(template.shape[0]),
            "pixels": template.flatten().tolist(),
        }
        for index, template in enumerate(templates, start=1)
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Position Template Diagnostics</title>
  <style>
    body {{
      font-family: Consolas, 'Courier New', monospace;
      background: #0c1016;
      color: #eaf2ff;
      margin: 24px;
    }}
    h1, h2 {{ color: #9bf6ff; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 32px;
      font-size: 12px;
    }}
    th, td {{
      border: 1px solid #243244;
      padding: 6px;
      text-align: center;
      vertical-align: middle;
    }}
    th {{
      background: #152133;
      position: sticky;
      top: 0;
    }}
    td.best {{
      background: #39ff14;
      color: #041006;
      font-weight: 700;
      box-shadow: inset 0 0 0 2px #8cff72;
    }}
    td.expected {{
      outline: 2px solid #00f5ff;
      outline-offset: -2px;
    }}
    img {{
      width: 124px;
      image-rendering: pixelated;
      border: 1px solid #33455c;
      background: #fff;
    }}
    canvas {{
      width: 124px;
      height: 114px;
      image-rendering: pixelated;
      border: 1px solid #33455c;
      background: #fff;
    }}
    .manual-fit {{
      display: grid;
      gap: 6px;
      min-width: 140px;
    }}
    .manual-fit-controls {{
      display: flex;
      justify-content: center;
      gap: 6px;
    }}
    .manual-fit-controls label {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }}
    .manual-fit-controls input {{
      width: 46px;
      background: #101926;
      color: #eaf2ff;
      border: 1px solid #36506f;
    }}
    .manual-fit-controls select {{
      background: #101926;
      color: #eaf2ff;
      border: 1px solid #36506f;
    }}
    .manual-fit-score {{
      color: #9bf6ff;
      font-weight: 700;
    }}
    .manual-fit-explain {{
      color: #b8c7dc;
      font-size: 11px;
    }}
    .manual-fit-warning {{
      color: #ff7b72;
      min-height: 16px;
    }}
    .manual-fit-range {{
      color: #b8c7dc;
      font-size: 11px;
    }}
    .legend span {{
      display: inline-block;
      margin-right: 16px;
      padding: 4px 8px;
    }}
    .legend .best {{
      background: #39ff14;
      color: #041006;
    }}
    .metric-selector {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 12px 0 18px;
    }}
    .metric-selector select {{
      background: #101926;
      color: #eaf2ff;
      border: 1px solid #36506f;
      padding: 4px 6px;
    }}
    td.metric-best {{
      background: #39ff14;
      color: #041006;
      font-weight: 700;
      box-shadow: inset 0 0 0 2px #8cff72;
    }}
    .legend .expected {{
      outline: 2px solid #00f5ff;
    }}
    .legend .overlap-hit {{
      background: #39ff14;
      color: #041006;
    }}
    .legend .overlap-miss {{
      background: #ff6b6b;
      color: #180404;
    }}
    .legend .overlap-window {{
      background: #ffe066;
      color: #221b00;
    }}
  </style>
</head>
<body>
  <h1>Position Template Diagnostics</h1>
  <p>Video prefix: {html.escape(video_prefix)}</p>
  <p>Template source: {html.escape(POSITION_TEMPLATE_FILENAME)}</p>
  <p>Weighted White IoU uses FN weight {FALSE_NEGATIVE_WEIGHT:.1f}, so template-white over ROI-black is penalized harder.</p>
  <div class="metric-selector">
    <label for="highlight-metric">Highlight top score by:</label>
    <select id="highlight-metric">
      <option value="coeff">Coeff</option>
      <option value="white-iou">White IoU</option>
      <option value="weighted-white-iou">Weighted White IoU</option>
    </select>
  </div>
  <div class="legend">
    <span class="best">Current selected winner</span>
    <span class="expected">Expected template for this row</span>
    <span class="overlap-hit">Template white over row white</span>
    <span class="overlap-miss">Template white over row dark</span>
    <span class="overlap-window">Compared window</span>
  </div>
  {''.join(sections)}
  <script>
    const ROW_IMAGES = {json.dumps(row_image_map)};
    const TEMPLATE_IMAGES = {json.dumps(template_map)};

    function computeMetricsAtOffset(rowImage, templateImage, offsetX, offsetY) {{
      const maxX = rowImage.width - templateImage.width;
      const maxY = rowImage.height - templateImage.height;
      if (offsetX < 0 || offsetY < 0 || offsetX > maxX || offsetY > maxY) {{
        return {{ valid: false, reason: `Outside ROI. Allowed x: 0..${{maxX}}, y: 0..${{maxY}}.` }};
      }}
      const count = templateImage.width * templateImage.height;
      let sumRow = 0;
      let sumTemplate = 0;
      for (let y = 0; y < templateImage.height; y += 1) {{
        for (let x = 0; x < templateImage.width; x += 1) {{
          const rowIndex = (offsetY + y) * rowImage.width + (offsetX + x);
          const templateIndex = y * templateImage.width + x;
          sumRow += rowImage.pixels[rowIndex];
          sumTemplate += templateImage.pixels[templateIndex];
        }}
      }}
      const meanRow = sumRow / count;
      const meanTemplate = sumTemplate / count;
      let numerator = 0;
      let denomRow = 0;
      let denomTemplate = 0;
      for (let y = 0; y < templateImage.height; y += 1) {{
        for (let x = 0; x < templateImage.width; x += 1) {{
          const rowIndex = (offsetY + y) * rowImage.width + (offsetX + x);
          const templateIndex = y * templateImage.width + x;
          const rowValue = rowImage.pixels[rowIndex] - meanRow;
          const templateValue = templateImage.pixels[templateIndex] - meanTemplate;
          numerator += rowValue * templateValue;
          denomRow += rowValue * rowValue;
          denomTemplate += templateValue * templateValue;
        }}
      }}
      const denominator = Math.sqrt(denomRow * denomTemplate);
      let tp = 0;
      let fp = 0;
      let fn = 0;
      for (let y = 0; y < templateImage.height; y += 1) {{
        for (let x = 0; x < templateImage.width; x += 1) {{
          const rowIndex = (offsetY + y) * rowImage.width + (offsetX + x);
          const templateIndex = y * templateImage.width + x;
          const rowWhite = rowImage.pixels[rowIndex] > 200;
          const templateWhite = templateImage.pixels[templateIndex] > 200;
          if (templateWhite && rowWhite) {{
            tp += 1;
          }} else if (!templateWhite && rowWhite) {{
            fp += 1;
          }} else if (templateWhite && !rowWhite) {{
            fn += 1;
          }}
        }}
      }}
      const union = tp + fp + fn;
      const weightedUnion = tp + fp + ({FALSE_NEGATIVE_WEIGHT:.1f} * fn);
      const whiteIou = union === 0 ? 0 : tp / union;
      const weightedWhiteIou = weightedUnion === 0 ? 0 : tp / weightedUnion;
      if (denominator === 0) {{
        return {{ valid: true, coefficient: 0, whiteIou, weightedWhiteIou, tp, fp, fn }};
      }}
      return {{ valid: true, coefficient: numerator / denominator, whiteIou, weightedWhiteIou, tp, fp, fn }};
    }}

    function drawBaseRow(canvas, rowImage) {{
      const ctx = canvas.getContext('2d');
      const imageData = ctx.createImageData(canvas.width, canvas.height);
      const scaleX = canvas.width / rowImage.width;
      const scaleY = canvas.height / rowImage.height;
      for (let y = 0; y < canvas.height; y += 1) {{
        for (let x = 0; x < canvas.width; x += 1) {{
          const srcX = Math.min(rowImage.width - 1, Math.floor(x / scaleX));
          const srcY = Math.min(rowImage.height - 1, Math.floor(y / scaleY));
          const value = rowImage.pixels[srcY * rowImage.width + srcX];
          const index = (y * canvas.width + x) * 4;
          imageData.data[index] = value;
          imageData.data[index + 1] = value;
          imageData.data[index + 2] = value;
          imageData.data[index + 3] = 255;
        }}
      }}
      ctx.putImageData(imageData, 0, 0);
      return {{ ctx, scaleX, scaleY }};
    }}

    function drawManualFit(canvas, rowImage, templateImage, offsetX, offsetY) {{
      const {{ ctx, scaleX, scaleY }} = drawBaseRow(canvas, rowImage);
      const drawX = offsetX * scaleX;
      const drawY = offsetY * scaleY;
      const drawW = templateImage.width * scaleX;
      const drawH = templateImage.height * scaleY;
      for (let y = 0; y < templateImage.height; y += 1) {{
        for (let x = 0; x < templateImage.width; x += 1) {{
          const value = templateImage.pixels[y * templateImage.width + x];
          if (value <= 200) {{
            continue;
          }}
          ctx.fillStyle = 'rgba(255,255,0,0.75)';
          ctx.fillRect((offsetX + x) * scaleX, (offsetY + y) * scaleY, Math.max(1, scaleX), Math.max(1, scaleY));
        }}
      }}
      ctx.strokeStyle = '#00f5ff';
      ctx.lineWidth = 1;
      ctx.strokeRect(drawX, drawY, drawW, drawH);
    }}

    function drawOverlapMap(canvas, rowImage, templateImage, offsetX, offsetY) {{
      const {{ ctx, scaleX, scaleY }} = drawBaseRow(canvas, rowImage);

      const x2 = offsetX + templateImage.width;
      const y2 = offsetY + templateImage.height;
      let greenCount = 0;
      let redCount = 0;
      let yellowCount = 0;
      for (let y = 0; y < templateImage.height; y += 1) {{
        for (let x = 0; x < templateImage.width; x += 1) {{
          const rowX = offsetX + x;
          const rowY = offsetY + y;
          if (rowX < 0 || rowY < 0 || rowX >= rowImage.width || rowY >= rowImage.height) {{
            continue;
          }}
          const rowIndex = rowY * rowImage.width + rowX;
          const templateIndex = y * templateImage.width + x;
          const rowValue = rowImage.pixels[rowIndex];
          const templateValue = templateImage.pixels[templateIndex];
          yellowCount += 1;
          const drawX = Math.floor(rowX * scaleX);
          const drawY = Math.floor(rowY * scaleY);
          const drawX2 = Math.max(drawX + 1, Math.ceil((rowX + 1) * scaleX));
          const drawY2 = Math.max(drawY + 1, Math.ceil((rowY + 1) * scaleY));
          for (let py = drawY; py < drawY2; py += 1) {{
            for (let px = drawX; px < drawX2; px += 1) {{
              if (templateValue > 200 && rowValue > 200) {{
                ctx.fillStyle = '#39ff14';
                greenCount += 1;
              }} else if (templateValue > 200) {{
                ctx.fillStyle = '#ff6b6b';
                redCount += 1;
              }} else {{
                ctx.fillStyle = 'rgba(255,224,102,0.55)';
              }}
              ctx.fillRect(px, py, 1, 1);
            }}
          }}
        }}
      }}
      ctx.strokeStyle = '#00f5ff';
      ctx.lineWidth = 1;
      ctx.strokeRect(offsetX * scaleX, offsetY * scaleY, templateImage.width * scaleX, templateImage.height * scaleY);
      return {{ greenCount, redCount, yellowCount }};
    }}

    async function initManualFit(node) {{
      const rowImage = ROW_IMAGES[node.dataset.rowKey];
      const templateSelect = node.querySelector('.fit-template');
      const inputX = node.querySelector('.fit-x');
      const inputY = node.querySelector('.fit-y');
      const coeffNode = node.querySelector('.fit-coeff');
      const iouNode = node.querySelector('.fit-white-iou');
      const weightedIouNode = node.querySelector('.fit-weighted-white-iou');
      const greenNode = node.querySelector('.fit-green');
      const redNode = node.querySelector('.fit-red');
      const yellowNode = node.querySelector('.fit-yellow');
      const rangeXNode = node.querySelector('.fit-range-x');
      const rangeYNode = node.querySelector('.fit-range-y');
      const deltaXNode = node.querySelector('.fit-delta-x');
      const deltaYNode = node.querySelector('.fit-delta-y');
      const warningNode = node.querySelector('.manual-fit-warning');
      const canvas = node.querySelector('.manual-fit-canvas');
      const overlapCanvas = node.querySelector('.manual-overlap-canvas');
      const defaultX = Number.parseInt(node.dataset.defaultX, 10) || 0;
      const defaultY = Number.parseInt(node.dataset.defaultY, 10) || 0;

      function refresh() {{
        const offsetX = Number.parseInt(inputX.value, 10) || 0;
        const offsetY = Number.parseInt(inputY.value, 10) || 0;
        const selectedTemplate = String(Number.parseInt(templateSelect.value, 10) || 1);
        const currentTemplateImage = TEMPLATE_IMAGES[selectedTemplate];
        const maxX = rowImage.width - currentTemplateImage.width;
        const maxY = rowImage.height - currentTemplateImage.height;
        rangeXNode.textContent = `0..${{maxX}}`;
        rangeYNode.textContent = `0..${{maxY}}`;
        deltaXNode.textContent = String(offsetX - defaultX);
        deltaYNode.textContent = String(offsetY - defaultY);
        const result = computeMetricsAtOffset(rowImage, currentTemplateImage, offsetX, offsetY);
        if (!result.valid) {{
          coeffNode.textContent = 'n/a';
          iouNode.textContent = 'n/a';
          weightedIouNode.textContent = 'n/a';
          greenNode.textContent = '0';
          redNode.textContent = '0';
          yellowNode.textContent = '0';
          warningNode.textContent = result.reason;
          drawBaseRow(canvas, rowImage);
          drawBaseRow(overlapCanvas, rowImage);
          return;
        }}
        coeffNode.textContent = result.coefficient.toFixed(4);
        iouNode.textContent = result.whiteIou.toFixed(4);
        weightedIouNode.textContent = result.weightedWhiteIou.toFixed(4);
        warningNode.textContent = '';
        drawManualFit(canvas, rowImage, currentTemplateImage, offsetX, offsetY);
        const overlapStats = drawOverlapMap(overlapCanvas, rowImage, currentTemplateImage, offsetX, offsetY);
        greenNode.textContent = String(overlapStats.greenCount);
        redNode.textContent = String(overlapStats.redCount);
        yellowNode.textContent = String(overlapStats.yellowCount);
      }}

      templateSelect.addEventListener('change', refresh);
      inputX.addEventListener('input', refresh);
      inputY.addEventListener('input', refresh);
      refresh();
    }}

    function updateMetricHighlight(metricName) {{
      document.querySelectorAll('tbody tr').forEach((row) => {{
        row.querySelectorAll('td[data-template-index]').forEach((cell) => {{
          cell.classList.remove('metric-best');
        }});
        let attrName = 'data-best-coeff-template';
        if (metricName === 'white-iou') {{
          attrName = 'data-best-white-iou-template';
        }} else if (metricName === 'weighted-white-iou') {{
          attrName = 'data-best-weighted-white-iou-template';
        }}
        const bestTemplate = row.getAttribute(attrName);
        if (!bestTemplate) {{
          return;
        }}
        const bestCell = row.querySelector(`td[data-template-index="${{bestTemplate}}"]`);
        if (bestCell) {{
          bestCell.classList.add('metric-best');
        }}
      }});
    }}

    window.addEventListener('load', () => {{
      const metricSelect = document.getElementById('highlight-metric');
      if (metricSelect) {{
        metricSelect.addEventListener('change', () => updateMetricHighlight(metricSelect.value));
        updateMetricHighlight(metricSelect.value);
      }}
      document.querySelectorAll('.manual-fit').forEach((node) => {{
        initManualFit(node);
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_html = build_report(args.video_prefix)
    report_path = REPORT_DIR / "position_template_diagnostics.html"
    report_path.write_text(report_html, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
