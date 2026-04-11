from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from mk8_local_play.extract_common import find_score_bundle_anchor_path
from mk8_local_play.extract_text import score_layout_id_from_filename
from mk8_local_play.ocr_scoreboard_consensus import best_character_matches, character_row_roi, load_character_templates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_CSV = PROJECT_ROOT / "Output_Results" / "Debug" / "20260410_092255_Tournament_Results_Debug.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output_Results" / "Debug" / "mii_memory_probe"


@dataclass(frozen=True)
class ProbeTarget:
    video: str
    player: str
    expected: str


DEFAULT_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget("2025-01-03__20250103_Groep2", "Wilco", "Mii"),
    ProbeTarget("2025-01-03__20250103_Groep2", "Player", "Mii"),
    ProbeTarget("Mario_Kart_Toernooien__Level_Level__2025-08-21__2025-08-21_20-44-50", "Floris", "Mii"),
    ProbeTarget("Mario_Kart_Toernooien__Level_Level__2025-08-21__2025-08-21_20-44-50", "jan willem", "Mii"),
    ProbeTarget("Mario_Kart_Toernooien__Level_Level__2025-08-21__2025-08-21_20-44-50", "Patrick", "Mii"),
    ProbeTarget("Mario_Kart_Toernooien__Stolk_staal__2025-05-16__2025-05-16_22-21-17", "BAwSer", "Inkling Boy"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate whether saved player-specific Mii crops are usable as runtime memory priors.")
    parser.add_argument(
        "--debug-csv",
        default=str(DEFAULT_DEBUG_CSV),
        help="Debug CSV to read OCR rows from.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write probe outputs into.",
    )
    parser.add_argument(
        "--target",
        action="append",
        default=[],
        help="Optional target in the format video|player|expected. Can be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top roster candidates to record per crop.",
    )
    return parser.parse_args()


def _parse_targets(values: list[str]) -> list[ProbeTarget]:
    if not values:
        return list(DEFAULT_TARGETS)
    targets: list[ProbeTarget] = []
    for value in values:
        parts = [part.strip() for part in str(value).split("|")]
        if len(parts) != 3 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid --target value: {value!r}. Expected video|player|expected")
        targets.append(ProbeTarget(parts[0], parts[1], parts[2]))
    return targets


def _normalize_roi(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] != (48, 48):
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LINEAR)
    return image


def _grayscale_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(left_gray - right_gray)))
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def _color_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_rgb = left.astype(np.float32)
    right_rgb = right.astype(np.float32)
    mean_abs_diff = float(np.mean(np.abs(left_rgb - right_rgb)))
    return max(0.0, 1.0 - (mean_abs_diff / 255.0))


def _edge_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_edges = cv2.Canny(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), 80, 160) > 0
    right_edges = cv2.Canny(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), 80, 160) > 0
    overlap = int(np.count_nonzero(left_edges & right_edges))
    total = int(np.count_nonzero(left_edges)) + int(np.count_nonzero(right_edges))
    if total <= 0:
        return 0.0
    return float((2.0 * overlap) / total)


def _blended_memory_score(left: np.ndarray, right: np.ndarray) -> float:
    return (0.45 * _grayscale_similarity(left, right)) + (0.35 * _edge_similarity(left, right)) + (0.20 * _color_similarity(left, right))


def _read_probe_rows(debug_csv_path: Path, targets: list[ProbeTarget]) -> list[dict[str, str]]:
    target_keys = {(target.video, target.player): target for target in targets}
    rows: list[dict[str, str]] = []
    with debug_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (str(row.get("Video") or ""), str(row.get("Standardized Player") or ""))
            target = target_keys.get(key)
            if target is None:
                continue
            row = dict(row)
            row["_Expected"] = target.expected
            rows.append(row)
    return rows


def _load_roi(row: dict[str, str]) -> np.ndarray | None:
    video = str(row.get("Video") or "")
    race = int(float(row.get("Race") or 0))
    position = int(float(row.get("Position") or 0))
    if not video or race <= 0 or position <= 0:
        return None
    anchor_path = find_score_bundle_anchor_path(video, race, "2RaceScore")
    if anchor_path is None:
        return None
    frame = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    score_layout_id = score_layout_id_from_filename(Path(anchor_path).name)
    (x1, y1), (x2, y2) = character_row_roi(position - 1, score_layout_id=score_layout_id)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return _normalize_roi(roi)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _summarize(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return round(min(values), 4), round(statistics.median(values), 4), round(max(values), 4)


def main() -> None:
    args = _parse_args()
    debug_csv_path = Path(args.debug_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = _parse_targets(list(args.target))
    templates = load_character_templates()
    probe_rows = _read_probe_rows(debug_csv_path, targets)
    if not probe_rows:
        raise RuntimeError(f"No matching probe rows found in {debug_csv_path}")

    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in probe_rows:
        grouped_rows[(str(row["Video"]), str(row["Standardized Player"]))].append(row)

    crop_records: list[dict[str, object]] = []
    player_memory_summary: list[dict[str, object]] = []

    for (video, player), rows in grouped_rows.items():
        loaded_rows: list[tuple[dict[str, str], np.ndarray]] = []
        for row in rows:
            roi = _load_roi(row)
            if roi is None:
                continue
            loaded_rows.append((row, roi))
        if not loaded_rows:
            continue

        reference_row, reference_roi = loaded_rows[0]
        same_scores: list[float] = []
        cross_scores: list[float] = []
        negative_scores: list[float] = []
        top_candidate_counter: Counter[int] = Counter()

        for row, roi in loaded_rows:
            roster_matches = best_character_matches(roi, templates, limit=max(2, int(args.limit)))
            top_indices = [int(match.get("CharacterIndex")) for match in roster_matches[: int(args.limit)] if match.get("CharacterIndex") is not None]
            top_candidate_counter.update(top_indices)
            best_conf = float(roster_matches[0].get("CharacterMatchConfidence", 0.0)) if roster_matches else 0.0
            second_conf = float(roster_matches[1].get("CharacterMatchConfidence", 0.0)) if len(roster_matches) > 1 else 0.0
            memory_score = _blended_memory_score(reference_roi, roi) * 100.0
            crop_records.append(
                {
                    "video": video,
                    "player": player,
                    "expected": str(row.get("_Expected") or ""),
                    "race": _safe_int(row.get("Race")),
                    "position": _safe_int(row.get("Position")),
                    "export_character": str(row.get("Character") or ""),
                    "reference_race": _safe_int(reference_row.get("Race")),
                    "memory_score_vs_reference": round(memory_score, 2),
                    "raw_best_closed_set": round(_safe_float(row.get("Character Match Confidence")), 2),
                    "raw_debug_best": round(_safe_float(row.get("Character Match Confidence")), 2),
                    "best_closed_set_probe": round(best_conf, 2),
                    "best_closed_set_margin_probe": round(best_conf - second_conf, 2),
                    "top_candidates": ",".join(str(index) for index in top_indices),
                    "top_candidate_names": ",".join(str(match.get("Character") or "") for match in roster_matches[: int(args.limit)]),
                }
            )
            if (video, player) == (str(reference_row["Video"]), str(reference_row["Standardized Player"])):
                same_scores.append(memory_score)

        for (other_video, other_player), other_rows in grouped_rows.items():
            if (other_video, other_player) == (video, player):
                continue
            for other_row in other_rows:
                other_roi = _load_roi(other_row)
                if other_roi is None:
                    continue
                score = _blended_memory_score(reference_roi, other_roi) * 100.0
                cross_scores.append(score)
                if str(other_row.get("_Expected") or "") != "Mii":
                    negative_scores.append(score)

        same_min, same_median, same_max = _summarize(same_scores)
        cross_min, cross_median, cross_max = _summarize(cross_scores)
        negative_min, negative_median, negative_max = _summarize(negative_scores)
        player_memory_summary.append(
            {
                "video": video,
                "player": player,
                "expected": str(reference_row.get("_Expected") or ""),
                "sample_count": len(loaded_rows),
                "reference_race": _safe_int(reference_row.get("Race")),
                "same_player_min": same_min,
                "same_player_median": same_median,
                "same_player_max": same_max,
                "cross_player_min": cross_min,
                "cross_player_median": cross_median,
                "cross_player_max": cross_max,
                "negative_min": negative_min,
                "negative_median": negative_median,
                "negative_max": negative_max,
                "top_memory_minus_cross_median": round(same_median - cross_median, 4),
                "top_memory_minus_negative_median": round(same_median - negative_median, 4),
                "recurring_top_candidates": ",".join(str(index) for index, _count in top_candidate_counter.most_common(8)),
            }
        )

    if not crop_records:
        raise RuntimeError("No probe crops could be loaded from the saved frames.")

    pair_rows: list[dict[str, object]] = []
    loaded_pairs: list[tuple[tuple[str, str, int], np.ndarray, str]] = []
    for (video, player), rows in grouped_rows.items():
        for row in rows:
            roi = _load_roi(row)
            if roi is None:
                continue
            loaded_pairs.append(((video, player, _safe_int(row.get("Race"))), roi, str(row.get("_Expected") or "")))
    for (left_key, left_roi, left_expected), (right_key, right_roi, right_expected) in itertools.combinations(loaded_pairs, 2):
        pair_rows.append(
            {
                "left_video": left_key[0],
                "left_player": left_key[1],
                "left_race": left_key[2],
                "left_expected": left_expected,
                "right_video": right_key[0],
                "right_player": right_key[1],
                "right_race": right_key[2],
                "right_expected": right_expected,
                "same_player": left_key[:2] == right_key[:2],
                "score_grayscale": round(_grayscale_similarity(left_roi, right_roi) * 100.0, 2),
                "score_edge": round(_edge_similarity(left_roi, right_roi) * 100.0, 2),
                "score_color": round(_color_similarity(left_roi, right_roi) * 100.0, 2),
                "score_blended": round(_blended_memory_score(left_roi, right_roi) * 100.0, 2),
            }
        )

    crop_output = output_dir / "mii_memory_probe_crops.csv"
    summary_output = output_dir / "mii_memory_probe_summary.csv"
    pairs_output = output_dir / "mii_memory_probe_pairs.csv"

    with crop_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(crop_records[0].keys()))
        writer.writeheader()
        writer.writerows(crop_records)

    with summary_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(player_memory_summary[0].keys()))
        writer.writeheader()
        writer.writerows(player_memory_summary)

    with pairs_output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    print(f"crop_report={crop_output}")
    print(f"summary_report={summary_output}")
    print(f"pair_report={pairs_output}")


if __name__ == "__main__":
    main()
