from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from mk8_local_play.data_paths import resolve_asset_file
from mk8_local_play.extract_common import crop_and_upscale_image, frame_to_timecode
from mk8_local_play.extract_frames import _prepare_video_context, _run_scan_phase_for_context
import mk8_local_play.extract_score_screen_selection as score_sel
from mk8_local_play.extract_video_io import seek_to_frame
from mk8_local_play.ocr_scoreboard_consensus import extract_points_transition_observation, parse_detected_int
from mk8_local_play.project_paths import PROJECT_ROOT


class _NullWriter:
    def writerow(self, _row: list[Any]) -> None:
        return None

    def writerows(self, _rows: list[list[Any]]) -> None:
        return None


def _load_templates():
    template_files = [
        "Trackname_template.png",
        "Race_template.png",
        "12th_pos_template.png",
        "ignore.png",
        "albumgallery_ignore.png",
        "ignore_2.png",
        "Race_template_NL_final.png",
        "12th_pos_templateNL.png",
    ]
    templates = []
    for filename in template_files:
        path = str(resolve_asset_file("templates", filename))
        template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if template is None:
            raise RuntimeError(f"Template not found: {path}")
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            _, alpha_mask = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
        elif len(template.shape) == 3 and template.shape[2] == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
            alpha_mask = None
        else:
            template_binary = template
            alpha_mask = None
        templates.append((template_binary, alpha_mask))
    return templates


@dataclass
class PatternHit:
    pattern_len: int
    first_true_frame: int
    confirm_frame: int
    anchor_frame: int
    true_count_at_confirm: int


def _apply_streak_pattern(
    frame_hits: list[tuple[int, bool]],
    *,
    pattern_len: int,
    max_false_gap: int,
    anchor_backoff_frames: int = 1,
) -> PatternHit | None:
    first_true_frame = None
    true_count = 0
    false_streak = 0
    for frame_number, is_hit in frame_hits:
        if is_hit:
            if first_true_frame is None:
                first_true_frame = int(frame_number)
                true_count = 1
            else:
                true_count += 1
            false_streak = 0
            if true_count >= int(pattern_len):
                anchor = max(0, int(first_true_frame) - int(anchor_backoff_frames))
                return PatternHit(
                    pattern_len=int(pattern_len),
                    first_true_frame=int(first_true_frame),
                    confirm_frame=int(frame_number),
                    anchor_frame=int(anchor),
                    true_count_at_confirm=int(true_count),
                )
            continue
        if first_true_frame is None:
            continue
        false_streak += 1
        if false_streak > int(max_false_gap):
            first_true_frame = None
            true_count = 0
            false_streak = 0
    return None


def _choose_winner(hits_by_pattern: dict[int, PatternHit | None]) -> PatternHit | None:
    for length in sorted(hits_by_pattern.keys(), reverse=True):
        hit = hits_by_pattern.get(int(length))
        if hit is not None:
            return hit
    return None


def _parse_points(values: list[Any], limit: int = 12) -> list[int | None]:
    parsed = []
    for value in list(values)[: int(limit)]:
        parsed.append(parse_detected_int(value))
    return parsed


def _collect_observations(
    cap: cv2.VideoCapture,
    *,
    context: dict[str, Any],
    start_frame: int,
    end_frame: int,
    score_layout_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seek_to_frame(cap, int(start_frame), {}, label="pattern_probe_start")
    previous = None
    for frame_number in range(int(start_frame), int(end_frame) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        image = crop_and_upscale_image(
            frame,
            int(context["median_left"]),
            int(context["median_top"]),
            int(context["median_crop_width"]),
            int(context["median_crop_height"]),
            1280,
            720,
        )
        observation = extract_points_transition_observation(image, score_layout_id=score_layout_id)
        race_points = _parse_points(observation.get("race_points") or [], limit=12)
        total_points = _parse_points(observation.get("total_points") or [], limit=12)
        changed_total_rows = 0
        changed_race_rows = 0
        changed_any_rows = 0
        if previous is not None:
            prev_race = previous["race_points"]
            prev_total = previous["total_points"]
            for idx in range(min(12, len(race_points), len(total_points), len(prev_race), len(prev_total))):
                race_changed = (
                    prev_race[idx] is not None
                    and race_points[idx] is not None
                    and int(prev_race[idx]) != int(race_points[idx])
                )
                total_changed = (
                    prev_total[idx] is not None
                    and total_points[idx] is not None
                    and int(prev_total[idx]) != int(total_points[idx])
                )
                changed_race_rows += int(race_changed)
                changed_total_rows += int(total_changed)
                changed_any_rows += int(race_changed or total_changed)
        raw_transition_hit = bool(
            changed_total_rows >= 2 and (changed_race_rows >= 1 or changed_any_rows >= 3)
        )
        unparsed_total_count = int(sum(1 for value in total_points[:12] if value is None))
        rows.append(
            {
                "frame": int(frame_number),
                "race_points": race_points,
                "total_points": total_points,
                "changed_total_rows": int(changed_total_rows),
                "changed_race_rows": int(changed_race_rows),
                "changed_any_rows": int(changed_any_rows),
                "raw_transition_hit": bool(raw_transition_hit),
                "unparsed_total_count": int(unparsed_total_count),
            }
        )
        previous = rows[-1]
    return rows


def _detect_resort_signal_rows(observations: list[dict[str, Any]], *, min_total_changes: int = 3) -> list[tuple[int, bool]]:
    frame_hits: list[tuple[int, bool]] = []
    for row in observations:
        raw_hit = bool(
            int(row.get("changed_total_rows", 0)) >= int(min_total_changes)
            and int(row.get("changed_race_rows", 0)) <= 1
        )
        frame_hits.append((int(row["frame"]), raw_hit))
    return frame_hits


def _analyze_race(
    cap: cv2.VideoCapture,
    *,
    context: dict[str, Any],
    candidate: dict[str, Any],
    templates: list[Any],
    pattern_lengths: list[int],
    max_false_gap: int,
) -> dict[str, Any]:
    fps = float(context["fps"])
    score_layout_id = str(candidate.get("score_layout_id") or "")
    task = {
        "video_path": str(context["processing_video_path"]),
        "race_number": int(candidate["race_number"]),
        "frame_number": int(candidate["frame_number"]),
        "fps": float(context["fps"]),
        "templates": templates,
        "scale_x": float(context["median_scale_x"]),
        "scale_y": float(context["median_scale_y"]),
        "left": int(context["median_left"]),
        "top": int(context["median_top"]),
        "crop_width": int(context["median_crop_width"]),
        "crop_height": int(context["median_crop_height"]),
        "source_height": int(context.get("source_height", 0) or 0),
        "ocr_consensus_frames": int(getattr(score_sel.APP_CONFIG, "ocr_consensus_frames", 7) or 7),
        "score_layout_id": score_layout_id,
    }
    result = score_sel.analyze_score_window_task(task, frame_to_timecode, capture=cap)
    analysis_trace = dict(result.get("analysis_trace") or {})
    score_hit_frame = analysis_trace.get("score_hit_frame")
    race_anchor_frame = analysis_trace.get("selected_points_anchor_frame")
    transition_frame = analysis_trace.get("transition_frame")
    total_anchor_frame = analysis_trace.get("stable_total_score_frame")

    if score_hit_frame is None:
        return {
            "race_number": int(candidate["race_number"]),
            "candidate_frame": int(candidate["frame_number"]),
            "status": "no_score_hit",
        }

    detail_start = max(0, int(candidate["frame_number"]) - int(3 * fps))
    detail_end = int(candidate["frame_number"]) + int(13 * fps)
    transition_start = int((result.get("race_score_frame") or score_hit_frame))
    transition_end = min(
        int(detail_end),
        int(transition_start + max(1, int(round(score_sel.POINTS_TRANSITION_SEARCH_END_SECONDS * max(fps, 1.0))))),
    )
    observations = _collect_observations(
        cap,
        context=context,
        start_frame=int(transition_start),
        end_frame=int(transition_end),
        score_layout_id=score_layout_id,
    )

    transition_hits_by_pattern: dict[int, PatternHit | None] = {}
    transition_frame_hits = [(int(row["frame"]), bool(row["raw_transition_hit"])) for row in observations]
    earliest_transition_true_frame = next(
        (int(frame_number) for frame_number, is_hit in transition_frame_hits if bool(is_hit)),
        None,
    )
    for length in pattern_lengths:
        transition_hits_by_pattern[int(length)] = _apply_streak_pattern(
            transition_frame_hits,
            pattern_len=int(length),
            max_false_gap=int(max_false_gap),
        )
    transition_winner = _choose_winner(transition_hits_by_pattern)

    resort_delay_frames = int(score_sel.fps_scaled_frames(40, fps))
    resort_start = int((transition_frame or transition_start) + resort_delay_frames)
    resort_end = min(int(detail_end), int(resort_start + int(round(6.0 * max(fps, 1.0)))))
    resort_obs = [
        row for row in observations
        if int(resort_start) <= int(row["frame"]) <= int(resort_end)
    ]
    resort_hits_by_pattern: dict[int, PatternHit | None] = {}
    resort_frame_hits = _detect_resort_signal_rows(resort_obs)
    for length in pattern_lengths:
        resort_hits_by_pattern[int(length)] = _apply_streak_pattern(
            resort_frame_hits,
            pattern_len=int(length),
            max_false_gap=int(max_false_gap),
            anchor_backoff_frames=0,
        )
    resort_winner = _choose_winner(resort_hits_by_pattern)

    return {
        "race_number": int(candidate["race_number"]),
        "candidate_frame": int(candidate["frame_number"]),
        "score_hit_frame": None if score_hit_frame is None else int(score_hit_frame),
        "points_anchor_frame": None if race_anchor_frame is None else int(race_anchor_frame),
        "production_transition_frame": None if transition_frame is None else int(transition_frame),
        "production_total_anchor_frame": None if total_anchor_frame is None else int(total_anchor_frame),
        "transition_scan_start": int(transition_start),
        "transition_scan_end": int(transition_end),
        "transition_patterns": {
            str(length): (
                None if hit is None else {
                    "first_true_frame": int(hit.first_true_frame),
                    "confirm_frame": int(hit.confirm_frame),
                    "anchor_frame": int(hit.anchor_frame),
                    "true_count_at_confirm": int(hit.true_count_at_confirm),
                }
            )
            for length, hit in sorted(transition_hits_by_pattern.items())
        },
        "transition_recommended": (
            None if transition_winner is None else {
                "pattern_len": int(transition_winner.pattern_len),
                "first_true_frame": int(transition_winner.first_true_frame),
                "confirm_frame": int(transition_winner.confirm_frame),
                # Requested behavior: when pattern confirms, report the very first raw TRUE frame.
                "anchor_frame": int(
                    earliest_transition_true_frame
                    if earliest_transition_true_frame is not None
                    else transition_winner.first_true_frame
                ),
            }
        ),
        "resort_scan_start": int(resort_start),
        "resort_scan_end": int(resort_end),
        "resort_patterns": {
            str(length): (
                None if hit is None else {
                    "first_true_frame": int(hit.first_true_frame),
                    "confirm_frame": int(hit.confirm_frame),
                    "anchor_frame": int(hit.anchor_frame),
                    "true_count_at_confirm": int(hit.true_count_at_confirm),
                }
            )
            for length, hit in sorted(resort_hits_by_pattern.items())
        },
        "resort_recommended": (
            None if resort_winner is None else {
                "pattern_len": int(resort_winner.pattern_len),
                "first_true_frame": int(resort_winner.first_true_frame),
                "confirm_frame": int(resort_winner.confirm_frame),
                "anchor_frame": int(resort_winner.anchor_frame),
            }
        ),
        "frame_trace": observations,
    }


def _write_outputs(output_dir: Path, video_label: str, race_results: list[dict[str, Any]]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{video_label}_pattern_analysis.json"
    csv_path = output_dir / f"{video_label}_pattern_summary.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "video_label": video_label,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "races": race_results,
            },
            f,
            indent=2,
        )
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Race",
                "CandidateFrame",
                "ProductionTransition",
                "RecommendedTransition",
                "RecommendedTransitionPattern",
                "ProductionTotalAnchor",
                "RecommendedResort",
                "RecommendedResortPattern",
            ]
        )
        for row in race_results:
            transition = row.get("transition_recommended") or {}
            resort = row.get("resort_recommended") or {}
            writer.writerow(
                [
                    int(row.get("race_number", 0)),
                    row.get("candidate_frame"),
                    row.get("production_transition_frame"),
                    transition.get("anchor_frame"),
                    transition.get("pattern_len"),
                    row.get("production_total_anchor_frame"),
                    resort.get("anchor_frame"),
                    resort.get("pattern_len"),
                ]
            )
    return json_path, csv_path


def _resolve_video_path(video_input: str) -> Path:
    candidate = Path(video_input)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    from_input = PROJECT_ROOT / "Input_Videos" / video_input
    if from_input.exists():
        return from_input
    direct = PROJECT_ROOT / video_input
    if direct.exists():
        return direct
    raise FileNotFoundError(f"Video not found: {video_input}")


def main():
    parser = argparse.ArgumentParser(description="Experimental transition/resort pattern analyzer.")
    parser.add_argument("--video", required=True, help="Video path (absolute or relative to Input_Videos).")
    parser.add_argument("--patterns", default="5,7,9", help="Comma-separated streak lengths.")
    parser.add_argument("--max-false-gap", type=int, default=2, help="Allowed consecutive FALSE gap in a streak.")
    parser.add_argument("--races", nargs="*", type=int, help="Optional race numbers to analyze.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "Output_Results" / "Debug" / "PatternAnalyzer"),
        help="Output directory for JSON/CSV results.",
    )
    args = parser.parse_args()

    video_path = _resolve_video_path(args.video)
    pattern_lengths = sorted({int(value.strip()) for value in str(args.patterns).split(",") if value.strip()})
    if not pattern_lengths:
        raise RuntimeError("No pattern lengths provided.")

    templates = _load_templates()
    context = _prepare_video_context(
        str(video_path),
        str(PROJECT_ROOT / "Input_Videos"),
        True,
        1,
        1,
        0.0,
        templates,
        video_label=video_path.stem,
        source_display_name=str(video_path),
    )
    if context is None:
        raise RuntimeError(f"Failed to prepare context for video: {video_path}")

    scan_result = _run_scan_phase_for_context(context, templates, _NullWriter(), _NullWriter())
    candidates = list(scan_result.get("score_candidates") or [])
    candidates.sort(key=lambda item: int(item.get("race_number", 0)))
    if args.races:
        selected = {int(value) for value in args.races}
        candidates = [row for row in candidates if int(row.get("race_number", 0)) in selected]
    if not candidates:
        raise RuntimeError("No score candidates to analyze.")

    cap = cv2.VideoCapture(str(context["processing_video_path"]))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {context['processing_video_path']}")

    try:
        race_results = []
        for candidate in candidates:
            race_result = _analyze_race(
                cap,
                context=context,
                candidate=candidate,
                templates=templates,
                pattern_lengths=pattern_lengths,
                max_false_gap=int(args.max_false_gap),
            )
            race_results.append(race_result)
            recommended = race_result.get("transition_recommended") or {}
            print(
                "[PatternAnalyzer] Race {race:03d} | prod_transition={prod} | rec_transition={rec} (pattern={pat})".format(
                    race=int(race_result.get("race_number", 0)),
                    prod=race_result.get("production_transition_frame"),
                    rec=recommended.get("anchor_frame"),
                    pat=recommended.get("pattern_len"),
                )
            )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / f"{video_path.stem}_{timestamp}"
        json_path, csv_path = _write_outputs(output_dir, video_path.stem, race_results)
        print(f"[PatternAnalyzer] JSON: {json_path}")
        print(f"[PatternAnalyzer] CSV : {csv_path}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
