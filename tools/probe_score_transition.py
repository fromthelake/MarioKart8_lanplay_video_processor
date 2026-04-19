import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2

from mk8_local_play.extract_common import frame_to_timecode
from mk8_local_play.extract_frames import _prepare_video_context, _run_scan_phase_for_context
from mk8_local_play.extract_score_screen_selection import (
    SMALL_FORWARD_GRAB_WINDOW_FRAMES,
    _find_points_transition_frame,
    _match_ignore_frame_target_detail,
    _raw_fixed_grid_prefix_confirm,
    crop_and_upscale_image,
)
from mk8_local_play.extract_video_io import position_capture_for_read, read_video_frame, seek_to_frame
from mk8_local_play.ocr_scoreboard_consensus import extract_points_transition_observation, parse_detected_int
from mk8_local_play.project_paths import PROJECT_ROOT
from mk8_local_play.data_paths import resolve_asset_file


class _NullWriter:
    def writerow(self, _row):
        return None

    def writerows(self, _rows):
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


def _resolve_video_path(video_arg: str, include_subfolders: bool) -> Path:
    candidate = Path(video_arg)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    input_root = PROJECT_ROOT / "Input_Videos"
    if include_subfolders:
        resolved = input_root / video_arg
        if resolved.exists():
            return resolved
    for path in (input_root.rglob("*") if include_subfolders else input_root.iterdir()):
        if not path.is_file():
            continue
        if path.name.lower() == candidate.name.lower():
            return path
    raise FileNotFoundError(f"Video not found under Input_Videos: {video_arg}")


def _probe_transition_window(
    cap,
    *,
    fps,
    start_frame,
    end_frame,
    left,
    top,
    crop_width,
    crop_height,
    score_layout_id,
):
    stats = defaultdict(float)
    rows = []
    previous = None
    seek_to_frame(cap, int(start_frame), stats, label="transition_probe_start")
    chosen_transition = None
    for frame_number in range(int(start_frame), int(end_frame) + 1):
        ret, frame = read_video_frame(cap, stats)
        if not ret:
            break
        image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, 1280, 720)
        obs = extract_points_transition_observation(image, score_layout_id=score_layout_id)
        changed_total_rows = 0
        changed_race_rows = 0
        changed_any_rows = 0
        if previous is not None:
            for row_index in range(6):
                prev_race = parse_detected_int(previous["race_points"][row_index])
                cur_race = parse_detected_int(obs["race_points"][row_index])
                prev_total = parse_detected_int(previous["total_points"][row_index])
                cur_total = parse_detected_int(obs["total_points"][row_index])
                race_changed = prev_race is not None and cur_race is not None and prev_race != cur_race
                total_changed = prev_total is not None and cur_total is not None and prev_total != cur_total
                changed_race_rows += int(race_changed)
                changed_total_rows += int(total_changed)
                changed_any_rows += int(race_changed or total_changed)
        triggered = (
            previous is not None
            and changed_total_rows >= 2
            and (changed_race_rows >= 1 or changed_any_rows >= 3)
        )
        if triggered and chosen_transition is None:
            chosen_transition = int(frame_number)
        rows.append(
            {
                "Frame": int(frame_number),
                "Timecode": frame_to_timecode(int(frame_number), fps),
                "RacePointsTop6": "|".join(str(x) for x in obs.get("race_points", [])[:6]),
                "TotalPointsTop6": "|".join(str(x) for x in obs.get("total_points", [])[:6]),
                "ChangedRaceRows": int(changed_race_rows),
                "ChangedTotalRows": int(changed_total_rows),
                "ChangedAnyRows": int(changed_any_rows),
                "TransitionRuleTriggered": int(bool(triggered)),
            }
        )
        previous = obs
    return rows, chosen_transition


def main():
    parser = argparse.ArgumentParser(description="Probe score-candidate transition logic frame-by-frame.")
    parser.add_argument("--video", required=True, help="Video file path or relative path under Input_Videos")
    parser.add_argument("--race", type=int, required=True, help="Race number to inspect (1-based)")
    parser.add_argument("--subfolders", action="store_true", help="Resolve video path under Input_Videos recursively")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "Output_Results" / "Debug"), help="Directory for CSV outputs")
    args = parser.parse_args()

    video_path = _resolve_video_path(args.video, args.subfolders)
    templates = _load_templates()
    input_root = str(PROJECT_ROOT / "Input_Videos")
    context = _prepare_video_context(
        str(video_path),
        input_root,
        args.subfolders,
        1,
        1,
        0.0,
        templates,
    )
    if context is None:
        raise RuntimeError("Failed to prepare video context")

    scan_result = _run_scan_phase_for_context(context, templates, _NullWriter(), _NullWriter())
    score_candidates = list(scan_result.get("score_candidates") or [])
    candidate = next((item for item in score_candidates if int(item.get("race_number", 0)) == int(args.race)), None)
    if candidate is None:
        available = ",".join(str(int(item.get("race_number", 0))) for item in score_candidates)
        raise RuntimeError(f"Race {args.race} candidate not found. Available races: {available}")

    fps = float(context["fps"])
    candidate_frame = int(candidate["frame_number"])
    start_frame = int(candidate_frame - int(3 * fps))
    end_frame = int(candidate_frame + int(13 * fps))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{context['video_label']}_race_{int(args.race):03d}"
    score_csv = output_dir / f"{prefix}_score_probe.csv"
    transition_csv = output_dir / f"{prefix}_transition_probe.csv"

    stats = defaultdict(float)
    cap = cv2.VideoCapture(context["processing_video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture: {context['processing_video_path']}")
    try:
        probe_rows = []
        race_score_frame = 0
        score_hit_frame = None
        transition_frame = None
        selected_points_anchor_frame = None
        seek_to_frame(cap, int(start_frame), stats, label="score_probe_start")
        frame_number = int(start_frame)
        while frame_number <= int(end_frame):
            position_capture_for_read(
                cap,
                int(frame_number),
                stats,
                max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
                label="score_probe_loop",
            )
            ret, frame = read_video_frame(cap, stats)
            if not ret:
                break
            image = crop_and_upscale_image(
                frame,
                context["median_left"],
                context["median_top"],
                context["median_crop_width"],
                context["median_crop_height"],
                1280,
                720,
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ignore_match = _match_ignore_frame_target_detail(gray, templates, stats) if race_score_frame == 0 else {
                "label": "",
                "max_val": 0.0,
                "match_threshold": 0.0,
                "rejected_as_blank": True,
            }
            raw_passed, raw_score = _raw_fixed_grid_prefix_confirm(
                image,
                required_players=6,
                score_layout_id=str(candidate.get("score_layout_id") or context.get("score_layout_id") or ""),
                stats=stats,
            )
            score_trigger = bool(raw_score > 0.3 and not ignore_match.get("rejected_as_blank", True))
            if score_trigger and race_score_frame == 0:
                score_hit_frame = int(frame_number)
                race_score_frame = int(score_hit_frame + int(0.7 * fps))
                transition_search_end = min(int(end_frame), int(score_hit_frame + max(1, int(round(6.0 * max(fps, 1.0))))))
                transition_frame, selected_points_anchor_frame = _find_points_transition_frame(
                    cap,
                    int(race_score_frame),
                    int(transition_search_end),
                    context["median_left"],
                    context["median_top"],
                    context["median_crop_width"],
                    context["median_crop_height"],
                    str(candidate.get("score_layout_id") or ""),
                    stats,
                    fps,
                    int(context.get("source_height", 0) or 0),
                )
            probe_rows.append(
                {
                    "Frame": int(frame_number),
                    "Timecode": frame_to_timecode(int(frame_number), fps),
                    "Race": int(args.race),
                    "CandidateFrame": int(candidate_frame),
                    "RawPrefixPassed": int(bool(raw_passed)),
                    "RawPrefixScore": float(raw_score),
                    "IgnoreLabel": str(ignore_match.get("label", "")),
                    "IgnoreMax": float(ignore_match.get("max_val", 0.0)),
                    "IgnoreThreshold": float(ignore_match.get("match_threshold", 0.0)),
                    "IgnoreRejectedAsBlank": int(bool(ignore_match.get("rejected_as_blank", False))),
                    "ScoreTrigger": int(bool(score_trigger)),
                    "RaceScoreFrameLocked": int(race_score_frame) if race_score_frame else "",
                    "ScoreHitFrame": int(score_hit_frame) if score_hit_frame is not None else "",
                    "TransitionFrame": int(transition_frame) if transition_frame is not None else "",
                    "PointsAnchorFrame": int(selected_points_anchor_frame) if selected_points_anchor_frame is not None else "",
                }
            )
            frame_number += 1

        transition_rows = []
        transition_probe_start = race_score_frame if race_score_frame else candidate_frame
        transition_probe_end = min(end_frame, transition_probe_start + int(round(6.0 * max(fps, 1.0))))
        transition_rows, transition_rule_frame = _probe_transition_window(
            cap,
            fps=fps,
            start_frame=int(transition_probe_start),
            end_frame=int(transition_probe_end),
            left=context["median_left"],
            top=context["median_top"],
            crop_width=context["median_crop_width"],
            crop_height=context["median_crop_height"],
            score_layout_id=str(candidate.get("score_layout_id") or ""),
        )
    finally:
        cap.release()

    with score_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(probe_rows[0].keys()) if probe_rows else [])
        if probe_rows:
            writer.writeheader()
            writer.writerows(probe_rows)
    with transition_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transition_rows[0].keys()) if transition_rows else [])
        if transition_rows:
            writer.writeheader()
            writer.writerows(transition_rows)

    print(f"Video: {video_path}")
    print(f"Race: {args.race}")
    print(f"Candidate frame: {candidate_frame}")
    print(f"Score hit frame: {score_hit_frame}")
    print(f"Race score frame: {race_score_frame}")
    print(f"Transition frame (selector): {transition_frame}")
    print(f"Points anchor frame (selector): {selected_points_anchor_frame}")
    print(f"Transition frame (rule probe): {transition_rule_frame}")
    print(f"Score probe CSV: {score_csv}")
    print(f"Transition probe CSV: {transition_csv}")


if __name__ == "__main__":
    main()
