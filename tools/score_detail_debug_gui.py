"""Primary GUI for per-race detail scan debugging (RaceScore/Transition/TotalScore).

This tool is the current reference GUI for detail-phase scan behavior.
"""

import threading
import time
from collections import defaultdict
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from mk8_local_play.data_paths import resolve_asset_file
from mk8_local_play.extract_common import crop_and_upscale_image, frame_to_timecode
from mk8_local_play.extract_frames import _prepare_video_context, _run_scan_phase_for_context
import mk8_local_play.extract_initial_scan as initial_scan
import mk8_local_play.extract_score_screen_selection as score_sel
from mk8_local_play.extract_video_io import position_capture_for_read, read_video_frame, seek_to_frame
from mk8_local_play.ocr_scoreboard_consensus import extract_points_transition_observation, parse_detected_int
from mk8_local_play.project_paths import PROJECT_ROOT
from mk8_local_play.score_layouts import get_score_layout


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


def _parse_pause_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        return 0.2
    return max(0.0, min(5.0, value))


def _simulate_transition_rows(observations, fps):
    pattern_lengths = (5,)
    # Keep p5 diagnostics aligned with production debounce/fallback behavior.
    pattern_max_false_gap = int(score_sel._points_transition_max_false_gap_for_fps(float(fps)))

    def _is_transition_hit(changed_total_rows, changed_race_rows, changed_any_rows):
        return bool(
            int(changed_total_rows) >= 2
            and (int(changed_race_rows) >= 1 or int(changed_any_rows) >= 3)
        )

    confirm_true_count = int(score_sel._points_transition_confirm_true_count_for_fps(float(fps)))
    max_false_gap_frames = int(score_sel._points_transition_max_false_gap_for_fps(float(fps)))
    fallback_max_false_gap_frames = int(score_sel._points_transition_fallback_max_false_gap_for_fps(float(fps)))

    rows = []
    row_index_by_frame = {}
    previous = None
    first_trigger = None
    pending_start_frame = None
    pending_true_count = 0
    pending_false_streak = 0
    pending_base_race_points = None
    pending_last_race_points = None
    pending_last_total_points = None
    pending_true_frames = []
    production_p5_confirm_frame = None
    production_p5_anchor_frame = None
    pattern_states = {
        int(length): {
            "start_frame": None,
            "true_count": 0,
            "false_streak": 0,
            "confirm_frame": None,
            "anchor_frame": None,
            "true_frames": [],
        }
        for length in pattern_lengths
    }
    raw_history = []
    for frame_number, obs in observations:
        changed_total_rows = 0
        changed_race_rows = 0
        changed_any_rows = 0
        fallback_keep_alive = False
        fallback_stable_race_rows = 0
        fallback_stable_total_rows = 0
        fallback_changed_from_base_rows = 0
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
        raw_trigger = bool(
            previous is not None
            and _is_transition_hit(changed_total_rows, changed_race_rows, changed_any_rows)
        )
        raw_history.append("T" if raw_trigger else "F")
        if len(raw_history) > 16:
            raw_history = raw_history[-16:]
        triggered = False
        trigger_confirm_frame = None
        if raw_trigger:
            current_race_points, current_total_points = score_sel._parse_transition_top6(obs)
            if pending_start_frame is None:
                pending_start_frame = int(frame_number)
                pending_true_count = 1
                pending_false_streak = 0
                pending_base_race_points, _ = score_sel._parse_transition_top6(previous)
                pending_last_race_points = list(current_race_points)
                pending_last_total_points = list(current_total_points)
                pending_true_frames = [int(frame_number)]
            else:
                pending_true_count += 1
                pending_false_streak = 0
                pending_last_race_points = list(current_race_points)
                pending_last_total_points = list(current_total_points)
                pending_true_frames = list(pending_true_frames) + [int(frame_number)]
            if pending_true_count >= int(confirm_true_count):
                triggered = True
                trigger_confirm_frame = int(frame_number)
                production_p5_confirm_frame = int(frame_number)
                production_p5_anchor_frame = int(pending_start_frame)
                anchor_index = row_index_by_frame.get(int(pending_start_frame))
                if anchor_index is not None:
                    rows[anchor_index]["triggered"] = True
                    rows[anchor_index]["trigger_confirm_frame"] = int(frame_number)
                if first_trigger is None:
                    first_trigger = int(pending_start_frame)
                pending_start_frame = None
                pending_true_count = 0
                pending_false_streak = 0
                pending_base_race_points = None
                pending_last_race_points = None
                pending_last_total_points = None
                pending_true_frames = []
        elif pending_start_frame is not None:
            pending_false_streak += 1
            if pending_false_streak > int(max_false_gap_frames):
                current_race_points, current_total_points = score_sel._parse_transition_top6(obs)
                (
                    fallback_keep_alive,
                    fallback_stable_race_rows,
                    fallback_stable_total_rows,
                    fallback_changed_from_base_rows,
                ) = score_sel._points_transition_fallback_keep_alive(
                    false_streak=int(pending_false_streak),
                    max_false_gap=int(fallback_max_false_gap_frames),
                    base_race_points=pending_base_race_points,
                    previous_race_points=pending_last_race_points,
                    previous_total_points=pending_last_total_points,
                    current_race_points=current_race_points,
                    current_total_points=current_total_points,
                )
                if bool(fallback_keep_alive):
                    pending_last_race_points = list(current_race_points)
                    pending_last_total_points = list(current_total_points)
                else:
                    pending_start_frame = None
                    pending_true_count = 0
                    pending_false_streak = 0
                    pending_base_race_points = None
                    pending_last_race_points = None
                    pending_last_total_points = None
                    pending_true_frames = []
        pattern_snapshot = {}
        for length in pattern_lengths:
            state = pattern_states[int(length)]
            if raw_trigger:
                if state["start_frame"] is None:
                    state["start_frame"] = int(frame_number)
                    state["true_count"] = 1
                    state["false_streak"] = 0
                    state["true_frames"] = [int(frame_number)]
                else:
                    state["true_count"] = int(state["true_count"]) + 1
                    state["false_streak"] = 0
                    state["true_frames"] = list(state["true_frames"]) + [int(frame_number)]
                if int(state["true_count"]) >= int(length) and state["confirm_frame"] is None:
                    state["confirm_frame"] = int(frame_number)
                    state["anchor_frame"] = int(state["start_frame"])
            elif state["start_frame"] is not None and state["confirm_frame"] is None:
                state["false_streak"] = int(state["false_streak"]) + 1
                if int(state["false_streak"]) > int(pattern_max_false_gap):
                    state["start_frame"] = None
                    state["true_count"] = 0
                    state["false_streak"] = 0
                    state["true_frames"] = []

            pattern_snapshot[int(length)] = {
                "true_count": int(state["true_count"]),
                "false_streak": int(state["false_streak"]),
                "confirmed": bool(state["confirm_frame"] is not None),
                "start_frame": state["start_frame"],
                "confirm_frame": state["confirm_frame"],
                "anchor_frame": state["anchor_frame"],
                "streak_alive": bool(state["start_frame"] is not None and state["confirm_frame"] is None),
                "true_frames": list(state["true_frames"]),
            }

        rows.append(
            {
                "frame": int(frame_number),
                "checked_by_code": True,
                "triggered": triggered,
                "raw_trigger": bool(raw_trigger),
                "pending_true_count": int(pending_true_count),
                "pending_false_streak": int(pending_false_streak),
                "pending_start_frame": None if pending_start_frame is None else int(pending_start_frame),
                "trigger_confirm_frame": trigger_confirm_frame,
                "fallback_keep_alive": bool(fallback_keep_alive),
                "fallback_stable_race_rows": int(fallback_stable_race_rows),
                "fallback_stable_total_rows": int(fallback_stable_total_rows),
                "fallback_changed_from_base_rows": int(fallback_changed_from_base_rows),
                "changed_race_rows": int(changed_race_rows),
                "changed_total_rows": int(changed_total_rows),
                "changed_any_rows": int(changed_any_rows),
                "race_points_top6": "|".join(str(x) for x in obs.get("race_points", [])[:6]),
                "total_points_top6": "|".join(str(x) for x in obs.get("total_points", [])[:6]),
                "raw_trigger_history": "".join(raw_history),
                "current_frame_contributes_to_streak": bool(raw_trigger),
                "pattern_max_false_gap": int(pattern_max_false_gap),
                "fallback_max_false_gap_frames": int(fallback_max_false_gap_frames),
            }
        )
        for length in pattern_lengths:
            snapshot = pattern_snapshot[int(length)]
            if int(length) == 5:
                rows[-1][f"p{int(length)}_true_count"] = int(pending_true_count)
                rows[-1][f"p{int(length)}_false_streak"] = int(pending_false_streak)
                rows[-1][f"p{int(length)}_confirmed"] = bool(production_p5_confirm_frame is not None)
                rows[-1][f"p{int(length)}_start_frame"] = pending_start_frame
                rows[-1][f"p{int(length)}_confirm_frame"] = production_p5_confirm_frame
                rows[-1][f"p{int(length)}_anchor_frame"] = production_p5_anchor_frame
                rows[-1][f"p{int(length)}_streak_alive"] = bool(pending_start_frame is not None)
                rows[-1][f"p{int(length)}_true_frames"] = "|".join(str(v) for v in pending_true_frames)
            else:
                rows[-1][f"p{int(length)}_true_count"] = int(snapshot["true_count"])
                rows[-1][f"p{int(length)}_false_streak"] = int(snapshot["false_streak"])
                rows[-1][f"p{int(length)}_confirmed"] = bool(snapshot["confirmed"])
                rows[-1][f"p{int(length)}_start_frame"] = snapshot["start_frame"]
                rows[-1][f"p{int(length)}_confirm_frame"] = snapshot["confirm_frame"]
                rows[-1][f"p{int(length)}_anchor_frame"] = snapshot["anchor_frame"]
                rows[-1][f"p{int(length)}_streak_alive"] = bool(snapshot["streak_alive"])
                rows[-1][f"p{int(length)}_true_frames"] = "|".join(str(v) for v in snapshot["true_frames"])
        row_index_by_frame[int(frame_number)] = len(rows) - 1
        previous = obs
    return rows, first_trigger


def build_detail_trace(context, candidate, templates, progress_cb=None):
    def report(phase, percent):
        if progress_cb is None:
            return
        try:
            progress_cb(str(phase), float(percent))
        except Exception:
            pass

    fps = float(context["fps"])
    score_layout_id = str(candidate.get("score_layout_id") or "")
    candidate_frame = int(candidate["frame_number"])
    start_frame = candidate_frame - int(3 * fps)
    end_frame = candidate_frame + int(13 * fps)
    race_num = int(candidate["race_number"])
    is_low_res_source = bool(context.get("is_low_res_source", False))

    cap = cv2.VideoCapture(context["processing_video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {context['processing_video_path']}")

    try:
        report("Preparing detail scan", 0.0)
        stats = defaultdict(float)
        coarse_step = max(1, int(score_sel.COARSE_SEARCH_STEP_FRAMES))
        coarse_rewind = max(1, int(score_sel.COARSE_SEARCH_REWIND_FRAMES))
        detail_frame_number = int(start_frame)
        race_score_frame = 0
        score_hit_frame = None
        transition_frame = None
        selected_points_anchor_frame = None
        expected_total_players = None
        raw_points_anchor_estimate = None
        max_detected_players = None
        cap_source_frame = None
        cap_source_visible_rows = None
        cap_source_label = ""
        cap_fallback_used = False
        total_score_frame = None
        visited = {}
        transition_observations = []

        position_capture_for_read(
            cap,
            detail_frame_number,
            stats,
            max_forward_grab_frames=score_sel.SMALL_FORWARD_GRAB_WINDOW_FRAMES,
            label="gui_detail_start",
        )
        last_report = -1
        while detail_frame_number < int(end_frame):
            ret, frame = read_video_frame(cap, stats)
            if not ret:
                break
            if end_frame > start_frame:
                progress = ((int(detail_frame_number) - int(start_frame)) / float(int(end_frame) - int(start_frame))) * 65.0
                progress_int = int(progress)
                if progress_int != last_report and progress_int % 2 == 0:
                    last_report = progress_int
                    report("Scanning RaceScore detail frames", progress)

            upscaled_image = crop_and_upscale_image(
                frame,
                context["median_left"],
                context["median_top"],
                context["median_crop_width"],
                context["median_crop_height"],
                1280,
                720,
            )
            gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

            ignore_label = ""
            ignore_max = 0.0
            ignore_threshold = 0.0
            ignore_rejected_as_blank = True
            if race_score_frame == 0:
                ignore_match = score_sel._match_ignore_frame_target_detail(gray_image, templates, stats)
                ignore_label = str(ignore_match.get("label", ""))
                ignore_max = float(ignore_match.get("max_val", 0.0))
                ignore_threshold = float(ignore_match.get("match_threshold", 0.0))
                ignore_rejected_as_blank = bool(ignore_match.get("rejected_as_blank", True))

            if race_score_frame == 0:
                raw_confirm_passed, raw_confirm_score = score_sel._raw_fixed_grid_prefix_confirm(
                    upscaled_image,
                    required_players=score_sel.POSITION_SCAN_MIN_PLAYERS,
                    score_layout_id=score_layout_id,
                    stats=stats,
                )
                max_val = float(raw_confirm_score)
                rejected_as_blank = not bool(raw_confirm_passed)
                detected_layout_id = score_layout_id
                gate_rows = []
                for row_number in range(1, int(score_sel.POSITION_SCAN_MIN_PLAYERS) + 1):
                    tile_gray = initial_scan._initial_scan_gate_tile_roi(
                        gray_image,
                        int(row_number),
                        score_layout_id=score_layout_id,
                    )
                    tile_size = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_SIZE)
                    x1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_X)
                    if str(score_layout_id or "").strip() == str(initial_scan.LAN1_SCORE_LAYOUT_ID):
                        x1 += int(initial_scan.SCORE_LAYOUT_SHIFT_X)
                    y1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_Y) + ((int(row_number) - 1) * tile_size)
                    bbox = (int(x1), int(y1), int(tile_size), int(tile_size))
                    if tile_gray is None or getattr(tile_gray, "size", 0) == 0:
                        gate_rows.append(
                            {
                                "row": int(row_number),
                                "score": 0.0,
                                "passed": False,
                                "bbox": bbox,
                            }
                        )
                        continue
                    coeff, _variant_name = initial_scan._best_initial_scan_gate_score(tile_gray, int(row_number))
                    gate_rows.append(
                        {
                            "row": int(row_number),
                            "score": float(coeff),
                            "passed": bool(float(coeff) >= float(initial_scan.POSITION_SCAN_MIN_ROW_COEFF)),
                            "bbox": bbox,
                        }
                    )
            else:
                max_val, rejected_as_blank, detected_layout_id, _layout_metrics = score_sel._match_score_target_layouts(
                    upscaled_image,
                    templates,
                    stats,
                    return_layout_metrics=True,
                    preferred_layout_ids=[score_layout_id] if score_layout_id else None,
                    stats_scope="detail",
                )
                gate_rows = []

            visited[int(detail_frame_number)] = {
                "frame": int(detail_frame_number),
                "checked_by_code": True,
                "max_val": float(max_val),
                "rejected_as_blank": bool(rejected_as_blank),
                "ignore_label": ignore_label,
                "ignore_max": float(ignore_max),
                "ignore_threshold": float(ignore_threshold),
                "ignore_rejected_as_blank": bool(ignore_rejected_as_blank),
                "raw_confirm_mode": bool(race_score_frame == 0),
                "detected_layout_id": str(detected_layout_id or ""),
                "race_score_frame_locked": int(race_score_frame) if race_score_frame else None,
                "score_hit_frame": int(score_hit_frame) if score_hit_frame is not None else None,
                "transition_frame": int(transition_frame) if transition_frame is not None else None,
                "points_anchor_frame": int(selected_points_anchor_frame) if selected_points_anchor_frame is not None else None,
                "gate_rows": gate_rows,
            }

            if max_val > 0.3 and not rejected_as_blank and race_score_frame == 0:
                if coarse_step > 1:
                    coarse_step = 1
                    detail_frame_number = max(int(start_frame), int(detail_frame_number) - coarse_rewind)
                    seek_to_frame(cap, detail_frame_number, stats, label="gui_detail_rewind")
                    continue
                score_hit_frame = int(detail_frame_number)
                race_score_frame = score_hit_frame + int(0.7 * fps)
                transition_search_end = min(
                    int(end_frame),
                    score_hit_frame + max(1, int(round(score_sel.POINTS_TRANSITION_SEARCH_END_SECONDS * max(fps, 1.0)))),
                )
                transition_frame, selected_points_anchor_frame = score_sel._find_points_transition_frame(
                    cap,
                    int(race_score_frame),
                    int(transition_search_end),
                    context["median_left"],
                    context["median_top"],
                    context["median_crop_width"],
                    context["median_crop_height"],
                    score_layout_id,
                    stats,
                    fps,
                    int(context.get("source_height", 0) or 0),
                )
                if transition_frame is not None:
                    if selected_points_anchor_frame is None:
                        selected_points_anchor_frame = max(0, int(transition_frame) - 2)
                    anchor_window_counts = []
                    for probe_frame in range(
                        int(selected_points_anchor_frame) - 2,
                        int(selected_points_anchor_frame) + 3,
                    ):
                        estimate = score_sel._estimate_visible_players_at_frame(
                            cap,
                            max(0, int(probe_frame)),
                            context["median_left"],
                            context["median_top"],
                            context["median_crop_width"],
                            context["median_crop_height"],
                            score_layout_id,
                            stats,
                            is_low_res_source=bool(is_low_res_source),
                        )
                        if estimate is not None and int(estimate) > 0:
                            anchor_window_counts.append(int(estimate))
                    if anchor_window_counts:
                        max_detected_players = int(max(anchor_window_counts))
                    raw_points_anchor_estimate = score_sel._estimate_visible_players_at_frame(
                        cap,
                        int(selected_points_anchor_frame),
                        context["median_left"],
                        context["median_top"],
                        context["median_crop_width"],
                        context["median_crop_height"],
                        score_layout_id,
                        stats,
                        is_low_res_source=bool(is_low_res_source),
                    )
                    transition_cap = score_sel._estimate_visible_players_window_median(
                        cap,
                        int(selected_points_anchor_frame),
                        context["median_left"],
                        context["median_top"],
                        context["median_crop_width"],
                        context["median_crop_height"],
                        score_layout_id,
                        stats,
                        is_low_res_source=bool(is_low_res_source),
                        radius_frames=2,
                    )
                    if transition_cap is not None:
                        cap_source_frame = int(selected_points_anchor_frame)
                        cap_source_visible_rows = int(transition_cap)
                        cap_source_label = "transition_rule_frame"
                    elif race_score_frame:
                        race_anchor_cap = score_sel._estimate_visible_players_window_median(
                            cap,
                            int(race_score_frame),
                            context["median_left"],
                            context["median_top"],
                            context["median_crop_width"],
                            context["median_crop_height"],
                            score_layout_id,
                            stats,
                            is_low_res_source=bool(is_low_res_source),
                            radius_frames=2,
                        )
                        if race_anchor_cap is not None:
                            cap_source_frame = int(race_score_frame)
                            cap_source_visible_rows = int(race_anchor_cap)
                            cap_source_label = "race_score_anchor_frame"
                            cap_fallback_used = True

                    if raw_points_anchor_estimate is None:
                        expected_total_players = (
                            None if cap_source_visible_rows is None else int(cap_source_visible_rows)
                        )
                    elif cap_source_visible_rows is None:
                        expected_total_players = int(raw_points_anchor_estimate)
                    else:
                        expected_total_players = int(
                            min(int(raw_points_anchor_estimate), int(cap_source_visible_rows))
                        )
                    total_score_frame = score_sel._find_total_score_stable_frame(
                        cap,
                        int(transition_frame),
                        fps,
                        context["median_left"],
                        context["median_top"],
                        context["median_crop_width"],
                        context["median_crop_height"],
                        score_layout_id,
                        stats,
                        is_low_res_source=bool(is_low_res_source),
                        min_players_expected=expected_total_players,
                    )
                    break

            if rejected_as_blank and race_score_frame == 0 and coarse_step > 1:
                next_frame = min(int(end_frame), int(detail_frame_number) + coarse_step)
                if next_frame <= int(detail_frame_number):
                    break
                detail_frame_number = next_frame
                position_capture_for_read(
                    cap,
                    int(detail_frame_number),
                    stats,
                    max_forward_grab_frames=score_sel.SMALL_FORWARD_GRAB_WINDOW_FRAMES,
                    label="gui_detail_skip",
                )
                continue

            detail_frame_number += 1

        if race_score_frame and transition_frame is not None:
            report("Scanning transition frames", 66.0)
            transition_start = int(race_score_frame)
            transition_end = min(
                int(end_frame),
                int(transition_start + max(1, int(round(score_sel.POINTS_TRANSITION_SEARCH_END_SECONDS * max(fps, 1.0))))),
            )
            seek_to_frame(cap, int(transition_start), stats, label="gui_transition_probe")
            for frame_number in range(int(transition_start), int(transition_end) + 1):
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
                obs = extract_points_transition_observation(image, score_layout_id=score_layout_id)
                transition_observations.append((int(frame_number), obs))

        transition_rows, transition_rule_frame = _simulate_transition_rows(transition_observations, fps)
        report("Scanning TotalScore stable frames", 78.0)

        total_rows_map = {}
        if transition_frame is not None:
            total_start = int(transition_frame)
            total_end = int(transition_frame) + max(1, int(round(score_sel.TOTAL_SCORE_STABLE_SEARCH_SECONDS * max(fps, 1.0))))
            stable_required_low = int(score_sel.fps_scaled_frames(score_sel.LOW_RES_TOTAL_SCORE_STABLE_FRAMES_30FPS, fps))
            stable_required_high = int(score_sel.fps_scaled_frames(score_sel.TOTAL_SCORE_STABLE_FRAMES_30FPS, fps))
            required_players = max(
                int(score_sel.POSITION_SCAN_MIN_PLAYERS),
                min(12, int(expected_total_players or score_sel.POSITION_SCAN_MIN_PLAYERS)),
            )
            stage1_required = max(1, min(6, int(required_players)))
            signature_rows = score_sel._select_total_signature_rows(
                required_players,
                max_rows=int(score_sel.TOTAL_SCORE_SIGNATURE_MAX_ROWS),
            )
            signal_thresholds = score_sel._tie_aware_signal_thresholds(is_low_res_source=bool(is_low_res_source))
            row1_threshold = float(signal_thresholds.get("row1_coeff", 0.0))
            row_threshold = float(signal_thresholds.get("row_coeff", 0.0))
            run_signature = None
            run_length = 0
            run_start_frame = None
            previous_total_points = None
            for frame_number in range(total_start, total_end + 1):
                seek_to_frame(cap, int(frame_number), stats, label="gui_total_probe")
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
                row_signal = score_sel._tie_aware_score_signal_present(
                    image,
                    score_layout_id=score_layout_id,
                    min_players=required_players,
                    is_low_res_source=bool(is_low_res_source),
                )
                position_metrics = score_sel._position_metrics_for_frame(
                    image,
                    score_layout_id=score_layout_id,
                )
                detected_players_current = score_sel._estimate_visible_players_for_total_checks(
                    position_metrics,
                    is_low_res_source=bool(is_low_res_source),
                )
                stage1_pass = score_sel._tie_aware_score_signal_present_from_metrics(
                    position_metrics,
                    min_players=stage1_required,
                    is_low_res_source=bool(is_low_res_source),
                )
                if int(required_players) > int(stage1_required):
                    stage2_pass = score_sel._tie_aware_score_signal_present_from_metrics(
                        position_metrics,
                        min_players=required_players,
                        is_low_res_source=bool(is_low_res_source),
                    )
                else:
                    stage2_pass = True
                position_roi_rows = []
                for row_number in range(1, 13):
                    metric = position_metrics[row_number - 1] if row_number - 1 < len(position_metrics) else {}
                    coeff = float(metric.get("best_position_score", 0.0) or 0.0)
                    best_template = int(metric.get("best_position_template", 0) or 0)
                    threshold = row1_threshold if int(row_number) == 1 else row_threshold
                    passed = bool(coeff >= float(threshold))
                    tile_size = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_SIZE)
                    x1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_X)
                    if str(score_layout_id or "").strip() == str(initial_scan.LAN1_SCORE_LAYOUT_ID):
                        x1 += int(initial_scan.SCORE_LAYOUT_SHIFT_X)
                    y1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_Y) + ((int(row_number) - 1) * tile_size)
                    position_roi_rows.append(
                        {
                            "row": int(row_number),
                            "score": float(coeff),
                            "threshold": float(threshold),
                            "best_template": int(best_template),
                            "passed": bool(passed),
                            "bbox": (int(x1), int(y1), int(tile_size), int(tile_size)),
                        }
                    )
                points_observation = extract_points_transition_observation(
                    image,
                    score_layout_id=score_layout_id,
                )
                total_points = list(points_observation.get("total_points") or [])
                score_layout = get_score_layout(score_layout_id)
                total_roi_rows = []
                parsed_totals = []
                for idx, coord in enumerate(list(score_layout.total_points_coords)[:12], start=1):
                    (x1, y1), (x2, y2) = coord
                    raw_value = total_points[idx - 1] if idx - 1 < len(total_points) else ""
                    parsed_value = parse_detected_int(raw_value)
                    parsed_totals.append(parsed_value)
                    total_roi_rows.append(
                        {
                            "row": int(idx),
                            "raw_value": str(raw_value),
                            "parsed_value": None if parsed_value is None else int(parsed_value),
                            "passed": bool(parsed_value is not None),
                            "bbox": (int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))),
                        }
                    )
                signature = score_sel._extract_total_score_stable_signature(
                    image,
                    score_layout_id=score_layout_id,
                    is_low_res_source=bool(is_low_res_source),
                    min_players_expected=expected_total_players,
                )
                signature_tuple = tuple(signature) if signature is not None else None
                changed_rows_all = []
                changed_rows_expected = []
                if previous_total_points is not None:
                    for idx, (prev_val, cur_val) in enumerate(zip(previous_total_points, parsed_totals), start=1):
                        if prev_val is not None and cur_val is not None and int(prev_val) != int(cur_val):
                            changed_rows_all.append(int(idx))
                            if idx <= int(required_players):
                                changed_rows_expected.append(int(idx))
                previous_total_points = parsed_totals
                prefix_row_pass_count = 0
                for item in position_roi_rows[: int(required_players)]:
                    if bool(item.get("passed", False)):
                        prefix_row_pass_count += 1
                parsed_prefix_count = sum(1 for value in parsed_totals[: int(required_players)] if value is not None)
                signature_parse_pass_count = sum(
                    1
                    for row_number in signature_rows
                    if 1 <= int(row_number) <= len(parsed_totals) and parsed_totals[int(row_number) - 1] is not None
                )
                reset_reason = ""
                if signature_tuple is None:
                    if run_length > 0:
                        reset_reason = "No stable signature on this frame."
                    run_signature = None
                    run_length = 0
                    run_start_frame = None
                elif run_signature == signature_tuple:
                    run_length += 1
                else:
                    if run_length > 0 and run_signature is not None:
                        reset_reason = "Signature changed from previous stable run."
                    run_signature = signature_tuple
                    run_length = 1
                    run_start_frame = int(frame_number)
                total_rows_map[int(frame_number)] = {
                    "frame": int(frame_number),
                    "checked_by_code": True,
                    "row_signal": bool(row_signal),
                    "signature": "" if signature is None else "|".join(str(v) for v in signature_tuple),
                    "stable_target_frame": int(total_score_frame) if total_score_frame is not None else None,
                    "stable_run_len": int(run_length),
                    "stable_run_start_frame": None if run_start_frame is None else int(run_start_frame),
                    "stable_run_signature": "" if run_signature is None else "|".join(str(v) for v in run_signature),
                    "stable_required_low": int(stable_required_low),
                    "stable_required_high": int(stable_required_high),
                    "stable_pass_low": bool(run_length >= stable_required_low),
                    "stable_pass_high": bool(run_length >= stable_required_high),
                    "stable_reset_reason": str(reset_reason),
                    "is_low_res_source": bool(is_low_res_source),
                    "expected_total_players": int(expected_total_players) if expected_total_players is not None else None,
                    "max_detected_players": int(max_detected_players) if max_detected_players is not None else None,
                    "raw_points_anchor_estimate": int(raw_points_anchor_estimate) if raw_points_anchor_estimate is not None else None,
                    "cap_source_frame": int(cap_source_frame) if cap_source_frame is not None else None,
                    "cap_source_visible_rows": int(cap_source_visible_rows) if cap_source_visible_rows is not None else None,
                    "cap_source_label": str(cap_source_label or ""),
                    "cap_fallback_used": bool(cap_fallback_used),
                    "detected_players_current": int(detected_players_current),
                    "required_players_for_total": int(required_players),
                    "stage1_required_players": int(stage1_required),
                    "stage1_pass": bool(stage1_pass),
                    "stage2_pass": bool(stage2_pass),
                    "signature_rows": "|".join(str(v) for v in signature_rows),
                    "signature_parse_pass_count": int(signature_parse_pass_count),
                    "position_prefix_pass_count": int(prefix_row_pass_count),
                    "total_parse_pass_prefix_count": int(parsed_prefix_count),
                    "changed_total_rows_all": "|".join(str(v) for v in changed_rows_all),
                    "changed_total_rows_expected": "|".join(str(v) for v in changed_rows_expected),
                    "position_roi_rows": position_roi_rows,
                    "total_roi_rows": total_roi_rows,
                }
                if total_end > total_start:
                    pct = 78.0 + (((int(frame_number) - int(total_start)) / float(int(total_end) - int(total_start))) * 20.0)
                    report("Scanning TotalScore stable frames", pct)

        race_rows = []
        for frame_number in range(int(start_frame), int(end_frame) + 1):
            row = {
                "frame": int(frame_number),
                "checked_by_code": frame_number in visited,
            }
            if frame_number in visited:
                row.update(visited[frame_number])
            race_rows.append(row)

        transition_rows_map = {int(row["frame"]): row for row in transition_rows}
        if transition_rows_map:
            tr_start = min(transition_rows_map.keys())
            tr_end = max(transition_rows_map.keys())
            transition_view_rows = []
            for frame_number in range(tr_start, tr_end + 1):
                row = {"frame": int(frame_number), "checked_by_code": frame_number in transition_rows_map}
                if frame_number in transition_rows_map:
                    row.update(transition_rows_map[frame_number])
                transition_view_rows.append(row)
        else:
            transition_view_rows = []

        if total_rows_map:
            total_start = min(total_rows_map.keys())
            total_end = max(total_rows_map.keys())
            total_view_rows = []
            for frame_number in range(total_start, total_end + 1):
                row = {"frame": int(frame_number), "checked_by_code": frame_number in total_rows_map}
                if frame_number in total_rows_map:
                    row.update(total_rows_map[frame_number])
                total_view_rows.append(row)
        else:
            total_view_rows = []

        summary = {
            "race_number": int(race_num),
            "candidate_frame": int(candidate_frame),
            "score_layout_id": str(score_layout_id or ""),
            "score_hit_frame": score_hit_frame,
            "race_score_frame": race_score_frame if race_score_frame else None,
            "transition_frame": transition_frame,
            "points_anchor_frame": selected_points_anchor_frame,
            "transition_rule_frame": transition_rule_frame,
            "total_score_frame": total_score_frame,
            "expected_total_players": int(expected_total_players) if expected_total_players is not None else None,
            "max_detected_players": int(max_detected_players) if max_detected_players is not None else None,
            "raw_points_anchor_estimate": int(raw_points_anchor_estimate) if raw_points_anchor_estimate is not None else None,
            "cap_source_frame": int(cap_source_frame) if cap_source_frame is not None else None,
            "cap_source_visible_rows": int(cap_source_visible_rows) if cap_source_visible_rows is not None else None,
            "cap_source_label": str(cap_source_label or ""),
            "cap_fallback_used": bool(cap_fallback_used),
        }
        report("Finalizing trace", 99.0)
        return race_rows, transition_view_rows, total_view_rows, summary
    finally:
        report("Detail trace ready", 100.0)
        cap.release()


class ScoreDetailDebugGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MK8 Score Detail Debug GUI")
        self.geometry("1840x980")

        self.templates = _load_templates()
        self.video_path = None
        self.context = None
        self.score_candidates = []

        self.race_rows = []
        self.transition_rows = []
        self.total_rows = []
        self.summary = {}

        self.current_rows = []
        self.current_row_map = {}
        self.range_start_frame = 0
        self.range_end_frame = 0
        self.current_frame = 0
        self.current_mode = tk.StringVar(value="RaceScore Detail")
        self.is_playing = False
        self.is_production_playing = False
        self.production_frames = []
        self.production_index = 0
        self.production_play_btn_text = tk.StringVar(value="Play Production Steps")
        self.pause_seconds_var = tk.StringVar(value="0.20")
        self.status_var = tk.StringVar(value="Select a video and run initial scan.")
        self.trace_state_var = tk.StringVar(value="Detail trace: NOT LOADED")
        self.initial_scan_progress_var = tk.StringVar(value="Initial scan: idle")
        self.initial_scan_progress_value = tk.DoubleVar(value=0.0)
        self.trace_progress_var = tk.StringVar(value="")
        self.trace_progress_value = tk.DoubleVar(value=0.0)
        self.code_parity_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="")
        self._trace_load_token = 0
        self.jump_frame_var = tk.StringVar(value="")
        self.production_checked_frames_all = set()
        self.production_checked_phase_map = {}
        self.production_frame_entries = []
        self._production_row_override = None
        self._production_phase_override = None
        self.trace_ready = False
        self._inspection_row_cache = {}
        self._trace_cache = {}
        self._play_after_id = None
        self._prod_play_after_id = None

        self.capture = None
        self.photo = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(top, text="Select Video", command=self._on_select_video).pack(side=tk.LEFT)
        self.video_label = ttk.Label(top, text="No video selected", width=120)
        self.video_label.pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Run Initial Scan", command=self._on_run_initial_scan).pack(side=tk.LEFT, padx=6)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(main, width=300)
        center = ttk.Frame(main)
        right = ttk.Frame(main, width=500)
        main.add(left, weight=1)
        main.add(center, weight=6)
        main.add(right, weight=2)

        ttk.Label(left, text="Detected RaceScore Candidates").pack(anchor=tk.W)
        self.candidate_list = tk.Listbox(left, height=25)
        self.candidate_list.pack(fill=tk.BOTH, expand=True, pady=4)
        self.candidate_list.bind("<<ListboxSelect>>", self._on_candidate_selected)
        ttk.Label(left, text="Mode").pack(anchor=tk.W, pady=(8, 2))
        mode_combo = ttk.Combobox(
            left,
            textvariable=self.current_mode,
            values=["RaceScore Detail", "Transition Scan", "TotalScore Scan"],
            state="readonly",
        )
        mode_combo.pack(fill=tk.X)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)
        self.trace_state_label = tk.Label(left, textvariable=self.trace_state_var, fg="#b00020", anchor="w", justify=tk.LEFT)
        self.trace_state_label.pack(fill=tk.X, pady=(8, 0))

        self.video_canvas = tk.Label(center, bg="#111111")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        self.video_canvas.bind("<Configure>", lambda _e: self._render_current_frame())

        controls_center = ttk.Frame(center)
        controls_center.pack(fill=tk.X, pady=6)
        row1 = ttk.Frame(controls_center)
        row1.pack(fill=tk.X, pady=2)
        ttk.Button(row1, text="Load Selected Race", command=self._load_selected_race).pack(side=tk.LEFT)
        ttk.Button(row1, text="Frame -1", command=lambda: self._step_frames(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Frame +1", command=lambda: self._step_frames(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Prod -1", command=lambda: self._step_production(-1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Prod +1", command=lambda: self._step_production(1)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Jump -1s", command=lambda: self._step_seconds(-1.0)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Jump +1s", command=lambda: self._step_seconds(1.0)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="Play/Pause", command=self._toggle_play).pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, textvariable=self.production_play_btn_text, command=self._toggle_play_production).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(controls_center)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Jump to frame:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.jump_frame_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Go", command=self._jump_to_frame_from_entry).pack(side=tk.LEFT)
        ttk.Label(row2, text="Pause/frame (s):").pack(side=tk.LEFT, padx=(16, 4))
        ttk.Entry(row2, textvariable=self.pause_seconds_var, width=7).pack(side=tk.LEFT)

        ttk.Label(right, text="Summary").pack(anchor=tk.W)
        self.summary_text = tk.Text(right, height=11, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.X, pady=(0, 6))
        self.summary_text.configure(state=tk.DISABLED)
        ttk.Label(right, text="Code Frame Parity").pack(anchor=tk.W)
        self.code_parity_label = ttk.Label(right, textvariable=self.code_parity_var, foreground="red")
        self.code_parity_label.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(right, text="Frame Metrics").pack(anchor=tk.W)
        self.metrics_text = tk.Text(right, height=45, wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.tag_config("bool_true", foreground="green")
        self.metrics_text.tag_config("bool_false", foreground="red")

        status = ttk.Label(self, textvariable=self.status_var)
        status.pack(fill=tk.X, padx=8, pady=(0, 6))

        initial_progress = ttk.Label(self, textvariable=self.initial_scan_progress_var)
        initial_progress.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.initial_progress = ttk.Progressbar(
            self,
            mode="determinate",
            maximum=100.0,
            variable=self.initial_scan_progress_value,
        )
        self.initial_progress.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.initial_scan_progress_value.set(0.0)

        trace_progress = ttk.Label(self, textvariable=self.trace_progress_var)
        trace_progress.pack(fill=tk.X, padx=8, pady=(0, 4))
        self.trace_progress = ttk.Progressbar(
            self,
            mode="determinate",
            maximum=100.0,
            variable=self.trace_progress_value,
        )
        self.trace_progress.pack(fill=tk.X, padx=8, pady=(0, 6))
        self.trace_progress_value.set(0.0)

    def _on_select_video(self):
        initial_dir = str(PROJECT_ROOT / "Input_Videos")
        path = filedialog.askopenfilename(
            title="Select MK8 Video",
            initialdir=initial_dir,
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*")],
        )
        if not path:
            return
        self.video_path = Path(path)
        self.video_label.configure(text=str(self.video_path))
        self.status_var.set("Video selected. Running initial scan...")
        self._on_run_initial_scan()

    def _set_trace_state(self, loaded, message):
        self.trace_state_var.set(message)
        self.trace_state_label.configure(fg=("#0a7a19" if bool(loaded) else "#b00020"))

    def _start_initial_scan_progress(self):
        self.initial_scan_progress_var.set("Initial scan: running...")
        self.initial_scan_progress_value.set(0.0)
        self.initial_progress.configure(mode="indeterminate")
        self.initial_progress.start(12)

    def _finish_initial_scan_progress(self, *, ok):
        self.initial_progress.stop()
        self.initial_progress.configure(mode="determinate")
        self.initial_scan_progress_value.set(100.0 if bool(ok) else 0.0)
        self.initial_scan_progress_var.set("Initial scan: done (100%)" if bool(ok) else "Initial scan: failed")

    def _on_run_initial_scan(self):
        if self.video_path is None:
            messagebox.showwarning("No video", "Select a video first.")
            return
        self._start_initial_scan_progress()
        self._set_trace_state(False, "Detail trace: NOT LOADED")
        self._trace_cache = {}

        def worker():
            try:
                self.status_var.set("Running initial scan...")
                folder_path = str(PROJECT_ROOT / "Input_Videos")
                include_subfolders = True
                context = _prepare_video_context(
                    str(self.video_path),
                    folder_path,
                    include_subfolders,
                    1,
                    1,
                    0.0,
                    self.templates,
                    video_label=self.video_path.stem,
                    source_display_name=str(self.video_path),
                )
                if context is None:
                    raise RuntimeError("Failed to prepare video context")
                scan_result = _run_scan_phase_for_context(context, self.templates, _NullWriter(), _NullWriter())
                candidates = list(scan_result.get("score_candidates") or [])
                candidates.sort(key=lambda item: int(item.get("race_number", 0)))
                self.after(0, lambda: self._on_scan_complete(context, candidates))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Scan failed", str(exc)))
                self.after(0, lambda: self.status_var.set("Initial scan failed."))
                self.after(0, lambda: self._finish_initial_scan_progress(ok=False))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_complete(self, context, candidates):
        self._finish_initial_scan_progress(ok=True)
        self.context = context
        self.score_candidates = candidates
        self.race_rows = []
        self.transition_rows = []
        self.total_rows = []
        self.summary = {}
        self.trace_ready = False
        self._set_trace_progress("idle", 0.0)
        self._set_trace_state(False, "Detail trace: NOT LOADED")
        self.candidate_list.delete(0, tk.END)
        for candidate in candidates:
            race = int(candidate.get("race_number", 0))
            frame = int(candidate.get("frame_number", 0))
            label = f"Race {race:03d} | Candidate frame {frame} | {frame_to_timecode(frame, float(context['fps']))}"
            self.candidate_list.insert(tk.END, label)
        self.candidate_list.selection_clear(0, tk.END)
        self.status_var.set(f"Initial scan done. Found {len(candidates)} race candidates. Select a race to load detail scan.")

    def _candidate_key(self, candidate):
        return (
            int(candidate.get("race_number", 0)),
            int(candidate.get("frame_number", 0)),
            str(candidate.get("score_layout_id") or ""),
        )

    def _apply_trace_data(self, race_rows, transition_rows, total_rows, summary, *, from_cache):
        self.race_rows = list(race_rows)
        self.transition_rows = list(transition_rows)
        self.total_rows = list(total_rows)
        self.summary = dict(summary)
        self._rebuild_production_checked_maps()
        self.trace_ready = True
        self._production_row_override = None
        self._production_phase_override = None
        self._set_mode_rows()
        fps = max(1.0, float(self.context["fps"]))
        candidate_frame = summary.get("candidate_frame")
        range_start = candidate_frame - int(3 * fps) if candidate_frame is not None else None
        range_end = candidate_frame + int(13 * fps) if candidate_frame is not None else None
        selected_frames = []
        for value in (
            summary.get("candidate_frame"),
            range_start,
            range_end,
            summary.get("score_hit_frame"),
            summary.get("race_score_frame"),
            summary.get("transition_frame"),
            summary.get("points_anchor_frame"),
            summary.get("transition_rule_frame"),
            summary.get("total_score_frame"),
        ):
            if value is None:
                continue
            frame_value = int(value)
            if frame_value not in selected_frames:
                selected_frames.append(frame_value)
        total_anchor_reason = self._build_total_anchor_reason(summary)
        self._set_summary_lines(
            [
                ("Race", f"{int(summary.get('race_number', 0)):03d}", None),
                ("Candidate Frame", str(summary.get("candidate_frame")), summary.get("candidate_frame")),
                ("Detail Range Start", f"{range_start} (-3.0s)", range_start),
                ("Detail Range End", f"{range_end} (+13.0s)", range_end),
                ("Score Hit Frame", str(summary.get("score_hit_frame")), summary.get("score_hit_frame")),
                ("Race Anchor Frame", str(summary.get("race_score_frame")), summary.get("race_score_frame")),
                ("Transition Frame", str(summary.get("transition_frame")), summary.get("transition_frame")),
                ("Points Anchor Frame", str(summary.get("points_anchor_frame")), summary.get("points_anchor_frame")),
                ("Expected Players (from points anchor)", str(summary.get("expected_total_players")), None),
                ("Max Detected Players", str(summary.get("max_detected_players")), None),
                ("Raw Points-Anchor Estimate", str(summary.get("raw_points_anchor_estimate")), None),
                ("Cap Source Label", str(summary.get("cap_source_label")), None),
                ("Cap Source Frame", str(summary.get("cap_source_frame")), summary.get("cap_source_frame")),
                ("Cap Source Visible Rows", str(summary.get("cap_source_visible_rows")), None),
                ("Cap Fallback Used", "TRUE" if bool(summary.get("cap_fallback_used", False)) else "FALSE", None),
                ("Transition Rule Frame", str(summary.get("transition_rule_frame")), summary.get("transition_rule_frame")),
                ("Total Anchor Frame", str(summary.get("total_score_frame")), summary.get("total_score_frame")),
                ("Total Anchor Status", total_anchor_reason, None),
                ("Selected Frames", ", ".join(str(v) for v in selected_frames), selected_frames),
            ]
        )
        self._set_trace_state(True, f"Detail trace: LOADED (Race {int(summary.get('race_number', 0)):03d}{' - cached' if from_cache else ''})")
        self.status_var.set("Detail trace loaded from cache." if from_cache else "Detail trace loaded.")
        self._set_trace_progress("ready", 100.0)

    def _on_candidate_selected(self, _event=None):
        index = self.candidate_list.curselection()
        if not index:
            return
        race_idx = int(index[0])
        if race_idx < len(self.score_candidates):
            candidate = self.score_candidates[race_idx]
            race_number = int(candidate.get("race_number", 0))
            self._reset_playback_state()
            self._show_race_preview(candidate)
            key = self._candidate_key(candidate)
            cached = self._trace_cache.get(key)
            if cached is not None:
                self.status_var.set(f"Selected race {race_number:03d}. Restored cached detail trace.")
                self._apply_trace_data(
                    cached.get("race_rows") or [],
                    cached.get("transition_rows") or [],
                    cached.get("total_rows") or [],
                    cached.get("summary") or {},
                    from_cache=True,
                )
            else:
                self.status_var.set(f"Selected race {race_number:03d}. Click 'Load Selected Race' to build detail trace.")
                self._set_trace_state(False, f"Detail trace: NOT LOADED (Race {race_number:03d})")

    def _show_race_preview(self, candidate):
        if self.context is None:
            return
        fps = float(self.context["fps"])
        candidate_frame = int(candidate.get("frame_number", 0))
        start_frame = max(0, int(candidate_frame - int(3 * fps)))
        end_frame = max(start_frame, int(candidate_frame + int(13 * fps)))
        self.current_rows = []
        self.current_row_map = {}
        self.trace_ready = False
        self._inspection_row_cache = {}
        self._production_row_override = None
        self._production_phase_override = None
        self.range_start_frame = int(start_frame)
        self.range_end_frame = int(end_frame)
        self.current_frame = int(start_frame)
        self._set_trace_progress("idle", 0.0)
        self._set_summary_lines(
            [
                ("Race", f"{int(candidate.get('race_number', 0)):03d}", None),
                ("Candidate Frame", str(candidate_frame), int(candidate_frame)),
                ("Detail Range Start", f"{start_frame} (-3.0s)", int(start_frame)),
                ("Detail Range End", f"{end_frame} (+13.0s)", int(end_frame)),
            ]
        )
        self._render_current_frame()

    def _load_selected_race(self):
        if self.context is None or not self.score_candidates:
            messagebox.showwarning("No scan data", "Run initial scan first.")
            return
        selected = self.candidate_list.curselection()
        if not selected:
            messagebox.showwarning("No race selected", "Select a race from the list.")
            return
        candidate = self.score_candidates[int(selected[0])]
        cache_key = self._candidate_key(candidate)
        cached = self._trace_cache.get(cache_key)
        if cached is not None:
            self._apply_trace_data(
                cached.get("race_rows") or [],
                cached.get("transition_rows") or [],
                cached.get("total_rows") or [],
                cached.get("summary") or {},
                from_cache=True,
            )
            return
        self._reset_playback_state()
        self._inspection_row_cache = {}
        self._trace_load_token += 1
        load_token = int(self._trace_load_token)
        self._set_trace_progress("starting", 1.0)
        self._set_trace_state(False, f"Detail trace: LOADING (Race {int(candidate.get('race_number', 0)):03d})")

        def worker():
            try:
                race_num = int(candidate.get("race_number", 0))
                self.after(0, lambda: self.status_var.set(f"Building detail trace for race {race_num:03d}..."))
                def progress_update(phase, percent):
                    self.after(0, lambda: self._set_trace_progress(phase, percent))

                race_rows, transition_rows, total_rows, summary = build_detail_trace(
                    self.context,
                    candidate,
                    self.templates,
                    progress_cb=progress_update,
                )
                self.after(
                    0,
                    lambda: self._on_trace_loaded(
                        load_token,
                        race_rows,
                        transition_rows,
                        total_rows,
                        summary,
                        cache_key,
                    ),
                )
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Trace failed", str(exc)))
                self.after(0, lambda: self.status_var.set("Detail trace failed."))
                self.after(0, lambda: self._set_trace_progress("failed", 0.0))
                self.after(0, lambda: self._set_trace_state(False, "Detail trace: FAILED"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_trace_loaded(self, load_token, race_rows, transition_rows, total_rows, summary, cache_key):
        if int(load_token) != int(self._trace_load_token):
            return
        self._trace_cache[cache_key] = {
            "race_rows": list(race_rows),
            "transition_rows": list(transition_rows),
            "total_rows": list(total_rows),
            "summary": dict(summary),
        }
        self._apply_trace_data(race_rows, transition_rows, total_rows, summary, from_cache=False)

    def _build_total_anchor_reason(self, summary):
        total_anchor = summary.get("total_score_frame")
        if total_anchor is not None:
            return f"Locked at frame {int(total_anchor)}."
        transition_frame = summary.get("transition_frame")
        if transition_frame is None:
            return "Not found: transition frame was not detected, so TotalScore search could not start."
        return "Not found: transition exists, but no stable TotalScore frame passed row-signal/signature checks in the search window."

    def _set_trace_progress(self, phase, percent):
        clamped = max(0.0, min(100.0, float(percent)))
        self.trace_progress_var.set(f"Detail trace: {phase} ({clamped:.0f}%)")
        self.trace_progress_value.set(clamped)

    def _on_mode_changed(self, _event=None):
        self._set_mode_rows()

    def _set_mode_rows(self):
        mode = self.current_mode.get()
        if mode == "Transition Scan":
            self.current_rows = list(self.transition_rows)
        elif mode == "TotalScore Scan":
            self.current_rows = list(self.total_rows)
        else:
            self.current_rows = list(self.race_rows)
        self.current_row_map = {int(row.get("frame", -1)): row for row in self.current_rows if row.get("frame") is not None}
        if self.current_rows:
            self.range_start_frame = int(min(self.current_row_map.keys()))
            self.range_end_frame = int(max(self.current_row_map.keys()))
            self.current_frame = int(self.range_start_frame)
            self.jump_frame_var.set(str(self.current_frame))
        self._render_current_frame()

    def _rebuild_production_checked_maps(self):
        self.production_checked_frames_all = set()
        self.production_checked_phase_map = {}
        self.production_frame_entries = []
        for phase_name, rows in (
            ("RaceScore Detail", self.race_rows),
            ("Transition Scan", self.transition_rows),
            ("TotalScore Scan", self.total_rows),
        ):
            for row in rows:
                frame = int(row.get("frame", -1))
                if frame < 0:
                    continue
                if bool(row.get("checked_by_code", False)):
                    self.production_checked_frames_all.add(frame)
                    phases = self.production_checked_phase_map.setdefault(frame, set())
                    phases.add(phase_name)
                    self.production_frame_entries.append(
                        {
                            "frame": int(frame),
                            "phase": str(phase_name),
                            "row": dict(row),
                        }
                    )
        self.production_frame_entries.sort(key=lambda item: int(item.get("frame", -1)))

    def _production_entries_for_active_mode(self):
        mode = str(self._production_phase_override or self.current_mode.get())
        if mode not in {"RaceScore Detail", "Transition Scan", "TotalScore Scan"}:
            mode = str(self.current_mode.get())
        entries = [
            item for item in self.production_frame_entries
            if str(item.get("phase") or "") == mode
        ]
        entries.sort(key=lambda item: int(item.get("frame", -1)))
        return entries

    def _ensure_capture(self):
        if self.capture is not None:
            return
        if self.context is None:
            return
        self.capture = cv2.VideoCapture(self.context["processing_video_path"])

    def _evaluate_frame_for_overlay(self, image, frame_number, mode):
        key = (int(frame_number), str(mode))
        cached = self._inspection_row_cache.get(key)
        if cached is not None:
            return dict(cached)

        row = {"frame": int(frame_number), "checked_by_code": False}
        if str(mode) == "RaceScore Detail":
            stats = defaultdict(float)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ignore_match = score_sel._match_ignore_frame_target_detail(gray, self.templates, stats)
            ignore_label = str(ignore_match.get("label", ""))
            ignore_max = float(ignore_match.get("max_val", 0.0))
            ignore_threshold = float(ignore_match.get("match_threshold", 0.0))
            ignore_rejected_as_blank = bool(ignore_match.get("rejected_as_blank", True))
            score_layout_id = str((self.summary or {}).get("score_layout_id") or "")
            raw_confirm_passed, raw_confirm_score = score_sel._raw_fixed_grid_prefix_confirm(
                image,
                required_players=score_sel.POSITION_SCAN_MIN_PLAYERS,
                score_layout_id=score_layout_id,
                stats=stats,
            )
            gate_rows = []
            for row_number in range(1, int(score_sel.POSITION_SCAN_MIN_PLAYERS) + 1):
                tile_gray = initial_scan._initial_scan_gate_tile_roi(
                    gray,
                    int(row_number),
                    score_layout_id=score_layout_id,
                )
                tile_size = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_SIZE)
                x1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_X)
                if str(score_layout_id or "").strip() == str(initial_scan.LAN1_SCORE_LAYOUT_ID):
                    x1 += int(initial_scan.SCORE_LAYOUT_SHIFT_X)
                y1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_Y) + ((int(row_number) - 1) * tile_size)
                bbox = (int(x1), int(y1), int(tile_size), int(tile_size))
                if tile_gray is None or getattr(tile_gray, "size", 0) == 0:
                    gate_rows.append({"row": int(row_number), "score": 0.0, "passed": False, "bbox": bbox})
                    continue
                coeff, _variant_name = initial_scan._best_initial_scan_gate_score(tile_gray, int(row_number))
                gate_rows.append(
                    {
                        "row": int(row_number),
                        "score": float(coeff),
                        "passed": bool(float(coeff) >= float(initial_scan.POSITION_SCAN_MIN_ROW_COEFF)),
                        "bbox": bbox,
                    }
                )
            row.update(
                {
                    "max_val": float(raw_confirm_score),
                    "rejected_as_blank": not bool(raw_confirm_passed),
                    "ignore_label": ignore_label,
                    "ignore_max": ignore_max,
                    "ignore_threshold": ignore_threshold,
                    "ignore_rejected_as_blank": ignore_rejected_as_blank,
                    "raw_confirm_mode": True,
                    "detected_layout_id": score_layout_id,
                    "race_score_frame_locked": None,
                    "score_hit_frame": None,
                    "transition_frame": None,
                    "points_anchor_frame": None,
                    "gate_rows": gate_rows,
                    "inspection_reason": "Inspection-only frame (not production-checked); values computed live.",
                }
            )
        else:
            score_layout_id = str((self.summary or {}).get("score_layout_id") or "")
            if str(mode) == "TotalScore Scan":
                is_low_res_source = bool(self.context.get("is_low_res_source", False)) if self.context else False
                expected_players = self.summary.get("expected_total_players")
                required_players = max(
                    int(score_sel.POSITION_SCAN_MIN_PLAYERS),
                    min(12, int(expected_players or score_sel.POSITION_SCAN_MIN_PLAYERS)),
                )
                stage1_required = max(1, min(6, int(required_players)))
                signature_rows = score_sel._select_total_signature_rows(
                    required_players,
                    max_rows=int(score_sel.TOTAL_SCORE_SIGNATURE_MAX_ROWS),
                )
                signal_thresholds = score_sel._tie_aware_signal_thresholds(is_low_res_source=bool(is_low_res_source))
                row1_threshold = float(signal_thresholds.get("row1_coeff", 0.0))
                row_threshold = float(signal_thresholds.get("row_coeff", 0.0))
                row_signal = score_sel._tie_aware_score_signal_present(
                    image,
                    score_layout_id=score_layout_id,
                    min_players=required_players,
                    is_low_res_source=bool(is_low_res_source),
                )
                position_metrics = score_sel._position_metrics_for_frame(
                    image,
                    score_layout_id=score_layout_id,
                )
                detected_players_current = score_sel._estimate_visible_players_for_total_checks(
                    position_metrics,
                    is_low_res_source=bool(is_low_res_source),
                )
                stage1_pass = score_sel._tie_aware_score_signal_present_from_metrics(
                    position_metrics,
                    min_players=stage1_required,
                    is_low_res_source=bool(is_low_res_source),
                )
                if int(required_players) > int(stage1_required):
                    stage2_pass = score_sel._tie_aware_score_signal_present_from_metrics(
                        position_metrics,
                        min_players=required_players,
                        is_low_res_source=bool(is_low_res_source),
                    )
                else:
                    stage2_pass = True
                position_roi_rows = []
                prefix_pass = 0
                for row_number in range(1, 13):
                    metric = position_metrics[row_number - 1] if row_number - 1 < len(position_metrics) else {}
                    coeff = float(metric.get("best_position_score", 0.0) or 0.0)
                    best_template = int(metric.get("best_position_template", 0) or 0)
                    threshold = row1_threshold if int(row_number) == 1 else row_threshold
                    passed = bool(coeff >= float(threshold))
                    if row_number <= required_players and passed:
                        prefix_pass += 1
                    tile_size = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_SIZE)
                    x1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_X)
                    if str(score_layout_id or "").strip() == str(initial_scan.LAN1_SCORE_LAYOUT_ID):
                        x1 += int(initial_scan.SCORE_LAYOUT_SHIFT_X)
                    y1 = int(initial_scan.INITIAL_SCAN_GATE_TEMPLATE_GRID_Y) + ((int(row_number) - 1) * tile_size)
                    position_roi_rows.append(
                        {
                            "row": int(row_number),
                            "score": float(coeff),
                            "threshold": float(threshold),
                            "best_template": int(best_template),
                            "passed": bool(passed),
                            "bbox": (int(x1), int(y1), int(tile_size), int(tile_size)),
                        }
                    )
                points_observation = extract_points_transition_observation(
                    image,
                    score_layout_id=score_layout_id,
                )
                total_points = list(points_observation.get("total_points") or [])
                score_layout = get_score_layout(score_layout_id)
                total_roi_rows = []
                parsed_prefix_count = 0
                parsed_by_row = {}
                for idx, coord in enumerate(list(score_layout.total_points_coords)[:12], start=1):
                    (x1, y1), (x2, y2) = coord
                    raw_value = total_points[idx - 1] if idx - 1 < len(total_points) else ""
                    parsed_value = parse_detected_int(raw_value)
                    if idx <= required_players and parsed_value is not None:
                        parsed_prefix_count += 1
                    parsed_by_row[int(idx)] = parsed_value
                    total_roi_rows.append(
                        {
                            "row": int(idx),
                            "raw_value": str(raw_value),
                            "parsed_value": None if parsed_value is None else int(parsed_value),
                            "passed": bool(parsed_value is not None),
                            "bbox": (int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))),
                        }
                    )
                signature = score_sel._extract_total_score_stable_signature(
                    image,
                    score_layout_id=score_layout_id,
                    is_low_res_source=bool(is_low_res_source),
                    min_players_expected=expected_players,
                )
                signature_text = "" if signature is None else "|".join(str(v) for v in tuple(signature))
                signature_parse_pass_count = sum(
                    1 for row_number in signature_rows if parsed_by_row.get(int(row_number)) is not None
                )
                row.update(
                    {
                        "row_signal": bool(row_signal),
                        "signature": signature_text,
                        "stable_target_frame": self.summary.get("total_score_frame"),
                        "stable_run_len": 0,
                        "stable_run_start_frame": None,
                        "stable_run_signature": "",
                        "stable_required_low": int(score_sel.fps_scaled_frames(score_sel.LOW_RES_TOTAL_SCORE_STABLE_FRAMES_30FPS, float(self.context.get("fps", 30.0) if self.context else 30.0))),
                        "stable_required_high": int(score_sel.fps_scaled_frames(score_sel.TOTAL_SCORE_STABLE_FRAMES_30FPS, float(self.context.get("fps", 30.0) if self.context else 30.0))),
                        "stable_pass_low": False,
                        "stable_pass_high": False,
                        "stable_reset_reason": "",
                        "expected_total_players": int(expected_players) if expected_players is not None else None,
                        "max_detected_players": self.summary.get("max_detected_players"),
                        "raw_points_anchor_estimate": self.summary.get("raw_points_anchor_estimate"),
                        "cap_source_frame": self.summary.get("cap_source_frame"),
                        "cap_source_visible_rows": self.summary.get("cap_source_visible_rows"),
                        "cap_source_label": self.summary.get("cap_source_label"),
                        "cap_fallback_used": bool(self.summary.get("cap_fallback_used", False)),
                        "detected_players_current": int(detected_players_current),
                        "required_players_for_total": int(required_players),
                        "stage1_required_players": int(stage1_required),
                        "stage1_pass": bool(stage1_pass),
                        "stage2_pass": bool(stage2_pass),
                        "signature_rows": "|".join(str(v) for v in signature_rows),
                        "signature_parse_pass_count": int(signature_parse_pass_count),
                        "position_prefix_pass_count": int(prefix_pass),
                        "total_parse_pass_prefix_count": int(parsed_prefix_count),
                        "changed_total_rows_all": "",
                        "changed_total_rows_expected": "",
                        "position_roi_rows": position_roi_rows,
                        "total_roi_rows": total_roi_rows,
                        "inspection_reason": "Inspection-only frame (not production-checked); values computed live.",
                    }
                )
            else:
                row.update(
                    {
                        "inspection_reason": (
                            "Inspection-only frame (not production-checked). "
                            "Transition/Total checks are evaluated only on production-checked frames in this tool."
                        ),
                    }
                )

        if len(self._inspection_row_cache) >= 128:
            self._inspection_row_cache.pop(next(iter(self._inspection_row_cache)))
        self._inspection_row_cache[key] = dict(row)
        return row

    def _render_current_frame(self):
        if self.range_end_frame < self.range_start_frame:
            self.code_parity_var.set("")
            self.metrics_text.delete("1.0", tk.END)
            return
        self.current_frame = max(int(self.range_start_frame), min(int(self.current_frame), int(self.range_end_frame)))
        frame_number = int(self.current_frame)
        if (
            self._production_row_override is not None
            and int(self._production_row_override.get("frame", -1)) == int(frame_number)
        ):
            row = dict(self._production_row_override)
            display_mode = str(self._production_phase_override or self.current_mode.get())
        else:
            row = self.current_row_map.get(frame_number, {"frame": frame_number, "checked_by_code": False})
            display_mode = self.current_mode.get()
        checked_by_code = bool(row.get("checked_by_code", False))

        self._ensure_capture()
        if self.capture is None or not self.capture.isOpened():
            return
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.capture.read()
        if ret and frame is not None:
            image = crop_and_upscale_image(
                frame,
                self.context["median_left"],
                self.context["median_top"],
                self.context["median_crop_width"],
                self.context["median_crop_height"],
                1280,
                720,
            )
            if not bool(row.get("checked_by_code", False)):
                row = self._evaluate_frame_for_overlay(image, frame_number, display_mode)
            self._draw_roi_overlays(image, row)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            canvas_w = max(1, int(self.video_canvas.winfo_width()))
            canvas_h = max(1, int(self.video_canvas.winfo_height()))
            scale = min(float(canvas_w) / float(pil.width), float(canvas_h) / float(pil.height))
            scale = max(0.05, min(2.0, scale))
            resized_w = max(1, int(round(pil.width * scale)))
            resized_h = max(1, int(round(pil.height * scale)))
            pil = pil.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(pil)
            self.video_canvas.configure(image=self.photo)

        checked_any_phase = frame_number in self.production_checked_frames_all
        checked_phases = sorted(self.production_checked_phase_map.get(frame_number, set()))
        if checked_any_phase:
            phase_text = ", ".join(checked_phases) if checked_phases else "unknown phase"
            self.code_parity_var.set(f"Frame {frame_number} is checked by production code ({phase_text}).")
            self.code_parity_label.configure(foreground="green")
        else:
            self.code_parity_var.set(f"Frame {frame_number} is NOT checked by production code.")
            self.code_parity_label.configure(foreground="red")

        self.metrics_text.delete("1.0", tk.END)
        total_span = int(self.range_end_frame - self.range_start_frame + 1)
        frame_offset = int(frame_number - self.range_start_frame + 1)
        lines = [
            f"Mode: {display_mode}",
            f"Frame {frame_offset}/{total_span}",
            f"Frame: {frame_number} ({frame_to_timecode(frame_number, float(self.context['fps']))})",
        ]
        lines.extend(self._friendly_metrics_lines(row, display_mode=display_mode))
        self.metrics_text.insert("1.0", "\n".join(lines))
        self._colorize_metric_bools()
        self.jump_frame_var.set(str(frame_number))

    def _step_frames(self, delta):
        if self.range_end_frame < self.range_start_frame:
            return
        self._cancel_play_jobs()
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Steps")
        self._production_row_override = None
        self._production_phase_override = None
        self.current_frame = int(self.current_frame) + int(delta)
        self._render_current_frame()

    def _step_seconds(self, seconds):
        if self.context is None:
            return
        fps = max(1.0, float(self.context["fps"]))
        delta = int(round(float(seconds) * fps))
        self._step_frames(delta)

    def _step_production(self, direction):
        entries = self._production_entries_for_active_mode()
        if not self.trace_ready or not entries:
            self.status_var.set(f"No production steps available for mode '{self.current_mode.get()}'.")
            return
        self._cancel_play_jobs()
        self.is_playing = False
        self.is_production_playing = False
        direction = 1 if int(direction) >= 0 else -1
        current = int(self.current_frame)
        target = None
        if direction > 0:
            for item in entries:
                if int(item.get("frame", -1)) > current:
                    target = item
                    break
            if target is None:
                target = entries[0]
        else:
            for item in reversed(entries):
                if int(item.get("frame", -1)) < current:
                    target = item
                    break
            if target is None:
                target = entries[-1]
        self.current_frame = int(target.get("frame", 0))
        self._production_row_override = dict(target.get("row") or {})
        self._production_phase_override = str(target.get("phase") or "")
        self._render_current_frame()

    def _toggle_play(self):
        if self.is_playing:
            self._cancel_play_jobs()
            self.is_playing = False
            return
        self._cancel_play_jobs()
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Steps")
        self._production_row_override = None
        self._production_phase_override = None
        self.is_playing = True
        self._play_loop()

    def _play_loop(self):
        if not self.is_playing:
            return
        if self.range_end_frame < self.range_start_frame:
            self.is_playing = False
            return
        self._render_current_frame()
        if int(self.current_frame) >= int(self.range_end_frame):
            self.is_playing = False
            return
        self.current_frame += 1
        pause_seconds = _parse_pause_seconds(self.pause_seconds_var.get())
        delay_ms = max(1, int(round(pause_seconds * 1000.0)))
        self._play_after_id = self.after(delay_ms, self._play_loop)

    def _toggle_play_production(self):
        if not self.trace_ready:
            self.status_var.set("Trace is still loading. Wait for 'Detail trace loaded'.")
            return
        if self.range_end_frame < self.range_start_frame:
            return
        entries = self._production_entries_for_active_mode()
        if not entries:
            self.status_var.set(f"No production-checked frames for mode '{self.current_mode.get()}'.")
            return
        if self.is_production_playing:
            self._cancel_play_jobs()
            self.is_production_playing = False
            self.production_play_btn_text.set("Play Production Steps")
            return
        self._cancel_play_jobs()
        self.is_playing = False
        self.is_production_playing = True
        self.production_play_btn_text.set("Pause Production Steps")
        self.production_frames = list(entries)
        try:
            next_idx = next(
                idx
                for idx, entry in enumerate(self.production_frames)
                if int(entry.get("frame", -1)) >= int(self.current_frame)
            )
        except StopIteration:
            next_idx = 0
        self.production_index = int(next_idx)
        self._play_production_loop()

    def _play_production_loop(self):
        if not self.is_production_playing:
            return
        if not self.production_frames or self.production_index >= len(self.production_frames):
            self.is_production_playing = False
            self.production_play_btn_text.set("Play Production Steps")
            self._production_row_override = None
            self._production_phase_override = None
            return
        entry = self.production_frames[self.production_index]
        self.current_frame = int(entry.get("frame", 0))
        self._production_row_override = dict(entry.get("row") or {})
        self._production_phase_override = str(entry.get("phase") or "")
        self._render_current_frame()
        self.production_index += 1
        if self.production_index >= len(self.production_frames):
            self.is_production_playing = False
            self.production_play_btn_text.set("Play Production Steps")
            self._production_row_override = None
            self._production_phase_override = None
            return
        pause_seconds = _parse_pause_seconds(self.pause_seconds_var.get())
        delay_ms = max(1, int(round(pause_seconds * 1000.0)))
        self._prod_play_after_id = self.after(delay_ms, self._play_production_loop)

    def _jump_to_frame_from_entry(self):
        if self.range_end_frame < self.range_start_frame:
            return
        try:
            target = int(str(self.jump_frame_var.get()).strip())
        except ValueError:
            return
        self._cancel_play_jobs()
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Steps")
        self._production_row_override = None
        self._production_phase_override = None
        self.current_frame = int(target)
        self._render_current_frame()

    def _cancel_play_jobs(self):
        if self._play_after_id is not None:
            try:
                self.after_cancel(self._play_after_id)
            except Exception:
                pass
            self._play_after_id = None
        if self._prod_play_after_id is not None:
            try:
                self.after_cancel(self._prod_play_after_id)
            except Exception:
                pass
            self._prod_play_after_id = None

    def _set_summary_lines(self, lines):
        self.summary_text.configure(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        for tag in self.summary_text.tag_names():
            if str(tag).startswith("jump_"):
                self.summary_text.tag_delete(tag)
        for idx, (label, display_value, jump_frame) in enumerate(lines):
            self.summary_text.insert(tk.END, f"{label}: ")
            if isinstance(jump_frame, list):
                frames = [int(v) for v in jump_frame]
                for item_index, frame_value in enumerate(frames):
                    if item_index > 0:
                        self.summary_text.insert(tk.END, ", ")
                    tag = f"jump_{idx}_{item_index}"
                    start = self.summary_text.index(tk.END)
                    frame_text = str(frame_value)
                    self.summary_text.insert(tk.END, frame_text, (tag,))
                    self.summary_text.tag_config(tag, foreground="blue", underline=1)
                    self.summary_text.tag_bind(
                        tag,
                        "<Button-1>",
                        lambda _e, frame=int(frame_value): self._jump_to_summary_frame(frame),
                    )
                if not frames:
                    self.summary_text.insert(tk.END, str(display_value))
                self.summary_text.insert(tk.END, "\n")
                continue
            value_text = str(display_value)
            if jump_frame is not None and value_text not in {"None", ""}:
                tag = f"jump_{idx}"
                self.summary_text.insert(tk.END, value_text, (tag,))
                self.summary_text.tag_config(tag, foreground="blue", underline=1)
                self.summary_text.tag_bind(tag, "<Button-1>", lambda _e, frame=int(jump_frame): self._jump_to_summary_frame(frame))
            else:
                self.summary_text.insert(tk.END, value_text)
            self.summary_text.insert(tk.END, "\n")
        self.summary_text.configure(state=tk.DISABLED)

    def _jump_to_summary_frame(self, frame_number):
        self._cancel_play_jobs()
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Steps")
        self._production_row_override = None
        self._production_phase_override = None
        self.current_frame = int(frame_number)
        self._render_current_frame()

    def _friendly_metrics_lines(self, row, *, display_mode=None):
        lines = []
        checked = bool(row.get("checked_by_code", False))
        mode = str(display_mode or self.current_mode.get())
        if not checked:
            lines.append("")
            lines.append("Inspection Note")
            lines.append(f"- {row.get('inspection_reason') or 'This frame is not part of the production step sequence.'}")
        lines.append("")
        lines.append("Goal")
        if mode == "RaceScore Detail":
            lines.append("- Find a stable RaceScore screen with enough position rows to lock score hit.")
        elif mode == "Transition Scan":
            lines.append("- Find the points animation transition (totals changing across rows).")
        else:
            lines.append("- Find a stable TotalScore frame after transition.")
        lines.append("")
        lines.append("Production Step")
        lines.append(f"- Production stepping scope: mode-only ({mode})")
        lines.append(f"- Production checked this frame: {'TRUE' if checked else 'FALSE'}")
        if mode == "RaceScore Detail":
            rejected = bool(row.get("rejected_as_blank", False))
            max_val = float(row.get("max_val", 0.0) or 0.0)
            score_trigger = bool(max_val > 0.30 and not rejected)
            gate_rows = row.get("gate_rows") or []
            passing_rows = sum(1 for gate in gate_rows if bool(gate.get("passed", False)))
            required_rows = int(score_sel.POSITION_SCAN_MIN_PLAYERS)
            score_margin = max_val - 0.30
            rows_pass = passing_rows >= required_rows
            lines.append(f"- Score-screen signal strong enough: {'TRUE' if score_trigger else 'FALSE'} (threshold > 0.30)")
            lines.append(f"- Trigger progress (score coeff): {max_val:.4f}/0.3000 => {'TRUE' if max_val > 0.30 else 'FALSE'} ({score_margin:+.4f})")
            lines.append(f"- Trigger progress (rows passing): {passing_rows}/{required_rows} => {'TRUE' if rows_pass else 'FALSE'}")
            lines.append(f"- Rejected as blank: {'TRUE' if rejected else 'FALSE'}")
            lines.append(f"- Ignore overlay label: {row.get('ignore_label') or 'None'}")
            lines.append(f"- Ignore match score: {float(row.get('ignore_max', 0.0) or 0.0):.4f}")
            lines.append(f"- Ignore threshold: {float(row.get('ignore_threshold', 0.0) or 0.0):.2f}")
            lines.append(f"- Raw prefix mode active: {'TRUE' if bool(row.get('raw_confirm_mode', False)) else 'FALSE'}")
            lines.append(f"- Detected layout: {row.get('detected_layout_id') or 'unknown'}")
            lines.append(f"- Score coefficient: {max_val:.4f}")
            lines.append(f"- Position rows passing gate: {passing_rows}/{required_rows}")
            lines.append(f"- Score hit frame: {row.get('score_hit_frame')}")
            lines.append(f"- Race anchor frame: {row.get('race_score_frame_locked')}")
            lines.append(f"- Transition frame: {row.get('transition_frame')}")
            lines.append(f"- Points anchor frame: {row.get('points_anchor_frame')}")
            if gate_rows:
                lines.append("")
                lines.append("ROI Checks (Position Tiles)")
                for gate in gate_rows:
                    lines.append(
                        "- Row {row}: score={score:.4f} pass={passed} bbox={bbox}".format(
                            row=int(gate.get("row", 0)),
                            score=float(gate.get("score", 0.0)),
                            passed=("TRUE" if bool(gate.get("passed", False)) else "FALSE"),
                            bbox=gate.get("bbox"),
                        )
                    )
                lines.append("- Trigger logic: if score signal is TRUE, code locks Score Hit and starts Transition scan.")
        elif mode == "Transition Scan":
            changed_race = int(row.get("changed_race_rows", 0) or 0)
            changed_total = int(row.get("changed_total_rows", 0) or 0)
            changed_any = int(row.get("changed_any_rows", 0) or 0)
            cond_totals = changed_total >= 2
            cond_race_any = (changed_race >= 1) or (changed_any >= 3)
            lines.append("- Step 1: check whether row changes look like animation start.")
            lines.append(f"- Trigger progress A (total changed rows >=2): {changed_total}/2 => {'TRUE' if cond_totals else 'FALSE'}")
            lines.append(f"- Trigger progress B (race>=1 OR any>=3): race={changed_race}, any={changed_any} => {'TRUE' if cond_race_any else 'FALSE'}")
            lines.append(f"- Changed race-point rows: {changed_race}")
            lines.append(f"- Changed total-point rows: {changed_total}")
            lines.append(f"- Changed rows (any): {changed_any}")
            lines.append("- Step 2: raw trigger result on this frame.")
            lines.append(f"- Raw trigger hit this frame: {'TRUE' if bool(row.get('raw_trigger', False)) else 'FALSE'}")
            lines.append(f"- Current frame contributes to streak: {'TRUE' if bool(row.get('current_frame_contributes_to_streak', False)) else 'FALSE'}")
            lines.append("- Step 3: production debounce/confirm state.")
            lines.append(
                "- Debounce progress: true_count={tc}, false_streak={fs}, start_frame={sf} (needs >= {need_true} true; max false gap {max_false})".format(
                    tc=int(row.get("pending_true_count", 0) or 0),
                    fs=int(row.get("pending_false_streak", 0) or 0),
                    sf=row.get("pending_start_frame"),
                    need_true=int(score_sel._points_transition_confirm_true_count_for_fps(float(self.context.get("fps", 30.0) if self.context else 30.0))),
                    max_false=int(score_sel._points_transition_max_false_gap_for_fps(float(self.context.get("fps", 30.0) if self.context else 30.0))),
                )
            )
            keep_state = "TRUE" if bool(row.get("fallback_keep_alive", False)) else "FALSE"
            lines.append(
                "- Fallback keep-alive (after max false gap): {keep_state} | stable race rows={sr}/6 | stable total rows={st}/6 | changed-from-base race rows={cb}/6 | max false gap fallback={fg}".format(
                    keep_state=keep_state,
                    sr=int(row.get("fallback_stable_race_rows", 0) or 0),
                    st=int(row.get("fallback_stable_total_rows", 0) or 0),
                    cb=int(row.get("fallback_changed_from_base_rows", 0) or 0),
                    fg=int(row.get("fallback_max_false_gap_frames", 0) or 0),
                )
            )
            lines.append(f"- Transition rule triggered: {'TRUE' if bool(row.get('triggered', False)) else 'FALSE'}")
            lines.append(f"- Race points top-6: {row.get('race_points_top6', '')}")
            lines.append(f"- Total points top-6: {row.get('total_points_top6', '')}")
            lines.append(f"- Raw trigger history (latest 16): {row.get('raw_trigger_history', '')}")
            lines.append("- Trigger logic: totals changed on >=2 rows and (race changed >=1 row OR any changed >=3 rows).")
            lines.append("")
            lines.append("Pattern Diagnostics (experimental)")
            lines.append(
                f"- Pattern false-gap tolerance: <= {int(row.get('pattern_max_false_gap', 2) or 2)} consecutive FALSE frames"
            )
            for pattern_len in (5,):
                p_true = int(row.get(f"p{pattern_len}_true_count", 0) or 0)
                p_false = int(row.get(f"p{pattern_len}_false_streak", 0) or 0)
                p_ok = bool(row.get(f"p{pattern_len}_confirmed", False))
                p_alive = bool(row.get(f"p{pattern_len}_streak_alive", False))
                p_start = row.get(f"p{pattern_len}_start_frame")
                p_confirm = row.get(f"p{pattern_len}_confirm_frame")
                p_anchor = row.get(f"p{pattern_len}_anchor_frame")
                p_true_frames = str(row.get(f"p{pattern_len}_true_frames") or "").strip()
                lines.append(
                    f"- p{pattern_len}: confirmed={'TRUE' if p_ok else 'FALSE'} | alive={'TRUE' if p_alive else 'FALSE'} | "
                    f"true_count={p_true}/{pattern_len} | false_streak={p_false}"
                )
                lines.append(f"- p{pattern_len} streak anchor candidate: {p_start}")
                lines.append(f"- p{pattern_len} true-hit frames in current streak: {p_true_frames or 'None'}")
                lines.append(f"- p{pattern_len} confirm frame: {p_confirm} | recommended anchor: {p_anchor}")
        else:
            row_signal = bool(row.get("row_signal", False))
            has_signature = bool(str(row.get("signature") or "").strip())
            total_candidate = row_signal and has_signature
            expected_players = row.get("expected_total_players")
            max_detected_players = row.get("max_detected_players")
            detected_players_current = int(row.get("detected_players_current", 0) or 0)
            required_players = int(row.get("required_players_for_total", score_sel.POSITION_SCAN_MIN_PLAYERS) or score_sel.POSITION_SCAN_MIN_PLAYERS)
            stage1_required = int(row.get("stage1_required_players", min(6, required_players)) or min(6, required_players))
            stage1_pass = bool(row.get("stage1_pass", False))
            stage2_pass = bool(row.get("stage2_pass", False))
            signature_rows = str(row.get("signature_rows") or "").strip()
            signature_pass = int(row.get("signature_parse_pass_count", 0) or 0)
            prefix_pass = int(row.get("position_prefix_pass_count", 0) or 0)
            parsed_prefix = int(row.get("total_parse_pass_prefix_count", 0) or 0)
            changed_all = str(row.get("changed_total_rows_all") or "").strip()
            changed_expected = str(row.get("changed_total_rows_expected") or "").strip()
            lines.append(f"- Row signal present: {'TRUE' if row_signal else 'FALSE'}")
            lines.append(f"- Stable total signature: {row.get('signature') or 'None'}")
            lines.append(f"- Trigger progress (row signal AND signature): {'TRUE' if total_candidate else 'FALSE'}")
            lines.append(f"- Expected players from points-anchor frame: {expected_players}")
            lines.append(f"- Max detected players (anchor): {max_detected_players}")
            lines.append(f"- Raw points-anchor estimate: {row.get('raw_points_anchor_estimate')}")
            lines.append(f"- Cap source: {row.get('cap_source_label') or 'None'}")
            lines.append(f"- Cap source frame: {row.get('cap_source_frame')}")
            lines.append(f"- Cap source visible rows: {row.get('cap_source_visible_rows')}")
            lines.append(f"- Cap fallback used: {'TRUE' if bool(row.get('cap_fallback_used', False)) else 'FALSE'}")
            lines.append(f"- Detected players on this frame: {detected_players_current}")
            lines.append(f"- Required rows for TotalScore checks: {required_players}")
            lines.append(f"- Stage 1 gate (rows 1..{stage1_required}): {'TRUE' if stage1_pass else 'FALSE'}")
            if required_players > stage1_required:
                lines.append(f"- Stage 2 gate (rows {stage1_required + 1}..{required_players}): {'TRUE' if stage2_pass else 'FALSE'}")
            else:
                lines.append("- Stage 2 gate: not required (required rows <= 6)")
            lines.append(f"- Signature rows used (max {int(score_sel.TOTAL_SCORE_SIGNATURE_MAX_ROWS)}): {signature_rows or 'None'}")
            total_sig_rows = len([p for p in signature_rows.split('|') if p.strip()]) if signature_rows else 0
            lines.append(f"- Signature parse progress: {signature_pass}/{total_sig_rows if total_sig_rows else 0} => {'TRUE' if total_sig_rows and signature_pass >= total_sig_rows else 'FALSE'}")
            lines.append(f"- Position prefix pass progress: {prefix_pass}/{required_players} => {'TRUE' if prefix_pass >= required_players else 'FALSE'}")
            lines.append(f"- Parsed total digits progress: {parsed_prefix}/{required_players} => {'TRUE' if parsed_prefix >= required_players else 'FALSE'}")
            lines.append(f"- Changed total rows (all 1..12) vs prev frame: {changed_all or 'None'}")
            lines.append(f"- Changed total rows (within required prefix): {changed_expected or 'None'}")
            run_len = int(row.get("stable_run_len", 0) or 0)
            req_low = int(row.get("stable_required_low", 8) or 8)
            req_high = int(row.get("stable_required_high", 20) or 20)
            pass_low = bool(row.get("stable_pass_low", False))
            pass_high = bool(row.get("stable_pass_high", False))
            lines.append(f"- Stable run progress for >=8 path: {run_len}/{req_low} => {'TRUE' if pass_low else 'FALSE'}")
            lines.append(f"- Stable run progress for >=20 path: {run_len}/{req_high} => {'TRUE' if pass_high else 'FALSE'}")
            lines.append("- Note: these are consecutive-frame counters, not matching coefficients.")
            lines.append(f"- Current stable run signature: {row.get('stable_run_signature') or 'None'}")
            lines.append(f"- Stable run started at frame: {row.get('stable_run_start_frame')}")
            reset_reason = str(row.get("stable_reset_reason") or "").strip()
            if reset_reason:
                lines.append(f"- Run reset reason on this frame: {reset_reason}")
            lines.append(f"- Stable target frame chosen by code: {row.get('stable_target_frame')}")
            if self.summary.get("total_score_frame") is None:
                lines.append("- If this stays FALSE across checked frames, Total Anchor Frame remains None.")
            lines.append("- Trigger logic: stable frame signature and row signal pass; then Total anchor is locked.")
            pos_rows = list(row.get("position_roi_rows") or [])
            if pos_rows:
                lines.append("")
                lines.append("ROI Checks (Position Gate Rows - Required)")
                required_pos_rows = [
                    item for item in pos_rows if int(item.get("row", 0) or 0) <= int(required_players)
                ]
                diagnostic_pos_rows = [
                    item for item in pos_rows if int(item.get("row", 0) or 0) > int(required_players)
                ]
                for item in required_pos_rows:
                    lines.append(
                        "- P{row}: score={score:.4f} threshold={thr:.4f} best_template={tpl} pass={passed} bbox={bbox}".format(
                            row=int(item.get("row", 0)),
                            score=float(item.get("score", 0.0) or 0.0),
                            thr=float(item.get("threshold", 0.0) or 0.0),
                            tpl=int(item.get("best_template", 0) or 0),
                            passed=("TRUE" if bool(item.get("passed", False)) else "FALSE"),
                            bbox=item.get("bbox"),
                        )
                    )
                if diagnostic_pos_rows:
                    lines.append("ROI Checks (Position Gate Rows - Diagnostic Only)")
                    for item in diagnostic_pos_rows:
                        lines.append(
                            "- P{row}: score={score:.4f} threshold={thr:.4f} best_template={tpl} pass={passed} bbox={bbox}".format(
                                row=int(item.get("row", 0)),
                                score=float(item.get("score", 0.0) or 0.0),
                                thr=float(item.get("threshold", 0.0) or 0.0),
                                tpl=int(item.get("best_template", 0) or 0),
                                passed=("TRUE" if bool(item.get("passed", False)) else "FALSE"),
                                bbox=item.get("bbox"),
                            )
                        )
            total_rows = list(row.get("total_roi_rows") or [])
            if total_rows:
                lines.append("")
                lines.append("ROI Checks (Total Points Digits - Required)")
                required_total_rows = [
                    item for item in total_rows if int(item.get("row", 0) or 0) <= int(required_players)
                ]
                diagnostic_total_rows = [
                    item for item in total_rows if int(item.get("row", 0) or 0) > int(required_players)
                ]
                for item in required_total_rows:
                    parsed = item.get("parsed_value")
                    lines.append(
                        "- T{row}: raw='{raw}' parsed={parsed} pass={passed} bbox={bbox}".format(
                            row=int(item.get("row", 0)),
                            raw=str(item.get("raw_value", "")),
                            parsed=("None" if parsed is None else int(parsed)),
                            passed=("TRUE" if bool(item.get("passed", False)) else "FALSE"),
                            bbox=item.get("bbox"),
                        )
                    )
                if diagnostic_total_rows:
                    lines.append("ROI Checks (Total Points Digits - Diagnostic Only)")
                    for item in diagnostic_total_rows:
                        parsed = item.get("parsed_value")
                        lines.append(
                            "- T{row}: raw='{raw}' parsed={parsed} pass={passed} bbox={bbox}".format(
                                row=int(item.get("row", 0)),
                                raw=str(item.get("raw_value", "")),
                                parsed=("None" if parsed is None else int(parsed)),
                                passed=("TRUE" if bool(item.get("passed", False)) else "FALSE"),
                                bbox=item.get("bbox"),
                            )
                        )

        lines.append("")
        lines.append("Technical Details")
        for key in sorted(row.keys()):
            if key in {"frame", "gate_rows"}:
                continue
            lines.append(f"- {key}: {row.get(key)}")
        return lines

    def _reset_playback_state(self):
        self._cancel_play_jobs()
        self.is_playing = False
        self.is_production_playing = False
        self.production_frames = []
        self.production_index = 0
        self.production_play_btn_text.set("Play Production Steps")
        self._production_row_override = None
        self._production_phase_override = None

    def _colorize_metric_bools(self):
        self.metrics_text.tag_remove("bool_true", "1.0", tk.END)
        self.metrics_text.tag_remove("bool_false", "1.0", tk.END)
        start = "1.0"
        while True:
            idx = self.metrics_text.search("TRUE", start, stopindex=tk.END)
            if not idx:
                break
            end = f"{idx}+4c"
            self.metrics_text.tag_add("bool_true", idx, end)
            start = end
        start = "1.0"
        while True:
            idx = self.metrics_text.search("FALSE", start, stopindex=tk.END)
            if not idx:
                break
            end = f"{idx}+5c"
            self.metrics_text.tag_add("bool_false", idx, end)
            start = end

    def _draw_roi_overlays(self, image, row):
        mode = str(self._production_phase_override or self.current_mode.get())
        if mode == "RaceScore Detail":
            gate_rows = row.get("gate_rows") or []
            for gate in gate_rows:
                bbox = gate.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = [int(v) for v in bbox]
                passed = bool(gate.get("passed", False))
                color = (0, 220, 0) if passed else (0, 140, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                cv2.putText(
                    image,
                    f"R{int(gate.get('row', 0))}:{float(gate.get('score', 0.0)):.2f}",
                    (x + 2, max(12, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            ignore_label = str(row.get("ignore_label") or "").strip()
            if ignore_label:
                for target in initial_scan.IGNORE_FRAME_TARGETS:
                    if str(target.get("label", "")) != ignore_label:
                        continue
                    x, y, w, h = [int(v) for v in target.get("roi", (0, 0, 0, 0))]
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        image,
                        f"{ignore_label}:{float(row.get('ignore_max', 0.0) or 0.0):.2f}",
                        (x + 2, max(12, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    break
            return

        if mode == "TotalScore Scan":
            pos_rows = list(row.get("position_roi_rows") or [])
            required_players = int(
                row.get("required_players_for_total", score_sel.POSITION_SCAN_MIN_PLAYERS)
                or score_sel.POSITION_SCAN_MIN_PLAYERS
            )
            for item in pos_rows:
                bbox = item.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = [int(v) for v in bbox]
                passed = bool(item.get("passed", False))
                row_number = int(item.get("row", 0) or 0)
                is_required = row_number <= required_players
                if is_required:
                    color = (0, 220, 0) if passed else (0, 0, 255)
                else:
                    color = (0, 200, 255) if passed else (120, 120, 120)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                cv2.putText(
                    image,
                    f"P{row_number}:{float(item.get('score', 0.0) or 0.0):.2f}",
                    (x + 2, max(12, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            total_rows = list(row.get("total_roi_rows") or [])
            for item in total_rows:
                bbox = item.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = [int(v) for v in bbox]
                passed = bool(item.get("passed", False))
                row_number = int(item.get("row", 0) or 0)
                is_required = row_number <= required_players
                if is_required:
                    color = (0, 220, 0) if passed else (0, 0, 255)
                else:
                    color = (0, 200, 255) if passed else (120, 120, 120)
                parsed = item.get("parsed_value")
                label_value = "?" if parsed is None else str(int(parsed))
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                cv2.putText(
                    image,
                    f"T{row_number}:{label_value}",
                    (x + 2, max(12, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def destroy(self):
        self._cancel_play_jobs()
        self.is_playing = False
        self.is_production_playing = False
        self.production_play_btn_text.set("Play Production Steps")
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        super().destroy()


def main():
    app = ScoreDetailDebugGui()
    app.mainloop()


if __name__ == "__main__":
    main()
