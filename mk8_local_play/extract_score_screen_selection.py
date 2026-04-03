import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .extract_common import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    crop_and_upscale_image,
    match_template,
    preprocess_roi,
    score_bundle_anchor_path,
    score_bundle_consensus_path,
    score_bundle_dir,
    score_bundle_race_context_path,
    score_bundle_points_anchor_path,
    score_bundle_points_context_path,
    write_export_image,
)
from .extract_initial_scan import (
    IGNORE_FRAME_TARGETS,
    POSITION_SCAN_MIN_AVG_COEFF,
    POSITION_SCAN_MIN_PLAYERS,
    POSITION_SCAN_MIN_ROW_COEFF,
    _best_initial_scan_gate_score,
    _initial_scan_gate_tile_roi,
    _match_score_target_layouts,
)
from .extract_video_io import (
    actual_frame_after_read,
    add_timing,
    increment_counter,
    log_exported_frame,
    position_capture_for_read,
    read_video_frame,
    seek_to_frame,
)
from .ocr_scoreboard_consensus import (
    POSITION_PRESENT_COEFF_THRESHOLD,
    POSITION_PRESENT_ROW1_COEFF_THRESHOLD,
    _template_match_score,
    build_position_signal_metrics,
    extract_points_transition_observation,
    parse_detected_int,
    process_image,
    stack_position_rows,
)
from .project_paths import PROJECT_ROOT
from .score_layouts import draw_score_layout_demo, get_score_layout, score_demo_output_path
from .app_runtime import load_app_config


def enhance_export_frame(upscaled_image, scale_x, scale_y):
    if scale_x > 1.3 and scale_y >= 1.3:
        from PIL import Image, ImageEnhance

        if isinstance(upscaled_image, np.ndarray):
            upscaled_image = Image.fromarray(upscaled_image)
        contrast_enhancer = ImageEnhance.Contrast(upscaled_image)
        high_contrast_image = contrast_enhancer.enhance(1.70)
        sharpness_enhancer = ImageEnhance.Sharpness(high_contrast_image)
        sharpened_image = sharpness_enhancer.enhance(1.23)
        return np.array(sharpened_image)
    return upscaled_image


def capture_export_frame(capture, target_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats, *, label="export_frame"):
    position_capture_for_read(
        capture,
        target_frame,
        stats,
        max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
        label=label,
    )
    ret, frame = read_video_frame(capture, stats)
    if not ret:
        return None, None
    actual_frame = actual_frame_after_read(capture)
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
    return actual_frame, enhance_export_frame(upscaled_image, scale_x, scale_y)


def fps_scaled_frames(base_frames_30fps, fps):
    return max(1, int(round(float(base_frames_30fps) * (max(float(fps), 1.0) / 30.0))))


RACE_SCORE_EXTRA_DELAY_FRAMES_30FPS = 1
RACE_SCORE_EARLY_EXPANSION_FRAMES = 7
RACE_SCORE_LATE_EXPANSION_FRAMES = 6
TOTAL_SCORE_TRANSITION_DROP_SECONDS = 5.0
STATIC_GALLERY_RACE_MIN_FIRST_FRAME_COEFF = 0.995
STATIC_GALLERY_RACE_AVG_FIRST_FRAME_COEFF = 0.997
TWELFTH_TEMPLATE_INDEX = 2
TWELFTH_TEMPLATE_NL_INDEX = 7
TWELFTH_NL_CHECK_ROI = (306, 658, 670, 41)
POSITION_PREFIX_GATE_MIN_COEFF = 0.20
POINTS_TRANSITION_SEARCH_END_SECONDS = 6.0
TOTAL_SCORE_FROM_TRANSITION_SECONDS = 3.6
TOTAL_SCORE_STABLE_SEARCH_SECONDS = 5.0
TOTAL_SCORE_STABLE_FRAMES_30FPS = 20
COARSE_SEARCH_STEP_FRAMES = 10
COARSE_SEARCH_REWIND_FRAMES = 10
SMALL_FORWARD_GRAB_WINDOW_FRAMES = 12
POINTS_TRANSITION_PRIMARY_OFFSET_FRAMES_30FPS = 23
POINTS_TRANSITION_PRIMARY_RADIUS_FRAMES_30FPS = 2
TOTAL_SCORE_EARLY_STABLE_START_FRAMES_30FPS = 35
TOTAL_SCORE_EARLY_STABLE_END_FRAMES_30FPS = 50
TOTAL_SCORE_LATE_STABLE_START_FRAMES_30FPS = 89
TOTAL_SCORE_LATE_STABLE_END_FRAMES_30FPS = 102
TOTAL_SCORE_EARLY_STABLE_PROBE_FRAMES_30FPS = 45
TOTAL_SCORE_LATE_STABLE_PROBE_FRAMES_30FPS = 95


APP_CONFIG = load_app_config()
# Experimental timing shortcuts stay opt-in until broad benchmark parity is proven.
TOTAL_SCORE_TIMING_FAST_PATH_ENABLED = os.environ.get("MK8_TOTAL_SCORE_TIMING_FAST_PATH", "0").strip().lower() not in {"0", "false", "no", "off"}
TOTAL_SCORE_TRANSITION_PRIMARY_ENABLED = os.environ.get("MK8_TOTAL_SCORE_TRANSITION_PRIMARY", "0").strip().lower() not in {"0", "false", "no", "off"}
TOTAL_SCORE_STABLE_HINT_ENABLED = os.environ.get("MK8_TOTAL_SCORE_STABLE_HINT", "0").strip().lower() not in {"0", "false", "no", "off"}


def _build_score_analysis_trace(
    task,
    *,
    start_frame,
    end_frame,
    score_hit_frame=None,
    transition_frame=None,
    stable_total_score_frame=None,
    selected_points_anchor_frame=None,
    total_score_used_fallback=False,
    ignored_candidate=False,
    ignore_label="",
):
    return {
        "race_number": int(task.get("race_number", 0) or 0),
        "score_layout_id": str(task.get("score_layout_id") or ""),
        "candidate_frame": int(task.get("frame_number", 0) or 0),
        "detail_start_frame": int(start_frame),
        "detail_end_frame": int(end_frame),
        "score_hit_frame": None if score_hit_frame is None else int(score_hit_frame),
        "transition_frame": None if transition_frame is None else int(transition_frame),
        "stable_total_score_frame": None if stable_total_score_frame is None else int(stable_total_score_frame),
        "selected_points_anchor_frame": None if selected_points_anchor_frame is None else int(selected_points_anchor_frame),
        "total_score_used_fallback": bool(total_score_used_fallback),
        "ignored_candidate": bool(ignored_candidate),
        "ignore_label": str(ignore_label or ""),
    }


def _match_ignore_frame_target_detail(gray_image, templates, stats):
    """Detect ignore/gallery overlays during second-pass score selection."""
    best_match = {
        "label": "",
        "max_val": 0.0,
        "match_threshold": 1.0,
        "rejected_as_blank": True,
    }
    for target in IGNORE_FRAME_TARGETS:
        template_index = int(target["template_index"])
        if len(templates) <= template_index:
            continue
        roi_x, roi_y, roi_width, roi_height = target["roi"]
        roi_x = max(int(roi_x), 0)
        roi_y = max(int(roi_y), 0)
        roi_width = min(int(roi_width), gray_image.shape[1] - roi_x)
        roi_height = min(int(roi_height), gray_image.shape[0] - roi_y)
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        template_binary, alpha_mask = templates[template_index]
        if roi.shape[0] < template_binary.shape[0] or roi.shape[1] < template_binary.shape[1]:
            roi = cv2.resize(
                roi,
                (max(template_binary.shape[1], roi.shape[1]), max(template_binary.shape[0], roi.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )
        stage_start = time.perf_counter()
        max_val = match_template(roi, template_binary, alpha_mask)
        add_timing(stats, "score_ignore_match_s", stage_start)
        if max_val > best_match["max_val"]:
            best_match = {
                "label": str(target["label"]),
                "max_val": float(max_val),
                "match_threshold": float(target["match_threshold"]),
                "rejected_as_blank": False,
            }
    return best_match


def _match_twelfth_presence(gray_image, templates, score_layout, stats):
    """Return the strongest 12th-place template hit across supported language variants."""
    candidates = [
        (score_layout.twelfth_place_check_roi, TWELFTH_TEMPLATE_INDEX),
        (TWELFTH_NL_CHECK_ROI, TWELFTH_TEMPLATE_NL_INDEX),
    ]
    best_match = 0.0
    for roi_definition, template_index in candidates:
        if template_index >= len(templates):
            continue
        roi_x, roi_y, roi_width, roi_height = roi_definition
        if roi_width <= 0 or roi_height <= 0:
            continue
        stage_start = time.perf_counter()
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        add_timing(stats, "score_detail_12th_roi_extract_s", stage_start)
        if roi.size == 0:
            continue
        stage_start = time.perf_counter()
        processed_roi = preprocess_roi(roi, 0)
        add_timing(stats, "score_detail_12th_preprocess_s", stage_start)
        template_binary, alpha_mask = templates[template_index]
        stage_start = time.perf_counter()
        max_val = match_template(processed_roi, template_binary, alpha_mask)
        add_timing(stats, "score_detail_match_12th_s", stage_start)
        if max_val > best_match:
            best_match = float(max_val)
    return best_match


def _position_metrics_for_frame(frame_image, score_layout_id=None, stats=None):
    processed_image = process_image(
        frame_image,
        score_layout_id=score_layout_id,
        stats=stats,
        stats_prefix="score_detail",
    )
    return build_position_signal_metrics(
        processed_image,
        score_layout_id=score_layout_id,
        stats=stats,
        stats_prefix="score_detail",
    )


def summarize_first_frame_similarity(consensus_frames):
    """Measure how static a saved score bundle is against the earliest frame."""
    if not consensus_frames or len(consensus_frames) < 2:
        return None

    grayscale_frames = []
    for _frame_number, frame_image in consensus_frames:
        if frame_image is None:
            continue
        if frame_image.ndim == 2:
            gray = frame_image
        else:
            gray = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
        grayscale_frames.append(gray)

    if len(grayscale_frames) < 2:
        return None

    first_frame = grayscale_frames[0]
    coefficients = []
    for current_frame in grayscale_frames[1:]:
        match = cv2.matchTemplate(first_frame, current_frame, cv2.TM_CCOEFF_NORMED)
        coefficients.append(float(match[0][0]))

    if not coefficients:
        return None

    return {
        "count": len(coefficients),
        "min": float(min(coefficients)),
        "avg": float(sum(coefficients) / len(coefficients)),
        "max": float(max(coefficients)),
    }


def is_static_gallery_race_bundle(consensus_frames):
    """Return True when a RaceScore bundle is effectively a static gallery image."""
    similarity = summarize_first_frame_similarity(consensus_frames)
    if similarity is None:
        return False, None

    is_static_gallery = (
        similarity["count"] >= 2
        and similarity["min"] >= STATIC_GALLERY_RACE_MIN_FIRST_FRAME_COEFF
        and similarity["avg"] >= STATIC_GALLERY_RACE_AVG_FIRST_FRAME_COEFF
    )
    return is_static_gallery, similarity


def _count_visible_rows_from_position_metrics(position_metrics):
    visible_rows = 0
    last_confirmed_rank = None
    for metric in position_metrics:
        row_number = int(metric.get('row_number', 0))
        best_position_score = float(metric.get('best_position_score', 0.0))
        best_rank = int(metric.get('best_position_template', 0))
        coeff_ranked_templates = [int(value) for value in metric.get('coeff_ranked_templates', [])]
        row_supported = best_position_score >= POSITION_PRESENT_COEFF_THRESHOLD
        if (
            not row_supported
            and row_number == 1
            and best_rank == 1
            and best_position_score >= POSITION_PRESENT_ROW1_COEFF_THRESHOLD
        ):
            row_supported = True
        if row_supported and last_confirmed_rank is not None and best_rank < last_confirmed_rank:
            fallback_rank = next((template_rank for template_rank in coeff_ranked_templates if template_rank >= last_confirmed_rank), 0)
            if fallback_rank > 0:
                best_rank = fallback_rank
            else:
                row_supported = False
        if not row_supported:
            break
        last_confirmed_rank = best_rank
        visible_rows = row_number
    return visible_rows


def _row_has_expected_template(position_metrics, row_number, min_score=0.4):
    if row_number <= 0 or row_number > len(position_metrics):
        return False
    metric = position_metrics[row_number - 1]
    return (
        int(metric.get('best_position_template', 0)) == int(row_number)
        and float(metric.get('best_position_score', 0.0)) >= float(min_score)
    )


def _count_tie_aware_prefix_rows(position_metrics, min_players):
    """Count rows whose rank badges remain plausible when totals are tied."""
    visible_rows = 0
    last_confirmed_rank = 1
    for row_number in range(1, min(int(min_players), len(position_metrics)) + 1):
        metric = position_metrics[row_number - 1]
        best_score = float(metric.get("best_position_score", 0.0))
        best_rank = int(metric.get("best_position_template", 0))
        coeff_ranked_templates = [int(value) for value in metric.get("coeff_ranked_templates", [])]
        threshold = POSITION_PRESENT_ROW1_COEFF_THRESHOLD if row_number == 1 else POSITION_PRESENT_COEFF_THRESHOLD

        chosen_rank = 0
        if best_score >= threshold and last_confirmed_rank <= best_rank <= row_number:
            chosen_rank = best_rank
        else:
            chosen_rank = next(
                (
                    template_rank
                    for template_rank in coeff_ranked_templates
                    if last_confirmed_rank <= int(template_rank) <= row_number
                ),
                0,
            )
        if chosen_rank <= 0:
            break
        last_confirmed_rank = int(chosen_rank)
        visible_rows = row_number
    return visible_rows


def _fast_prefix_gate_score(frame_image, score_layout_id=None, min_players=None, stats=None):
    """Cheap pre-gate: reuse the raw fixed-grid 5/6 tile check from the initial scan."""
    gate_start = time.perf_counter()
    row_scores = {}
    for row_number in (5, 6):
        tile_roi = _initial_scan_gate_tile_roi(frame_image, row_number, score_layout_id=score_layout_id)
        if tile_roi.size == 0:
            row_scores[int(row_number)] = 0.0
            continue
        tile_gray = cv2.cvtColor(tile_roi, cv2.COLOR_BGR2GRAY)
        coeff, _variant_name = _best_initial_scan_gate_score(tile_gray, row_number)
        row_scores[int(row_number)] = float(coeff)
        if stats is not None:
            stats["score_prefix_gate_template_checks"] += 1
    result = all(float(row_scores.get(row_number, 0.0)) >= float(POSITION_PREFIX_GATE_MIN_COEFF) for row_number in (5, 6))
    if stats is not None:
        stats["score_prefix_gate_calls"] += 1
        stats["score_prefix_gate_s"] += time.perf_counter() - gate_start
        if result:
            stats["score_prefix_gate_passes"] += 1
    return result


def _raw_fixed_grid_prefix_confirm(frame_image, required_players=None, score_layout_id=None, stats=None):
    """Confirm a score screen using the fixed-grid raw position tiles for rows 1..N."""
    confirm_start = time.perf_counter()
    prefix_count = max(2, int(required_players or POSITION_SCAN_MIN_PLAYERS))
    row_scores = []
    for row_number in range(1, prefix_count + 1):
        tile_roi = _initial_scan_gate_tile_roi(frame_image, row_number, score_layout_id=score_layout_id)
        if tile_roi.size == 0:
            row_scores.append(0.0)
            continue
        tile_gray = cv2.cvtColor(tile_roi, cv2.COLOR_BGR2GRAY)
        coeff, _variant_name = _best_initial_scan_gate_score(tile_gray, row_number)
        row_scores.append(float(coeff))
        if stats is not None:
            stats["score_raw_prefix_confirm_template_checks"] += 1
    passed = all(score >= float(POSITION_PREFIX_GATE_MIN_COEFF) for score in row_scores)
    average_score = float(sum(row_scores) / len(row_scores)) if row_scores else 0.0
    if stats is not None:
        stats["score_raw_prefix_confirm_calls"] += 1
        stats["score_raw_prefix_confirm_s"] += time.perf_counter() - confirm_start
        if passed:
            stats["score_raw_prefix_confirm_passes"] += 1
    return passed, average_score


def _tie_aware_score_signal_present_from_metrics(position_metrics, min_players=None):
    required_players = max(2, int(min_players or POSITION_SCAN_MIN_PLAYERS))
    if _count_tie_aware_prefix_rows(position_metrics, required_players) < required_players:
        return False
    prefix_scores = [
        float(position_metrics[row_number - 1].get("best_position_score", 0.0))
        for row_number in range(1, required_players + 1)
    ]
    average_score = float(sum(prefix_scores) / len(prefix_scores)) if prefix_scores else 0.0
    return average_score >= float(POSITION_SCAN_MIN_AVG_COEFF)


def _tie_aware_score_signal_present(frame_image, score_layout_id=None, min_players=None, stats=None):
    position_metrics = _position_metrics_for_frame(frame_image, score_layout_id=score_layout_id, stats=stats)
    return _tie_aware_score_signal_present_from_metrics(position_metrics, min_players=min_players)


def _extract_total_score_stable_signature(frame_image, score_layout_id=None):
    observation = extract_points_transition_observation(
        frame_image,
        score_layout_id=score_layout_id,
    )
    totals = observation.get("total_points") or []
    signature = []
    for row_index in range(3):
        if row_index >= len(totals):
            return None
        parsed_value = parse_detected_int(totals[row_index])
        if parsed_value is None:
            return None
        signature.append(int(parsed_value))
    if not (signature[0] >= signature[1] >= signature[2]):
        return None
    return tuple(signature)


def _find_points_transition_frame_in_range(
    local_cap,
    start_frame,
    end_frame,
    left,
    top,
    crop_width,
    crop_height,
    score_layout_id,
    stats,
    *,
    seek_label,
):
    previous_observation = None
    seek_to_frame(local_cap, start_frame, stats, label=seek_label)
    for frame_number in range(int(start_frame), int(end_frame) + 1):
        ret, frame = read_video_frame(local_cap, stats)
        if not ret:
            break
        upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
        observation = extract_points_transition_observation(
            upscaled_image,
            score_layout_id=score_layout_id,
        )
        if previous_observation is not None:
            changed_total_rows = 0
            changed_race_rows = 0
            changed_any_rows = 0
            for row_index in range(6):
                previous_race = parse_detected_int(previous_observation["race_points"][row_index])
                current_race = parse_detected_int(observation["race_points"][row_index])
                previous_total = parse_detected_int(previous_observation["total_points"][row_index])
                current_total = parse_detected_int(observation["total_points"][row_index])
                race_changed = previous_race is not None and current_race is not None and previous_race != current_race
                total_changed = previous_total is not None and current_total is not None and previous_total != current_total
                changed_race_rows += int(race_changed)
                changed_total_rows += int(total_changed)
                changed_any_rows += int(race_changed or total_changed)
            if changed_total_rows >= 2 and (changed_race_rows >= 1 or changed_any_rows >= 3):
                return int(frame_number), int(frame_number)
        previous_observation = observation
    return None, None


def _find_points_transition_frame(local_cap, start_frame, end_frame, left, top, crop_width, crop_height, score_layout_id, stats, fps=30.0):
    if not TOTAL_SCORE_TIMING_FAST_PATH_ENABLED or not TOTAL_SCORE_TRANSITION_PRIMARY_ENABLED:
        return _find_points_transition_frame_in_range(
            local_cap,
            start_frame,
            end_frame,
            left,
            top,
            crop_width,
            crop_height,
            score_layout_id,
            stats,
            seek_label="points_transition_start",
        )
    primary_offset = fps_scaled_frames(POINTS_TRANSITION_PRIMARY_OFFSET_FRAMES_30FPS, fps)
    primary_radius = fps_scaled_frames(POINTS_TRANSITION_PRIMARY_RADIUS_FRAMES_30FPS, fps)
    primary_start_frame = max(int(start_frame), int(start_frame) + primary_offset - primary_radius - 1)
    primary_end_frame = min(int(end_frame), int(start_frame) + primary_offset + primary_radius)
    if primary_start_frame < primary_end_frame:
        primary_transition_frame, primary_points_anchor_frame = _find_points_transition_frame_in_range(
            local_cap,
            primary_start_frame,
            primary_end_frame,
            left,
            top,
            crop_width,
            crop_height,
            score_layout_id,
            stats,
            seek_label="points_transition_primary",
        )
        if primary_transition_frame is not None:
            return primary_transition_frame, primary_points_anchor_frame
    return _find_points_transition_frame_in_range(
        local_cap,
        start_frame,
        end_frame,
        left,
        top,
        crop_width,
        crop_height,
        score_layout_id,
        stats,
        seek_label="points_transition_start",
    )


def _frame_has_total_score_signature(
    local_cap,
    frame_number,
    left,
    top,
    crop_width,
    crop_height,
    score_layout_id,
    stats,
    *,
    position_label,
):
    position_capture_for_read(
        local_cap,
        int(frame_number),
        stats,
        max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
        label=position_label,
    )
    ret, frame = read_video_frame(local_cap, stats)
    if not ret:
        return False
    upscaled_image = crop_and_upscale_image(
        frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT
    )
    return _extract_total_score_stable_signature(
        upscaled_image,
        score_layout_id=score_layout_id,
    ) is not None


def _find_total_score_stable_frame(local_cap, transition_frame, fps, left, top, crop_width, crop_height, score_layout_id, stats):
    stable_frames_required = max(1, fps_scaled_frames(TOTAL_SCORE_STABLE_FRAMES_30FPS, fps))
    search_end_frame = int(transition_frame) + max(1, int(round(TOTAL_SCORE_STABLE_SEARCH_SECONDS * max(float(fps), 1.0))))
    if TOTAL_SCORE_TIMING_FAST_PATH_ENABLED and TOTAL_SCORE_STABLE_HINT_ENABLED:
        early_probe_frame = int(transition_frame) + fps_scaled_frames(TOTAL_SCORE_EARLY_STABLE_PROBE_FRAMES_30FPS, fps)
        late_probe_frame = int(transition_frame) + fps_scaled_frames(TOTAL_SCORE_LATE_STABLE_PROBE_FRAMES_30FPS, fps)
        early_search_start_frame = int(transition_frame) + fps_scaled_frames(TOTAL_SCORE_EARLY_STABLE_START_FRAMES_30FPS, fps)
        late_search_start_frame = int(transition_frame) + fps_scaled_frames(TOTAL_SCORE_LATE_STABLE_START_FRAMES_30FPS, fps)
        if early_probe_frame <= int(search_end_frame) and _frame_has_total_score_signature(
            local_cap,
            early_probe_frame,
            left,
            top,
            crop_width,
            crop_height,
            score_layout_id,
            stats,
            position_label="total_stable_probe_early",
        ):
            frame_number = max(int(transition_frame), int(early_search_start_frame))
        elif late_probe_frame <= int(search_end_frame) and _frame_has_total_score_signature(
            local_cap,
            late_probe_frame,
            left,
            top,
            crop_width,
            crop_height,
            score_layout_id,
            stats,
            position_label="total_stable_probe_late",
        ):
            frame_number = max(int(transition_frame), int(late_search_start_frame))
        else:
            frame_number = int(transition_frame)
    else:
        frame_number = int(transition_frame)
    coarse_step = max(1, int(COARSE_SEARCH_STEP_FRAMES))
    coarse_rewind = max(1, int(COARSE_SEARCH_REWIND_FRAMES))
    position_capture_for_read(
        local_cap,
        frame_number,
        stats,
        max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
        label="total_stable_search",
    )
    stable_run_start = None
    stable_run_count = 0
    stable_signature = None
    while frame_number <= int(search_end_frame):
        ret, frame = read_video_frame(local_cap, stats)
        if not ret:
            break
        upscaled_image = crop_and_upscale_image(
            frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT
        )
        signature = _extract_total_score_stable_signature(
            upscaled_image,
            score_layout_id=score_layout_id,
        )
        if coarse_step > 1:
            if signature is None:
                next_frame = min(int(search_end_frame), int(frame_number) + coarse_step)
                if next_frame <= int(frame_number):
                    break
                position_capture_for_read(
                    local_cap,
                    next_frame,
                    stats,
                    max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
                    label="total_stable_search",
                )
                frame_number = next_frame
                continue
            rewind_frame = max(int(transition_frame), int(frame_number) - coarse_rewind)
            coarse_step = 1
            stable_run_start = None
            stable_run_count = 0
            stable_signature = None
            seek_to_frame(local_cap, rewind_frame, stats, label="total_stable_rewind")
            frame_number = rewind_frame
            continue
        if signature is not None:
            if stable_run_start is None or signature != stable_signature:
                stable_run_start = int(frame_number)
                stable_run_count = 1
                stable_signature = signature
            else:
                stable_run_count += 1
            if stable_run_count >= stable_frames_required:
                return int(frame_number)
        else:
            stable_run_start = None
            stable_run_count = 0
            stable_signature = None
        frame_number += 1
    return None


def _captured_frames_cover_range(captured_frames, start_frame, end_frame):
    """Return True when existing captured frames already span an inclusive target range."""
    if not captured_frames:
        return False
    frame_numbers = sorted(
        int(frame_number)
        for frame_number, _frame_image in captured_frames
        if frame_number is not None
    )
    if not frame_numbers:
        return False
    return frame_numbers[0] <= int(start_frame) and frame_numbers[-1] >= int(end_frame)


def count_visible_position_rows(frame_image, score_layout_id=None):
    return _count_visible_rows_from_position_metrics(_position_metrics_for_frame(frame_image, score_layout_id=score_layout_id))


def refine_race_score_result_for_expected_players(result, expected_players):
    score_layout_id = result.get("candidate", {}).get("score_layout_id")
    race_score_image = result.get('race_score_image')
    if race_score_image is None:
        return result

    candidate = result.get('candidate', {})
    fps = float(candidate.get('fps', 0) or 0)
    race_number = int(candidate.get('race_number', 0) or 0)
    position_metrics = _position_metrics_for_frame(race_score_image, score_layout_id=score_layout_id)
    visible_rows = _count_visible_rows_from_position_metrics(position_metrics)
    row11_present = _row_has_expected_template(position_metrics, 11, POSITION_PRESENT_COEFF_THRESHOLD)
    row12_present = _row_has_expected_template(position_metrics, 12, POSITION_PRESENT_COEFF_THRESHOLD)
    twelfth_template_detected = bool(result.get("twelfth_template_detected", False))

    should_search = False
    if expected_players == 12:
        should_search = not row12_present
    elif race_number == 1 and visible_rows == 11 and row11_present:
        should_search = not row12_present

    needs_early_expansion = twelfth_template_detected

    if not should_search and not needs_early_expansion:
        return result

    extra_search_frames = fps_scaled_frames(10, fps)
    video_path = candidate.get('video_path')
    left = candidate.get('left')
    top = candidate.get('top')
    crop_width = candidate.get('crop_width')
    crop_height = candidate.get('crop_height')
    scale_x = candidate.get('scale_x')
    scale_y = candidate.get('scale_y')
    consensus_frame_count = int(candidate.get('ocr_consensus_frames', 0) or 0)
    stats = defaultdict(float, result.get('stats') or {})
    result['stats'] = stats
    start_frame = int(result.get('race_score_frame', 0) or 0)

    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        return result

    try:
        if needs_early_expansion and result.get('actual_race_score_frame') is not None:
            expanded_start_frame = max(
                0,
                int(result['actual_race_score_frame']) - RACE_SCORE_EARLY_EXPANSION_FRAMES,
            )
            expanded_end_frame = int(result['actual_race_score_frame']) + RACE_SCORE_LATE_EXPANSION_FRAMES
            result['race_consensus_frames'] = collect_frame_range_from_capture(
                local_cap,
                expanded_start_frame,
                expanded_end_frame,
                left,
                top,
                crop_width,
                crop_height,
                stats=stats,
                label="refine_early_expand",
            )
        if not should_search:
            return result

        for frame_offset in range(1, extra_search_frames + 1):
            candidate_frame = start_frame + frame_offset
            actual_frame, refined_image = capture_export_frame(
                local_cap, candidate_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats, label="refine_probe"
            )
            if actual_frame is None or refined_image is None:
                continue
            refined_metrics = _position_metrics_for_frame(refined_image, score_layout_id=score_layout_id)
            if not _row_has_expected_template(refined_metrics, 12, POSITION_PRESENT_COEFF_THRESHOLD):
                continue
            final_frame = candidate_frame + fps_scaled_frames(RACE_SCORE_EXTRA_DELAY_FRAMES_30FPS, fps)
            final_actual_frame, final_image = capture_export_frame(
                local_cap, final_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats, label="refine_final_export"
            )
            if final_actual_frame is None or final_image is None:
                final_frame = candidate_frame
                final_actual_frame = actual_frame
                final_image = refined_image
            result['race_score_frame'] = final_frame
            result['actual_race_score_frame'] = final_actual_frame
            result['race_score_image'] = final_image
            expanded_start_frame = max(0, int(final_actual_frame) - RACE_SCORE_EARLY_EXPANSION_FRAMES)
            expanded_end_frame = int(final_actual_frame) + RACE_SCORE_LATE_EXPANSION_FRAMES
            result['race_consensus_frames'] = collect_frame_range_from_capture(
                local_cap,
                expanded_start_frame,
                expanded_end_frame,
                left,
                top,
                crop_width,
                crop_height,
                stats=stats,
                label="refine_final_expand",
            )
            break
    finally:
        local_cap.release()

    return result


def expand_race_score_consensus_window(result, minimum_expected_players):
    """Extend RaceScore bundles so OCR can use early frames for points and late frames for presence."""
    if not bool(result.get("twelfth_template_detected", False)):
        return result

    candidate = result.get('candidate', {})
    actual_race_score_frame = result.get('actual_race_score_frame')
    if actual_race_score_frame is None:
        return result

    left = candidate.get('left')
    top = candidate.get('top')
    crop_width = candidate.get('crop_width')
    crop_height = candidate.get('crop_height')
    consensus_frame_count = int(candidate.get('ocr_consensus_frames', 0) or 0)
    video_path = candidate.get('video_path')
    stats = defaultdict(float, result.get('stats') or {})
    result['stats'] = stats
    if not video_path or any(value is None for value in (left, top, crop_width, crop_height)):
        return result

    expanded_start_frame = max(0, int(actual_race_score_frame) - RACE_SCORE_EARLY_EXPANSION_FRAMES)
    expanded_end_frame = int(actual_race_score_frame) + RACE_SCORE_LATE_EXPANSION_FRAMES
    if _captured_frames_cover_range(
        result.get('race_consensus_frames', []),
        expanded_start_frame,
        expanded_end_frame,
    ):
        return result

    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        return result

    try:
        result['race_consensus_frames'] = collect_frame_range_from_capture(
            local_cap,
            expanded_start_frame,
            expanded_end_frame,
            left,
            top,
            crop_width,
            crop_height,
            stats=stats,
            label="expand_consensus_window",
        )
    finally:
        local_cap.release()

    return result


def analyze_score_window_task(task, frame_to_timecode, capture=None):
    """Analyze one score candidate window and decide which frames to export."""
    video_path = task["video_path"]
    frame_number = task["frame_number"]
    fps = task["fps"]
    templates = task["templates"]
    left = task["left"]
    top = task["top"]
    crop_width = task["crop_width"]
    crop_height = task["crop_height"]
    scale_x = task["scale_x"]
    scale_y = task["scale_y"]
    score_layout = get_score_layout(task.get("score_layout_id"))

    start_frame = frame_number - int(3 * fps)
    end_frame = frame_number + int(13 * fps)
    race_score_frame = 0
    total_score_frame = 0
    score_hit_frame = None
    stable_total_score_frame = None
    total_score_used_fallback = False
    twelfth_template_detected = False
    drop_start_frame = None
    selected_points_anchor_frame = None
    transition_frame = None
    debug_rows = []
    stats = {}
    from collections import defaultdict

    stats = defaultdict(float)
    owned_capture = capture is None
    local_cap = capture if capture is not None else cv2.VideoCapture(video_path)
    if local_cap is None or not local_cap.isOpened():
        return {
            "candidate": task,
            "race_score_frame": 0,
            "total_score_frame": 0,
            "debug_rows": [],
            "stats": stats,
            "analysis_trace": _build_score_analysis_trace(task, start_frame=start_frame, end_frame=end_frame),
        }

    coarse_search_step = max(1, int(COARSE_SEARCH_STEP_FRAMES))
    coarse_search_rewind = max(1, int(COARSE_SEARCH_REWIND_FRAMES))
    detail_frame_number = start_frame
    position_capture_for_read(
        local_cap,
        detail_frame_number,
        stats,
        max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
        label="detail_loop_start",
    )
    while detail_frame_number < end_frame:
        stats["score_detail_frames"] += 1
        ret, frame = read_video_frame(local_cap, stats)
        if not ret:
            break

        frame_prepare_start = time.perf_counter()
        upscaled_image = crop_and_upscale_image(
            frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT
        )
        gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
        add_timing(stats, "score_detail_frame_prepare_s", frame_prepare_start)

        if race_score_frame == 0:
            ignore_match = _match_ignore_frame_target_detail(gray_image, templates, stats)
            if (
                not ignore_match["rejected_as_blank"]
                and ignore_match["max_val"] > ignore_match["match_threshold"]
                and not np.isinf(ignore_match["max_val"])
            ):
                stats["score_ignore_rejects"] += 1
                timecode = frame_to_timecode(detail_frame_number, fps)
                debug_rows.append(
                    [
                        os.path.basename(video_path),
                        str(ignore_match["label"] or "Ignore"),
                        detail_frame_number,
                        float(ignore_match["max_val"]),
                        timecode,
                    ]
                )
                if owned_capture:
                    local_cap.release()
                return {
                    "candidate": task,
                    "race_score_frame": 0,
                    "total_score_frame": 0,
                    "debug_rows": debug_rows,
                    "stats": stats,
                    "ignored_candidate": True,
                    "ignore_label": str(ignore_match["label"] or ""),
                    "analysis_trace": _build_score_analysis_trace(
                        task,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        ignored_candidate=True,
                        ignore_label=str(ignore_match["label"] or ""),
                    ),
                }

        fast_prefix_gate = _fast_prefix_gate_score(
            upscaled_image,
            score_layout_id=score_layout.layout_id,
            min_players=POSITION_SCAN_MIN_PLAYERS,
            stats=stats,
        )
        if fast_prefix_gate:
            stats["score_prefix_gate_pass_calls"] += 1
        else:
            stats["score_prefix_gate_fail_calls"] += 1
            max_val = 0.0
            rejected_as_blank = False
            detected_layout_id = score_layout.layout_id
            layout_metrics = {}
            timecode = frame_to_timecode(detail_frame_number, fps)
            debug_rows.append([os.path.basename(video_path), "Score", detail_frame_number, 0.0, timecode])
            if race_score_frame != 0:
                if drop_start_frame is None:
                    drop_start_frame = int(detail_frame_number)
                drop_duration_frames = int(detail_frame_number) - int(drop_start_frame) + 1
                if drop_duration_frames >= max(1, int(round(TOTAL_SCORE_TRANSITION_DROP_SECONDS * max(float(fps), 1.0)))):
                    total_score_frame = int(drop_start_frame) - int(2.7 * fps)
                    break
            if race_score_frame == 0 and coarse_search_step > 1:
                next_frame = min(end_frame, int(detail_frame_number) + coarse_search_step)
                if next_frame <= int(detail_frame_number):
                    break
                detail_frame_number = next_frame
                position_capture_for_read(
                    local_cap,
                    detail_frame_number,
                    stats,
                    max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
                    label="detail_loop_skip",
                )
                continue
            detail_frame_number += 1
            continue

        layout_metrics = {}
        detected_layout_id = score_layout.layout_id
        if race_score_frame == 0:
            stage_start = time.perf_counter()
            raw_confirm_passed, raw_confirm_score = _raw_fixed_grid_prefix_confirm(
                upscaled_image,
                required_players=POSITION_SCAN_MIN_PLAYERS,
                score_layout_id=score_layout.layout_id,
                stats=stats,
            )
            add_timing(stats, "score_detail_match_score_s", stage_start)
            stats["score_detail_score_match_calls"] += 1
            max_val = float(raw_confirm_score)
            rejected_as_blank = not bool(raw_confirm_passed)
        else:
            stage_start = time.perf_counter()
            max_val, rejected_as_blank, detected_layout_id, layout_metrics = _match_score_target_layouts(
                upscaled_image,
                templates,
                stats,
                return_layout_metrics=True,
                preferred_layout_ids=[score_layout.layout_id],
                stats_scope="detail",
            )
            add_timing(stats, "score_detail_match_score_s", stage_start)
            stats["score_detail_score_match_calls"] += 1
        timecode = frame_to_timecode(detail_frame_number, fps)
        debug_rows.append([os.path.basename(video_path), "Score", detail_frame_number, 0 if rejected_as_blank else max_val, timecode])

        if rejected_as_blank:
            if race_score_frame != 0:
                if drop_start_frame is None:
                    drop_start_frame = int(detail_frame_number)
                drop_duration_frames = int(detail_frame_number) - int(drop_start_frame) + 1
                if drop_duration_frames >= max(1, int(round(TOTAL_SCORE_TRANSITION_DROP_SECONDS * max(float(fps), 1.0)))):
                    total_score_frame = int(drop_start_frame) - int(2.7 * fps)
                    break
            if race_score_frame == 0 and coarse_search_step > 1:
                next_frame = min(end_frame, int(detail_frame_number) + coarse_search_step)
                if next_frame <= int(detail_frame_number):
                    break
                detail_frame_number = next_frame
                position_capture_for_read(
                    local_cap,
                    detail_frame_number,
                    stats,
                    max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
                    label="detail_loop_skip",
                )
                continue
            detail_frame_number += 1
            continue

        if max_val > 0.3 and not np.isinf(max_val) and race_score_frame == 0:
            if coarse_search_step > 1:
                coarse_search_step = 1
                detail_frame_number = max(start_frame, int(detail_frame_number) - coarse_search_rewind)
                seek_to_frame(local_cap, detail_frame_number, stats, label="detail_coarse_rewind")
                continue
            score_hit_frame = int(detail_frame_number)
            race_score_frame = score_hit_frame + int(0.7 * fps)
            score_layout = get_score_layout(detected_layout_id or score_layout.layout_id)
            drop_start_frame = None
            transition_search_end = min(
                end_frame,
                score_hit_frame + max(1, int(round(POINTS_TRANSITION_SEARCH_END_SECONDS * max(float(fps), 1.0)))),
            )
            transition_frame, selected_points_anchor_frame = _find_points_transition_frame(
                local_cap,
                int(race_score_frame),
                int(transition_search_end),
                left,
                top,
                crop_width,
                crop_height,
                score_layout.layout_id,
                stats,
                fps,
            )
            if transition_frame is not None:
                if selected_points_anchor_frame is None:
                    selected_points_anchor_frame = max(0, int(transition_frame) - 2)
                total_score_frame = _find_total_score_stable_frame(
                    local_cap,
                    int(transition_frame),
                    fps,
                    left,
                    top,
                    crop_width,
                    crop_height,
                    score_layout.layout_id,
                    stats,
                )
                if total_score_frame is None:
                    total_score_used_fallback = True
                    total_score_frame = int(transition_frame) + int(TOTAL_SCORE_FROM_TRANSITION_SECONDS * fps)
                else:
                    stable_total_score_frame = int(total_score_frame)
                break
            position_capture_for_read(
                local_cap,
                int(detail_frame_number) + 1,
                stats,
                max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
                label="detail_post_transition_continue",
            )
            continue

        if race_score_frame != 0:
            selected_layout_metrics = layout_metrics.get(str(score_layout.layout_id), {})
            selected_position_metrics = selected_layout_metrics.get("position_metrics") or []
            if selected_position_metrics and _tie_aware_score_signal_present_from_metrics(
                selected_position_metrics,
                min_players=POSITION_SCAN_MIN_PLAYERS,
            ):
                stats["score_tie_aware_reuse_calls"] += 1
                drop_start_frame = None
            else:
                stats["score_tie_aware_drop_checks"] += 1
                if drop_start_frame is None:
                    drop_start_frame = int(detail_frame_number)
                drop_duration_frames = int(detail_frame_number) - int(drop_start_frame) + 1
                if drop_duration_frames >= max(1, int(round(TOTAL_SCORE_TRANSITION_DROP_SECONDS * max(float(fps), 1.0)))):
                    total_score_frame = int(drop_start_frame) - int(2.7 * fps)
                    break

        detail_frame_number += 1

    race_score_image = None
    total_score_image = None
    total_score_visible_players = None
    actual_race_score_frame = None
    actual_total_score_frame = None
    actual_points_anchor_frame = None
    points_anchor_image = None
    points_context_frames = []
    race_consensus_frames = []
    total_consensus_frames = []
    if race_score_frame > 0 and total_score_frame > 0:
        export_stage_start = time.perf_counter()
        actual_race_score_frame, race_score_image = capture_export_frame(
            local_cap, race_score_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats, label="race_anchor_export"
        )
        if actual_race_score_frame is not None:
            race_consensus_frames = collect_consensus_frames_from_capture(
                local_cap, actual_race_score_frame, left, top, crop_width, crop_height, task["ocr_consensus_frames"], stats=stats, label="race_consensus"
            )
        actual_total_score_frame, total_score_image = capture_export_frame(
            local_cap, total_score_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats, label="total_anchor_export"
        )
        if actual_total_score_frame is not None:
            total_consensus_frames = collect_consensus_frames_from_capture(
                local_cap, actual_total_score_frame, left, top, crop_width, crop_height, task["ocr_consensus_frames"], stats=stats, label="total_consensus"
            )
            total_score_visible_players = count_visible_position_rows(
                total_score_image,
                task.get("score_layout_id"),
            )
        if selected_points_anchor_frame is not None:
            actual_points_anchor_frame, points_anchor_image = capture_export_frame(
                local_cap, int(selected_points_anchor_frame), left, top, crop_width, crop_height, scale_x, scale_y, stats, label="points_anchor_export"
            )
            if actual_points_anchor_frame is not None:
                points_context_frames = collect_frame_range_from_capture(
                    local_cap,
                    int(actual_points_anchor_frame) - 3,
                    int(actual_points_anchor_frame) + 3,
                    left,
                    top,
                    crop_width,
                    crop_height,
                    stats=stats,
                    label="points_context",
                )
        _record_score_capture_usage(
            stats,
            fps=fps,
            actual_race_score_frame=actual_race_score_frame,
            actual_total_score_frame=actual_total_score_frame,
            actual_points_anchor_frame=actual_points_anchor_frame,
            race_consensus_frames=race_consensus_frames,
            total_consensus_frames=total_consensus_frames,
            points_context_frames=points_context_frames,
        )
        add_timing(stats, "output_frame_capture_s", export_stage_start)

    if owned_capture:
        local_cap.release()
    return {
        "candidate": task,
        "race_score_frame": race_score_frame,
        "total_score_frame": total_score_frame,
        "actual_race_score_frame": actual_race_score_frame,
        "actual_total_score_frame": actual_total_score_frame,
        "race_score_image": race_score_image,
        "total_score_image": total_score_image,
        "total_score_visible_players": total_score_visible_players,
        "race_consensus_frames": race_consensus_frames,
        "total_consensus_frames": total_consensus_frames,
        "selected_points_anchor_frame": selected_points_anchor_frame,
        "actual_points_anchor_frame": actual_points_anchor_frame,
        "points_anchor_image": points_anchor_image,
        "points_context_frames": points_context_frames,
        "twelfth_template_detected": twelfth_template_detected,
        "debug_rows": debug_rows,
        "stats": stats,
        "analysis_trace": _build_score_analysis_trace(
            task,
            start_frame=start_frame,
            end_frame=end_frame,
            score_hit_frame=score_hit_frame,
            transition_frame=transition_frame,
            stable_total_score_frame=stable_total_score_frame,
            selected_points_anchor_frame=selected_points_anchor_frame,
            total_score_used_fallback=total_score_used_fallback,
            ignored_candidate=bool(False),
            ignore_label="",
        ),
    }


def collect_consensus_frames_from_capture(capture, center_frame, left, top, crop_width, crop_height, consensus_frame_count, stats=None, label=None):
    """Collect neighbouring upscaled frames for OCR voting from an open capture."""
    radius = max(0, consensus_frame_count // 2)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bundled_frames = []
    start_frame = max(0, center_frame - radius)
    end_frame = min(total_frames, center_frame + radius + 1)
    if stats is None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        position_capture_for_read(
            capture,
            start_frame,
            stats,
            max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
            label=label,
        )
    for _frame_number in range(start_frame, end_frame):
        if stats is None:
            ret, frame = capture.read()
        else:
            ret, frame = read_video_frame(capture, stats)
        if not ret:
            continue
        bundled_frames.append(
            (
                actual_frame_after_read(capture),
                crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT),
            )
        )
    return bundled_frames


def collect_frame_range_from_capture(capture, start_frame, end_frame, left, top, crop_width, crop_height, stats=None, label=None):
    """Collect a specific inclusive frame range as OCR inputs."""
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bundled_frames = []
    start_frame = max(0, int(start_frame))
    end_frame = min(total_frames - 1, int(end_frame))
    if end_frame < start_frame:
        return bundled_frames
    if stats is None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        position_capture_for_read(
            capture,
            start_frame,
            stats,
            max_forward_grab_frames=SMALL_FORWARD_GRAB_WINDOW_FRAMES,
            label=label,
        )
    for _frame_number in range(start_frame, end_frame + 1):
        if stats is None:
            ret, frame = capture.read()
        else:
            ret, frame = read_video_frame(capture, stats)
        if not ret:
            continue
        bundled_frames.append(
            (
                actual_frame_after_read(capture),
                crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT),
            )
        )
    return bundled_frames


def _record_score_capture_usage(
    stats,
    *,
    fps,
    actual_race_score_frame=None,
    actual_total_score_frame=None,
    actual_points_anchor_frame=None,
    race_consensus_frames=None,
    total_consensus_frames=None,
    points_context_frames=None,
):
    """Record per-race capture usage so profiler output can quantify overlap and possible waste."""
    if stats is None:
        return

    race_consensus_frames = list(race_consensus_frames or [])
    total_consensus_frames = list(total_consensus_frames or [])
    points_context_frames = list(points_context_frames or [])

    race_consensus_numbers = [int(frame_number) for frame_number, _image in race_consensus_frames]
    total_consensus_numbers = [int(frame_number) for frame_number, _image in total_consensus_frames]
    points_context_numbers = [int(frame_number) for frame_number, _image in points_context_frames]

    capture_events = []
    if actual_race_score_frame is not None:
        capture_events.append(int(actual_race_score_frame))
        increment_counter(stats, "score_capture_race_anchor_frames")
    if actual_total_score_frame is not None:
        capture_events.append(int(actual_total_score_frame))
        increment_counter(stats, "score_capture_total_anchor_frames")
    if actual_points_anchor_frame is not None:
        capture_events.append(int(actual_points_anchor_frame))
        increment_counter(stats, "score_capture_points_anchor_frames")

    capture_events.extend(race_consensus_numbers)
    capture_events.extend(total_consensus_numbers)
    capture_events.extend(points_context_numbers)

    increment_counter(stats, "score_capture_race_consensus_frames", len(race_consensus_numbers))
    increment_counter(stats, "score_capture_total_consensus_frames", len(total_consensus_numbers))
    increment_counter(stats, "score_capture_points_context_frames", len(points_context_numbers))

    unique_capture_frames = set(capture_events)
    duplicate_capture_frames = max(0, len(capture_events) - len(unique_capture_frames))

    increment_counter(stats, "score_capture_frame_events_total", len(capture_events))
    increment_counter(stats, "score_capture_unique_frames_total", len(unique_capture_frames))
    increment_counter(stats, "score_capture_duplicate_frames_total", duplicate_capture_frames)

    same_run_ocr_frames = list(race_consensus_numbers) + list(total_consensus_numbers)
    same_run_ocr_unique_frames = set(same_run_ocr_frames)
    increment_counter(stats, "score_same_run_ocr_frames_total", len(same_run_ocr_frames))
    increment_counter(stats, "score_same_run_ocr_unique_frames_total", len(same_run_ocr_unique_frames))

    persisted_ocr_frames = []
    if actual_race_score_frame is not None:
        persisted_ocr_frames.append(int(actual_race_score_frame))
    if actual_total_score_frame is not None:
        persisted_ocr_frames.append(int(actual_total_score_frame))
    persisted_ocr_frames.extend(points_context_numbers)
    persisted_ocr_frames.extend(total_consensus_numbers)
    persisted_ocr_unique_frames = set(persisted_ocr_frames)
    increment_counter(stats, "score_persisted_ocr_frames_total", len(persisted_ocr_frames))
    increment_counter(stats, "score_persisted_ocr_unique_frames_total", len(persisted_ocr_unique_frames))

    capture_outside_same_run_cache_frames = unique_capture_frames - same_run_ocr_unique_frames
    increment_counter(stats, "score_capture_frames_outside_same_run_cache_total", len(capture_outside_same_run_cache_frames))

    fps_value = max(float(fps or 0.0), 1.0)
    stats["score_capture_duplicate_source_seconds_total"] += duplicate_capture_frames / fps_value
    stats["score_capture_outside_same_run_cache_source_seconds_total"] += len(capture_outside_same_run_cache_frames) / fps_value


def _write_export_image_tracked(path, image, stats=None):
    write_start = time.perf_counter()
    output_path = write_export_image(path, image)
    if stats is not None:
        add_timing(stats, "score_save_image_write_s", write_start)
        increment_counter(stats, "score_save_image_writes")
    return output_path


def _remove_legacy_bundle_files(bundle_dir: Path, patterns, stats=None, *, preexisting=True):
    if not preexisting:
        return 0
    cleanup_start = time.perf_counter()
    removed = 0
    for pattern in patterns:
        for existing_frame_path in bundle_dir.glob(pattern):
            try:
                existing_frame_path.unlink()
                removed += 1
            except OSError:
                pass
    if stats is not None:
        add_timing(stats, "score_save_cleanup_s", cleanup_start)
        increment_counter(stats, "score_save_cleanup_removed", removed)
        if removed > 0:
            increment_counter(stats, "score_save_cleanup_runs")
    return removed


def save_score_frames(video_path, video_label, race_number, race_score_frame, total_score_frame, actual_race_score_frame,
                      actual_total_score_frame, race_score_image, total_score_image, race_consensus_frames,
                      total_consensus_frames, fps, metadata_writer, consensus_frame_cache, frame_to_timecode, *, video_source_path=None,
                      score_layout_id=None, actual_points_anchor_frame=None, points_anchor_image=None,
                      points_context_frames=None, stats=None):
    """Persist the chosen race-score and total-score screenshots for one race."""
    if race_score_image is None or total_score_image is None:
        return False
    score_layout = get_score_layout(score_layout_id)
    race_bundle_dir = Path(score_bundle_dir(video_label, race_number, "2RaceScore"))
    race_bundle_preexisting = race_bundle_dir.exists()
    frame_filename = score_bundle_anchor_path(
        video_label,
        race_number,
        "2RaceScore",
        actual_race_score_frame,
        score_layout.layout_id,
    )
    write_start = time.perf_counter()
    frame_filename = _write_export_image_tracked(frame_filename, race_score_image, stats=stats)
    if stats is not None:
        add_timing(stats, "score_save_race_anchor_s", write_start)
    _remove_legacy_bundle_files(
        race_bundle_dir,
        ("frame_*", "12point_*", "12point_frame_*"),
        stats=stats,
        preexisting=race_bundle_preexisting,
    )
    if APP_CONFIG.write_debug_score_images:
        draw_score_layout_demo(
            race_score_image,
            score_layout.layout_id,
            "2RaceScore",
            score_demo_output_path(video_label, race_number, "2RaceScore", score_layout.layout_id),
        )
    video_stem = video_label
    if race_consensus_frames:
        consensus_frame_cache[(video_stem, int(race_number), "RaceScore")] = [image for _frame, image in race_consensus_frames]
    log_exported_frame(
        metadata_writer,
        video_path,
        race_number,
        "RaceScore",
        race_score_frame,
        actual_race_score_frame,
        fps,
        frame_to_timecode,
        video_label=video_label,
        video_source_path=video_source_path,
        score_layout_id=score_layout.layout_id,
        bundle_path=str(Path(frame_filename).parent),
        anchor_path=str(frame_filename),
    )

    if actual_points_anchor_frame is not None and points_context_frames:
        _remove_legacy_bundle_files(
            race_bundle_dir,
            (f"Race_{int(race_number):03d}_F*",),
            stats=stats,
            preexisting=race_bundle_preexisting,
        )
        for context_frame_number, context_frame_image in points_context_frames:
            _write_export_image_tracked(
                score_bundle_race_context_path(
                    video_label,
                    race_number,
                    "2RaceScore",
                    context_frame_number,
                    score_layout.layout_id,
                ),
                context_frame_image,
                stats=stats,
            )

    total_bundle_dir = Path(score_bundle_dir(video_label, race_number, "3TotalScore"))
    total_bundle_preexisting = total_bundle_dir.exists()
    frame_filename = score_bundle_anchor_path(
        video_label,
        race_number,
        "3TotalScore",
        actual_total_score_frame,
        score_layout.layout_id,
    )
    write_start = time.perf_counter()
    frame_filename = _write_export_image_tracked(frame_filename, total_score_image, stats=stats)
    if stats is not None:
        add_timing(stats, "score_save_total_anchor_s", write_start)
    _remove_legacy_bundle_files(
        total_bundle_dir,
        ("frame_*",),
        stats=stats,
        preexisting=total_bundle_preexisting,
    )
    for consensus_frame_number, consensus_frame_image in total_consensus_frames:
        _write_export_image_tracked(
            score_bundle_consensus_path(video_label, race_number, "3TotalScore", consensus_frame_number),
            consensus_frame_image,
            stats=stats,
        )
    if APP_CONFIG.write_debug_score_images:
        draw_score_layout_demo(
            total_score_image,
            score_layout.layout_id,
            "3TotalScore",
            score_demo_output_path(video_label, race_number, "3TotalScore", score_layout.layout_id),
        )
    if total_consensus_frames:
        consensus_frame_cache[(video_stem, int(race_number), "TotalScore")] = [image for _frame, image in total_consensus_frames]
    log_exported_frame(
        metadata_writer,
        video_path,
        race_number,
        "TotalScore",
        total_score_frame,
        actual_total_score_frame,
        fps,
        frame_to_timecode,
        video_label=video_label,
        video_source_path=video_source_path,
        score_layout_id=score_layout.layout_id,
        bundle_path=str(Path(frame_filename).parent),
        anchor_path=str(frame_filename),
    )
    return True
