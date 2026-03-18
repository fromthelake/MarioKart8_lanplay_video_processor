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
    crop_to_gray_and_upscale_image,
    match_template,
    preprocess_roi,
    score_bundle_anchor_path,
    score_bundle_consensus_path,
    write_export_image,
)
from .extract_video_io import actual_frame_after_read, add_timing, log_exported_frame, read_video_frame, seek_to_frame
from .ocr_scoreboard_consensus import (
    POSITION_PRESENT_COEFF_THRESHOLD,
    POSITION_PRESENT_ROW1_COEFF_THRESHOLD,
    build_position_signal_metrics,
    process_image,
)
from .project_paths import PROJECT_ROOT
from .score_layouts import draw_score_layout_demo, get_score_layout, score_demo_output_path


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


def capture_export_frame(capture, target_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats):
    seek_to_frame(capture, target_frame, stats)
    ret, frame = read_video_frame(capture, stats)
    if not ret:
        return None, None
    actual_frame = actual_frame_after_read(capture)
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
    return actual_frame, enhance_export_frame(upscaled_image, scale_x, scale_y)


def fps_scaled_frames(base_frames_30fps, fps):
    return max(1, int(round(float(base_frames_30fps) * (max(float(fps), 1.0) / 30.0))))


RACE_SCORE_EXTRA_DELAY_FRAMES_30FPS = 1


def _position_metrics_for_frame(frame_image, score_layout_id=None):
    processed_image = process_image(frame_image, score_layout_id=score_layout_id)
    return build_position_signal_metrics(processed_image)


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

    should_search = False
    if expected_players == 12:
        should_search = not row12_present
    elif race_number == 1 and visible_rows == 11 and row11_present:
        should_search = not row12_present

    if not should_search:
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
        for frame_offset in range(1, extra_search_frames + 1):
            candidate_frame = start_frame + frame_offset
            actual_frame, refined_image = capture_export_frame(
                local_cap, candidate_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats
            )
            if actual_frame is None or refined_image is None:
                continue
            refined_metrics = _position_metrics_for_frame(refined_image, score_layout_id=score_layout_id)
            if not _row_has_expected_template(refined_metrics, 12, POSITION_PRESENT_COEFF_THRESHOLD):
                continue
            final_frame = candidate_frame + fps_scaled_frames(RACE_SCORE_EXTRA_DELAY_FRAMES_30FPS, fps)
            final_actual_frame, final_image = capture_export_frame(
                local_cap, final_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats
            )
            if final_actual_frame is None or final_image is None:
                final_frame = candidate_frame
                final_actual_frame = actual_frame
                final_image = refined_image
            result['race_score_frame'] = final_frame
            result['actual_race_score_frame'] = final_actual_frame
            result['race_score_image'] = final_image
            result['race_consensus_frames'] = collect_consensus_frames_from_capture(
                local_cap,
                final_actual_frame,
                left,
                top,
                crop_width,
                crop_height,
                consensus_frame_count,
            )
            break
    finally:
        local_cap.release()

    return result


def analyze_score_window_task(task, frame_to_timecode):
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
    player12 = 0
    check_player_12 = 0
    debug_rows = []
    stats = {}
    from collections import defaultdict

    stats = defaultdict(float)

    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        return {"candidate": task, "race_score_frame": 0, "total_score_frame": 0, "debug_rows": [], "stats": stats}

    detail_frame_number = start_frame
    seek_to_frame(local_cap, detail_frame_number, stats)
    template_binary, alpha_mask = templates[0]

    while detail_frame_number < end_frame:
        ret, frame = read_video_frame(local_cap, stats)
        if not ret:
            break

        frame_prepare_start = time.perf_counter()
        gray_image, crop_upscale_time, grayscale_time = crop_to_gray_and_upscale_image(
            frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT
        )
        stats["score_detail_crop_upscale_s"] += crop_upscale_time
        stats["score_detail_grayscale_s"] += grayscale_time

        stage_start = time.perf_counter()
        roi_x, roi_y, roi_width, roi_height = score_layout.scoreboard_points_roi
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        add_timing(stats, "score_detail_score_roi_extract_s", stage_start)

        stage_start = time.perf_counter()
        processed_roi = preprocess_roi(roi, 0)
        add_timing(stats, "score_detail_score_preprocess_s", stage_start)
        add_timing(stats, "score_detail_frame_prepare_s", frame_prepare_start)

        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            detail_frame_number += 1
            timecode = frame_to_timecode(detail_frame_number, fps)
            debug_rows.append([os.path.basename(video_path), "Score", detail_frame_number, 0, timecode])
            if race_score_frame != 0:
                total_score_frame = detail_frame_number - int(2.7 * fps)
                break
            continue

        stage_start = time.perf_counter()
        max_val = match_template(processed_roi, template_binary, alpha_mask)
        add_timing(stats, "score_detail_match_score_s", stage_start)
        timecode = frame_to_timecode(detail_frame_number, fps)
        debug_rows.append([os.path.basename(video_path), "Score", detail_frame_number, max_val, timecode])

        if max_val > 0.3 and not np.isinf(max_val) and race_score_frame == 0:
            race_score_frame = detail_frame_number + int(0.6 * fps)
            check_player_12 = 1
            continue

        if max_val > 0.3 and not np.isinf(max_val) and check_player_12 == 1:
            stage_start = time.perf_counter()
            roi_x2, roi_y2, roi_width2, roi_height2 = score_layout.twelfth_place_check_roi
            roi2 = gray_image[roi_y2:roi_y2 + roi_height2, roi_x2:roi_x2 + roi_width2]
            add_timing(stats, "score_detail_12th_roi_extract_s", stage_start)

            stage_start = time.perf_counter()
            processed_roi2 = preprocess_roi(roi2, 0)
            add_timing(stats, "score_detail_12th_preprocess_s", stage_start)
            template_binary2, alpha_mask2 = templates[3]

            stage_start = time.perf_counter()
            max_val2 = match_template(processed_roi2, template_binary2, alpha_mask2)
            add_timing(stats, "score_detail_match_12th_s", stage_start)

            if max_val2 > 0.4 and not np.isinf(max_val2):
                player12 = 1

            if player12 == 1 and max_val2 < 0.1:
                race_score_frame = detail_frame_number + fps_scaled_frames(16 + RACE_SCORE_EXTRA_DELAY_FRAMES_30FPS, fps)
                detail_frame_number += int(3.9 * fps)
                seek_to_frame(local_cap, detail_frame_number, stats)
                check_player_12 = 2
                continue

        if max_val <= 0 and race_score_frame != 0:
            total_score_frame = detail_frame_number - int(2.7 * fps)
            break

        detail_frame_number += 1

    race_score_image = None
    total_score_image = None
    actual_race_score_frame = None
    actual_total_score_frame = None
    race_consensus_frames = []
    total_consensus_frames = []
    if race_score_frame > 0 and total_score_frame > 0:
        export_stage_start = time.perf_counter()
        actual_race_score_frame, race_score_image = capture_export_frame(
            local_cap, race_score_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats
        )
        if actual_race_score_frame is not None:
            race_consensus_frames = collect_consensus_frames_from_capture(
                local_cap, actual_race_score_frame, left, top, crop_width, crop_height, task["ocr_consensus_frames"]
            )
        actual_total_score_frame, total_score_image = capture_export_frame(
            local_cap, total_score_frame, left, top, crop_width, crop_height, scale_x, scale_y, stats
        )
        if actual_total_score_frame is not None:
            total_consensus_frames = collect_consensus_frames_from_capture(
                local_cap, actual_total_score_frame, left, top, crop_width, crop_height, task["ocr_consensus_frames"]
            )
        add_timing(stats, "output_frame_capture_s", export_stage_start)

    local_cap.release()
    return {
        "candidate": task,
        "race_score_frame": race_score_frame,
        "total_score_frame": total_score_frame,
        "actual_race_score_frame": actual_race_score_frame,
        "actual_total_score_frame": actual_total_score_frame,
        "race_score_image": race_score_image,
        "total_score_image": total_score_image,
        "race_consensus_frames": race_consensus_frames,
        "total_consensus_frames": total_consensus_frames,
        "debug_rows": debug_rows,
        "stats": stats,
    }


def collect_consensus_frames_from_capture(capture, center_frame, left, top, crop_width, crop_height, consensus_frame_count):
    """Collect neighbouring upscaled frames for OCR voting from an open capture."""
    radius = max(0, consensus_frame_count // 2)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bundled_frames = []
    start_frame = max(0, center_frame - radius)
    end_frame = min(total_frames, center_frame + radius + 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _frame_number in range(start_frame, end_frame):
        ret, frame = capture.read()
        if not ret:
            continue
        bundled_frames.append(
            (
                actual_frame_after_read(capture),
                crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT),
            )
        )
    return bundled_frames


def save_score_frames(video_path, video_label, race_number, race_score_frame, total_score_frame, actual_race_score_frame,
                      actual_total_score_frame, race_score_image, total_score_image, race_consensus_frames,
                      total_consensus_frames, fps, metadata_writer, consensus_frame_cache, frame_to_timecode, *, video_source_path=None,
                      score_layout_id=None):
    """Persist the chosen race-score and total-score screenshots for one race."""
    if race_score_image is None or total_score_image is None:
        return False
    score_layout = get_score_layout(score_layout_id)
    frame_filename = score_bundle_anchor_path(video_label, race_number, "2RaceScore", actual_race_score_frame)
    write_export_image(frame_filename, race_score_image)
    for consensus_frame_number, consensus_frame_image in race_consensus_frames:
        write_export_image(
            score_bundle_consensus_path(video_label, race_number, "2RaceScore", consensus_frame_number),
            consensus_frame_image,
        )
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

    frame_filename = score_bundle_anchor_path(video_label, race_number, "3TotalScore", actual_total_score_frame)
    write_export_image(frame_filename, total_score_image)
    for consensus_frame_number, consensus_frame_image in total_consensus_frames:
        write_export_image(
            score_bundle_consensus_path(video_label, race_number, "3TotalScore", consensus_frame_number),
            consensus_frame_image,
        )
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
