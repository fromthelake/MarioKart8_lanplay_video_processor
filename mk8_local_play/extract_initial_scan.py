import multiprocessing as mp
import os
import time
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from queue import Empty, Queue

import cv2
import numpy as np

from .console_logging import LOGGER
from .extract_common import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    build_video_identity,
    crop_and_upscale_image,
    frame_to_timecode,
    match_template,
    preprocess_roi,
    race_anchor_frame_path,
    relative_video_path,
    write_export_image,
)
from .ocr_scoreboard_consensus import build_position_signal_metrics, process_image
from .project_paths import PROJECT_ROOT
from .score_layouts import DEFAULT_SCORE_LAYOUT_ID, all_score_layouts
from . import extract_video_io as video_io

INITIAL_SCAN_DIAGNOSTICS_ENABLED = os.environ.get("MK8_INITIAL_SCAN_DIAGNOSTICS", "0").strip().lower() in {"1", "true", "yes", "on"}
POSITION_SCAN_MIN_PLAYERS = max(2, int(os.environ.get("MK8_POSITION_SCAN_MIN_PLAYERS", "6")))
POSITION_SCAN_MIN_ROW_COEFF = float(os.environ.get("MK8_POSITION_SCAN_MIN_ROW_COEFF", "0.4"))
POSITION_SCAN_MIN_AVG_COEFF = float(os.environ.get("MK8_POSITION_SCAN_MIN_AVG_COEFF", "0.6"))


INITIAL_SCAN_TARGETS = (
    {
        "kind": "score",
        "label": "Score",
        "match_threshold": 0.3,
        "skip_seconds": 20,
        "roi": (315, 57, 52, 610),
    },
    {
        "kind": "track",
        "label": "TrackName",
        "match_threshold": 0.6,
        "skip_seconds": 0,
        "roi": (141, 607, 183, 101),
        "template_index": 1,
    },
    {
        "kind": "race",
        "label": "RaceNumber",
        "match_threshold": 0.6,
        "skip_seconds": 60,
        "roi": (540, 590, 144, 48),
        "template_index": 2,
        "alternate_matches": (
            {"roi": (640, 590, 144, 48), "template_index": 2},
            {"roi": (694, 594, 130, 40), "template_index": 7},
        ),
    },
)

IGNORE_FRAME_TARGETS = (
    {
        "kind": "ignore",
        "label": "Ignore",
        "match_threshold": 0.75,
        "skip_seconds": 5,
        "roi": (413, 667, 808, 36),
        "template_index": 4,
    },
    {
        "kind": "ignore",
        "label": "IgnoreAlbumGallery",
        "match_threshold": 0.62,
        "skip_seconds": 5,
        # User-provided ROI (662, 669, 557, 29), expanded by 2 pixels on all sides.
        "roi": (660, 667, 561, 33),
        "template_index": 5,
    },
    {
        "kind": "ignore",
        "label": "IgnoreAlbumGalleryAlt",
        "match_threshold": 0.62,
        "skip_seconds": 5,
        # User-provided ROI (558, 669, 660, 30), expanded by 2 pixels on all sides.
        "roi": (556, 667, 664, 34),
        "template_index": 6,
    },
)

def _match_score_target_layouts(image_source, templates, stats):
    """Evaluate score presence using the per-row position boxes for both supported layouts."""
    best_match = {
        "max_val": 0.0,
        "layout_id": DEFAULT_SCORE_LAYOUT_ID,
    }
    saw_supported_layout = False
    if len(image_source.shape) == 2:
        bgr_image = cv2.cvtColor(image_source, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = image_source
    for layout in all_score_layouts():
        stage_start = time.perf_counter()
        processed_image = process_image(bgr_image, score_layout_id=layout.layout_id)
        video_io.add_timing(stats, "initial_roi_preprocess_s", stage_start)

        stage_start = time.perf_counter()
        position_metrics = build_position_signal_metrics(processed_image, score_layout_id=layout.layout_id)
        video_io.add_timing(stats, "initial_match_s", stage_start)

        if not position_metrics:
            continue
        saw_supported_layout = True
        prefix_scores = []
        for row_number in range(1, POSITION_SCAN_MIN_PLAYERS + 1):
            if row_number > len(position_metrics):
                break
            metric = position_metrics[row_number - 1]
            best_rank = int(metric.get("best_position_template", 0))
            best_score = float(metric.get("best_position_score", 0.0))
            if best_rank != row_number or best_score < POSITION_SCAN_MIN_ROW_COEFF:
                break
            prefix_scores.append(best_score)
        if len(prefix_scores) < POSITION_SCAN_MIN_PLAYERS:
            continue
        average_score = float(sum(prefix_scores) / len(prefix_scores))
        if average_score < POSITION_SCAN_MIN_AVG_COEFF:
            continue
        max_val = average_score
        if max_val > best_match["max_val"]:
            best_match = {
                "max_val": max_val,
                "layout_id": layout.layout_id,
            }
    return best_match["max_val"], not saw_supported_layout, str(best_match["layout_id"])


def update_segment_progress(progress_queue, segment_index, frame_number, emit_start, emit_end, force=False, video_label=None):
    """Send best-effort progress from worker segments back to the parent process."""
    if progress_queue is None:
        return
    completed = max(0, min(max(frame_number, emit_start), emit_end) - emit_start)
    total = max(1, emit_end - emit_start)
    progress_queue.put(
        {
            "type": "progress",
            "segment_index": segment_index,
            "video_label": str(video_label or ""),
            "completed_frames": completed,
            "total_frames": total,
            "force": force,
        }
    )


def update_segment_detection(progress_queue, segment_index, kind, frame_number, video_label=None):
    """Stream worker detections back to the parent process for live scan visibility."""
    if progress_queue is None:
        return
    progress_queue.put(
        {
            "type": "detection",
            "segment_index": segment_index,
            "video_label": str(video_label or ""),
            "kind": str(kind),
            "frame_number": int(frame_number),
        }
    )


def _expanded_roi(gray_image, roi_definition):
    """Allow small capture shifts by padding the ROI before template matching."""
    roi_x, roi_y, roi_width, roi_height = roi_definition
    roi_x = max(roi_x - 25, 0)
    roi_y = max(roi_y - 25, 0)
    roi_width = min(roi_width + 50, gray_image.shape[1] - roi_x)
    roi_height = min(roi_height + 50, gray_image.shape[0] - roi_y)
    return roi_x, roi_y, roi_width, roi_height


def _score_anchor_roi_for_template(gray_image, roi_definition, template_binary):
    """Keep the anchor aligned to the top of the scoreboard for shorter score templates."""
    roi_x, roi_y, roi_width, _roi_height = roi_definition
    return _expanded_roi(gray_image, (roi_x, roi_y, roi_width, int(template_binary.shape[0])))


def _bounded_roi(gray_image, roi_definition):
    """Clamp a fixed ROI without adding the broader scan padding used by normal anchors."""
    roi_x, roi_y, roi_width, roi_height = roi_definition
    roi_x = max(int(roi_x), 0)
    roi_y = max(int(roi_y), 0)
    roi_width = min(int(roi_width), gray_image.shape[1] - roi_x)
    roi_height = min(int(roi_height), gray_image.shape[0] - roi_y)
    return roi_x, roi_y, roi_width, roi_height


def _match_ignore_frame_target(gray_image, templates, stats):
    """Detect gallery/review UI frames that should never become race detections."""
    best_match = {
        "label": "",
        "max_val": 0.0,
        "match_threshold": 1.0,
        "skip_seconds": 0,
        "rejected_as_blank": True,
    }
    for target in IGNORE_FRAME_TARGETS:
        template_index = int(target["template_index"])
        if len(templates) <= template_index:
            continue
        roi_x, roi_y, roi_width, roi_height = _bounded_roi(gray_image, target["roi"])
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
        video_io.add_timing(stats, "initial_match_s", stage_start)
        if max_val > best_match["max_val"]:
            best_match = {
                "label": str(target["label"]),
                "max_val": float(max_val),
                "match_threshold": float(target["match_threshold"]),
                "skip_seconds": int(target["skip_seconds"]),
                "rejected_as_blank": False,
            }
    return best_match


def _match_initial_scan_target(gray_image, target, templates, stats):
    best_processed_roi = None
    best_match_val = 0.0
    saw_non_blank_roi = False
    candidate_matches = (
        {"roi": target["roi"], "template_index": int(target["template_index"])},
        *tuple(target.get("alternate_matches", ())),
    )
    for candidate in candidate_matches:
        template_binary, alpha_mask = templates[int(candidate["template_index"])]
        roi_definition = candidate["roi"]
        roi_x, roi_y, roi_width, roi_height = _expanded_roi(gray_image, roi_definition)
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        stage_start = time.perf_counter()
        processed_roi = preprocess_roi(roi, INITIAL_SCAN_TARGETS.index(target))
        video_io.add_timing(stats, "initial_roi_preprocess_s", stage_start)

        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            if best_processed_roi is None:
                best_processed_roi = processed_roi
            continue
        saw_non_blank_roi = True

        if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
            processed_roi = cv2.resize(
                processed_roi,
                (max(template_binary.shape[1], processed_roi.shape[1]), max(template_binary.shape[0], processed_roi.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )

        stage_start = time.perf_counter()
        max_val = match_template(processed_roi, template_binary, alpha_mask)
        video_io.add_timing(stats, "initial_match_s", stage_start)
        if best_processed_roi is None or max_val > best_match_val:
            best_processed_roi = processed_roi
            best_match_val = float(max_val)

    return best_match_val, not saw_non_blank_roi, best_processed_roi


def process_frame(frame, frame_number, video_path, video_label, video_source_path, templates, fps, csv_writer, scale_x, scale_y, left, top,
                  crop_width, crop_height, stats, runtime_state, score_candidates, metadata_writer):
    """Run the single-process initial scan and export supporting frames immediately."""
    process_frame_start = time.perf_counter()
    stage_start = time.perf_counter()
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
    gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    video_io.add_timing(stats, "initial_frame_prepare_s", stage_start)

    ignore_match = _match_ignore_frame_target(gray_image, templates, stats)
    ignore_timecode = frame_to_timecode(frame_number, fps)
    if not ignore_match["rejected_as_blank"]:
        csv_writer.writerow([video_source_path or os.path.basename(video_path), ignore_match["label"], frame_number, ignore_match["max_val"], ignore_timecode])
        if ignore_match["max_val"] > ignore_match["match_threshold"] and not np.isinf(ignore_match["max_val"]):
            video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
            return int(fps * ignore_match["skip_seconds"])

    for target in INITIAL_SCAN_TARGETS:
        if target["kind"] == "score":
            max_val, rejected_as_blank, score_layout_id = _match_score_target_layouts(upscaled_image, templates, stats)
        else:
            max_val, rejected_as_blank, _processed_roi = _match_initial_scan_target(gray_image, target, templates, stats)
            score_layout_id = ""
        timecode = frame_to_timecode(frame_number, fps)
        if rejected_as_blank:
            csv_writer.writerow([video_source_path or os.path.basename(video_path), target["label"], frame_number, 0, timecode])
            if target["kind"] == "race":
                return 0
            continue

        csv_writer.writerow([video_source_path or os.path.basename(video_path), target["label"], frame_number, max_val, timecode])
        if max_val <= target["match_threshold"] or np.isinf(max_val):
            continue

        if target["kind"] == "score":
            score_candidates.append(
                {
                    "race_number": runtime_state["next_race_number"],
                    "frame_number": frame_number,
                    "score_layout_id": score_layout_id or DEFAULT_SCORE_LAYOUT_ID,
                    }
            )
            runtime_state["next_race_number"] += 1
            video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
            return int(fps * target["skip_seconds"])

        if target["kind"] == "track":
            if runtime_state["last_track_frame"] < max(1, frame_number - int(fps * 20)):
                stage_start = time.perf_counter()
                save_frame_number = frame_number + int(fps * 1)
                video_io.seek_to_frame(runtime_state["capture"], save_frame_number, stats)
                ret, save_frame = video_io.read_video_frame(runtime_state["capture"], stats)
                if not ret:
                    break
                actual_track_frame = video_io.actual_frame_after_read(runtime_state["capture"])
                saved_image = crop_and_upscale_image(save_frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
                runtime_state["last_track_frame"] = frame_number
                race_number = runtime_state["next_race_number"]
                LOGGER.log("", _color_detection_message(video_label, race_number, " | track screen found at source ", timecode))
                frame_filename = race_anchor_frame_path(video_label, race_number, "0TrackName")
                write_export_image(frame_filename, saved_image)
                video_io.log_exported_frame(
                    metadata_writer,
                    video_path,
                    race_number,
                    "TrackName",
                    save_frame_number,
                    actual_track_frame,
                    fps,
                    frame_to_timecode,
                    video_source_path=video_source_path,
                )
                video_io.seek_to_frame(runtime_state["capture"], frame_number, stats)
                ret, _frame = video_io.read_video_frame(runtime_state["capture"], stats)
                if not ret:
                    break
                video_io.add_timing(stats, "output_frame_capture_s", stage_start)
            video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
            return 0

        if target["kind"] == "race":
            if runtime_state["last_race_frame"] < max(1, frame_number - int(fps * 20)):
                runtime_state["last_race_frame"] = frame_number
                race_number = runtime_state["next_race_number"]
                LOGGER.log("", _color_detection_message(video_label, race_number, " | race number found at source ", timecode))
                frame_filename = race_anchor_frame_path(video_label, race_number, "1RaceNumber")
                write_export_image(frame_filename, upscaled_image)
                video_io.log_exported_frame(
                    metadata_writer,
                    video_path,
                    race_number,
                    "RaceNumber",
                    frame_number,
                    frame_number,
                    fps,
                    frame_to_timecode,
                    video_source_path=video_source_path,
                )
            video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
            return int(fps * target["skip_seconds"])

    video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
    return 0


def process_segment_frame(frame, frame_number, video_path, video_source_path, templates, fps, scale_x, scale_y, left, top,
                          crop_width, crop_height, stats, state, debug_rows, emit_results, progress_queue=None,
                          segment_index=None, video_label=None):
    """Run the worker-safe initial scan without writing files or mutating shared state."""
    process_frame_start = time.perf_counter()
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
    gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    stats["initial_frame_prepare_s"] += time.perf_counter() - process_frame_start

    ignore_match = _match_ignore_frame_target(gray_image, templates, stats)
    if emit_results and not ignore_match["rejected_as_blank"]:
        timecode = frame_to_timecode(frame_number, fps)
        debug_rows.append([video_source_path or os.path.basename(video_path), ignore_match["label"], frame_number, ignore_match["max_val"], timecode])
    if not ignore_match["rejected_as_blank"] and ignore_match["max_val"] > ignore_match["match_threshold"] and not np.isinf(ignore_match["max_val"]):
        video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
        return int(fps * ignore_match["skip_seconds"])

    for target in INITIAL_SCAN_TARGETS:
        if target["kind"] == "score":
            max_val, rejected_as_blank, score_layout_id = _match_score_target_layouts(upscaled_image, templates, stats)
        else:
            max_val, rejected_as_blank, _processed_roi = _match_initial_scan_target(gray_image, target, templates, stats)
            score_layout_id = ""
        timecode = frame_to_timecode(frame_number, fps) if emit_results else ""
        if emit_results:
            debug_rows.append([video_source_path or os.path.basename(video_path), target["label"], frame_number, 0 if rejected_as_blank else max_val, timecode])
        if rejected_as_blank:
            if target["kind"] == "race":
                video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
                return 0
            continue

        if max_val <= target["match_threshold"] or np.isinf(max_val):
            continue

        if target["kind"] == "score":
            if emit_results:
                state["score_detections"].append(
                    {
                        "frame_number": frame_number,
                        "confidence": max_val,
                        "score_layout_id": score_layout_id or DEFAULT_SCORE_LAYOUT_ID,
                    }
                )
                update_segment_detection(progress_queue, segment_index, "score", frame_number, video_label)
            video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
            return int(fps * target["skip_seconds"])

        if target["kind"] == "track":
            if state["last_track_frame"] < max(1, frame_number - int(fps * 20)):
                state["last_track_frame"] = frame_number
                if emit_results:
                    save_frame_number = frame_number + int(fps * 1)
                    saved_image = None
                    if state["capture"] is not None:
                        video_io.seek_to_frame(state["capture"], save_frame_number, stats)
                        ret, save_frame = video_io.read_video_frame(state["capture"], stats)
                        if ret:
                            saved_image = crop_and_upscale_image(
                                save_frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT
                            )
                        video_io.seek_to_frame(state["capture"], frame_number, stats)
                        video_io.read_video_frame(state["capture"], stats)
                    state["track_detections"].append(
                        {
                            "frame_number": frame_number,
                            "save_frame_number": save_frame_number,
                            "confidence": max_val,
                            "saved_image": saved_image,
                        }
                    )
                    update_segment_detection(progress_queue, segment_index, "track", frame_number, video_label)
            video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
            return 0

        if target["kind"] == "race":
            if state["last_race_frame"] < max(1, frame_number - int(fps * 20)):
                state["last_race_frame"] = frame_number
                if emit_results:
                    state["race_detections"].append(
                        {
                            "frame_number": frame_number,
                            "save_frame_number": frame_number,
                            "confidence": max_val,
                            "saved_image": upscaled_image.copy(),
                        }
                    )
                    update_segment_detection(progress_queue, segment_index, "race", frame_number, video_label)
            video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
            return int(fps * target["skip_seconds"])

    video_io.add_timing(stats, "process_frame_total_s", process_frame_start)
    return 0


def scan_detection_segment(task):
    """Scan one overlapped initial-scan segment in isolation for safe parallel work."""
    video_path = task["video_path"]
    fps = task["fps"]
    scan_start = task["scan_start"]
    scan_end = task["scan_end"]
    emit_start = task["emit_start"]
    emit_end = task["emit_end"]
    include_debug_rows = task["include_debug_rows"]
    progress_queue = task.get("progress_queue")
    video_label = task.get("video_label")

    local_cap = cv2.VideoCapture(video_path)
    frame_skip = int(3 * max(1, int(fps)))
    stats = defaultdict(float)
    state = {
        "last_track_frame": 0,
        "last_race_frame": 0,
        "score_detections": [],
        "track_detections": [],
        "race_detections": [],
        "capture": local_cap,
    }
    debug_rows = []

    if not local_cap.isOpened():
        return {
            "segment_index": task["segment_index"],
            "stats": stats,
            "debug_rows": [],
            "score_detections": [],
            "track_detections": [],
            "race_detections": [],
        }

    video_io.seek_to_frame(local_cap, scan_start, stats)
    frame_count = scan_start
    last_progress_report = time.perf_counter()
    update_segment_progress(progress_queue, task["segment_index"], frame_count, emit_start, emit_end, force=True, video_label=video_label)

    while local_cap.isOpened() and frame_count < scan_end:
        window_interrupted = False
        for _ in range(task["window_steps"]):
            ret, frame = video_io.read_video_frame(local_cap, stats)
            if not ret:
                window_interrupted = True
                break

            emit_results = emit_start <= frame_count < emit_end
            frames_to_skip = process_segment_frame(
                frame,
                frame_count,
                video_path,
                task.get("video_source_path"),
                task["templates"],
                fps,
                task["scale_x"],
                task["scale_y"],
                task["left"],
                task["top"],
                task["crop_width"],
                task["crop_height"],
                stats,
                state,
                debug_rows,
                emit_results and include_debug_rows,
                progress_queue=progress_queue,
                segment_index=task["segment_index"],
                video_label=video_label,
            )

            if frames_to_skip > 0:
                frame_count += frames_to_skip + frame_skip
                if frame_count < scan_end:
                    video_io.seek_to_frame(local_cap, frame_count, stats)
                window_interrupted = True
                break

            if not video_io.advance_frames_by_grab(local_cap, frame_skip - 1, stats):
                frame_count = scan_end
                window_interrupted = True
                break

            frame_count += frame_skip
            if frame_count >= scan_end:
                window_interrupted = True
                break

            now = time.perf_counter()
            if now - last_progress_report >= task["progress_report_seconds"]:
                update_segment_progress(progress_queue, task["segment_index"], frame_count, emit_start, emit_end, video_label=video_label)
                last_progress_report = now

        if window_interrupted and frame_count >= scan_end:
            break

    local_cap.release()
    update_segment_progress(progress_queue, task["segment_index"], emit_end, emit_start, emit_end, force=True, video_label=video_label)
    return {
        "segment_index": task["segment_index"],
        "stats": dict(stats),
        "debug_rows": debug_rows,
        "score_detections": state["score_detections"],
        "track_detections": state["track_detections"],
        "race_detections": state["race_detections"],
    }


def choose_detection_segment_count(total_frames, requested_workers, minimum_segment_frames):
    """Only split the video when each worker gets a meaningful chunk of work."""
    if requested_workers <= 1 or total_frames < minimum_segment_frames:
        return 1
    segment_count = int(total_frames // minimum_segment_frames)
    return max(1, min(requested_workers, segment_count if segment_count > 0 else 1))


def build_detection_segment_tasks(video_path, video_label, video_source_path, total_frames, fps, templates, scale_x, scale_y, left, top, crop_width,
                                  crop_height, requested_workers, include_debug_rows, overlap_frames,
                                  minimum_segment_frames, window_steps, progress_report_seconds):
    """Build overlapped scan segments but keep output ownership non-overlapping."""
    segment_count = choose_detection_segment_count(total_frames, requested_workers, minimum_segment_frames)
    if segment_count == 1:
        return []

    segment_size = (total_frames + segment_count - 1) // segment_count
    tasks = []
    for segment_index in range(segment_count):
        emit_start = segment_index * segment_size
        emit_end = min(total_frames, emit_start + segment_size)
        scan_start = max(0, emit_start - overlap_frames)
        scan_end = min(total_frames, emit_end + overlap_frames)
        tasks.append(
            {
                "segment_index": segment_index,
                "video_path": video_path,
                "video_label": video_label,
                "video_source_path": video_source_path,
                "fps": fps,
                "scan_start": scan_start,
                "scan_end": scan_end,
                "emit_start": emit_start,
                "emit_end": emit_end,
                "templates": templates,
                "scale_x": scale_x,
                "scale_y": scale_y,
                "left": left,
                "top": top,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "include_debug_rows": include_debug_rows,
                "window_steps": window_steps,
                "progress_report_seconds": progress_report_seconds,
            }
        )
    return tasks


def merge_nearby_detections(detections, min_gap_frames):
    """Keep only the strongest detection inside each overlap window."""
    if not detections:
        return []

    merged = []
    for detection in sorted(detections, key=lambda item: item["frame_number"]):
        if not merged:
            merged.append(detection)
            continue
        previous = merged[-1]
        if detection["frame_number"] - previous["frame_number"] <= min_gap_frames:
            if detection["confidence"] > previous["confidence"]:
                merged[-1] = detection
            continue
        merged.append(detection)
    return merged


def assign_race_number(frame_number, score_frame_numbers):
    """Attach supporting frames to the next score screen that follows them."""
    return bisect_left(score_frame_numbers, frame_number) + 1


def save_auxiliary_detection_frames(capture, video_path, video_label, video_source_path, detections, score_frame_numbers, left, top,
                                    crop_width, crop_height, fps, stats, metadata_writer):
    """Persist merged track-name and race-number frames after worker results are combined."""
    if not detections:
        return

    for detection in detections:
        race_number = assign_race_number(detection["frame_number"], score_frame_numbers)
        stage_start = time.perf_counter()
        if detection.get("saved_image") is not None:
            upscaled_image = detection["saved_image"]
            actual_saved_frame = detection["save_frame_number"]
        else:
            video_io.seek_to_frame(capture, detection["save_frame_number"], stats)
            ret, frame = video_io.read_video_frame(capture, stats)
            if not ret:
                continue
            actual_saved_frame = video_io.actual_frame_after_read(capture)
            upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)

        if detection["kind"] == "track":
            timecode = frame_to_timecode(detection["frame_number"], fps)
            LOGGER.log("", _color_detection_message(video_label, race_number, " | track screen found at source ", timecode))
            suffix = "0TrackName"
            kind = "TrackName"
        else:
            timecode = frame_to_timecode(detection["frame_number"], fps)
            LOGGER.log("", _color_detection_message(video_label, race_number, " | race number found at source ", timecode))
            suffix = "1RaceNumber"
            kind = "RaceNumber"

        frame_filename = race_anchor_frame_path(video_label, race_number, suffix)
        write_export_image(frame_filename, upscaled_image)
        video_io.log_exported_frame(
            metadata_writer,
            video_path,
            race_number,
            kind,
            detection["save_frame_number"],
            actual_saved_frame,
            fps,
            frame_to_timecode,
            video_source_path=video_source_path,
        )
        video_io.add_timing(stats, "output_frame_capture_s", stage_start)


def run_parallel_detection_segments(segment_tasks, progress=None, diagnostics=None):
    """Prefer processes for the CPU-bound initial scan, then fall back to threads if needed."""
    progress_video_label = str(segment_tasks[0].get("video_label", "")) if segment_tasks else ""

    def _drain_progress(progress_queue, segment_state, live_counts, live_state):
        changed = False
        detection_changed = False
        while True:
            try:
                message = progress_queue.get_nowait()
            except Empty:
                break
            message_type = str(message.get("type", "progress"))
            if message_type == "detection":
                kind = str(message.get("kind", "")).strip().lower()
                if kind in live_counts:
                    live_counts[kind] += 1
                    detection_changed = True
                continue
            segment_state[message["segment_index"]] = (
                message["completed_frames"],
                message["total_frames"],
            )
            changed = True
        if changed and progress is not None:
            completed_frames = sum(item[0] for item in segment_state.values())
            total_frames = sum(item[1] for item in segment_state.values())
            progress.total_units = max(1, total_frames)
            detail = _color_live_detection_detail(progress_video_label, live_counts['score'], live_counts['track'], live_counts['race'])
            progress.update(completed_frames, detail, value_color_token=progress_video_label)
        elif detection_changed and progress is not None:
            now = time.perf_counter()
            if now - float(live_state["last_detection_log_time"]) >= 2.0:
                completed_frames = sum(item[0] for item in segment_state.values())
                total_frames = sum(item[1] for item in segment_state.values())
                progress.total_units = max(1, total_frames)
                detail = _color_live_detection_detail(progress_video_label, live_counts['score'], live_counts['track'], live_counts['race'])
                progress.update(completed_frames, detail, force=True, value_color_token=progress_video_label)
                live_state["last_detection_log_time"] = now

    def _run_with_executor(executor_factory, progress_queue, executor_label):
        run_start = time.perf_counter()
        segment_state = {task["segment_index"]: (0, max(1, task["emit_end"] - task["emit_start"])) for task in segment_tasks}
        live_counts = {"score": 0, "track": 0, "race": 0}
        live_state = {"last_detection_log_time": 0.0}
        task_payloads = [{**task, "progress_queue": progress_queue} for task in segment_tasks]
        first_result_elapsed_s = None
        startup_elapsed_s = None
        with executor_factory(max_workers=len(segment_tasks)) as executor:
            submit_start = time.perf_counter()
            pending = {
                executor.submit(scan_detection_segment, task): task
                for task in task_payloads
            }
            startup_elapsed_s = time.perf_counter() - submit_start
            results = []
            while pending:
                done, _ = wait(pending.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
                _drain_progress(progress_queue, segment_state, live_counts, live_state)
                for future in done:
                    pending.pop(future)
                    if first_result_elapsed_s is None:
                        first_result_elapsed_s = time.perf_counter() - run_start
                    results.append(future.result())
                    if progress is not None:
                        _drain_progress(progress_queue, segment_state, live_counts, live_state)
            _drain_progress(progress_queue, segment_state, live_counts, live_state)
            if diagnostics is not None:
                diagnostics.update(
                    {
                        "executor": executor_label,
                        "segment_count": len(segment_tasks),
                        "submit_startup_s": float(startup_elapsed_s or 0.0),
                        "first_result_s": float(first_result_elapsed_s or 0.0),
                        "parallel_wait_s": float(time.perf_counter() - run_start),
                        "live_score_detections": int(live_counts["score"]),
                        "live_track_detections": int(live_counts["track"]),
                        "live_race_detections": int(live_counts["race"]),
                    }
                )
            return results

    try:
        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            return _run_with_executor(ProcessPoolExecutor, progress_queue, "process")
    except (PermissionError, OSError):
        LOGGER.log("[Extract - Settings]", "Process pool unavailable, falling back to threads", color_name="yellow")
        progress_queue = Queue()
        if diagnostics is not None:
            diagnostics["fallback"] = "thread"
        return _run_with_executor(ThreadPoolExecutor, progress_queue, "thread")


def run_parallel_detection_segments_shared(segment_tasks, progress_by_video, diagnostics_by_video=None, max_workers=None, total_video_count=None):
    """Run detection segments from multiple videos in one shared executor."""
    if not segment_tasks:
        return {}

    status_state = {"last_print_s": 0.0}

    def _drain_progress(progress_queue, segment_state, live_counts, live_state, segment_totals):
        changed_videos = set()
        detection_changed_videos = set()
        while True:
            try:
                message = progress_queue.get_nowait()
            except Empty:
                break
            message_type = str(message.get("type", "progress"))
            video_label = str(message.get("video_label", ""))
            segment_key = (video_label, int(message.get("segment_index", -1)))
            if message_type == "detection":
                kind = str(message.get("kind", "")).strip().lower()
                if video_label in live_counts and kind in live_counts[video_label]:
                    live_counts[video_label][kind] += 1
                    detection_changed_videos.add(video_label)
                continue
            segment_state[segment_key] = (
                int(message.get("completed_frames", 0)),
                int(message.get("total_frames", 0)),
            )
            changed_videos.add(video_label)

        update_videos = changed_videos | detection_changed_videos
        if update_videos:
            now = time.perf_counter()
            if now - float(status_state["last_print_s"]) >= 2.0:
                active_videos = 0
                for video_label, video_segment_keys in segment_totals.items():
                    completed_frames = sum(segment_state.get(key, (0, 0))[0] for key in video_segment_keys)
                    total_frames = sum(segment_state.get(key, (0, 0))[1] for key in video_segment_keys)
                    if completed_frames < max(1, total_frames):
                        active_videos += 1
                LOGGER.log(
                    "[Extract - Scan Status]",
                    f"Active videos: {active_videos}/{int(total_video_count or len(progress_by_video))} | {LOGGER.resource_text()}",
                )
                status_state["last_print_s"] = now
        for video_label in update_videos:
            progress = progress_by_video.get(video_label)
            if progress is None:
                continue
            video_segment_keys = segment_totals.get(video_label, [])
            completed_frames = sum(segment_state.get(key, (0, 0))[0] for key in video_segment_keys)
            total_frames = sum(segment_state.get(key, (0, 0))[1] for key in video_segment_keys)
            progress.total_units = max(1, total_frames)
            counts = live_counts.get(video_label, {"score": 0, "track": 0, "race": 0})
            detail = _color_live_detection_detail(video_label, counts['score'], counts['track'], counts['race'])
            force = False
            if video_label in detection_changed_videos and video_label not in changed_videos:
                now = time.perf_counter()
                if now - float(live_state.get(video_label, 0.0)) >= 2.0:
                    force = True
                    live_state[video_label] = now
            progress.update(completed_frames, detail, force=force, value_color_token=video_label)

    def _run_with_executor(executor_factory, progress_queue, executor_label):
        run_start = time.perf_counter()
        task_payloads = [{**task, "progress_queue": progress_queue} for task in segment_tasks]
        if max_workers is None:
            executor_workers = len(task_payloads)
        else:
            executor_workers = max(1, min(int(max_workers), len(task_payloads)))
        segment_state = {
            (str(task["video_label"]), int(task["segment_index"])): (0, max(1, int(task["emit_end"]) - int(task["emit_start"])))
            for task in task_payloads
        }
        segment_totals = defaultdict(list)
        live_counts = {}
        live_state = {}
        results_by_video = defaultdict(list)
        first_result_s_by_video = {}
        pending_segments_by_video = defaultdict(int)
        completed_s_by_video = {}
        for task in task_payloads:
            video_label = str(task["video_label"])
            segment_totals[video_label].append((video_label, int(task["segment_index"])))
            live_counts.setdefault(video_label, {"score": 0, "track": 0, "race": 0})
            live_state.setdefault(video_label, 0.0)
            pending_segments_by_video[video_label] += 1

        with executor_factory(max_workers=executor_workers) as executor:
            submit_start = time.perf_counter()
            pending = {
                executor.submit(scan_detection_segment, task): task
                for task in task_payloads
            }
            submit_startup_s = time.perf_counter() - submit_start
            while pending:
                done, _ = wait(pending.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
                _drain_progress(progress_queue, segment_state, live_counts, live_state, segment_totals)
                for future in done:
                    task = pending.pop(future)
                    video_label = str(task["video_label"])
                    if video_label not in first_result_s_by_video:
                        first_result_s_by_video[video_label] = time.perf_counter() - run_start
                    results_by_video[video_label].append(future.result())
                    pending_segments_by_video[video_label] -= 1
                    if pending_segments_by_video[video_label] <= 0 and video_label not in completed_s_by_video:
                        completed_s_by_video[video_label] = time.perf_counter() - run_start
                    _drain_progress(progress_queue, segment_state, live_counts, live_state, segment_totals)
            _drain_progress(progress_queue, segment_state, live_counts, live_state, segment_totals)

        if diagnostics_by_video is not None:
            total_elapsed_s = time.perf_counter() - run_start
            for video_label, counts in live_counts.items():
                diagnostics_by_video.setdefault(video_label, {})
                diagnostics_by_video[video_label].update(
                    {
                        "executor": executor_label,
                        "submit_startup_s": float(submit_startup_s),
                        "first_result_s": float(first_result_s_by_video.get(video_label, 0.0)),
                        "parallel_wait_s": float(total_elapsed_s),
                        "video_complete_s": float(completed_s_by_video.get(video_label, total_elapsed_s)),
                        "live_score_detections": int(counts["score"]),
                        "live_track_detections": int(counts["track"]),
                        "live_race_detections": int(counts["race"]),
                        "segment_count": int(len(segment_totals.get(video_label, []))),
                    }
                )
        return results_by_video

    try:
        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            return _run_with_executor(ProcessPoolExecutor, progress_queue, "process")
    except (PermissionError, OSError):
        LOGGER.log("[Extract - Settings]", "Shared process pool unavailable, falling back to threads", color_name="yellow")
        progress_queue = Queue()
        if diagnostics_by_video is not None:
            for diagnostics in diagnostics_by_video.values():
                diagnostics["fallback"] = "thread"
        return _run_with_executor(ThreadPoolExecutor, progress_queue, "thread")
def _color_detection_message(video_label, race_number: int, description: str, timecode: str) -> str:
    return (
        "Race "
        + LOGGER.video_value(f"{race_number:03}", video_label)
        + description
        + LOGGER.video_value(timecode, video_label)
    )


def _color_live_detection_detail(video_label, score_count: int, track_count: int, race_count: int) -> str:
    return (
        "Live detections: score "
        + LOGGER.video_value(f"{int(score_count):2d}", video_label)
        + " | track "
        + LOGGER.video_value(f"{int(track_count):2d}", video_label)
        + " | race "
        + LOGGER.video_value(f"{int(race_count):2d}", video_label)
    )
