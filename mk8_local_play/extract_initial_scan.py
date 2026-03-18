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
from .project_paths import PROJECT_ROOT
from .score_layouts import DEFAULT_SCORE_LAYOUT_ID, all_score_layouts
from . import extract_video_io as video_io


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
    },
    {
        "kind": "race",
        "label": "RaceNumber",
        "match_threshold": 0.6,
        "skip_seconds": 60,
        "roi": (640, 590, 144, 48),
    },
)


def _match_score_target_layouts(gray_image, templates, stats):
    """Evaluate the score anchor against both supported score layouts."""
    template_binary, alpha_mask = templates[0]
    best_match = {
        "max_val": 0.0,
        "layout_id": DEFAULT_SCORE_LAYOUT_ID,
        "rejected_as_blank": False,
    }
    for layout in all_score_layouts():
        roi_x, roi_y, roi_width, roi_height = _expanded_roi(gray_image, layout.score_anchor_roi)
        roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        stage_start = time.perf_counter()
        processed_roi = preprocess_roi(roi, 0)
        video_io.add_timing(stats, "initial_roi_preprocess_s", stage_start)

        black_pixel_percentage = np.mean(processed_roi == 0)
        if black_pixel_percentage >= 0.97:
            if not best_match["rejected_as_blank"]:
                best_match["rejected_as_blank"] = True
            continue

        if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
            processed_roi = cv2.resize(
                processed_roi,
                (max(template_binary.shape[1], processed_roi.shape[1]), max(template_binary.shape[0], processed_roi.shape[0])),
                interpolation=cv2.INTER_LINEAR,
            )

        stage_start = time.perf_counter()
        max_val = match_template(processed_roi, template_binary, alpha_mask)
        video_io.add_timing(stats, "initial_match_s", stage_start)
        if max_val > best_match["max_val"]:
            best_match = {
                "max_val": float(max_val),
                "layout_id": layout.layout_id,
                "rejected_as_blank": False,
            }
    return best_match["max_val"], bool(best_match["rejected_as_blank"]), str(best_match["layout_id"])


def update_segment_progress(progress_queue, segment_index, frame_number, emit_start, emit_end, force=False):
    """Send best-effort progress from worker segments back to the parent process."""
    if progress_queue is None:
        return
    completed = max(0, min(max(frame_number, emit_start), emit_end) - emit_start)
    total = max(1, emit_end - emit_start)
    progress_queue.put(
        {
            "segment_index": segment_index,
            "completed_frames": completed,
            "total_frames": total,
            "force": force,
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


def _match_initial_scan_target(gray_image, target, templates, stats):
    """Prepare, filter, and match one anchor target inside the current frame."""
    roi_x, roi_y, roi_width, roi_height = _expanded_roi(gray_image, target["roi"])
    roi = gray_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    stage_start = time.perf_counter()
    processed_roi = preprocess_roi(roi, INITIAL_SCAN_TARGETS.index(target))
    video_io.add_timing(stats, "initial_roi_preprocess_s", stage_start)

    black_pixel_percentage = np.mean(processed_roi == 0)
    if black_pixel_percentage >= 0.97:
        return 0.0, True, processed_roi

    template_binary, alpha_mask = templates[INITIAL_SCAN_TARGETS.index(target)]
    if processed_roi.shape[0] < template_binary.shape[0] or processed_roi.shape[1] < template_binary.shape[1]:
        processed_roi = cv2.resize(
            processed_roi,
            (max(template_binary.shape[1], processed_roi.shape[1]), max(template_binary.shape[0], processed_roi.shape[0])),
            interpolation=cv2.INTER_LINEAR,
        )

    stage_start = time.perf_counter()
    max_val = match_template(processed_roi, template_binary, alpha_mask)
    video_io.add_timing(stats, "initial_match_s", stage_start)
    return max_val, False, processed_roi


def process_frame(frame, frame_number, video_path, video_label, video_source_path, templates, fps, csv_writer, scale_x, scale_y, left, top,
                  crop_width, crop_height, stats, runtime_state, score_candidates, metadata_writer):
    """Run the single-process initial scan and export supporting frames immediately."""
    process_frame_start = time.perf_counter()
    stage_start = time.perf_counter()
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
    gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    video_io.add_timing(stats, "initial_frame_prepare_s", stage_start)

    for target in INITIAL_SCAN_TARGETS:
        if target["kind"] == "score":
            max_val, rejected_as_blank, score_layout_id = _match_score_target_layouts(gray_image, templates, stats)
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
                LOGGER.log("", f"Race {race_number:03} | track screen found at source {timecode}", color_name="green")
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
                LOGGER.log("", f"Race {race_number:03} | race number found at source {timecode}", color_name="green")
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
                          crop_width, crop_height, stats, state, debug_rows, emit_results):
    """Run the worker-safe initial scan without writing files or mutating shared state."""
    process_frame_start = time.perf_counter()
    upscaled_image = crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT)
    gray_image = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    stats["initial_frame_prepare_s"] += time.perf_counter() - process_frame_start

    for target in INITIAL_SCAN_TARGETS:
        if target["kind"] == "score":
            max_val, rejected_as_blank, score_layout_id = _match_score_target_layouts(gray_image, templates, stats)
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
    update_segment_progress(progress_queue, task["segment_index"], frame_count, emit_start, emit_end, force=True)

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
                update_segment_progress(progress_queue, task["segment_index"], frame_count, emit_start, emit_end)
                last_progress_report = now

        if window_interrupted and frame_count >= scan_end:
            break

    local_cap.release()
    update_segment_progress(progress_queue, task["segment_index"], emit_end, emit_start, emit_end, force=True)
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
            LOGGER.log("", f"Race {race_number:03} | track screen found at source {timecode}", color_name="green")
            suffix = "0TrackName"
            kind = "TrackName"
        else:
            timecode = frame_to_timecode(detection["frame_number"], fps)
            LOGGER.log("", f"Race {race_number:03} | race number found at source {timecode}", color_name="green")
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


def run_parallel_detection_segments(segment_tasks, progress=None):
    """Prefer processes for the CPU-bound initial scan, then fall back to threads if needed."""
    def _drain_progress(progress_queue, segment_state):
        changed = False
        while True:
            try:
                message = progress_queue.get_nowait()
            except Empty:
                break
            segment_state[message["segment_index"]] = (
                message["completed_frames"],
                message["total_frames"],
            )
            changed = True
        if changed and progress is not None:
            completed_frames = sum(item[0] for item in segment_state.values())
            total_frames = sum(item[1] for item in segment_state.values())
            progress.total_units = max(1, total_frames)
            progress.update(completed_frames)

    def _run_with_executor(executor_factory, progress_queue):
        segment_state = {task["segment_index"]: (0, max(1, task["emit_end"] - task["emit_start"])) for task in segment_tasks}
        task_payloads = [{**task, "progress_queue": progress_queue} for task in segment_tasks]
        with executor_factory(max_workers=len(segment_tasks)) as executor:
            pending = {
                executor.submit(scan_detection_segment, task): task
                for task in task_payloads
            }
            results = []
            while pending:
                done, _ = wait(pending.keys(), timeout=0.5, return_when=FIRST_COMPLETED)
                _drain_progress(progress_queue, segment_state)
                for future in done:
                    pending.pop(future)
                    results.append(future.result())
                    if progress is not None:
                        _drain_progress(progress_queue, segment_state)
            _drain_progress(progress_queue, segment_state)
            return results

    try:
        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            return _run_with_executor(ProcessPoolExecutor, progress_queue)
    except (PermissionError, OSError):
        LOGGER.log("[Extract - Settings]", "Process pool unavailable, falling back to threads", color_name="yellow")
        progress_queue = Queue()
        return _run_with_executor(ThreadPoolExecutor, progress_queue)
