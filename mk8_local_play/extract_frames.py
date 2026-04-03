import cv2
import numpy as np
import os
import argparse
import csv
import time
import threading
import queue
from pathlib import Path
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from .app_runtime import detect_easyocr_runtime, effective_overlap_ocr_mode, load_app_config
from .console_logging import LOGGER
from .data_paths import resolve_asset_file
from .project_paths import PROJECT_ROOT
from . import extract_initial_scan as initial_scan
from . import extract_score_screen_selection as score_screen_selection
from . import extract_video_io as video_io
from .extract_common import (
    GPU_RUNTIME,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    build_video_identity,
    count_exported_detection_files,
    crop_and_upscale_image,
    determine_scaling,
    frame_to_timecode,
    load_videos_from_folder,
    relative_video_path,
)
from .ocr_common import load_exported_frame_metadata

# Record the start time
start_run_time = time.time()
APP_CONFIG = load_app_config()
EASYOCR_RUNTIME = detect_easyocr_runtime(APP_CONFIG)
INITIAL_SCAN_DIAGNOSTICS_ENABLED = os.environ.get("MK8_INITIAL_SCAN_DIAGNOSTICS", "0").strip().lower() in {"1", "true", "yes", "on"}

SCORE_ANALYSIS_WORKERS = APP_CONFIG.score_analysis_workers
PARALLEL_VIDEO_SCORE_WORKERS = max(1, int(APP_CONFIG.parallel_video_score_workers))
INITIAL_SCAN_WORKERS = APP_CONFIG.pass1_scan_workers
INITIAL_SCAN_WINDOW_STEPS = 2
INITIAL_SCAN_SEGMENT_OVERLAP_FRAMES = APP_CONFIG.pass1_segment_overlap_frames
INITIAL_SCAN_MIN_SEGMENT_FRAMES = APP_CONFIG.pass1_min_segment_frames
INITIAL_SCAN_PROGRESS_REPORT_SECONDS = 2.0
INITIAL_SCAN_EOF_GUARD_SECONDS = 10.0
INITIAL_SCAN_EOF_GUARD_PROGRESS = 0.99
INITIAL_SCAN_EOF_READ_TIMEOUT_SECONDS = 10.0
INITIAL_SCAN_FFPROBE_COUNTFRAME_MIN_DELTA_FRAMES = 30
DEFAULT_PARALLEL_VIDEO_SCAN_WORKERS = 2
DISABLE_STATIC_GALLERY_RACE_FILTER = os.environ.get("MK8_DISABLE_STATIC_GALLERY_RACE_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}
PARALLEL_VIDEO_SCAN_WORKERS = max(
    1,
    int(os.environ.get("MK8_PARALLEL_VIDEO_SCAN_WORKERS", str(DEFAULT_PARALLEL_VIDEO_SCAN_WORKERS))),
)

CONSENSUS_FRAME_CACHE = {}


def color_video_scope(scope: str, video_identity: object) -> str:
    return LOGGER.color_video_identity(scope, video_identity)


def color_video_detail(label: str, value: object, video_identity: object) -> str:
    return f"{label}{LOGGER.video_value(value, video_identity)}"


def color_video_message(parts: list[str | tuple[str, object]]) -> str:
    formatted_parts: list[str] = []
    for part in parts:
        if isinstance(part, tuple):
            value, video_identity = part
            formatted_parts.append(LOGGER.video_value(value, video_identity))
        else:
            formatted_parts.append(part)
    return "".join(formatted_parts)


def _format_total_score_progress_detail(video_label: str, completed_count: int, total_tasks: int, race_number: int) -> str:
    return (
        color_video_detail("races ", f"{completed_count:02}/{total_tasks:02}", video_label)
        + " | last R"
        + LOGGER.video_value(f"{int(race_number):03}", video_label)
    )


class NullCsvWriter:
    """Drop debug CSV rows when debug output is disabled."""

    def writerow(self, _row):
        return None


class MetadataCsvWriter:
    """Write exported frame metadata when enabled."""

    def __init__(self, csv_writer):
        self._csv_writer = csv_writer

    def writerow(self, row):
        self._csv_writer.writerow(row)


class ProgressPrinter:
    """Print throttled progress updates for long-running stages."""

    def __init__(
        self,
        scope,
        total_units,
        percent_step=5,
        min_interval_s=3.0,
        *,
        unit_formatter=None,
        include_resources=True,
    ):
        self.scope = scope
        self.total_units = max(1, int(total_units))
        self.percent_step = max(1, int(percent_step))
        self.min_interval_s = float(min_interval_s)
        self.unit_formatter = unit_formatter
        self.include_resources = bool(include_resources)
        self.last_percent = -1
        self.last_print_time = 0.0
        self.start_time = time.perf_counter()
        self.phase_peak = {
            "cpu_percent": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
            "gpu_percent": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
        }
        self.sample_count = 0
        self.cpu_percent_sum = 0.0
        self.ram_used_gb_sum = 0.0

    def update(self, completed_units, detail="", force=False, value_color_token=None):
        percent = min(100, int((max(0, completed_units) / self.total_units) * 100))
        now = time.perf_counter()
        should_print = force or percent >= 100 or self.last_percent < 0
        if not should_print and percent >= self.last_percent + self.percent_step:
            should_print = True
        if not should_print and now - self.last_print_time >= self.min_interval_s:
            should_print = True
        if not should_print:
            return
        snapshot = LOGGER.resources.sample()
        self._update_phase_peak(snapshot)
        percent_text = (
            LOGGER.video_value(f"{percent:3d}%", value_color_token)
            if value_color_token is not None else
            f"{percent:3d}%"
        )
        completed_value = (
            self.unit_formatter(min(completed_units, self.total_units), self.total_units)
            if self.unit_formatter is not None else
            f"{min(completed_units, self.total_units):,}/{self.total_units:,}"
        )
        completed_text = (
            LOGGER.video_value(completed_value, value_color_token)
            if value_color_token is not None else
            completed_value
        )
        message = f"{percent_text} | {completed_text}"
        if self.include_resources:
            resource_text = LOGGER.resource_text(snapshot, value_color_token=value_color_token)
            message = f"{message} | {resource_text}"
        detail_suffix = f" | {detail}" if detail else ""
        LOGGER.log(
            self.scope,
            f"{message}{detail_suffix}",
        )
        self.last_percent = percent
        self.last_print_time = now

    def _update_phase_peak(self, snapshot):
        self.sample_count += 1
        if snapshot.cpu_percent is not None:
            self.cpu_percent_sum += float(snapshot.cpu_percent)
        if snapshot.ram_used_gb is not None:
            self.ram_used_gb_sum += float(snapshot.ram_used_gb)
        for field in ("cpu_percent", "ram_used_gb", "gpu_percent", "vram_used_gb"):
            current = getattr(snapshot, field)
            peak_value = self.phase_peak[field]
            if current is not None and (peak_value is None or current > peak_value):
                self.phase_peak[field] = current
        for field in ("ram_total_gb", "vram_total_gb"):
            current = getattr(snapshot, field)
            if current is not None:
                self.phase_peak[field] = current

    def peak_lines(self):
        lines = []
        if self.sample_count > 0 and self.cpu_percent_sum > 0:
            lines.append(f"Avg CPU: {self.cpu_percent_sum / self.sample_count:.0f}%")
        if self.sample_count > 0 and self.ram_used_gb_sum > 0 and self.phase_peak["ram_total_gb"] is not None:
            lines.append(
                f"Avg RAM: {self.ram_used_gb_sum / self.sample_count:.1f} / {self.phase_peak['ram_total_gb']:.1f} GB"
            )
        if self.phase_peak["cpu_percent"] is not None:
            lines.append(f"Peak CPU: {self.phase_peak['cpu_percent']:.0f}%")
        if self.phase_peak["ram_used_gb"] is not None and self.phase_peak["ram_total_gb"] is not None:
            lines.append(f"Peak RAM: {self.phase_peak['ram_used_gb']:.1f} / {self.phase_peak['ram_total_gb']:.1f} GB")
        if self.phase_peak["gpu_percent"] is not None:
            lines.append(f"Peak GPU: {self.phase_peak['gpu_percent']:.0f}%")
        if self.phase_peak["vram_used_gb"] is not None and self.phase_peak["vram_total_gb"] is not None:
            lines.append(f"Peak VRAM: {self.phase_peak['vram_used_gb']:.1f} / {self.phase_peak['vram_total_gb']:.1f} GB")
        return lines


def format_duration(seconds):
    """Format seconds as HH:MM:SS for status messages."""
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def format_scan_time_progress(completed_frames: int, total_frames: int, fps: float) -> str:
    return f"{frame_to_timecode(completed_frames, fps)} / {frame_to_timecode(total_frames, fps)}"


def _assign_display_video_indices(prepared_contexts):
    total_videos = len(prepared_contexts)
    parallel_contexts = [context for context in prepared_contexts if context["detection_segment_tasks"]]
    serial_contexts = [context for context in prepared_contexts if not context["detection_segment_tasks"]]
    ordered_contexts = [*parallel_contexts, *serial_contexts]
    for display_index, context in enumerate(ordered_contexts, start=1):
        context["display_video_index"] = display_index
        context["display_total_videos"] = total_videos
    return parallel_contexts, serial_contexts


def build_workflow_video_plan(video_paths, folder_path, *, include_subfolders=False):
    total_videos = len(video_paths)
    plan_entries = []
    total_source_seconds = 0.0
    for input_index, video_path in enumerate(video_paths, start=1):
        video_label = build_video_identity(Path(video_path), input_root=folder_path, include_subfolders=include_subfolders)
        source_display_name = relative_video_path(Path(video_path), folder_path) if include_subfolders else os.path.basename(video_path)
        nominal_total_frames = 0
        nominal_fps = 1.0
        source_length_s = 0.0
        parallel_capable = False
        with video_io.suppress_native_stderr():
            probe = cv2.VideoCapture(str(video_path))
        try:
            if probe.isOpened():
                nominal_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                nominal_fps = probe.get(cv2.CAP_PROP_FPS) or 1.0
                source_length_s = nominal_total_frames / max(nominal_fps, 1.0)
                parallel_capable = (
                    initial_scan.choose_detection_segment_count(
                        nominal_total_frames,
                        INITIAL_SCAN_WORKERS,
                        INITIAL_SCAN_MIN_SEGMENT_FRAMES,
                    ) > 1
                )
        finally:
            probe.release()
        total_source_seconds += source_length_s
        plan_entries.append(
            {
                "video_path": video_path,
                "input_index": input_index,
                "display_video_index": input_index,
                "display_total_videos": total_videos,
                "video_label": video_label,
                "source_display_name": source_display_name,
                "source_length_s": source_length_s,
                "nominal_total_frames": nominal_total_frames,
                "nominal_fps": nominal_fps,
                "parallel_capable": parallel_capable,
            }
        )
    ordered_entries = sorted(
        plan_entries,
        key=lambda item: (0 if item["parallel_capable"] else 1, item["input_index"]),
    )
    for display_index, entry in enumerate(ordered_entries, start=1):
        entry["display_video_index"] = display_index
    return ordered_entries, total_source_seconds


def print_timing_summary(video_name, stats):
    """Print a user-facing summary of where time went."""
    total_time = float(stats.get("video_total_s", 0.0))
    if total_time <= 0:
        return

    major_buckets = [
        ("Video loading (seek/read/grab)", float(stats.get("seek_time_s", 0.0) + stats.get("read_time_s", 0.0) + stats.get("grab_time_s", 0.0))),
        ("Total Score analysis", float(stats.get("score_candidate_pass_s", 0.0))),
        ("Saving frame images", float(stats.get("output_frame_capture_s", 0.0))),
        ("Initial scan", float(stats.get("main_scan_loop_s", 0.0))),
        ("Calibration", float(stats.get("scaling_scan_s", 0.0))),
    ]
    ranked = [(label, value) for label, value in major_buckets if value > 0.0]
    ranked.sort(key=lambda item: item[1], reverse=True)

    lines = []
    if ranked:
        lines.append(f"Likely bottleneck: {ranked[0][0]}")
        lines.append("Biggest time sinks")
    for label, value in ranked[:3]:
        percent = (value / total_time) * 100 if total_time > 0 else 0.0
        lines.append(f"- {label}: {format_duration(value)} ({percent:.0f}%)")
    LOGGER.summary_block(f"[{video_name} - Time Summary]", lines, color_name="dim")


def _top_profiler_costs(stats, *, limit=4):
    candidates = [
        ("video grab", float(stats.get("grab_time_s", 0.0))),
        ("video read", float(stats.get("read_time_s", 0.0))),
        ("video seek", float(stats.get("seek_time_s", 0.0))),
        ("frame export", float(stats.get("output_frame_capture_s", 0.0))),
        ("flush save score frames", float(stats.get("score_flush_save_frames_s", 0.0))),
        ("save image writes", float(stats.get("score_save_image_write_s", 0.0))),
        ("flush refine/expand", float(stats.get("score_flush_refine_expand_s", 0.0))),
        ("static gallery check", float(stats.get("score_flush_static_gallery_check_s", 0.0))),
        ("initial scan", float(stats.get("main_scan_loop_s", 0.0))),
        ("score screen selection", float(stats.get("score_candidate_pass_s", 0.0))),
    ]
    ranked = [(label, seconds) for label, seconds in candidates if seconds > 0.0]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def _describe_video_bottleneck(stats) -> str:
    seek_read_grab = float(stats.get("seek_time_s", 0.0)) + float(stats.get("read_time_s", 0.0)) + float(stats.get("grab_time_s", 0.0))
    frame_export = float(stats.get("output_frame_capture_s", 0.0))
    save_frames = float(stats.get("score_flush_save_frames_s", 0.0))
    refine_expand = float(stats.get("score_flush_refine_expand_s", 0.0))
    score_pass = float(stats.get("score_candidate_pass_s", 0.0))
    if seek_read_grab >= max(frame_export + save_frames, refine_expand, score_pass * 0.35):
        return "Video loading and decoder seeks"
    if frame_export + save_frames >= max(seek_read_grab, refine_expand):
        return "Frame export and image writes"
    if refine_expand >= max(seek_read_grab, frame_export + save_frames):
        return "Result refinement"
    if score_pass > 0:
        return "Total Score analysis"
    return "Mixed extraction work"


def print_extract_profiler_summary(video_name, stats):
    """Print per-function/bucket call counts, total time, and average milliseconds."""
    profile_rows = [
        ("score selection match", "score_detail_match_score_s", "score_detail_score_match_calls"),
        ("flush save score frames", "score_flush_save_frames_s", None),
        ("flush refine/expand", "score_flush_refine_expand_s", None),
        ("flush callback", "score_flush_callback_s", None),
        ("flush static gallery check", "score_flush_static_gallery_check_s", None),
        ("flush visible row count", "score_flush_row_count_s", None),
        ("flush debug CSV write", "score_flush_debug_write_s", None),
        ("save image writes", "score_save_image_write_s", "score_save_image_writes"),
        ("save cleanup", "score_save_cleanup_s", "score_save_cleanup_runs"),
        ("save race anchor", "score_save_race_anchor_s", None),
        ("save total anchor", "score_save_total_anchor_s", None),
        ("scan 5/6 gate", "scan_gate_s", "scan_gate_calls"),
        ("scan gate row crops", "scan_gate_position_row_crop_s", "scan_gate_position_row_crop_calls"),
        ("scan score preprocess", "scan_score_preprocess_s", "scan_score_process_image_calls"),
        ("scan score metrics", "scan_score_metrics_s", "scan_score_position_metrics_calls"),
        ("detail score preprocess", "detail_score_preprocess_s", "detail_score_process_image_calls"),
        ("detail score metrics", "detail_score_metrics_s", "detail_score_position_metrics_calls"),
        ("initial score metrics", "initial_match_s", "score_position_metrics_calls"),
        ("score preprocess", "initial_roi_preprocess_s", "score_process_image_calls"),
        ("score prefix gate", "score_prefix_gate_s", "score_prefix_gate_calls"),
        ("gate row crops", "score_gate_position_row_crop_s", "score_gate_position_row_crop_calls"),
        ("position row crops", "score_position_row_crop_s", "score_position_row_crop_calls"),
        ("position metrics total", "score_position_metrics_total_s", "score_position_metrics_total_calls"),
        ("position metrics fast", "score_position_metrics_fast_s", "score_position_metrics_fast_calls"),
        ("position metrics slow", "score_position_metrics_slow_s", "score_position_metrics_slow_calls"),
        ("score preprocess internal", "score_process_image_total_s", "score_process_image_internal_calls"),
        ("score rewrite blocks", "score_process_image_block_rewrite_s", "score_process_image_internal_calls"),
        ("video seek", "seek_time_s", "seek_calls"),
        ("video read", "read_time_s", "read_calls"),
        ("video grab", "grab_time_s", "grab_calls"),
        ("score frame prepare", "score_detail_frame_prepare_s", "score_detail_frames"),
        ("12th template match", "score_detail_match_12th_s", None),
        ("12th preprocess", "score_detail_12th_preprocess_s", None),
        ("frame export", "output_frame_capture_s", None),
    ]
    profile_entries = []
    for label, seconds_key, calls_key in profile_rows:
        total_s = float(stats.get(seconds_key, 0.0))
        calls = int(stats.get(calls_key, 0)) if calls_key is not None else 0
        if total_s <= 0 and calls <= 0:
            continue
        avg_ms = (total_s / calls * 1000.0) if calls > 0 else 0.0
        if calls > 0:
            line = f"{label}: calls {calls:,} | total {total_s:.2f}s | avg {avg_ms:.1f} ms"
        else:
            line = f"{label}: total {total_s:.2f}s"
        profile_entries.append((label, total_s, calls, line))
    profile_entries.sort(key=lambda item: (-item[1], item[0]))
    lines = [item[3] for item in profile_entries]

    summary_lines = [f"Likely bottleneck: {_describe_video_bottleneck(stats)}"]
    top_costs = _top_profiler_costs(stats)
    if top_costs:
        summary_lines.append("Top measured costs")
        for label, seconds in top_costs:
            summary_lines.append(f"- {label}: {seconds:.2f}s")

    extra_lines = []
    if int(stats.get("score_layout_evaluation_calls", 0)) > 0:
        extra_lines.append(f"score layout evaluations: {int(stats.get('score_layout_evaluation_calls', 0)):,}")
    if int(stats.get("scan_score_layout_evaluation_calls", 0)) > 0 or int(stats.get("detail_score_layout_evaluation_calls", 0)) > 0:
        extra_lines.append(
            f"score layout evaluations by phase: "
            f"scan {int(stats.get('scan_score_layout_evaluation_calls', 0)):,} | "
            f"detail {int(stats.get('detail_score_layout_evaluation_calls', 0)):,}"
        )
    if int(stats.get("scan_gate_passes", 0)) > 0 or int(stats.get("scan_gate_calls", 0)) > 0:
        extra_lines.append(
            f"scan 5/6 gate passes: {int(stats.get('scan_gate_passes', 0)):,}/{int(stats.get('scan_gate_calls', 0)):,}"
        )
    if int(stats.get("scan_gate_template_checks", 0)) > 0:
        extra_lines.append(f"scan gate template checks: {int(stats.get('scan_gate_template_checks', 0)):,}")
    if int(stats.get("scan_gate_position_rows_requested", 0)) > 0:
        extra_lines.append(
            f"scan gate rows requested/extracted: "
            f"{int(stats.get('scan_gate_position_rows_requested', 0)):,}/"
            f"{int(stats.get('scan_gate_position_rows_extracted', 0)):,}"
        )
    if int(stats.get("score_prefix_gate_passes", 0)) > 0:
        extra_lines.append(f"score prefix gate passes: {int(stats.get('score_prefix_gate_passes', 0)):,}")
    if int(stats.get("score_gate_position_rows_requested", 0)) > 0:
        extra_lines.append(
            f"gate rows requested/extracted: "
            f"{int(stats.get('score_gate_position_rows_requested', 0)):,}/"
            f"{int(stats.get('score_gate_position_rows_extracted', 0)):,}"
        )
    if int(stats.get("score_position_rows_requested", 0)) > 0:
        extra_lines.append(
            f"position rows requested/extracted: "
            f"{int(stats.get('score_position_rows_requested', 0)):,}/"
            f"{int(stats.get('score_position_rows_extracted', 0)):,}"
        )
    if int(stats.get("score_position_metrics_rows_processed", 0)) > 0:
        extra_lines.append(
            f"position metric rows processed: {int(stats.get('score_position_metrics_rows_processed', 0)):,}"
        )
    if int(stats.get("score_position_metrics_template_candidates", 0)) > 0:
        extra_lines.append(
            f"position template candidates scored: "
            f"{int(stats.get('score_position_metrics_template_candidates', 0)):,}"
        )
    if int(stats.get("score_tie_aware_reuse_calls", 0)) > 0:
        extra_lines.append(f"tie-aware metric reuses: {int(stats.get('score_tie_aware_reuse_calls', 0)):,}")
    if int(stats.get("score_tie_aware_drop_checks", 0)) > 0:
        extra_lines.append(f"drop-window checks: {int(stats.get('score_tie_aware_drop_checks', 0)):,}")
    if int(stats.get("score_save_cleanup_removed", 0)) > 0:
        extra_lines.append(f"legacy bundle files removed: {int(stats.get('score_save_cleanup_removed', 0)):,}")
    if int(stats.get("score_capture_frame_events_total", 0)) > 0:
        extra_lines.append(
            f"score capture events/unique frames: "
            f"{int(stats.get('score_capture_frame_events_total', 0)):,}/"
            f"{int(stats.get('score_capture_unique_frames_total', 0)):,}"
        )
    if int(stats.get("score_capture_duplicate_frames_total", 0)) > 0:
        extra_lines.append(
            f"score capture overlap: "
            f"{int(stats.get('score_capture_duplicate_frames_total', 0)):,} duplicate frame reads "
            f"({float(stats.get('score_capture_duplicate_source_seconds_total', 0.0)):.2f}s source)"
        )
    if int(stats.get("score_capture_race_consensus_frames", 0)) > 0 or int(stats.get("score_capture_total_consensus_frames", 0)) > 0:
        extra_lines.append(
            f"captured score windows: "
            f"race consensus {int(stats.get('score_capture_race_consensus_frames', 0)):,} | "
            f"total consensus {int(stats.get('score_capture_total_consensus_frames', 0)):,} | "
            f"race context {int(stats.get('score_capture_points_context_frames', 0)):,}"
        )
    if int(stats.get("score_capture_race_anchor_frames", 0)) > 0 or int(stats.get("score_capture_total_anchor_frames", 0)) > 0:
        extra_lines.append(
            f"captured score anchors: "
            f"race {int(stats.get('score_capture_race_anchor_frames', 0)):,} | "
            f"total {int(stats.get('score_capture_total_anchor_frames', 0)):,} | "
            f"points {int(stats.get('score_capture_points_anchor_frames', 0)):,}"
        )
    if int(stats.get("score_same_run_ocr_frames_total", 0)) > 0:
        extra_lines.append(
            f"same-run OCR frame inputs: "
            f"{int(stats.get('score_same_run_ocr_frames_total', 0)):,} "
            f"({int(stats.get('score_same_run_ocr_unique_frames_total', 0)):,} unique)"
        )
    if int(stats.get("score_persisted_ocr_frames_total", 0)) > 0:
        extra_lines.append(
            f"persisted rerun OCR frame inputs: "
            f"{int(stats.get('score_persisted_ocr_frames_total', 0)):,} "
            f"({int(stats.get('score_persisted_ocr_unique_frames_total', 0)):,} unique)"
        )
    if int(stats.get("score_capture_frames_outside_same_run_cache_total", 0)) > 0:
        extra_lines.append(
            f"captured frames outside same-run in-memory OCR cache: "
            f"{int(stats.get('score_capture_frames_outside_same_run_cache_total', 0)):,} "
            f"({float(stats.get('score_capture_outside_same_run_cache_source_seconds_total', 0.0)):.2f}s source)"
        )
    if int(stats.get("position_calls", 0)) > 0:
        extra_lines.append(
            f"capture positioning: "
            f"{int(stats.get('position_calls', 0)):,} calls | "
            f"no-op {int(stats.get('position_noop_calls', 0)):,} | "
            f"grab-advance {int(stats.get('position_forward_grab_calls', 0)):,} "
            f"({int(stats.get('position_forward_grab_frames', 0)):,} frames) | "
            f"seek fallback {int(stats.get('position_seek_fallback_calls', 0)):,}"
        )
    if int(stats.get("seek_calls", 0)) > 0:
        extra_lines.append(
            f"seek profile: "
            f"forward {int(stats.get('seek_forward_calls', 0)):,} | "
            f"backward {int(stats.get('seek_backward_calls', 0)):,} | "
            f"short {int(stats.get('seek_short_calls', 0)):,} | "
            f"medium {int(stats.get('seek_medium_calls', 0)):,} | "
            f"long {int(stats.get('seek_long_calls', 0)):,} | "
            f"distance {int(stats.get('seek_frame_distance_total', 0)):,} frames"
        )
    labeled_seek_calls = {
        key.split("__", 1)[1]: int(value)
        for key, value in stats.items()
        if key.startswith("seek_calls__") and int(value) > 0
    }
    if labeled_seek_calls:
        top_seek_labels = sorted(labeled_seek_calls.items(), key=lambda item: (-item[1], item[0]))[:5]
        formatted_labels = []
        for label, call_count in top_seek_labels:
            backward_calls = int(stats.get(f"seek_backward_calls__{label}", 0))
            long_calls = int(stats.get(f"seek_long_calls__{label}", 0))
            distance = int(stats.get(f"seek_frame_distance_total__{label}", 0))
            formatted_labels.append(
                f"{label} {call_count}c/{backward_calls}b/{long_calls}l/{distance:,}f"
            )
        extra_lines.append("seek hotspots: " + " | ".join(formatted_labels))
    scheduler_strategy = stats.get("score_scheduler_strategy")
    if scheduler_strategy:
        extra_lines.append(f"score scheduler: {scheduler_strategy}")
    scheduler_layouts = stats.get("score_scheduler_worker_layouts")
    if scheduler_layouts:
        extra_lines.append("score worker layout: " + " | ".join(str(item) for item in list(scheduler_layouts)))
    scheduler_backends = stats.get("score_scheduler_worker_backends")
    if scheduler_backends:
        extra_lines.append("score worker backends: " + " | ".join(str(item) for item in list(scheduler_backends)))
    if int(stats.get("score_scheduler_capture_opens", 0)) > 0:
        extra_lines.append(
            f"score worker-local capture opens: {int(stats.get('score_scheduler_capture_opens', 0)):,}"
        )
    if int(stats.get("score_ready_results_max", 0)) > 1 or int(stats.get("score_out_of_order_results", 0)) > 0:
        extra_lines.append(
            f"parallel score result backlog: "
            f"max ready {int(stats.get('score_ready_results_max', 0)):,} | "
            f"out-of-order completions {int(stats.get('score_out_of_order_results', 0)):,}"
        )
    if int(stats.get("score_flush_io_lock_acquires", 0)) > 0:
        extra_lines.append(
            f"flush IO lock wait: {float(stats.get('score_flush_io_lock_wait_s', 0.0)):.2f}s "
            f"across {int(stats.get('score_flush_io_lock_acquires', 0)):,} acquires"
        )
    if int(stats.get("score_callback_io_lock_acquires", 0)) > 0:
        extra_lines.append(
            f"callback IO lock wait: {float(stats.get('score_callback_io_lock_wait_s', 0.0)):.2f}s "
            f"across {int(stats.get('score_callback_io_lock_acquires', 0)):,} acquires"
        )

    if not lines and not extra_lines:
        return
    LOGGER.summary_block(
        f"[{video_name} - Extract Profiler]",
        [*summary_lines, "", *lines, *extra_lines],
        color_name="dim",
    )


def _accumulate_result_stats(global_stats, result):
    accounted_stats = result.setdefault("_accounted_stats", {})
    current_stats = dict(result.get("stats") or {})
    for key, value in current_stats.items():
        current_value = float(value)
        accounted_value = float(accounted_stats.get(key, 0.0))
        delta = current_value - accounted_value
        if abs(delta) <= 1e-12:
            continue
        global_stats[key] += delta
        accounted_stats[key] = current_value


def _score_task_start_frame(task):
    fps = float(task.get("fps", 0) or 0)
    return max(0, int(task.get("frame_number", 0) or 0) - int(3 * fps))


def _score_task_estimated_cost(task):
    fps = max(1.0, float(task.get("fps", 0) or 0))
    analysis_window_frames = max(1, int(round(16.0 * fps)))
    return analysis_window_frames


def _partition_score_tasks_contiguous(tasks, worker_count):
    if not tasks:
        return []
    sorted_tasks = sorted(
        tasks,
        key=lambda task: (_score_task_start_frame(task), int(task.get("race_number", 0) or 0)),
    )
    worker_count = max(1, min(int(worker_count), len(sorted_tasks)))
    if worker_count == 1:
        return [sorted_tasks]

    weighted_tasks = []
    previous_start = None
    total_cost = 0.0
    for task in sorted_tasks:
        start_frame = _score_task_start_frame(task)
        base_cost = float(_score_task_estimated_cost(task))
        gap_cost = 0.0 if previous_start is None else max(0.0, float(start_frame - previous_start))
        task_cost = base_cost + gap_cost
        weighted_tasks.append((task, start_frame, task_cost))
        total_cost += task_cost
        previous_start = start_frame

    target_cost = total_cost / float(worker_count)
    blocks = []
    current_block = []
    current_cost = 0.0
    remaining_tasks = len(weighted_tasks)
    remaining_workers = worker_count
    for weighted_index, (task, _start_frame, task_cost) in enumerate(weighted_tasks):
        current_block.append(task)
        current_cost += task_cost
        remaining_tasks = len(weighted_tasks) - weighted_index - 1
        remaining_workers = worker_count - len(blocks) - 1
        must_split = remaining_workers > 0 and remaining_tasks >= remaining_workers
        if must_split and current_cost >= target_cost:
            blocks.append(current_block)
            current_block = []
            current_cost = 0.0
    if current_block:
        blocks.append(current_block)
    while len(blocks) < worker_count:
        blocks.append([])
    return [block for block in blocks if block]


def _chunk_score_tasks_contiguous(tasks, chunk_size):
    sorted_tasks = sorted(
        tasks,
        key=lambda task: (_score_task_start_frame(task), int(task.get("race_number", 0) or 0)),
    )
    chunk_size = max(1, int(chunk_size))
    return [
        sorted_tasks[index:index + chunk_size]
        for index in range(0, len(sorted_tasks), chunk_size)
    ]


def _score_worker_layout_text(block_index, tasks):
    race_numbers = [int(task.get("race_number", 0) or 0) for task in tasks]
    if not race_numbers:
        return f"w{block_index + 1}: none"
    start_frames = [_score_task_start_frame(task) for task in tasks]
    return (
        f"w{block_index + 1}: races {race_numbers[0]:03}-{race_numbers[-1]:03} "
        f"| span {min(start_frames):,}-{max(start_frames):,}"
    )


def _run_score_task_block(video_path, block_index, tasks, frame_to_timecode, result_queue):
    local_cap = cv2.VideoCapture(video_path)
    backend_name = ""
    if local_cap is not None and local_cap.isOpened():
        try:
            backend_name = str(local_cap.getBackendName() or "")
        except Exception:
            backend_name = ""
    for task in tasks:
        result = score_screen_selection.analyze_score_window_task(
            task,
            frame_to_timecode,
            capture=local_cap if local_cap is not None and local_cap.isOpened() else None,
        )
        result_queue.put(
            {
                "task": task,
                "result": result,
                "block_index": block_index,
            }
        )
    if local_cap is not None and local_cap.isOpened():
        local_cap.release()
    race_numbers = [int(task.get("race_number", 0) or 0) for task in tasks]
    start_frames = [_score_task_start_frame(task) for task in tasks]
    return {
        "block_index": int(block_index),
        "backend_name": backend_name,
        "race_numbers": race_numbers,
        "frame_start": min(start_frames) if start_frames else None,
        "frame_end": max(start_frames) if start_frames else None,
        "capture_opens": 1 if tasks else 0,
    }


def _run_score_task_chunk_worker(video_path, worker_id, chunk_queue, frame_to_timecode, result_queue):
    local_cap = cv2.VideoCapture(video_path)
    backend_name = ""
    if local_cap is not None and local_cap.isOpened():
        try:
            backend_name = str(local_cap.getBackendName() or "")
        except Exception:
            backend_name = ""
    processed_chunks = []
    processed_races = []
    while True:
        try:
            chunk = chunk_queue.get_nowait()
        except queue.Empty:
            break
        chunk_index = int(chunk["chunk_index"])
        chunk_tasks = list(chunk["tasks"])
        processed_chunks.append(chunk_index)
        processed_races.extend(int(task.get("race_number", 0) or 0) for task in chunk_tasks)
        for task in chunk_tasks:
            result = score_screen_selection.analyze_score_window_task(
                task,
                frame_to_timecode,
                capture=local_cap if local_cap is not None and local_cap.isOpened() else None,
            )
            result_queue.put(
                {
                    "task": task,
                    "result": result,
                    "worker_id": worker_id,
                    "chunk_index": chunk_index,
                }
            )
    if local_cap is not None and local_cap.isOpened():
        local_cap.release()
    return {
        "worker_id": int(worker_id),
        "backend_name": backend_name,
        "race_numbers": processed_races,
        "chunk_indices": processed_chunks,
        "capture_opens": 1 if processed_races else 0,
    }


def collect_consensus_frames(video_path, video_label, center_frame, fps, left, top, crop_width, crop_height, bundle_kind):
    """Collect nearby upscaled frames for in-memory OCR consensus during --all runs."""
    radius = max(0, APP_CONFIG.ocr_consensus_frames // 2)
    cache_key = (video_label, int(bundle_kind[0]), bundle_kind[1])
    local_cap = cv2.VideoCapture(video_path)
    if not local_cap.isOpened():
        return

    total_frames = int(local_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bundled_frames = []
    start_frame = max(0, center_frame - radius)
    end_frame = min(total_frames, center_frame + radius + 1)
    local_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_number in range(start_frame, end_frame):
        ret, frame = local_cap.read()
        if not ret:
            continue
        bundled_frames.append(crop_and_upscale_image(frame, left, top, crop_width, crop_height, TARGET_WIDTH, TARGET_HEIGHT))
    local_cap.release()
    if bundled_frames:
        CONSENSUS_FRAME_CACHE[cache_key] = bundled_frames


def process_score_candidates(video_path, video_label, video_source_path, score_candidates, templates, fps, csv_writer, scale_x, scale_y, left, top,
                             crop_width, crop_height, stats, metadata_writer, video_index=None, total_videos=None, progress=None,
                             per_race_complete_callback=None, analysis_workers_override=None, io_lock=None):
    """Second pass over recorded score candidates."""
    if not score_candidates:
        return

    stage_start = time.perf_counter()
    tasks = [
        {
            "video_path": video_path,
            "race_number": candidate["race_number"],
            "frame_number": candidate["frame_number"],
            "fps": fps,
            "templates": templates,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "left": left,
            "top": top,
            "crop_width": crop_width,
            "crop_height": crop_height,
            "ocr_consensus_frames": APP_CONFIG.ocr_consensus_frames,
            "score_layout_id": candidate.get("score_layout_id"),
        }
        for candidate in score_candidates
    ]

    worker_count = min(max(1, int(analysis_workers_override or SCORE_ANALYSIS_WORKERS)), len(tasks))
    local_progress = progress or ProgressPrinter(
        color_video_scope(f"[Video {video_index}/{total_videos} - Total Score]", video_label)
        if video_index is not None else "[Total Score]",
        len(tasks),
        percent_step=5,
        min_interval_s=2.0,
        include_resources=False,
    )

    def _acquire_optional_io_lock():
        if io_lock is None:
            return None
        lock_wait_start = time.perf_counter()
        io_lock.acquire()
        video_io.add_timing(stats, "score_flush_io_lock_wait_s", lock_wait_start)
        stats["score_flush_io_lock_acquires"] += 1
        return io_lock

    def _write_debug_rows(result):
        debug_write_stage_start = time.perf_counter()
        if io_lock is None:
            for row in result["debug_rows"]:
                csv_writer.writerow(row)
        else:
            lock_handle = _acquire_optional_io_lock()
            try:
                for row in result["debug_rows"]:
                    csv_writer.writerow(row)
            finally:
                lock_handle.release()
        video_io.add_timing(stats, "score_flush_debug_write_s", debug_write_stage_start)

    def _persist_saved_result(result, *, revision):
        race_number = int(result["candidate"].get("race_number", 0) or 0)
        if result["race_score_frame"] <= 0 or result["total_score_frame"] <= 0:
            return False
        static_gallery_stage_start = time.perf_counter()
        is_static_gallery_bundle, similarity = score_screen_selection.is_static_gallery_race_bundle(
            result.get("race_consensus_frames", [])
        )
        video_io.add_timing(stats, "score_flush_static_gallery_check_s", static_gallery_stage_start)
        if is_static_gallery_bundle and not DISABLE_STATIC_GALLERY_RACE_FILTER:
            similarity_text = (
                f"min {similarity['min']:.6f} | avg {similarity['avg']:.6f} | max {similarity['max']:.6f}"
                if similarity is not None
                else "no similarity summary"
            )
            LOGGER.log(
                "[Score Screen Filter]",
                (
                    f"Skipping Race {race_number:03} for {video_label}: "
                    f"RaceScore bundle is effectively static from first-frame comparison ({similarity_text})"
                ),
                color_name="yellow",
            )
            return False
        def _save_frames():
            return score_screen_selection.save_score_frames(
                video_path,
                video_label,
                result["candidate"]["race_number"],
                result["race_score_frame"],
                result["total_score_frame"],
                result.get("actual_race_score_frame"),
                result.get("actual_total_score_frame"),
                result.get("race_score_image"),
                result.get("total_score_image"),
                result.get("race_consensus_frames", []),
                result.get("total_consensus_frames", []),
                fps,
                metadata_writer,
                CONSENSUS_FRAME_CACHE,
                frame_to_timecode,
                video_source_path=video_source_path,
                score_layout_id=result["candidate"].get("score_layout_id"),
                actual_points_anchor_frame=result.get("actual_points_anchor_frame"),
                points_anchor_image=result.get("points_anchor_image"),
                points_context_frames=result.get("points_context_frames", []),
                stats=stats,
            )
        save_stage_start = time.perf_counter()
        if io_lock is None:
            saved = _save_frames()
        else:
            lock_handle = _acquire_optional_io_lock()
            try:
                saved = _save_frames()
            finally:
                lock_handle.release()
        video_io.add_timing(stats, "score_flush_save_frames_s", save_stage_start)
        if saved and per_race_complete_callback is not None:
            callback_stage_start = time.perf_counter()
            per_race_complete_callback(
                {
                    "video_label": video_label,
                    "race_number": int(result["candidate"]["race_number"]),
                    "ocr_revision": int(revision),
                }
            )
            video_io.add_timing(stats, "score_flush_callback_s", callback_stage_start)
        return bool(saved)

    def _prepare_result_for_save(result, expected_players):
        refine_stage_start = time.perf_counter()
        result = score_screen_selection.refine_race_score_result_for_expected_players(result, expected_players)
        result = score_screen_selection.expand_race_score_consensus_window(
            result,
            expected_players,
        )
        video_io.add_timing(stats, "score_flush_refine_expand_s", refine_stage_start)
        _accumulate_result_stats(stats, result)
        return result

    if worker_count == 1:
        immediate_expected_players = None
        for completed_count, task in enumerate(tasks, start=1):
            result = score_screen_selection.analyze_score_window_task(
                task,
                frame_to_timecode,
            )
            _accumulate_result_stats(stats, result)
            total_score_image = result.get("total_score_image")
            immediate_visible_players = None
            if total_score_image is not None:
                row_count_stage_start = time.perf_counter()
                immediate_visible_players = result.get("total_score_visible_players")
                if immediate_visible_players is None:
                    immediate_visible_players = score_screen_selection.count_visible_position_rows(
                        total_score_image,
                        result["candidate"].get("score_layout_id"),
                    )
                    result["total_score_visible_players"] = immediate_visible_players
                video_io.add_timing(stats, "score_flush_row_count_s", row_count_stage_start)
            if immediate_visible_players is not None:
                immediate_expected_players = max(
                    int(immediate_expected_players or 0),
                    int(immediate_visible_players),
                )
            if result["total_score_frame"] > 0:
                LOGGER.log(
                    "",
                    "Race "
                    + LOGGER.video_value(f"{task['race_number']:03}", video_label)
                    + " | total score found at "
                    + LOGGER.video_value(frame_to_timecode(result['total_score_frame'], fps), video_label),
                )
            local_progress.update(
                completed_count,
                _format_total_score_progress_detail(video_label, completed_count, len(tasks), int(task["race_number"])),
                value_color_token=video_label,
            )
            _write_debug_rows(result)
            result = _prepare_result_for_save(
                result,
                immediate_expected_players if int(task["race_number"]) >= 2 else None,
            )
            _persist_saved_result(result, revision=1)
    else:
        chunk_size = 2
        score_task_chunks = _chunk_score_tasks_contiguous(tasks, chunk_size)
        stats["score_scheduler_strategy"] = f"chunked queue (size {chunk_size})"
        stats["score_scheduler_worker_layouts"] = [
            _score_worker_layout_text(chunk_index, chunk_tasks)
            for chunk_index, chunk_tasks in enumerate(score_task_chunks)
        ]
        chunk_queue = queue.Queue()
        for chunk_index, chunk_tasks in enumerate(score_task_chunks):
            chunk_queue.put({"chunk_index": chunk_index, "tasks": chunk_tasks})
        result_queue = queue.Queue()
        active_worker_count = min(worker_count, len(score_task_chunks))
        with ThreadPoolExecutor(max_workers=active_worker_count) as executor:
            futures = [
                executor.submit(
                    _run_score_task_chunk_worker,
                    video_path,
                    worker_id,
                    chunk_queue,
                    frame_to_timecode,
                    result_queue,
                )
                for worker_id in range(active_worker_count)
            ]
            immediate_expected_players = None
            completed_count = 0
            while completed_count < len(tasks):
                try:
                    queue_item = result_queue.get(timeout=0.1)
                except queue.Empty:
                    for future in futures:
                        exception = future.exception() if future.done() else None
                        if exception is not None:
                            raise exception
                    continue
                task = queue_item["task"]
                result = queue_item["result"]
                _accumulate_result_stats(stats, result)
                race_number = int(task["race_number"])
                completed_count += 1
                total_score_image = result.get("total_score_image")
                immediate_visible_players = None
                if total_score_image is not None:
                    row_count_stage_start = time.perf_counter()
                    immediate_visible_players = result.get("total_score_visible_players")
                    if immediate_visible_players is None:
                        immediate_visible_players = score_screen_selection.count_visible_position_rows(
                            total_score_image,
                            result["candidate"].get("score_layout_id"),
                        )
                        result["total_score_visible_players"] = immediate_visible_players
                    video_io.add_timing(stats, "score_flush_row_count_s", row_count_stage_start)
                if immediate_visible_players is not None:
                    immediate_expected_players = max(
                        int(immediate_expected_players or 0),
                        int(immediate_visible_players),
                    )
                if result["total_score_frame"] > 0:
                    LOGGER.log(
                        "",
                        "Race "
                        + LOGGER.video_value(f"{task['race_number']:03}", video_label)
                        + " | total score found at "
                        + LOGGER.video_value(frame_to_timecode(result['total_score_frame'], fps), video_label),
                    )
                local_progress.update(
                    completed_count,
                    _format_total_score_progress_detail(video_label, completed_count, len(tasks), int(task["race_number"])),
                    value_color_token=video_label,
                )
                _write_debug_rows(result)
                result = _prepare_result_for_save(
                    result,
                    immediate_expected_players if race_number >= 2 else None,
                )
                _persist_saved_result(result, revision=1)
            worker_summaries = [future.result() for future in futures]
        stats["score_scheduler_capture_opens"] = sum(
            int(summary.get("capture_opens", 0) or 0) for summary in worker_summaries
        )
        stats["score_scheduler_worker_backends"] = [
            f"w{int(summary.get('worker_id', 0)) + 1}:{summary.get('backend_name') or 'unknown'}"
            for summary in worker_summaries
        ]
    video_io.add_timing(stats, "score_candidate_pass_s", stage_start)


def _run_total_score_phase_for_context(
    context,
    score_candidates,
    templates,
    metadata_context,
    csv_writer,
    metadata_writer,
    per_race_complete_callback,
    per_video_complete_callback,
    include_subfolders,
    *,
    io_lock=None,
    analysis_workers_override=None,
):
    video_index = int(context.get("display_video_index", context["video_index"]))
    total_videos = int(context.get("display_total_videos", context["total_videos"]))
    video_label = context["video_label"]
    video_name = context["video_name"]
    source_display_name = context["source_display_name"]
    processing_video_path = context["processing_video_path"]
    video_stats = context["video_stats"]
    fps = context["fps"]
    total_frames = context["total_frames"]

    LOGGER.log(color_video_scope(f"[Video {video_index}/{total_videos} - Total Score - Phase Start]", video_label), "")
    configured_workers = max(1, int(analysis_workers_override or SCORE_ANALYSIS_WORKERS))
    active_workers = min(configured_workers, max(1, len(score_candidates)))
    LOGGER.log(
        color_video_scope(f"[Video {video_index}/{total_videos} - Total Score]", video_label),
        f"Races queued: {len(score_candidates)} | Workers: {active_workers}",
        color_name="cyan",
    )
    total_score_progress = ProgressPrinter(
        color_video_scope(f"[Video {video_index}/{total_videos} - Total Score]", video_label),
        max(1, len(score_candidates)),
        percent_step=5,
        min_interval_s=2.0,
        include_resources=False,
    )
    total_score_progress.update(0)

    race_complete_callback = per_race_complete_callback
    if race_complete_callback is not None and metadata_context is not None:
        def race_complete_callback(payload, _callback=race_complete_callback):
            if io_lock is None:
                metadata_context.flush()
                _callback(payload)
            else:
                lock_wait_start = time.perf_counter()
                io_lock.acquire()
                video_io.add_timing(video_stats, "score_callback_io_lock_wait_s", lock_wait_start)
                video_stats["score_callback_io_lock_acquires"] += 1
                try:
                    metadata_context.flush()
                    _callback(payload)
                finally:
                    io_lock.release()

    process_score_candidates(
        processing_video_path,
        video_label,
        source_display_name,
        score_candidates,
        templates,
        fps,
        csv_writer,
        context["median_scale_x"],
        context["median_scale_y"],
        context["median_left"],
        context["median_top"],
        context["median_crop_width"],
        context["median_crop_height"],
        video_stats,
        metadata_writer,
        video_index=video_index,
        total_videos=total_videos,
        progress=total_score_progress,
        per_race_complete_callback=race_complete_callback,
        analysis_workers_override=analysis_workers_override,
        io_lock=io_lock,
    )
    exported_counts = count_exported_detection_files(video_label)
    video_stats["video_total_s"] = time.perf_counter() - context["video_start"]
    if total_score_progress.last_percent < 100:
        total_score_progress.update(len(score_candidates))
    LOGGER.summary_block(
        color_video_scope(f"[Video {video_index}/{total_videos} - Total Score - Phase Complete]", video_label),
        [
            f"Duration: {format_duration(video_stats.get('score_candidate_pass_s', 0.0))}",
            f"Total score screens found: {exported_counts['total']}",
            *total_score_progress.peak_lines(),
        ] if total_score_progress.peak_lines() else [
            f"Duration: {format_duration(video_stats.get('score_candidate_pass_s', 0.0))}",
            f"Total score screens found: {exported_counts['total']}",
        ],
        color_name="green",
    )
    print_timing_summary(video_name, video_stats)
    print_extract_profiler_summary(video_name, video_stats)
    LOGGER.log(
        color_video_scope(f"[Video {video_index}/{total_videos} - Complete]", video_label),
        f"{video_name} | Elapsed until complete: {format_duration(video_stats['video_total_s'])} | "
        f"Source length: {format_duration(total_frames / max(fps, 1))} | Track screens: {exported_counts['track']} | "
        f"Race numbers: {exported_counts['race']} | Total score screens: {exported_counts['total']}",
        color_name="green",
    )

    per_video_summary = {
        "video_name": video_name,
        "video_label": video_label,
        "source_display_name": context["source_display_name"],
        "source_length_s": total_frames / max(fps, 1),
        "processing_duration_s": float(video_stats["video_total_s"]),
        "scan_duration_s": float(video_stats.get("main_scan_loop_s", 0.0)),
        "total_score_duration_s": float(video_stats.get("score_candidate_pass_s", 0.0)),
        "corrupt_check_duration_s": float(video_stats.get("corrupt_check_duration_s", 0.0)),
        "corrupt_check_status": context["corrupt_check_status"],
        "repair_duration_s": float(video_stats.get("repair_duration_s", 0.0)),
        "repair_created": bool(video_stats.get("repair_created", 0)),
        "track_screens": exported_counts["track"],
        "race_numbers": exported_counts["race"],
        "total_score_screens": exported_counts["total"],
        "display_video_index": int(context.get("display_video_index", context["video_index"])),
        "display_total_videos": int(context.get("display_total_videos", context["total_videos"])),
    }

    if per_video_complete_callback is not None:
        if metadata_context is not None:
            if io_lock is None:
                metadata_context.flush()
            else:
                with io_lock:
                    metadata_context.flush()
        if io_lock is None:
            per_video_cache = {
                key: value[:]
                for key, value in CONSENSUS_FRAME_CACHE.items()
                if str(key[0]) == str(video_label)
            }
            metadata_index = load_exported_frame_metadata(Path(PROJECT_ROOT))
        else:
            with io_lock:
                per_video_cache = {
                    key: value[:]
                    for key, value in CONSENSUS_FRAME_CACHE.items()
                    if str(key[0]) == str(video_label)
                }
        metadata_index = load_exported_frame_metadata(Path(PROJECT_ROOT))
        per_video_complete_callback(
            {
                "video_name": video_name,
                "video_label": video_label,
                "display_video_index": video_index,
                "display_total_videos": total_videos,
                "summary": per_video_summary,
                "frame_bundle_cache": per_video_cache,
                "metadata_index": {
                    key: value
                    for key, value in metadata_index.items()
                    if str(value.get("video_label", "")).strip() == str(video_label)
                    or Path(str(value.get("video", ""))).stem == str(video_label)
                },
            }
        )

    return {
        "video_label": video_label,
        "exported_counts": exported_counts,
        "per_video_summary": per_video_summary,
    }


def _format_parallel_scan_diagnostics(parallel_scan_diag, merged_score_detections, merged_track_detections, merged_race_detections, parallel_merge_s, parallel_dedupe_s, auxiliary_save_s):
    def _format_seconds_precise(value):
        return f"{float(value):.2f}s"

    return [
        f"Mode: parallel {parallel_scan_diag.get('executor', 'unknown')} x {parallel_scan_diag.get('segment_count', 0)}",
        f"Task submit/startup: {_format_seconds_precise(parallel_scan_diag.get('submit_startup_s', 0.0))}",
        f"First segment result: {_format_seconds_precise(parallel_scan_diag.get('first_result_s', 0.0))}",
        f"Parallel wait/collect: {_format_seconds_precise(parallel_scan_diag.get('parallel_wait_s', 0.0))}",
        f"Merge worker results: {_format_seconds_precise(parallel_merge_s)}",
        f"Deduplicate detections: {_format_seconds_precise(parallel_dedupe_s)}",
        f"Save auxiliary frames: {_format_seconds_precise(auxiliary_save_s)}",
        f"Raw merged detections: score {len(merged_score_detections)} | track {len(merged_track_detections)} | race {len(merged_race_detections)}",
    ]


def _log_corrupt_preflight_outcome(video_index, total_videos, preflight_result, *, fps=None, video_label=None):
    if not preflight_result:
        return

    status = str(preflight_result.get("status", "checked"))
    reason = str(preflight_result.get("reason", "") or "")
    probe_count = int(preflight_result.get("probe_count", 0) or 0)
    usable_total_frames = preflight_result.get("usable_total_frames")
    scope = color_video_scope(f"[Video {video_index}/{total_videos} - Start]", video_label) if video_label else f"[Video {video_index}/{total_videos} - Start]"

    if preflight_result.get("is_suspect"):
        LOGGER.log(
            scope,
            f"Corrupt preflight suspect ({status}) | action: try mp4 remux, then full transcode only if still unreadable | {reason}",
            color_name="yellow",
        )
        return

    if status == "head_clamped":
        usable_start_frame = int(preflight_result.get("usable_start_frame", 0) or 0)
        usable_total_frames = preflight_result.get("usable_total_frames")
        readable_detail = "the readable part only"
        if usable_total_frames is not None:
            readable_detail = (
                f"{int(usable_total_frames):,} frames "
                f"(about {format_duration(int(usable_total_frames) / max(float(fps or 1), 1.0))})"
            )
        failed_frame = preflight_result.get("failed_frame")
        debug_detail = ""
        if failed_frame is not None and usable_total_frames is not None:
            end_frame = max(usable_start_frame, usable_start_frame + int(usable_total_frames) - 1)
            debug_detail = (
                f"Debug: head probe failed at requested frame {int(failed_frame):,}; "
                f"extraction will use frames {usable_start_frame:,}-{end_frame:,} "
                f"({int(usable_total_frames):,} total)."
            )
        LOGGER.log(
            scope,
            (
                f"Video readability check: the file becomes unreadable right at the start. "
                f"Extraction will skip that damaged opening and continue with {readable_detail}. "
                f"({probe_count} probe checks)"
            ),
            color_name="yellow",
        )
        if debug_detail:
            LOGGER.log(scope, debug_detail, color_name="dim")
        return

    if status == "tail_clamped":
        readable_detail = "the readable part only"
        if usable_total_frames is not None:
            readable_detail = (
                f"{int(usable_total_frames):,} frames "
                f"(about {format_duration(int(usable_total_frames) / max(float(fps or 1), 1.0))})"
            )
        failed_frame = preflight_result.get("failed_frame")
        debug_detail = ""
        if failed_frame is not None and usable_total_frames is not None:
            debug_detail = (
                f"Debug: tail probe failed at requested frame {int(failed_frame):,}; "
                f"extraction will use frames 0-{int(usable_total_frames) - 1:,} "
                f"({int(usable_total_frames):,} total)."
            )
        LOGGER.log(
            scope,
            (
                f"Video readability check: the file becomes unreadable at the very end. "
                f"Extraction will continue with {readable_detail}. "
                f"({probe_count} probe checks)"
            ),
            color_name="yellow",
        )
        if debug_detail:
            LOGGER.log(scope, debug_detail, color_name="dim")
        return

    LOGGER.log(
        scope,
        f"Video readability check passed. Extraction will continue unchanged. ({probe_count} probe checks)",
        color_name="green",
    )


def _prepare_video_context(video_path, folder_path, include_subfolders, video_index, total_videos, template_load_time_s, templates, *, video_label=None, source_display_name=None):
    video_start = time.perf_counter()
    video_stats = defaultdict(float)
    video_stats["template_load_s"] = template_load_time_s
    capture_poisoned = False
    video_label = video_label or build_video_identity(Path(video_path), input_root=folder_path, include_subfolders=include_subfolders)

    probe = cv2.VideoCapture(video_path)
    if not probe.isOpened():
        LOGGER.log(f"[Video {video_index}/{total_videos} - Start]", f"Could not open video: {video_path}", color_name="red")
        return None
    nominal_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    nominal_fps = probe.get(cv2.CAP_PROP_FPS) or 1
    nominal_duration_s = nominal_total_frames / max(nominal_fps, 1)
    corrupt_check_status = "checked"
    preflight_result = video_io.sample_video_readability(
        video_path,
        nominal_total_frames,
        stats=video_stats,
        video_identity=video_label,
    )
    corrupt_check_status = str(preflight_result.get("status", "checked"))
    probe.release()
    _log_corrupt_preflight_outcome(
        video_index,
        total_videos,
        preflight_result,
        fps=nominal_fps,
        video_label=video_label,
    )
    processing_video_path = video_io.repair_video_if_needed(
        video_path,
        nominal_total_frames,
        preflight_result,
        duration_s=nominal_duration_s,
        stats=video_stats,
    )

    cap = cv2.VideoCapture(processing_video_path)
    if not cap.isOpened():
        LOGGER.log(
            f"[Video {video_index}/{total_videos} - Start]",
            f"Could not open video: {processing_video_path}",
            color_name="red",
        )
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(3 * max(1, int(fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nominal_processing_total_frames = total_frames
    usable_start_frame = int(preflight_result.get("usable_start_frame") or 0) if preflight_result else 0
    effective_total_frames = int(preflight_result.get("usable_total_frames") or total_frames) if preflight_result else total_frames
    total_frames = max(0, min(total_frames, effective_total_frames))
    if total_frames < nominal_processing_total_frames:
        LOGGER.log(
            color_video_scope(f"[Video {video_index}/{total_videos} - Start]", video_label),
            (
                f"Readable portion selected for extraction: {total_frames:,} of {nominal_processing_total_frames:,} "
                f"frames ({format_duration(total_frames / max(fps, 1))})"
            ),
            color_name="yellow",
        )
    elif usable_start_frame > 0:
        LOGGER.log(
            color_video_scope(f"[Video {video_index}/{total_videos} - Start]", video_label),
            (
                f"Readable portion selected for extraction: frames {usable_start_frame:,}-"
                f"{max(usable_start_frame, usable_start_frame + total_frames - 1):,} "
                f"({total_frames:,} total, {format_duration(total_frames / max(fps, 1))})"
            ),
            color_name="yellow",
        )
    processing_is_suspect = bool(preflight_result and preflight_result.get("is_suspect") and processing_video_path == video_path)
    processing_has_head_offset = usable_start_frame > 0
    scan_worker_count = 1 if (processing_is_suspect or processing_has_head_offset) else INITIAL_SCAN_WORKERS
    if processing_is_suspect:
        LOGGER.log(
            f"[Video {video_index}/{total_videos} - Start]",
            "Using conservative single-process scan for suspect video",
            color_name="yellow",
        )
    if processing_has_head_offset:
        LOGGER.log(
            color_video_scope(f"[Video {video_index}/{total_videos} - Start]", video_label),
            "Using conservative single-process scan because extraction starts after a damaged opening section",
            color_name="yellow",
        )
    readable_end_frame = usable_start_frame + total_frames
    sample_frames = np.linspace(usable_start_frame, max(usable_start_frame, readable_end_frame - 1), 19).astype(int)
    scales = []
    video_name = os.path.basename(processing_video_path)
    original_source_display_name = (
        relative_video_path(Path(video_path), folder_path) if include_subfolders else os.path.basename(video_path)
    )
    processing_source_display_name = (
        relative_video_path(Path(processing_video_path), folder_path)
        if include_subfolders else
        os.path.basename(processing_video_path)
    )
    source_display_name = source_display_name or processing_source_display_name
    stage_start = time.perf_counter()
    eof_guard_frames = max(frame_skip, int(fps * INITIAL_SCAN_EOF_GUARD_SECONDS))
    if total_frames <= 0:
        cap.release()
        LOGGER.log(
            f"[Video {video_index}/{total_videos} - Start]",
            f"No readable frames available after preflight clamp: {processing_video_path}",
            color_name="red",
        )
        return None
    for frame_num in sample_frames:
        video_io.seek_to_frame(cap, frame_num, video_stats)
        read_timeout_s = (
            INITIAL_SCAN_EOF_READ_TIMEOUT_SECONDS
            if readable_end_frame - int(frame_num) <= eof_guard_frames
            else None
        )
        if read_timeout_s is None:
            ret, frame = video_io.read_video_frame(cap, video_stats)
            timed_out = False
        else:
            ret, frame, timed_out = video_io.read_video_frame_with_timeout(cap, video_stats, read_timeout_s)
        if timed_out:
            LOGGER.log(
                color_video_scope(f"[Video {video_index}/{total_videos} - Start]", video_label),
                (
                    f"Aborting video after frame-read stall during scaling scan "
                    f"(requested frame {int(frame_num)}/{total_frames})"
                ),
                color_name="yellow",
            )
            capture_poisoned = True
            break
        if not ret:
            continue
        scale_x, scale_y, left, top, crop_width, crop_height = determine_scaling(frame)
        scales.append((scale_x, scale_y, left, top, crop_width, crop_height))
    video_io.add_timing(video_stats, "scaling_scan_s", stage_start)
    cap.release()

    if not scales:
        if capture_poisoned:
            LOGGER.log(
                f"[Video {video_index}/{total_videos} - Start]",
                f"Skipping poisoned capture after read timeout: {processing_video_path}",
                color_name="yellow",
            )
        else:
            LOGGER.log(
                f"[Video {video_index}/{total_videos} - Start]",
                f"No valid frames found for scaling: {processing_video_path}",
                color_name="red",
            )
        return None

    context = {
        "video_path": video_path,
        "video_index": video_index,
        "total_videos": total_videos,
        "video_start": video_start,
        "video_stats": video_stats,
        "corrupt_check_status": corrupt_check_status,
        "processing_video_path": processing_video_path,
        "video_name": video_name,
        "video_label": video_label,
        "source_display_name": source_display_name,
        "original_source_display_name": original_source_display_name,
        "display_video_index": video_index,
        "display_total_videos": total_videos,
        "fps": fps,
        "frame_skip": frame_skip,
        "total_frames": total_frames,
        "usable_start_frame": usable_start_frame,
        "readable_end_frame": readable_end_frame,
        "eof_guard_frames": eof_guard_frames,
        "capture_poisoned": capture_poisoned,
        "median_scale_x": np.median([s[0] for s in scales]),
        "median_scale_y": np.median([s[1] for s in scales]),
        "median_left": int(np.median([s[2] for s in scales])),
        "median_top": int(np.median([s[3] for s in scales])),
        "median_crop_width": int(np.median([s[4] for s in scales])),
        "median_crop_height": int(np.median([s[5] for s in scales])),
    }
    context["detection_segment_tasks"] = initial_scan.build_detection_segment_tasks(
        context["processing_video_path"],
        context["video_label"],
        context["source_display_name"] if include_subfolders else os.path.basename(video_path),
        context["total_frames"],
        context["fps"],
        templates,
        context["median_scale_x"],
        context["median_scale_y"],
        context["median_left"],
        context["median_top"],
        context["median_crop_width"],
        context["median_crop_height"],
        scan_worker_count,
        APP_CONFIG.write_debug_csv,
        INITIAL_SCAN_SEGMENT_OVERLAP_FRAMES,
        INITIAL_SCAN_MIN_SEGMENT_FRAMES,
        INITIAL_SCAN_WINDOW_STEPS,
        INITIAL_SCAN_PROGRESS_REPORT_SECONDS,
    )
    return context


def _run_serial_initial_scan(context, templates, csv_writer, metadata_writer):
    display_video_index = int(context.get("display_video_index", context["video_index"]))
    display_total_videos = int(context.get("display_total_videos", context["total_videos"]))
    cap = cv2.VideoCapture(context["processing_video_path"])
    if not cap.isOpened():
        LOGGER.log(
            f"[Video {display_video_index}/{display_total_videos} - Start]",
            f"Could not open video: {context['processing_video_path']}",
            color_name="red",
        )
        return {"aborted": True, "capture_poisoned": False}

    video_stats = context["video_stats"]
    fps = context["fps"]
    total_frames = context["total_frames"]
    usable_start_frame = int(context.get("usable_start_frame", 0) or 0)
    readable_end_frame = int(context.get("readable_end_frame", usable_start_frame + total_frames) or (usable_start_frame + total_frames))
    frame_skip = context["frame_skip"]
    eof_guard_frames = context["eof_guard_frames"]
    score_candidates = []
    runtime_state = {
        "last_track_frame": 0,
        "last_race_frame": 0,
        "next_race_number": 1,
        "capture": cap,
    }
    frame_count = usable_start_frame
    stage_start = time.perf_counter()
    scan_progress = ProgressPrinter(
        color_video_scope(
            f"[Video {display_video_index}/{display_total_videos} - Scan]",
            context["video_label"],
        ),
        total_frames,
        percent_step=5,
        min_interval_s=2.0,
        unit_formatter=lambda completed, total, fps=fps: format_scan_time_progress(completed, total, fps),
        include_resources=False,
    )
    scan_progress.update(0)
    capture_poisoned = False
    video_io.seek_to_frame(cap, usable_start_frame, video_stats)

    while cap.isOpened() and frame_count < readable_end_frame:
        completed_frames = max(0, frame_count - usable_start_frame)
        remaining_frames = max(0, readable_end_frame - frame_count)
        if (
            total_frames > 0
            and completed_frames / total_frames >= INITIAL_SCAN_EOF_GUARD_PROGRESS
            and remaining_frames <= eof_guard_frames
        ):
            LOGGER.log(
                color_video_scope(
                    f"[Video {display_video_index}/{display_total_videos} - Scan]",
                    context["video_label"],
                ),
                (
                    f"Stopping scan near EOF to avoid decoder stalls "
                    f"({remaining_frames} frames remaining, "
                    f"{remaining_frames / max(fps, 1):.1f}s tail)"
                ),
                color_name="yellow",
            )
            frame_count = readable_end_frame
            break
        window_interrupted = False
        for _ in range(INITIAL_SCAN_WINDOW_STEPS):
            read_timeout_s = (
                INITIAL_SCAN_EOF_READ_TIMEOUT_SECONDS
                if remaining_frames <= eof_guard_frames
                else None
            )
            if read_timeout_s is None:
                ret, frame = video_io.read_video_frame(cap, video_stats)
                timed_out = False
            else:
                ret, frame, timed_out = video_io.read_video_frame_with_timeout(cap, video_stats, read_timeout_s)
            if timed_out:
                LOGGER.log(
                    color_video_scope(
                        f"[Video {display_video_index}/{display_total_videos} - Scan]",
                        context["video_label"],
                    ),
                    (
                        f"Aborting video after frame-read stall near EOF "
                        f"({remaining_frames} frames remaining, "
                        f"{remaining_frames / max(fps, 1):.1f}s tail)"
                    ),
                    color_name="yellow",
                )
                frame_count = readable_end_frame
                window_interrupted = True
                capture_poisoned = True
                break
            if not ret:
                window_interrupted = True
                break

            frames_to_skip = initial_scan.process_frame(
                frame,
                frame_count,
                context["processing_video_path"],
                context["video_label"],
                context["source_display_name"],
                templates,
                fps,
                csv_writer,
                context["median_scale_x"],
                context["median_scale_y"],
                context["median_left"],
                context["median_top"],
                context["median_crop_width"],
                context["median_crop_height"],
                video_stats,
                runtime_state,
                score_candidates,
                metadata_writer,
            )
            if frames_to_skip > 0:
                frame_count += frames_to_skip + frame_skip
                if frame_count < readable_end_frame:
                    video_io.seek_to_frame(cap, frame_count, video_stats)
                window_interrupted = True
                scan_progress.update(
                    min(max(0, frame_count - usable_start_frame), total_frames),
                    color_video_detail("Score candidates: ", len(score_candidates), context["video_label"]),
                    value_color_token=context["video_label"],
                )
                break

            if not video_io.advance_frames_by_grab(cap, frame_skip - 1, video_stats):
                window_interrupted = True
                frame_count = readable_end_frame
                break

            frame_count += frame_skip
            if frame_count >= readable_end_frame:
                window_interrupted = True
                break
            scan_progress.update(
                min(max(0, frame_count - usable_start_frame), total_frames),
                color_video_detail("Score candidates: ", len(score_candidates), context["video_label"]),
                value_color_token=context["video_label"],
            )
        if window_interrupted and frame_count >= readable_end_frame:
            break

    if scan_progress.last_percent < 100:
        scan_progress.update(
            total_frames,
            color_video_detail("Score candidates: ", len(score_candidates), context["video_label"]),
            value_color_token=context["video_label"],
        )
    video_io.add_timing(video_stats, "main_scan_loop_s", stage_start)
    pre_pass2_counts = count_exported_detection_files(context["video_label"])
    cap.release()
    return {
        "aborted": False,
        "capture_poisoned": capture_poisoned,
        "cap": None,
        "score_candidates": score_candidates,
        "scan_progress": scan_progress,
        "scan_track_count": pre_pass2_counts["track"],
        "scan_race_count": pre_pass2_counts["race"],
        "parallel_scan_diag": None,
    }


def _run_scan_phase_for_context(context, templates, csv_writer, metadata_writer):
    video_stats = context["video_stats"]
    video_index = int(context.get("display_video_index", context["video_index"]))
    total_videos = int(context.get("display_total_videos", context["total_videos"]))
    video_name = context["video_name"]
    video_label = context["video_label"]
    fps = context["fps"]
    total_frames = context["total_frames"]
    scan_progress = context.get("scan_progress")

    if context["detection_segment_tasks"]:
        LOGGER.log(color_video_scope(f"[Video {video_index}/{total_videos} - Scan - Phase Start]", video_label), "")
        LOGGER.log(
            color_video_scope(f"[Video {video_index}/{total_videos} - Scan]", video_label),
            f"Source: {format_duration(total_frames / max(fps, 1.0))} | Mode: segmented | Workers: {len(context['detection_segment_tasks'])}",
            color_name="cyan",
        )
        scan_progress = ProgressPrinter(
            color_video_scope(f"[Video {video_index}/{total_videos} - Scan]", video_label),
            total_frames,
            percent_step=5,
            min_interval_s=1.0,
            unit_formatter=lambda completed, total, fps=fps: format_scan_time_progress(completed, total, fps),
            include_resources=False,
        )
        scan_progress.update(0)
        parallel_scan_diag = {}
        stage_start = time.perf_counter()
        segment_results = initial_scan.run_parallel_detection_segments(
            context["detection_segment_tasks"],
            scan_progress,
            diagnostics=parallel_scan_diag,
        )
        video_io.add_timing(video_stats, "main_scan_loop_s", stage_start)
        parallel_result = _finalize_parallel_initial_scan(
            context,
            segment_results,
            csv_writer,
            metadata_writer,
        )
        score_candidates = parallel_result["score_candidates"]
        scan_track_count = int(parallel_result["scan_track_count"])
        scan_race_count = int(parallel_result["scan_race_count"])
        if INITIAL_SCAN_DIAGNOSTICS_ENABLED:
            LOGGER.summary_block(
                f"[Video {video_index}/{total_videos} - Scan Diagnostics]",
                _format_parallel_scan_diagnostics(
                    parallel_scan_diag or {},
                    parallel_result["merged_score_detections"],
                    parallel_result["merged_track_detections"],
                    parallel_result["merged_race_detections"],
                    parallel_result["parallel_merge_s"],
                    parallel_result["parallel_dedupe_s"],
                    parallel_result["auxiliary_save_s"],
                ),
                color_name="dim",
            )
        return {
            "aborted": False,
            "capture_poisoned": False,
            "score_candidates": score_candidates,
            "scan_track_count": scan_track_count,
            "scan_race_count": scan_race_count,
            "scan_progress": scan_progress,
        }

    LOGGER.log(color_video_scope(f"[Video {video_index}/{total_videos} - Scan - Phase Start]", video_label), "")
    LOGGER.log(
        color_video_scope(f"[Video {video_index}/{total_videos} - Scan]", video_label),
        f"Source: {format_duration(total_frames / max(fps, 1.0))} | Mode: sequential | Workers: 1",
        color_name="cyan",
    )
    scan_result = _run_serial_initial_scan(context, templates, csv_writer, metadata_writer)
    if scan_result.get("aborted"):
        return scan_result
    if scan_result.get("capture_poisoned"):
        LOGGER.log(
            f"[Video {video_index}/{total_videos} - Complete]",
            f"{video_name} | Aborted after timed-out read to avoid reusing a poisoned decoder",
            color_name="yellow",
        )
        video_stats["video_total_s"] = time.perf_counter() - context["video_start"]
    return scan_result


def _finalize_parallel_initial_scan(context, segment_results, csv_writer, metadata_writer):
    video_stats = context["video_stats"]
    fps = context["fps"]
    merged_score_detections = []
    merged_track_detections = []
    merged_race_detections = []
    merged_debug_rows = []

    merge_start = time.perf_counter()
    for result in sorted(segment_results, key=lambda item: item["segment_index"]):
        for key, value in result["stats"].items():
            video_stats[key] += value
        merged_score_detections.extend(result["score_detections"])
        merged_track_detections.extend(result["track_detections"])
        merged_race_detections.extend(result["race_detections"])
        merged_debug_rows.extend(result["debug_rows"])
    parallel_merge_s = time.perf_counter() - merge_start

    dedupe_start = time.perf_counter()
    min_gap_frames = int(fps * 20)
    merged_score_detections = initial_scan.merge_nearby_detections(merged_score_detections, min_gap_frames)
    merged_track_detections = initial_scan.merge_nearby_detections(merged_track_detections, min_gap_frames)
    merged_race_detections = initial_scan.merge_nearby_detections(merged_race_detections, min_gap_frames)
    parallel_dedupe_s = time.perf_counter() - dedupe_start

    score_frame_numbers = [item["frame_number"] for item in merged_score_detections]
    score_candidates = [
        {
            "race_number": index + 1,
            "frame_number": item["frame_number"],
            "score_layout_id": item.get("score_layout_id"),
        }
        for index, item in enumerate(merged_score_detections)
    ]

    if APP_CONFIG.write_debug_csv and merged_debug_rows:
        csv_writer.writerows(merged_debug_rows)

    auxiliary_detections = [{"kind": "track", **item} for item in merged_track_detections]
    auxiliary_detections.extend({"kind": "race", **item} for item in merged_race_detections)
    auxiliary_detections.sort(key=lambda item: item["frame_number"])
    if auxiliary_detections:
        LOGGER.log(
                color_video_scope(
                    f"[Video {context['video_index']}/{context['total_videos']} - Scan - Confirmed Results]",
                    context["video_label"],
                ),
            "",
        )

    cap = cv2.VideoCapture(context["processing_video_path"])
    auxiliary_save_start = time.perf_counter()
    initial_scan.save_auxiliary_detection_frames(
        cap,
        context["processing_video_path"],
        context["video_label"],
        context["source_display_name"],
        auxiliary_detections,
        score_frame_numbers,
        context["median_left"],
        context["median_top"],
        context["median_crop_width"],
        context["median_crop_height"],
        fps,
        video_stats,
        metadata_writer,
    )
    auxiliary_save_s = time.perf_counter() - auxiliary_save_start
    cap.release()

    return {
        "score_candidates": score_candidates,
        "scan_track_count": len(merged_track_detections),
        "scan_race_count": len(merged_race_detections),
        "parallel_merge_s": parallel_merge_s,
        "parallel_dedupe_s": parallel_dedupe_s,
        "auxiliary_save_s": auxiliary_save_s,
        "merged_score_detections": merged_score_detections,
        "merged_track_detections": merged_track_detections,
        "merged_race_detections": merged_race_detections,
    }


def _extract_frames_parallel_video_scan(
    video_paths,
    folder_path,
    include_subfolders,
    templates,
    template_load_time_s,
    csv_writer,
    metadata_writer,
    metadata_context,
    per_video_complete_callback,
    per_race_complete_callback,
    total_source_seconds,
    return_frame_cache,
    phase_start_time,
):
    workflow_plan, total_source_seconds = build_workflow_video_plan(
        video_paths,
        folder_path,
        include_subfolders=include_subfolders,
    )
    total_videos = len(workflow_plan)
    total_score_screens_found = 0
    total_track_screens_found = 0
    total_race_numbers_found = 0
    per_video_summaries = []

    prepared_contexts = []
    prepare_workers = 1 if len(workflow_plan) <= 1 else max(1, min(PARALLEL_VIDEO_SCAN_WORKERS, len(workflow_plan)))
    with ThreadPoolExecutor(max_workers=prepare_workers) as prepare_executor:
        prepare_futures = {
            prepare_executor.submit(
                _prepare_video_context,
                plan_entry["video_path"],
                folder_path,
                include_subfolders,
                plan_entry["display_video_index"],
                total_videos,
                template_load_time_s,
                templates,
                video_label=plan_entry["video_label"],
                source_display_name=plan_entry["source_display_name"],
            ): plan_entry
            for plan_entry in workflow_plan
        }
        for future in as_completed(prepare_futures):
            context = future.result()
            if context is not None:
                prepared_contexts.append(context)
    prepared_contexts.sort(key=lambda item: int(item.get("display_video_index", item["video_index"])))

    parallel_contexts = [context for context in prepared_contexts if context["detection_segment_tasks"]]
    for context in prepared_contexts:
        display_video_index = int(context.get("display_video_index", context["video_index"]))
        display_total_videos = int(context.get("display_total_videos", context["total_videos"]))
        LOGGER.log(
            color_video_scope(f"[Video {display_video_index}/{display_total_videos} - Start]", context["video_label"]),
            color_video_message([
                (context["source_display_name"] if include_subfolders else context["video_name"], context["video_label"]),
                " | Source length: ",
                (format_duration(context["total_frames"] / max(context["fps"], 1)), context["video_label"]),
            ]),
        )
    parallel_score_workers = 1
    if len(prepared_contexts) > 1:
        parallel_score_workers = max(1, min(PARALLEL_VIDEO_SCORE_WORKERS, len(prepared_contexts)))
    per_video_score_analysis_workers = max(1, SCORE_ANALYSIS_WORKERS)
    io_lock = threading.Lock() if parallel_score_workers > 1 else None
    scan_workers = 1 if len(prepared_contexts) <= 1 else max(1, min(PARALLEL_VIDEO_SCAN_WORKERS, len(prepared_contexts)))

    with ThreadPoolExecutor(max_workers=scan_workers) as scan_executor, ThreadPoolExecutor(max_workers=parallel_score_workers) as score_executor:
        pending_scan_futures = {
            scan_executor.submit(_run_scan_phase_for_context, context, templates, csv_writer, metadata_writer): context
            for context in prepared_contexts
        }
        pending_score_futures = {}

        while pending_scan_futures or pending_score_futures:
            if not any(future.done() for future in pending_scan_futures) and not any(future.done() for future in pending_score_futures):
                pending_any = list(pending_scan_futures.keys()) + list(pending_score_futures.keys())
                if pending_any:
                    wait(pending_any, return_when=FIRST_COMPLETED)

            completed_scan_futures = [future for future in list(pending_scan_futures.keys()) if future.done()]
            for future in completed_scan_futures:
                context = pending_scan_futures.pop(future)
                scan_result = future.result()
                if scan_result.get("aborted"):
                    continue

                video_stats = context["video_stats"]
                video_index = int(context.get("display_video_index", context["video_index"]))
                total_videos = int(context.get("display_total_videos", context["total_videos"]))
                video_name = context["video_name"]
                fps = context["fps"]
                total_frames = context["total_frames"]
                scan_progress = scan_result.get("scan_progress")
                score_candidates = scan_result["score_candidates"]
                scan_track_count = int(scan_result["scan_track_count"])
                scan_race_count = int(scan_result["scan_race_count"])

                if scan_result.get("capture_poisoned"):
                    LOGGER.log(
                        f"[Video {video_index}/{total_videos} - Complete]",
                        f"{video_name} | Aborted after timed-out read to avoid reusing a poisoned decoder",
                        color_name="yellow",
                    )
                    video_stats["video_total_s"] = time.perf_counter() - context["video_start"]
                    continue

                scan_duration = float(video_stats.get("main_scan_loop_s", 0.0))
                scan_speed = total_frames / scan_duration if scan_duration > 0 else 0.0
                scan_summary_lines = [
                    f"Duration: {format_duration(scan_duration)}",
                    f"Source length: {format_duration(total_frames / max(fps, 1))}",
                    f"Frames scanned: {total_frames:,}",
                    f"Scan speed: {scan_speed:,.0f} frames/s",
                    f"Track screens found: {scan_track_count}",
                    f"Race numbers found: {scan_race_count}",
                    f"Total score screens queued: {len(score_candidates)}",
                ]
                if scan_progress is not None:
                    scan_summary_lines.extend(scan_progress.peak_lines())
                LOGGER.summary_block(
                    f"[Video {video_index}/{total_videos} - Scan - Phase Complete]",
                    scan_summary_lines,
                    color_name="green",
                )

                pending_score_futures[
                    score_executor.submit(
                        _run_total_score_phase_for_context,
                        context,
                        score_candidates,
                        templates,
                        metadata_context,
                        csv_writer,
                        metadata_writer,
                        per_race_complete_callback,
                        per_video_complete_callback,
                        include_subfolders,
                        io_lock=io_lock,
                        analysis_workers_override=per_video_score_analysis_workers,
                    )
                ] = context

            completed_score_futures = [future for future in list(pending_score_futures.keys()) if future.done()]
            for future in completed_score_futures:
                pending_score_futures.pop(future, None)
                result = future.result()
                exported_counts = result["exported_counts"]
                total_score_screens_found += exported_counts["score"]
                total_track_screens_found += exported_counts["track"]
                total_race_numbers_found += exported_counts["race"]
                per_video_summaries.append(result["per_video_summary"])

    elapsed_time = time.time() - phase_start_time
    per_video_summaries.sort(key=lambda item: int(item.get("display_video_index", 0)))
    extract_summary = {
        "duration_s": elapsed_time,
        "total_source_seconds": total_source_seconds,
        "track_screens": total_track_screens_found,
        "race_numbers": total_race_numbers_found,
        "total_score_screens": total_score_screens_found,
        "corrupt_check_duration_s": sum(float(item.get("corrupt_check_duration_s", 0.0)) for item in per_video_summaries),
        "repair_duration_s": sum(float(item.get("repair_duration_s", 0.0)) for item in per_video_summaries),
        "repair_count": sum(1 for item in per_video_summaries if item.get("repair_created")),
        "per_video_summaries": per_video_summaries,
    }
    extract_lines = [
        f"Duration: {format_duration(elapsed_time)}",
        f"Source video total: {format_duration(total_source_seconds)}",
        f"Track screens found: {total_track_screens_found}",
        f"Race numbers found: {total_race_numbers_found}",
        f"Total score screens found: {total_score_screens_found}",
    ]
    resource_snapshot = LOGGER.resources.sample()
    extract_lines.extend(LOGGER.peak_lines(resource_snapshot))
    LOGGER.summary_block("[Extract - Phase Complete]", extract_lines, color_name="green")
    return {"frame_bundle_cache": CONSENSUS_FRAME_CACHE, "summary": extract_summary} if return_frame_cache else {"summary": extract_summary}

def extract_frames(
    return_frame_cache=False,
    selected_videos=None,
    include_subfolders=False,
    per_video_complete_callback=None,
    per_race_complete_callback=None,
):
    """Main function to process videos and apply template matching."""
    phase_start_time = time.time()
    CONSENSUS_FRAME_CACHE.clear()
    empty_summary = {
        "duration_s": 0.0,
        "total_source_seconds": 0.0,
        "track_screens": 0,
        "race_numbers": 0,
        "total_score_screens": 0,
        "per_video_summaries": [],
    }
    LOGGER.log("[Extract - Phase Start]", "Extract race and score screens", color_name="cyan")

    folder_path = os.path.join(PROJECT_ROOT, 'Input_Videos')

    template_paths = [
        (str(resolve_asset_file('templates', 'Trackname_template.png')), None),
        (str(resolve_asset_file('templates', 'Race_template.png')), None),
        (str(resolve_asset_file('templates', '12th_pos_template.png')), None),
        (str(resolve_asset_file('templates', 'ignore.png')), None),
        (str(resolve_asset_file('templates', 'albumgallery_ignore.png')), None),
        (str(resolve_asset_file('templates', 'ignore_2.png')), None),
        (str(resolve_asset_file('templates', 'Race_template_NL_final.png')), None),
        (str(resolve_asset_file('templates', '12th_pos_templateNL.png')), None),
    ]

    templates = []
    template_load_start = time.perf_counter()
    for template_path, _ in template_paths:
        template = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)
        if template is None:
            LOGGER.log("[Extract - Phase Start]", f"Template image could not be loaded: {template_path}", color_name="red")
            return {"frame_bundle_cache": {}, "summary": empty_summary} if return_frame_cache else {"summary": empty_summary}
        if len(template.shape) == 3 and template.shape[2] == 4:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGRA2GRAY)
            _, alpha_mask = cv2.threshold(template[:, :, 3], 0, 255, cv2.THRESH_BINARY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
        elif len(template.shape) == 3 and template.shape[2] == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_binary = cv2.threshold(template_gray, 180, 255, cv2.THRESH_BINARY)
            alpha_mask = None
        elif len(template.shape) == 2:
            template_binary = template
            alpha_mask = None
        else:
            LOGGER.log("[Extract - Phase Start]", f"Template image has an unexpected number of channels: {template_path}", color_name="red")
            return {"frame_bundle_cache": {}, "summary": empty_summary} if return_frame_cache else {"summary": empty_summary}
        templates.append((template_binary, alpha_mask))
    template_load_time_s = time.perf_counter() - template_load_start

    csv_output_path = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug', 'debug_max_val.csv')
    metadata_output_path = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug', 'exported_frame_metadata.csv')
    video_paths = load_videos_from_folder(folder_path, include_subfolders=include_subfolders)
    if selected_videos:
        selected_names = {str(name).replace("\\", "/").lower() for name in selected_videos}
        filtered_video_paths = []
        for path in video_paths:
            path_obj = Path(path)
            basename = path_obj.name.lower()
            relative_name = relative_video_path(path_obj, folder_path).lower()
            if basename in selected_names or relative_name in selected_names:
                filtered_video_paths.append(path)
        video_paths = filtered_video_paths
    if not video_paths:
        LOGGER.log("[Extract - Phase Start]", "No videos found in Input_Videos", color_name="red")
        return {"frame_bundle_cache": {}, "summary": empty_summary} if return_frame_cache else {"summary": empty_summary}

    if APP_CONFIG.write_debug_csv:
        os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
        csv_context = open(csv_output_path, mode='w', newline='')
        csv_writer = csv.writer(csv_context, delimiter=';')
        csv_writer.writerow(["Video", "Target", "Frame Number", "Max Value", "Timecode"])
        metadata_context = open(metadata_output_path, mode='w', newline='')
        metadata_writer = MetadataCsvWriter(csv.writer(metadata_context, delimiter=';'))
        metadata_writer.writerow(
            [
                "VideoLabel",
                "Video",
                "Race",
                "Kind",
                "Requested Frame",
                "Requested Timecode",
                "Actual Frame",
                "Actual Timecode",
                "Score Layout",
                "Bundle Path",
                "Anchor Path",
            ]
        )
    else:
        csv_context = None
        csv_writer = NullCsvWriter()
        metadata_context = None
        metadata_writer = None

    total_source_seconds = 0.0
    for video_path in video_paths:
        probe = cv2.VideoCapture(video_path)
        if probe.isOpened():
            fps = probe.get(cv2.CAP_PROP_FPS) or 1
            frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            total_source_seconds += frames / max(fps, 1)
        probe.release()
    effective_parallel_scan_workers = PARALLEL_VIDEO_SCAN_WORKERS if len(video_paths) > 1 else 1
    effective_parallel_score_workers = PARALLEL_VIDEO_SCORE_WORKERS if len(video_paths) > 1 else 1
    LOGGER.log(
        "[Extract - Settings]",
        (
            f"Extraction backend: {GPU_RUNTIME['backend']} | "
            f"Initial scan workers/video: {INITIAL_SCAN_WORKERS} | "
            f"Parallel video scan workers: {effective_parallel_scan_workers} | "
            f"Parallel video total score workers: {effective_parallel_score_workers}"
        ),
        color_name="cyan",
    )
    LOGGER.log(
        "[Extract - Settings]",
        (
            f"OCR backend: {EASYOCR_RUNTIME['backend']} | "
            f"OCR overlap mode: {APP_CONFIG.overlap_ocr_mode} -> {effective_overlap_ocr_mode(APP_CONFIG)} | "
            f"OCR consumers: {APP_CONFIG.overlap_ocr_consumers} | "
            f"Score analysis workers/video: {SCORE_ANALYSIS_WORKERS}"
        ),
        color_name="cyan",
    )

    try:
        if PARALLEL_VIDEO_SCAN_WORKERS > 1 and len(video_paths) > 1:
            return _extract_frames_parallel_video_scan(
                video_paths,
                folder_path,
                include_subfolders,
                templates,
                template_load_time_s,
                csv_writer,
                metadata_writer,
                metadata_context,
                per_video_complete_callback,
                per_race_complete_callback,
                total_source_seconds,
                return_frame_cache,
                phase_start_time,
            )
        total_videos = len(video_paths)
        total_score_screens_found = 0
        total_track_screens_found = 0
        total_race_numbers_found = 0
        per_video_summaries = []
        for video_index, video_path in enumerate(video_paths, start=1):
            video_start = time.perf_counter()
            video_stats = defaultdict(float)
            video_stats["template_load_s"] = template_load_time_s
            capture_poisoned = False

            probe = cv2.VideoCapture(video_path)
            if not probe.isOpened():
                LOGGER.log(f"[Video {video_index}/{total_videos} - Start]", f"Could not open video: {video_path}", color_name="red")
                continue
            nominal_total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
            nominal_fps = probe.get(cv2.CAP_PROP_FPS) or 1
            nominal_duration_s = nominal_total_frames / max(nominal_fps, 1)
            corrupt_check_status = "checked"
            preflight_result = video_io.sample_video_readability(video_path, nominal_total_frames, stats=video_stats)
            corrupt_check_status = str(preflight_result.get("status", "checked"))
            probe.release()
            _log_corrupt_preflight_outcome(
                video_index,
                total_videos,
                preflight_result,
                fps=nominal_fps,
            )
            processing_video_path = video_io.repair_video_if_needed(
                video_path,
                nominal_total_frames,
                preflight_result,
                duration_s=nominal_duration_s,
                stats=video_stats,
            )

            processing_readable_frames = nominal_total_frames

            cap = cv2.VideoCapture(processing_video_path)
            if not cap.isOpened():
                LOGGER.log(
                    f"[Video {video_index}/{total_videos} - Start]",
                    f"Could not open video: {processing_video_path}",
                    color_name="red",
                )
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = int(3 * max(1, int(fps)))
            score_candidates = []
            runtime_state = {
                "last_track_frame": 0,
                "last_race_frame": 0,
                "next_race_number": 1,
                "capture": cap,
            }
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            nominal_processing_total_frames = total_frames
            effective_total_frames = int(preflight_result.get("usable_total_frames") or total_frames) if preflight_result else total_frames
            total_frames = max(0, min(total_frames, effective_total_frames))
            if total_frames < nominal_processing_total_frames:
                LOGGER.log(
                    f"[Video {video_index}/{total_videos} - Start]",
                    (
                        f"Using readable frame count {total_frames:,} / {nominal_processing_total_frames:,} "
                        f"({format_duration(total_frames / max(fps, 1))}) for extraction"
                    ),
                    color_name="yellow",
                )
            processing_is_suspect = bool(preflight_result and preflight_result.get("is_suspect") and processing_video_path == video_path)
            scan_worker_count = 1 if processing_is_suspect else INITIAL_SCAN_WORKERS
            if processing_is_suspect:
                LOGGER.log(
                    f"[Video {video_index}/{total_videos} - Start]",
                    "Using conservative single-process scan for suspect video",
                    color_name="yellow",
                )
            sample_frames = np.linspace(0, total_frames - 1, 19).astype(int)
            scales = []
            video_name = os.path.basename(processing_video_path)
            video_label = build_video_identity(Path(video_path), input_root=folder_path, include_subfolders=include_subfolders)
            original_source_display_name = (
                relative_video_path(Path(video_path), folder_path) if include_subfolders else os.path.basename(video_path)
            )
            source_display_name = (
                relative_video_path(Path(processing_video_path), folder_path)
                if include_subfolders else
                os.path.basename(processing_video_path)
            )
            LOGGER.log(
                color_video_scope(f"[Video {video_index}/{total_videos} - Start]", video_index),
                color_video_message([
                    (source_display_name if include_subfolders else video_name, video_index),
                    " | Source length: ",
                    (format_duration(total_frames / max(fps, 1)), video_index),
                ]),
            )
            stage_start = time.perf_counter()
            eof_guard_frames = max(frame_skip, int(fps * INITIAL_SCAN_EOF_GUARD_SECONDS))
            for frame_num in sample_frames:
                video_io.seek_to_frame(cap, frame_num, video_stats)
                read_timeout_s = (
                    INITIAL_SCAN_EOF_READ_TIMEOUT_SECONDS
                    if total_frames - int(frame_num) <= eof_guard_frames
                    else None
                )
                if read_timeout_s is None:
                    ret, frame = video_io.read_video_frame(cap, video_stats)
                    timed_out = False
                else:
                    ret, frame, timed_out = video_io.read_video_frame_with_timeout(cap, video_stats, read_timeout_s)
                if timed_out:
                    LOGGER.log(
                        color_video_scope(f"[Video {video_index}/{total_videos} - Start]", video_index),
                        (
                            f"Aborting video after frame-read stall during scaling scan "
                            f"(requested frame {int(frame_num)}/{total_frames})"
                        ),
                        color_name="yellow",
                    )
                    capture_poisoned = True
                    break
                if not ret:
                    continue
                scale_x, scale_y, left, top, crop_width, crop_height = determine_scaling(frame)
                scales.append((scale_x, scale_y, left, top, crop_width, crop_height))
            video_io.add_timing(video_stats, "scaling_scan_s", stage_start)

            if not scales:
                if capture_poisoned:
                    LOGGER.log(
                        f"[Video {video_index}/{total_videos} - Start]",
                        f"Skipping poisoned capture after read timeout: {processing_video_path}",
                        color_name="yellow",
                    )
                else:
                    LOGGER.log(
                        f"[Video {video_index}/{total_videos} - Start]",
                        f"No valid frames found for scaling: {processing_video_path}",
                        color_name="red",
                    )
                    cap.release()
                continue

            median_scale_x = np.median([s[0] for s in scales])
            median_scale_y = np.median([s[1] for s in scales])
            median_left = int(np.median([s[2] for s in scales]))
            median_top = int(np.median([s[3] for s in scales]))
            median_crop_width = int(np.median([s[4] for s in scales]))
            median_crop_height = int(np.median([s[5] for s in scales]))

            video_io.seek_to_frame(cap, 0, video_stats)
            detection_segment_tasks = initial_scan.build_detection_segment_tasks(
                processing_video_path,
                video_label,
                source_display_name if include_subfolders else os.path.basename(video_path),
                total_frames,
                fps,
                templates,
                median_scale_x,
                median_scale_y,
                median_left,
                median_top,
                median_crop_width,
                median_crop_height,
                scan_worker_count,
                APP_CONFIG.write_debug_csv,
                INITIAL_SCAN_SEGMENT_OVERLAP_FRAMES,
                INITIAL_SCAN_MIN_SEGMENT_FRAMES,
                INITIAL_SCAN_WINDOW_STEPS,
                INITIAL_SCAN_PROGRESS_REPORT_SECONDS,
            )

            LOGGER.log(color_video_scope(f"[Video {video_index}/{total_videos} - Scan - Phase Start]", video_index), "")
            scan_progress = None
            scan_track_count = 0
            scan_race_count = 0
            if not detection_segment_tasks:
                frame_count = 0
                stage_start = time.perf_counter()
                scan_progress = ProgressPrinter(
                    color_video_scope(f"[Video {video_index}/{total_videos} - Scan]", video_index),
                    total_frames,
                    percent_step=5,
                    min_interval_s=2.0,
                )
                scan_progress.update(0)
                while cap.isOpened() and frame_count < total_frames:
                    remaining_frames = max(0, total_frames - frame_count)
                    if (
                        total_frames > 0
                        and frame_count / total_frames >= INITIAL_SCAN_EOF_GUARD_PROGRESS
                        and remaining_frames <= eof_guard_frames
                    ):
                        LOGGER.log(
                            color_video_scope(f"[Video {video_index}/{total_videos} - Scan]", video_index),
                            (
                                f"Stopping scan near EOF to avoid decoder stalls "
                                f"({remaining_frames} frames remaining, "
                                f"{remaining_frames / max(fps, 1):.1f}s tail)"
                            ),
                            color_name="yellow",
                        )
                        frame_count = total_frames
                        break
                    window_interrupted = False

                    for _ in range(INITIAL_SCAN_WINDOW_STEPS):
                        read_timeout_s = (
                            INITIAL_SCAN_EOF_READ_TIMEOUT_SECONDS
                            if remaining_frames <= eof_guard_frames
                            else None
                        )
                        if read_timeout_s is None:
                            ret, frame = video_io.read_video_frame(cap, video_stats)
                            timed_out = False
                        else:
                            ret, frame, timed_out = video_io.read_video_frame_with_timeout(cap, video_stats, read_timeout_s)
                        if timed_out:
                            LOGGER.log(
                                color_video_scope(f"[Video {video_index}/{total_videos} - Scan]", video_index),
                                (
                                    f"Aborting video after frame-read stall near EOF "
                                    f"({remaining_frames} frames remaining, "
                                    f"{remaining_frames / max(fps, 1):.1f}s tail)"
                                ),
                                color_name="yellow",
                            )
                            frame_count = total_frames
                            window_interrupted = True
                            capture_poisoned = True
                            break
                        if not ret:
                            window_interrupted = True
                            break

                        frames_to_skip = initial_scan.process_frame(
                            frame,
                            frame_count,
                            processing_video_path,
                            video_label,
                            source_display_name if include_subfolders else os.path.basename(video_path),
                            templates,
                            fps,
                            csv_writer,
                            median_scale_x,
                            median_scale_y,
                            median_left,
                            median_top,
                            median_crop_width,
                            median_crop_height,
                            video_stats,
                            runtime_state,
                            score_candidates,
                            metadata_writer,
                        )
                        if frames_to_skip > 0:
                            frame_count += frames_to_skip + frame_skip
                            if frame_count < total_frames:
                                video_io.seek_to_frame(cap, frame_count, video_stats)
                            window_interrupted = True
                            scan_progress.update(
                                min(frame_count, total_frames),
                                color_video_detail("Score candidates: ", len(score_candidates), video_label),
                                value_color_token=video_label,
                            )
                            break

                        if not video_io.advance_frames_by_grab(cap, frame_skip - 1, video_stats):
                            window_interrupted = True
                            frame_count = total_frames
                            break

                        frame_count += frame_skip
                        if frame_count >= total_frames:
                            window_interrupted = True
                            break

                        scan_progress.update(
                            frame_count,
                            color_video_detail("Score candidates: ", len(score_candidates), video_label),
                            value_color_token=video_label,
                        )

                    if window_interrupted and frame_count >= total_frames:
                        break
                if scan_progress.last_percent < 100:
                    scan_progress.update(
                        total_frames,
                        color_video_detail("Score candidates: ", len(score_candidates), video_label),
                        value_color_token=video_label,
                    )
                video_io.add_timing(video_stats, "main_scan_loop_s", stage_start)
                pre_pass2_counts = count_exported_detection_files(processing_video_path if not include_subfolders else video_label)
                scan_track_count = pre_pass2_counts["track"]
                scan_race_count = pre_pass2_counts["race"]
            else:
                stage_start = time.perf_counter()
                scan_progress = ProgressPrinter(
                    color_video_scope(f"[Video {video_index}/{total_videos} - Scan]", video_index),
                    total_frames,
                    percent_step=5,
                    min_interval_s=1.0,
                )
                scan_progress.update(0)
                parallel_scan_diag = {} if INITIAL_SCAN_DIAGNOSTICS_ENABLED else None
                segment_results = initial_scan.run_parallel_detection_segments(
                    detection_segment_tasks,
                    scan_progress,
                    diagnostics=parallel_scan_diag,
                )
                video_io.add_timing(video_stats, "main_scan_loop_s", stage_start)

                merged_score_detections = []
                merged_track_detections = []
                merged_race_detections = []
                merged_debug_rows = []

                merge_start = time.perf_counter()
                for result in sorted(segment_results, key=lambda item: item["segment_index"]):
                    for key, value in result["stats"].items():
                        video_stats[key] += value
                    merged_score_detections.extend(result["score_detections"])
                    merged_track_detections.extend(result["track_detections"])
                    merged_race_detections.extend(result["race_detections"])
                    merged_debug_rows.extend(result["debug_rows"])
                parallel_merge_s = time.perf_counter() - merge_start

                dedupe_start = time.perf_counter()
                min_gap_frames = int(fps * 20)
                merged_score_detections = initial_scan.merge_nearby_detections(merged_score_detections, min_gap_frames)
                merged_track_detections = initial_scan.merge_nearby_detections(merged_track_detections, min_gap_frames)
                merged_race_detections = initial_scan.merge_nearby_detections(merged_race_detections, min_gap_frames)
                parallel_dedupe_s = time.perf_counter() - dedupe_start

                score_frame_numbers = [item["frame_number"] for item in merged_score_detections]
                score_candidates = [
                    {
                        "race_number": index + 1,
                        "frame_number": item["frame_number"],
                        "score_layout_id": item.get("score_layout_id"),
                    }
                    for index, item in enumerate(merged_score_detections)
                ]

                if APP_CONFIG.write_debug_csv and merged_debug_rows:
                    csv_writer.writerows(merged_debug_rows)

                auxiliary_detections = [
                    {"kind": "track", **item}
                    for item in merged_track_detections
                ]
                auxiliary_detections.extend({"kind": "race", **item} for item in merged_race_detections)
                auxiliary_detections.sort(key=lambda item: item["frame_number"])
                if auxiliary_detections:
                    LOGGER.log(color_video_scope(f"[Video {video_index}/{total_videos} - Scan - Confirmed Results]", video_label), "")
                scan_track_count = len(merged_track_detections)
                scan_race_count = len(merged_race_detections)
                auxiliary_save_start = time.perf_counter()
                initial_scan.save_auxiliary_detection_frames(
                    cap,
                    processing_video_path,
                    video_label,
                    source_display_name if include_subfolders else os.path.basename(video_path),
                    auxiliary_detections,
                    score_frame_numbers,
                    median_left,
                    median_top,
                    median_crop_width,
                    median_crop_height,
                    fps,
                    video_stats,
                    metadata_writer,
                )
                auxiliary_save_s = time.perf_counter() - auxiliary_save_start
                if INITIAL_SCAN_DIAGNOSTICS_ENABLED:
                    def _format_seconds_precise(value):
                        return f"{float(value):.2f}s"

                    diag_lines = [
                        f"Mode: parallel {parallel_scan_diag.get('executor', 'unknown')} x {parallel_scan_diag.get('segment_count', len(detection_segment_tasks))}",
                        f"Task submit/startup: {_format_seconds_precise(parallel_scan_diag.get('submit_startup_s', 0.0))}",
                        f"First segment result: {_format_seconds_precise(parallel_scan_diag.get('first_result_s', 0.0))}",
                        f"Parallel wait/collect: {_format_seconds_precise(parallel_scan_diag.get('parallel_wait_s', 0.0))}",
                        f"Merge worker results: {_format_seconds_precise(parallel_merge_s)}",
                        f"Deduplicate detections: {_format_seconds_precise(parallel_dedupe_s)}",
                        f"Save auxiliary frames: {_format_seconds_precise(auxiliary_save_s)}",
                        f"Raw merged detections: score {len(merged_score_detections)} | track {len(merged_track_detections)} | race {len(merged_race_detections)}",
                    ]
                    LOGGER.summary_block(
                        f"[Video {video_index}/{total_videos} - Scan Diagnostics]",
                        diag_lines,
                        color_name="dim",
                    )
            if capture_poisoned:
                LOGGER.log(
                    color_video_scope(f"[Video {video_index}/{total_videos} - Complete]", video_index),
                    f"{video_name} | Aborted after timed-out read to avoid reusing a poisoned decoder",
                    color_name="yellow",
                )
                video_stats["video_total_s"] = time.perf_counter() - video_start
                continue
            scan_duration = float(video_stats.get("main_scan_loop_s", 0.0))
            scan_speed = total_frames / scan_duration if scan_duration > 0 else 0.0
            scan_summary_lines = [
                f"Duration: {format_duration(scan_duration)}",
                f"Source length: {format_duration(total_frames / max(fps, 1))}",
                f"Frames scanned: {total_frames:,}",
                f"Scan speed: {scan_speed:,.0f} frames/s",
                f"Track screens found: {scan_track_count}",
                f"Race numbers found: {scan_race_count}",
                f"Total score screens queued: {len(score_candidates)}",
            ]
            if scan_progress is not None:
                scan_summary_lines.extend(scan_progress.peak_lines())
            LOGGER.summary_block(
                color_video_scope(f"[Video {video_index}/{total_videos} - Scan - Phase Complete]", video_index),
                scan_summary_lines,
                color_name="green",
            )
            LOGGER.log(color_video_scope(f"[Video {video_index}/{total_videos} - Total Score Screen - Phase Start]", video_index), "")
            total_score_progress = ProgressPrinter(
                color_video_scope(f"[Video {video_index}/{total_videos} - Total Score Screen]", video_index),
                max(1, len(score_candidates)),
                percent_step=5,
                min_interval_s=2.0,
            )
            total_score_progress.update(0)
            race_complete_callback = per_race_complete_callback
            if race_complete_callback is not None and metadata_context is not None:
                def race_complete_callback(payload, _callback=race_complete_callback):
                    metadata_context.flush()
                    _callback(payload)
            process_score_candidates(
                processing_video_path,
                video_label,
                source_display_name if include_subfolders else os.path.basename(video_path),
                score_candidates,
                templates,
                fps,
                csv_writer,
                median_scale_x,
                median_scale_y,
                median_left,
                median_top,
                median_crop_width,
                median_crop_height,
                video_stats,
                metadata_writer,
                video_index=video_index,
                total_videos=total_videos,
                progress=total_score_progress,
                per_race_complete_callback=race_complete_callback,
            )
            exported_counts = count_exported_detection_files(processing_video_path if not include_subfolders else video_label)
            total_score_screens_found += exported_counts["score"]
            total_track_screens_found += exported_counts["track"]
            total_race_numbers_found += exported_counts["race"]
            if not capture_poisoned:
                cap.release()
            video_stats["video_total_s"] = time.perf_counter() - video_start
            if total_score_progress.last_percent < 100:
                total_score_progress.update(len(score_candidates))
            LOGGER.summary_block(
                color_video_scope(f"[Video {video_index}/{total_videos} - Total Score Screen - Phase Complete]", video_index),
                [
                    f"Duration: {format_duration(video_stats.get('score_candidate_pass_s', 0.0))}",
                    f"Total score screens found: {exported_counts['total']}",
                    *total_score_progress.peak_lines(),
                ] if total_score_progress.peak_lines() else [
                    f"Duration: {format_duration(video_stats.get('score_candidate_pass_s', 0.0))}",
                    f"Total score screens found: {exported_counts['total']}",
                ],
                color_name="green",
            )
            print_extract_profiler_summary(video_name, video_stats)
            LOGGER.log(
                color_video_scope(f"[Video {video_index}/{total_videos} - Complete]", video_label),
                f"{video_name} | Elapsed until complete: {format_duration(video_stats['video_total_s'])} | "
                f"Source length: {format_duration(total_frames / max(fps, 1))} | Track screens: {exported_counts['track']} | "
                f"Race numbers: {exported_counts['race']} | Total score screens: {exported_counts['total']}",
                color_name="green",
            )
            if video_index < total_videos:
                LOGGER.blank_lines(2)
            per_video_summaries.append(
                {
                    "video_name": video_name,
                    "video_label": video_label,
                    "source_display_name": source_display_name,
                    "source_length_s": total_frames / max(fps, 1),
                    "processing_duration_s": float(video_stats["video_total_s"]),
                    "scan_duration_s": float(video_stats.get("main_scan_loop_s", 0.0)),
                    "total_score_duration_s": float(video_stats.get("score_candidate_pass_s", 0.0)),
                    "corrupt_check_duration_s": float(video_stats.get("corrupt_check_duration_s", 0.0)),
                    "corrupt_check_status": corrupt_check_status,
                    "repair_duration_s": float(video_stats.get("repair_duration_s", 0.0)),
                    "repair_created": bool(video_stats.get("repair_created", 0)),
                    "track_screens": exported_counts["track"],
                    "race_numbers": exported_counts["race"],
                    "total_score_screens": exported_counts["total"],
                }
            )
            if per_video_complete_callback is not None:
                if metadata_context is not None:
                    metadata_context.flush()
                per_video_cache = {
                    key: value[:]
                    for key, value in CONSENSUS_FRAME_CACHE.items()
                    if str(key[0]) == str(video_label)
                }
                metadata_index = load_exported_frame_metadata(Path(PROJECT_ROOT))
                per_video_complete_callback(
                    {
                        "video_name": video_name,
                        "video_label": video_label,
                        "summary": per_video_summaries[-1],
                        "frame_bundle_cache": per_video_cache,
                        "metadata_index": {
                            key: value
                            for key, value in metadata_index.items()
                            if str(value.get("video_label", "")).strip() == str(video_label)
                            or Path(str(value.get("video", ""))).stem == str(video_label)
                        },
                    }
                )
    finally:
        if csv_context is not None:
            csv_context.close()
        if metadata_context is not None:
            metadata_context.close()

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - phase_start_time

    # Convert elapsed time to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # Print the elapsed time in mm:ss format
    extract_summary = {
        "duration_s": elapsed_time,
        "total_source_seconds": total_source_seconds,
        "track_screens": total_track_screens_found,
        "race_numbers": total_race_numbers_found,
        "total_score_screens": total_score_screens_found,
        "corrupt_check_duration_s": sum(float(item.get("corrupt_check_duration_s", 0.0)) for item in per_video_summaries),
        "repair_duration_s": sum(float(item.get("repair_duration_s", 0.0)) for item in per_video_summaries),
        "repair_count": sum(1 for item in per_video_summaries if item.get("repair_created")),
        "per_video_summaries": per_video_summaries,
    }
    extract_lines = [
        f"Duration: {format_duration(elapsed_time)}",
        f"Source video total: {format_duration(total_source_seconds)}",
        f"Track screens found: {total_track_screens_found}",
        f"Race numbers found: {total_race_numbers_found}",
        f"Total score screens found: {total_score_screens_found}",
    ]
    extract_lines.extend(LOGGER.peak_lines())
    LOGGER.summary_block("[Extract - Phase Complete]", extract_lines, color_name="green")
    if return_frame_cache:
        return {
            "frame_bundle_cache": {key: value[:] for key, value in CONSENSUS_FRAME_CACHE.items()},
            "summary": extract_summary,
        }
    return {"summary": extract_summary}


def main():
    parser = argparse.ArgumentParser(description="Extract Mario Kart 8 race and score screens")
    parser.add_argument("--video", help="Process only a specific video filename")
    args = parser.parse_args()
    extract_frames(return_frame_cache=False, selected_videos=[args.video] if args.video else None)

if __name__ == "__main__":
    main()
