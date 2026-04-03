import argparse
import cProfile
import importlib.util
import json
import os
import pstats
import re
import shutil
import subprocess
import sys
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
os.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")
import cv2
import colorsys
import multiprocessing as mp
import threading
import queue
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None

from PIL import Image

from .app_runtime import (
    check_runtime,
    detect_easyocr_runtime,
    detect_gpu_runtime,
    easyocr_gpu_enabled as runtime_easyocr_gpu_enabled,
    effective_overlap_ocr_mode as runtime_effective_overlap_ocr_mode,
    load_app_config,
    open_path,
    update_app_config_values,
)
from .console_logging import LOGGER
from .data_paths import resolve_asset_file
from .extract_common import EXPORT_IMAGE_FORMAT, remove_tree_contents, should_include_input_video_path
from .ocr_export import build_completion_payload
from .ocr_scoreboard_consensus import build_race_warning_messages
from .project_paths import PROJECT_ROOT
from . import extract_initial_scan as initial_scan
from . import extract_frames
from . import extract_video_io


APP_CONFIG = load_app_config()
SCRIPT_DIR = PROJECT_ROOT
INPUT_DIR = SCRIPT_DIR / "Input_Videos"
OUTPUT_DIR = SCRIPT_DIR / "Output_Results"
FRAMES_DIR = OUTPUT_DIR / "Frames"
DEBUG_DIR = OUTPUT_DIR / "Debug"
DEBUG_SCORE_FRAMES_DIR = DEBUG_DIR / "Score_Frames"
EXTRACT_MODULE = "mk8_local_play.extract_frames"
OCR_MODULE = "mk8_local_play.extract_text"
PROFILE_OUTPUT = DEBUG_DIR / "performance_profile.txt"
OCR_TRACE_DIR = DEBUG_DIR / "OCR_Tracing"


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mkv", ".mov", ".avi", ".webm"}


def configure_headless_debug_outputs(enabled: bool | None) -> None:
    if enabled is None:
        return
    value = "1" if enabled else "0"
    os.environ["MK8_WRITE_DEBUG_CSV"] = value
    os.environ["MK8_WRITE_DEBUG_SCORE_IMAGES"] = value
    os.environ["MK8_WRITE_DEBUG_LINKING_EXCEL"] = value


def _workflow_sorted_source_summaries(video_files: list[Path], *, include_subfolders: bool) -> tuple[list[tuple[str, str]], float]:
    workflow_plan, total_source_seconds = extract_frames.build_workflow_video_plan(
        video_files,
        INPUT_DIR,
        include_subfolders=include_subfolders,
    )
    source_summaries = [
        (
            entry["video_label"],
            f"{entry['source_display_name']} ({extract_frames.format_duration(entry['source_length_s'])})",
        )
        for entry in workflow_plan
    ]
    return source_summaries, total_source_seconds


def ocr_trace_enabled() -> bool:
    return os.environ.get("MK8_TRACE_OCR_LINKING", "0").strip().lower() in {"1", "true", "yes", "on"}


def ensure_ocr_trace_env(*, run_mode: str) -> tuple[str, str]:
    trace_label = os.environ.get("MK8_OCR_TRACE_LABEL", "").strip()
    if not trace_label:
        trace_label = time.strftime("%Y%m%d_%H%M%S")
        os.environ["MK8_OCR_TRACE_LABEL"] = trace_label
    os.environ.setdefault("MK8_OCR_TRACE_MODE", run_mode)
    return trace_label, os.environ["MK8_OCR_TRACE_MODE"]


def append_ocr_trace_event(filename: str, payload: dict) -> None:
    if not ocr_trace_enabled():
        return
    trace_label = os.environ.get("MK8_OCR_TRACE_LABEL", "").strip() or "adhoc"
    trace_mode = os.environ.get("MK8_OCR_TRACE_MODE", "").strip() or "unspecified"
    trace_path = OCR_TRACE_DIR / trace_label / trace_mode / filename
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    event_payload = dict(payload)
    event_payload.setdefault("trace_label", trace_label)
    event_payload.setdefault("trace_mode", trace_mode)
    event_payload.setdefault("pid", os.getpid())
    event_payload.setdefault("ts", time.time())
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")


def build_video_identity(video_path: Path, *, include_subfolders: bool = False) -> str:
    if not include_subfolders:
        return video_path.stem
    try:
        relative_path = video_path.relative_to(INPUT_DIR)
    except ValueError:
        relative_path = video_path if not video_path.is_absolute() else Path(video_path.name)
    path_without_suffix = relative_path.with_suffix("")
    sanitized_parts = [
        re.sub(r"[^A-Za-z0-9._-]+", "_", part).strip("._-") or "part"
        for part in path_without_suffix.parts
    ]
    return "__".join(sanitized_parts)


def discover_input_video_files(*, include_subfolders: bool = False) -> list[Path]:
    if not INPUT_DIR.exists():
        return []
    iterator = INPUT_DIR.rglob("*") if include_subfolders else INPUT_DIR.iterdir()
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
        and should_include_input_video_path(path, INPUT_DIR, include_subfolders=include_subfolders)
    )


def _selected_input_video_files_single(selected_video: str | None = None, *, include_subfolders: bool = False) -> list[Path]:
    all_video_files = discover_input_video_files(include_subfolders=include_subfolders)
    if not selected_video:
        return all_video_files
    selected_relative = selected_video.replace("\\", "/").lower()
    if include_subfolders and ("/" in selected_relative or "\\" in selected_video):
        relative_matches = [
            path for path in all_video_files
            if str(path.relative_to(INPUT_DIR)).replace("\\", "/").lower() == selected_relative
        ]
        if relative_matches:
            return relative_matches
    selected_name = Path(selected_video).name.lower()
    exact_matches = [path for path in all_video_files if path.name.lower() == selected_name]
    if exact_matches:
        return exact_matches
    relative_matches = [
        path for path in all_video_files
        if str(path.relative_to(INPUT_DIR)).replace("\\", "/").lower() == selected_relative
    ]
    if relative_matches:
        return relative_matches
    selected_stem = Path(selected_video).stem.lower()
    stem_matches = [path for path in all_video_files if path.stem.lower() == selected_stem]
    if not include_subfolders:
        return stem_matches
    selected_identity = build_video_identity(Path(selected_video), include_subfolders=True).lower()
    return [
        path for path in all_video_files
        if build_video_identity(path, include_subfolders=True).lower() == selected_identity
    ] or stem_matches


def selected_input_video_files(
    selected_video: str | list[str] | None = None,
    *,
    include_subfolders: bool = False,
) -> list[Path]:
    if isinstance(selected_video, (list, tuple)):
        resolved_files: list[Path] = []
        seen_paths: set[str] = set()
        missing_requests: list[str] = []
        for requested_video in selected_video:
            requested_text = str(requested_video or "").strip()
            if not requested_text:
                continue
            matches = _selected_input_video_files_single(
                requested_text,
                include_subfolders=include_subfolders,
            )
            if not matches:
                missing_requests.append(requested_text)
                continue
            for match in matches:
                normalized_path = str(match.resolve())
                if normalized_path in seen_paths:
                    continue
                seen_paths.add(normalized_path)
                resolved_files.append(match)
        if missing_requests:
            missing_text = ", ".join(missing_requests)
            raise RuntimeError(f"No supported videos found for selection: {missing_text}")
        return resolved_files
    return _selected_input_video_files_single(selected_video, include_subfolders=include_subfolders)


def selected_race_classes_for_videos(video_files: list[Path], *, include_subfolders: bool = False) -> list[str]:
    return [build_video_identity(path, include_subfolders=include_subfolders) for path in video_files]


def resolve_project_python() -> str:
    """Prefer the repo-local virtualenv so CLI runs use the installed project dependencies."""
    candidate_paths = [
        SCRIPT_DIR / ".venv" / "Scripts" / "python.exe",
        SCRIPT_DIR / ".venv" / "bin" / "python",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def find_latest_results_xlsx(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("*_Tournament_Results.xlsx"))
    if candidates:
        return candidates[-1]
    return output_dir / "No_Tournament_Results_Found.xlsx"


def show_info(title: str, message: str) -> None:
    if messagebox is not None:
        messagebox.showinfo(title, message)
    else:
        print(f"{title}: {message}")


def show_warning(title: str, message: str) -> None:
    if messagebox is not None:
        messagebox.showwarning(title, message)
    else:
        print(f"{title}: {message}")


def show_error(title: str, message: str) -> None:
    if messagebox is not None:
        messagebox.showerror(title, message)
    else:
        print(f"{title}: {message}", file=sys.stderr)


def _pad_table_cell(value: object, width: int, alignment: str = "left") -> str:
    text = str(value)
    if alignment == "right":
        return text.rjust(width)
    return text.ljust(width)


def _format_simple_table(headers: list[str], rows: list[list[str]], alignments: list[str] | None = None) -> list[str]:
    if not rows:
        return []
    alignments = list(alignments or ["left"] * len(headers))
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    table_lines = [
        "  " + "  ".join(_pad_table_cell(headers[index], widths[index], alignments[index]) for index in range(len(headers))),
        "  " + "  ".join(("-" * widths[index]).ljust(widths[index]) for index in range(len(headers))),
    ]
    for row in rows:
        table_lines.append(
            "  " + "  ".join(_pad_table_cell(row[index], widths[index], alignments[index]) for index in range(len(headers)))
        )
    return table_lines


def _format_metric_table(rows: list[tuple[str, str]]) -> list[str]:
    if not rows:
        return []
    metric_width = max(len("Metric"), max(len(metric) for metric, _ in rows))
    value_width = max(len("Value"), max(len(value) for _, value in rows))
    lines = [
        "  " + "  ".join(["Metric".ljust(metric_width), "Value".ljust(value_width)]),
        "  " + "  ".join([("-" * metric_width).ljust(metric_width), ("-" * value_width).ljust(value_width)]),
    ]
    for metric, value in rows:
        lines.append(
            "  " + metric.ljust(metric_width) + "  " + LOGGER.bold(value.ljust(value_width))
        )
    return lines


def _format_simple_table_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))
    return widths


def _format_colored_table_row(values: list[str], widths: list[int], video_identity: object, alignments: list[str] | None = None) -> str:
    alignments = list(alignments or ["left"] * len(values))
    padded_values = [
        LOGGER.video_value(_pad_table_cell(values[index], widths[index], alignments[index]), video_identity)
        for index in range(len(values))
    ]
    return "  " + "  ".join(padded_values)


def _format_input_summary_lines(source_summaries: list[tuple[str, str]], total_source_seconds: float) -> list[str]:
    lines = [
        f"Videos selected: {len(source_summaries)}",
        f"Total source length: {extract_frames.format_duration(total_source_seconds)}",
    ]
    if source_summaries:
        lines.extend(["", "Selection"])
        for index, (_video_identity, summary) in enumerate(source_summaries, start=1):
            lines.append(f"{index:02}. {summary}")
    return lines


def _summarize_pipeline_bottleneck(
    *,
    extract_duration_s: float,
    ocr_duration_s: float,
    total_processing_seconds: float,
) -> str:
    if total_processing_seconds <= 0:
        return "n/a"
    extract_share = extract_duration_s / total_processing_seconds
    ocr_share = ocr_duration_s / total_processing_seconds
    if extract_share >= 0.75 and extract_duration_s >= (ocr_duration_s * 1.15):
        return "Video loading and frame extraction"
    if ocr_share >= 0.65 and ocr_duration_s >= (extract_duration_s * 0.90):
        return "OCR and workbook export"
    if extract_share >= 0.55:
        return "Mostly video loading and frame extraction"
    if ocr_share >= 0.45:
        return "Mixed, leaning OCR and export"
    return "Mixed pipeline"


def _review_summary_text(video_ocr_summary: dict) -> str:
    if not video_ocr_summary:
        return "n/a"
    review_race_count = int(video_ocr_summary.get("review_race_count", 0))
    review_row_count = int(video_ocr_summary.get("review_row_count", 0))
    if review_race_count <= 0 and review_row_count <= 0:
        return "none"
    return f"{review_race_count} races / {review_row_count} rows"


def _video_labeled_value(prefix: str, value: object, video_identity: object, suffix: str = "") -> str:
    return prefix + LOGGER.video_value(value, video_identity) + suffix


OVERLAP_SCOPE_OCR = "[Run - Overlap OCR]"
OVERLAP_SCOPE = "[Run - Overlap]".ljust(len(OVERLAP_SCOPE_OCR))


def _short_overlap_video_label(video_label: str) -> str:
    text = str(video_label or "").strip()
    if "__" in text:
        text = text.split("__")[-1]
    return text or "unknown"


def _format_overlap_subject(video_label: str, *, width: int = 28) -> str:
    return LOGGER.video_value(_short_overlap_video_label(video_label).ljust(width), video_label)


def _format_overlap_queue_status(*, queued_for_video: int, queued_total: int) -> str:
    return f"Q {int(queued_for_video):>2} | GQ {int(queued_total):>2}"


def _format_overlap_race_event(
    video_label: str,
    action: str,
    race_number: int,
    *,
    queued_for_video: int | None = None,
    queued_total: int | None = None,
) -> str:
    status_text = ""
    if queued_for_video is not None and queued_total is not None:
        status_text = " | " + _format_overlap_queue_status(
            queued_for_video=queued_for_video,
            queued_total=queued_total,
        )
    return (
        _format_overlap_subject(video_label)
        + f" | OCR {str(action).ljust(8)}"
        + f" | R{int(race_number):03}"
        + status_text
    )


def _format_overlap_extraction_complete(video_label: str, race_count: int) -> str:
    return _format_overlap_subject(video_label) + " | Extract ready | races " + LOGGER.video_value(race_count, video_label)


def _format_overlap_queue_video(video_label: str) -> str:
    return _format_overlap_subject(video_label) + " | OCR queued"


def _format_overlap_start_video(video_label: str) -> str:
    return _format_overlap_subject(video_label) + " | OCR started"


def _format_overlap_complete(video_label: str, race_count: int, duration_text: str) -> str:
    return (
        _format_overlap_subject(video_label)
        + " | OCR done"
        + " | races "
        + LOGGER.video_value(race_count, video_label)
        + " | "
        + LOGGER.video_value(duration_text, video_label)
    )


def _format_overlap_finalize_start(video_label: str, race_count: int) -> str:
    return (
        _format_overlap_subject(video_label)
        + " | OCR finalize"
        + " | races "
        + LOGGER.video_value(race_count, video_label)
    )


def _format_overlap_complete_with_finalize(video_label: str, race_count: int, duration_text: str, finalize_duration_text: str) -> str:
    return (
        _format_overlap_complete(video_label, race_count, duration_text)
        + " | finalize "
        + LOGGER.video_value(finalize_duration_text, video_label)
    )


def _format_overlap_progress_line(video_label: str, detail_text: str) -> str:
    return _format_overlap_subject(video_label) + " | " + detail_text


def confirm_yes_no(title: str, message: str) -> bool:
    if messagebox is not None:
        return bool(messagebox.askyesno(title, message))
    reply = input(f"{title}: {message} [Yes/No] ").strip().lower()
    return reply in {"y", "yes"}


def ensure_output_results_structure() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_SCORE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)


def clear_output_results_for_videos(video_files: list[Path], *, include_subfolders: bool = False) -> bool:
    ensure_output_results_structure()
    deleted_anything = False
    for video_path in video_files:
        video_label = build_video_identity(video_path, include_subfolders=include_subfolders)
        candidate_paths = [
            FRAMES_DIR / video_label,
            DEBUG_SCORE_FRAMES_DIR / video_label,
            DEBUG_DIR / "Identity_Linking" / video_label,
            DEBUG_DIR / "Low_Res" / video_label,
        ]
        for candidate in candidate_paths:
            if not candidate.exists():
                continue
            if candidate.is_dir():
                shutil.rmtree(candidate)
                deleted_anything = True
            else:
                if candidate.name != ".gitkeep":
                    candidate.unlink()
                    deleted_anything = True
    return deleted_anything


def clear_output_results(*, require_confirmation: bool = True) -> bool:
    if require_confirmation and not confirm_yes_no(
        "Confirm",
        "Are you sure you want to clear all files in Output_Results?",
    ):
        return False

    if OUTPUT_DIR.exists():
        for child in OUTPUT_DIR.iterdir():
            try:
                if child.is_dir():
                    remove_tree_contents(child)
                else:
                    if child.name == ".gitkeep":
                        continue
                    child.unlink()
            except Exception as exc:
                raise RuntimeError(f"Unable to delete {child}: {exc}") from exc

    ensure_output_results_structure()
    return True


def run_python_module(module_name: str, extra_args: list[str] | None = None) -> None:
    # Always prefer the repo-local virtualenv for child scripts so extraction and OCR
    # run with the same dependencies a user installed during setup.
    command = [resolve_project_python(), "-m", module_name]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)


def ensure_runtime_or_raise(require_ffmpeg: bool = False) -> None:
    issues = check_runtime(APP_CONFIG, require_ffmpeg=require_ffmpeg)
    if issues:
        raise RuntimeError("\n".join(issues))


def select_video() -> None:
    try:
        run_extract()
        show_info("Success", "Video analyzed and races found.")
    except Exception as exc:
        show_error("Error", str(exc))


def export_to_excel() -> None:
    try:
        run_ocr()
        show_info("Success", "Races exported to Excel.")
    except Exception as exc:
        show_error("Error", str(exc))


def open_excel_scores() -> None:
    latest_results_xlsx = find_latest_results_xlsx(OUTPUT_DIR)
    if latest_results_xlsx.exists():
        try:
            open_path(latest_results_xlsx)
        except Exception as exc:
            show_error("Error", f"Unable to open the Excel file: {exc}")
    else:
        show_warning("Warning", "Please perform Step 3 first.")


def open_frames_folder() -> None:
    if FRAMES_DIR.exists():
        try:
            open_path(FRAMES_DIR)
        except Exception as exc:
            show_error("Error", f"Unable to open the folder: {exc}")
    else:
        show_warning("Warning", "The frames folder does not exist.")


def clear_all_races_found() -> None:
    deleted_anything = False
    if FRAMES_DIR.exists():
        try:
            deleted_anything = remove_tree_contents(FRAMES_DIR) or deleted_anything
        except Exception as exc:
            show_error("Error", f"Unable to clear frames folder: {exc}")
            return
    else:
        show_warning("Warning", "The frames folder does not exist.")
        return

    if DEBUG_SCORE_FRAMES_DIR.exists():
        try:
            deleted_anything = remove_tree_contents(DEBUG_SCORE_FRAMES_DIR) or deleted_anything
        except Exception as exc:
            show_error("Error", f"Unable to clear debug score frames folder: {exc}")
            return

    if deleted_anything:
        show_info("Success", "Found race screenshots and annotated screenshots have been deleted.")
    else:
        show_info("Info", "No found race screenshots were present to delete.")


def clear_output_results_gui() -> None:
    try:
        deleted = clear_output_results(require_confirmation=True)
        if deleted:
            show_info("Success", "Output_Results has been cleared.")
        else:
            show_info("Cancelled", "Output_Results was not cleared.")
    except Exception as exc:
        show_error("Error", str(exc))


def open_videos_folder() -> None:
    if INPUT_DIR.exists():
        try:
            open_path(INPUT_DIR)
        except Exception as exc:
            show_error("Error", f"Unable to open the folder: {exc}")
    else:
        show_warning("Warning", "The Input_Videos folder does not exist.")


def merge_videos() -> None:
    if filedialog is None:
        raise RuntimeError("Tk file dialogs are unavailable in this environment.")

    ensure_runtime_or_raise(require_ffmpeg=True)

    file_paths = filedialog.askopenfilenames(
        title="Select Videos to Merge",
        initialdir=str(INPUT_DIR),
        filetypes=[("Video Files", "*.mp4;*.mkv;*.avi")],
    )
    if not file_paths:
        return

    output_file = filedialog.asksaveasfilename(
        title="Save Merged Video As",
        defaultextension=".mp4",
        filetypes=[("MP4 Files", "*.mp4")],
    )
    if not output_file:
        return

    temp_file = SCRIPT_DIR / "file_list.txt"
    try:
        with temp_file.open("w", encoding="utf-8") as handle:
            for file_path in file_paths:
                handle.write(f"file '{file_path}'\n")

        ffmpeg_command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", str(temp_file), "-c", "copy", "-y", output_file]
        subprocess.run(ffmpeg_command, check=True)
        show_info("Success", f"Videos merged successfully into {output_file}")
    except subprocess.CalledProcessError as exc:
        show_error("Error", f"An error occurred while merging videos: {exc}")
    except Exception as exc:
        show_error("Error", f"An unexpected error occurred: {exc}")
    finally:
        if temp_file.exists():
            temp_file.unlink()


def run_extract(
    selected_video: str | list[str] | None = None,
    *,
    include_subfolders: bool = False,
    debug: bool | None = None,
) -> None:
    configure_headless_debug_outputs(debug)
    ensure_runtime_or_raise()
    video_files = selected_input_video_files(
        selected_video=selected_video,
        include_subfolders=include_subfolders,
    )
    if selected_video:
        clear_output_results_for_videos(video_files, include_subfolders=include_subfolders)
    extra_args = []
    if isinstance(selected_video, str) and selected_video:
        extra_args.extend(["--video", selected_video])
    if include_subfolders:
        extra_args.append("--subfolders")
    run_python_module(EXTRACT_MODULE, extra_args=extra_args)


def run_ocr(
    selected_video: str | list[str] | None = None,
    *,
    include_subfolders: bool = False,
    selection_mode: bool = False,
    debug: bool | None = None,
) -> None:
    configure_headless_debug_outputs(debug)
    ensure_runtime_or_raise()
    extra_args = []
    if selection_mode:
        video_files = selected_input_video_files(
            selected_video=selected_video,
            include_subfolders=include_subfolders,
        )
        if not video_files:
            target = selected_video or "Input_Videos"
            raise RuntimeError(f"No supported videos found for selection: {target}")
        for race_class in selected_race_classes_for_videos(
            video_files,
            include_subfolders=include_subfolders,
        ):
            extra_args.extend(["--race-class", race_class])
    elif isinstance(selected_video, str) and selected_video:
        selected_identifier = (
            build_video_identity(Path(selected_video), include_subfolders=True)
            if include_subfolders else Path(selected_video).stem
        )
        extra_args.extend(["--video", selected_identifier])
    run_python_module(OCR_MODULE, extra_args=extra_args)


def _format_overlap_ocr_detail(event: dict, format_duration) -> str:
    completed = int(event.get("completed", 0))
    total = max(1, int(event.get("total", 1)))
    percent = min(100, int((completed / total) * 100))
    elapsed_text = format_duration(float(event.get("elapsed_s", 0.0)))
    if event.get("event") == "progress":
        race_id = event.get("race_id")
        track_name = str(event.get("track_name") or "UNKNOWN").strip() or "UNKNOWN"
        if len(track_name) > 28:
            track_name = track_name[:25] + "..."
        race_players = event.get("race_score_players")
        total_players = event.get("total_score_players")
        players_text = ""
        if race_players is not None or total_players is not None:
            players_text = f" | P {int(race_players or 0):>2}/{int(total_players or 0):>2}"
        race_text = f" | R{int(race_id):03}" if race_id is not None else ""
        return f"OCR {completed:02}/{total:02} ({percent:>3}%)" + race_text + f" | {track_name:<28}{players_text} | {elapsed_text}"
    detail = str(event.get("detail") or "").strip()
    pending = event.get("pending")
    global_pending = event.get("global_pending")
    pending_text = f" | Q {int(pending):>2}" if pending is not None else ""
    global_pending_text = f" | GQ {int(global_pending):>2}" if global_pending is not None else ""
    detail_text = f" | {detail}" if detail else ""
    return f"OCR {completed:02}/{total:02} ({percent:>3}%) | {elapsed_text}{pending_text}{global_pending_text}{detail_text}"


def _make_overlap_ocr_progress_callback(video_label: str, format_duration, display_video_label):
    def _callback(event: dict) -> None:
        LOGGER.log(
            OVERLAP_SCOPE_OCR,
            _format_overlap_progress_line(
                video_label,
                _format_overlap_ocr_detail(event, format_duration),
            ),
        )

    return _callback


def _overlap_ocr_process_worker(job_queue, result_queue, progress_queue) -> None:
    from . import extract_text

    while True:
        job = job_queue.get()
        if job is None:
            append_ocr_trace_event("scheduler_events.jsonl", {"event": "video_worker_exit"})
            result_queue.put({"event": "worker_exit"})
            progress_queue.put({"event": "worker_exit"})
            return
        video_label = str(job["video_label"])
        video_name = str(job["video_name"])

        def _progress_callback(event: dict) -> None:
            payload = dict(event)
            payload["video_label"] = video_label
            progress_queue.put(payload)

        try:
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {"event": "video_worker_start", "video_label": video_label, "video_name": video_name},
            )
            progress_queue.put({"event": "worker_start", "video_label": video_label})
            result = extract_text.process_images_in_folder(
                str(FRAMES_DIR),
                selected_race_classes=[video_label],
                write_outputs=False,
                emit_logs=False,
                progress_callback=_progress_callback,
            )
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {
                    "event": "video_worker_result",
                    "video_label": video_label,
                    "video_name": video_name,
                    "race_count": int(result.get("race_count", 0)),
                    "duration_s": float(result.get("duration_s", 0.0)),
                },
            )
            result_queue.put(
                {
                    "event": "result",
                    "video_label": video_label,
                    "video_name": video_name,
                    "result": result,
                }
            )
        except BaseException as exc:
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {"event": "video_worker_error", "video_label": video_label, "video_name": video_name, "error": repr(exc)},
            )
            result_queue.put(
                {
                    "event": "error",
                    "video_label": video_label,
                    "video_name": video_name,
                    "error": repr(exc),
                }
            )


def _overlap_ocr_race_process_worker(job_queue, result_queue, progress_queue) -> None:
    from . import extract_text

    base_dir = Path(PROJECT_ROOT)
    input_videos_folder = base_dir / "Input_Videos"
    text_detected_folder = os.path.join(PROJECT_ROOT, 'Output_Results', 'Debug', 'Score_Frames')
    if extract_text.APP_CONFIG.write_debug_score_images and not os.path.exists(text_detected_folder):
        os.makedirs(text_detected_folder, exist_ok=True)

    while True:
        job = job_queue.get()
        if job is None:
            append_ocr_trace_event("scheduler_events.jsonl", {"event": "race_worker_exit"})
            result_queue.put({"event": "worker_exit"})
            progress_queue.put({"event": "worker_exit"})
            return
        video_label = str(job["video_label"])
        race_id = int(job["race_id"])
        ocr_revision = max(1, int(job.get("ocr_revision", 1) or 1))
        grouped_item = job["grouped_item"]
        try:
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {"event": "race_worker_start", "video_label": video_label, "race_id": race_id, "ocr_revision": ocr_revision},
            )
            progress_queue.put({"event": "race_start", "video_label": video_label, "race_id": race_id, "ocr_revision": ocr_revision})
            metadata_index = extract_text.load_exported_frame_metadata(base_dir)
            race_result = extract_text.process_race_group(
                grouped_item,
                text_detected_folder,
                metadata_index,
                input_videos_folder,
                None,
            )
            result_queue.put(
                {
                    "event": "race_result",
                    "video_label": video_label,
                    "race_id": race_id,
                    "ocr_revision": ocr_revision,
                    "race_result": race_result,
                }
            )
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {
                    "event": "race_worker_result",
                    "video_label": video_label,
                    "race_id": race_id,
                    "ocr_revision": ocr_revision,
                    "duration_s": float(race_result.get("duration_s", 0.0)),
                    "row_count": len(race_result.get("rows", [])),
                },
            )
        except BaseException as exc:
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {"event": "race_worker_error", "video_label": video_label, "race_id": race_id, "ocr_revision": ocr_revision, "error": repr(exc)},
            )
            result_queue.put(
                {
                    "event": "error",
                    "video_label": video_label,
                    "race_id": race_id,
                    "ocr_revision": ocr_revision,
                    "error": repr(exc),
                }
            )


def _run_all_with_video_overlap(video_files: list[Path], *, selection_mode: bool, include_subfolders: bool = False) -> None:
    from . import extract_frames, extract_text
    runtime_config = load_app_config()
    overlap_ocr_consumers = max(1, int(runtime_config.overlap_ocr_consumers))
    overlap_ocr_mode = runtime_effective_overlap_ocr_mode(runtime_config)
    diagnostics_enabled = os.environ.get("MK8_OVERLAP_OCR_DIAGNOSTICS", "0").strip().lower() in {"1", "true", "yes", "on"}
    video_index_by_label = {
        build_video_identity(path, include_subfolders=include_subfolders): index
        for index, path in enumerate(video_files, start=1)
    }

    def display_video_label(video_label: str) -> str:
        return LOGGER.color_video_text(str(video_label), video_index_by_label.get(str(video_label), 1))

    selected_video_names = [
        str(path.relative_to(INPUT_DIR)).replace("\\", "/") if include_subfolders else path.name
        for path in video_files
    ]
    ocr_results: list[dict] = []
    ocr_errors: list[BaseException] = []
    ocr_lock = threading.Lock()
    ocr_start_time_holder = {"value": None}
    use_multi_process_ocr = overlap_ocr_consumers > 1
    use_race_overlap = use_multi_process_ocr and overlap_ocr_mode == "race"
    finalize_executor = None
    overlap_diag = {
        "queued_at": {},
        "started_at": {},
        "first_progress_at": {},
        "completed_at": {},
        "last_completed": {},
        "active_videos": set(),
        "max_active": 0,
        "samples": [],
    }
    overlap_diag_lock = threading.Lock()

    def record_ocr_queued(video_label: str) -> None:
        if not diagnostics_enabled:
            return
        with overlap_diag_lock:
            overlap_diag["queued_at"].setdefault(video_label, time.time())

    def record_ocr_started(video_label: str) -> None:
        if not diagnostics_enabled:
            return
        now = time.time()
        with overlap_diag_lock:
            overlap_diag["started_at"].setdefault(video_label, now)
            overlap_diag["active_videos"].add(video_label)
            overlap_diag["max_active"] = max(overlap_diag["max_active"], len(overlap_diag["active_videos"]))

    def record_ocr_progress(video_label: str, event: dict) -> None:
        if not diagnostics_enabled:
            return
        if event.get("event") not in {"heartbeat", "progress"}:
            return
        with overlap_diag_lock:
            overlap_diag["first_progress_at"].setdefault(video_label, time.time())
            overlap_diag["last_completed"][video_label] = int(event.get("completed", 0))

    def record_ocr_completed(video_label: str) -> None:
        if not diagnostics_enabled:
            return
        with overlap_diag_lock:
            overlap_diag["completed_at"][video_label] = time.time()
            overlap_diag["active_videos"].discard(video_label)

    sampler_stop = threading.Event()

    def diagnostic_sampler():
        while not sampler_stop.wait(2.0):
            snapshot = LOGGER.resources.sample()
            with overlap_diag_lock:
                overlap_diag["samples"].append(
                    {
                        "time_s": LOGGER.elapsed_seconds(),
                        "active_ocr": len(overlap_diag["active_videos"]),
                        "cpu_percent": snapshot.cpu_percent,
                        "ram_used_gb": snapshot.ram_used_gb,
                    }
                )

    sampler_thread = None
    if diagnostics_enabled:
        sampler_thread = threading.Thread(target=diagnostic_sampler, name="mk8-overlap-diag", daemon=True)
        sampler_thread.start()

    if use_race_overlap:
        ctx = mp.get_context("spawn")
        ocr_jobs = ctx.Queue()
        ocr_result_queue = ctx.Queue()
        ocr_progress_queue = ctx.Queue()
        worker_processes = [
            ctx.Process(
                target=_overlap_ocr_race_process_worker,
                args=(ocr_jobs, ocr_result_queue, ocr_progress_queue),
                name=f"mk8-ocr-race-overlap-{index + 1}",
                daemon=True,
            )
            for index in range(overlap_ocr_consumers)
        ]
        for process in worker_processes:
            process.start()

        video_expected_races = {}
        video_extraction_complete = set()
        video_expected_race_ids = defaultdict(set)
        video_expected_race_revisions = defaultdict(dict)
        video_completed_race_revisions = defaultdict(dict)
        video_race_results = defaultdict(dict)
        video_race_durations = defaultdict(float)
        finalized_videos = set()
        finalizing_videos = set()
        finalize_futures = {}
        finalize_executor = ThreadPoolExecutor(max_workers=max(1, min(2, overlap_ocr_consumers)))
        race_overlap_lock = threading.Lock()
        queued_races_by_video = defaultdict(int)
        active_races_by_video = defaultdict(int)

        def summarize_race_overlap_state(video_label: str) -> tuple[int, int, int, int]:
            with race_overlap_lock:
                queued_for_video = int(queued_races_by_video.get(video_label, 0))
                active_for_video = int(active_races_by_video.get(video_label, 0))
                queued_total = int(sum(queued_races_by_video.values()))
                active_total = int(sum(active_races_by_video.values()))
            return queued_for_video, active_for_video, queued_total, active_total

        def _finalize_video_result(video_label: str) -> dict:
            video_label = str(video_label)
            finalize_start = time.time()
            finalized = extract_text.finalize_ocr_results(
                [
                    row
                    for _race_id, item in sorted(
                        video_race_results[video_label].items(),
                        key=lambda value: int(value[0]),
                    )
                    for row in item.get("rows", [])
                ],
                folder_path=str(FRAMES_DIR),
                phase_start_time=finalize_start,
                per_video_ocr_durations={video_label: float(video_race_durations[video_label])},
                progress_peak_lines=[],
                ocr_profiler_lines=[
                    "Overlapped per-race OCR mode",
                    f"OCR consumers: {overlap_ocr_consumers}",
                    "Detailed OCR call profiling is omitted from the combined summary in overlap mode.",
                ],
                write_outputs=False,
                emit_logs=False,
            )
            finalize_duration_s = float(time.time() - finalize_start)
            finalized["ocr_duration_s"] = float(video_race_durations[video_label])
            finalized["finalize_duration_s"] = finalize_duration_s
            finalized["duration_s"] = float(video_race_durations[video_label]) + finalize_duration_s
            finalized["video_label"] = video_label
            finalized["video_name"] = next(
                (
                    str(path.relative_to(INPUT_DIR)).replace("\\", "/") if include_subfolders else path.name
                    for path in video_files
                    if build_video_identity(path, include_subfolders=include_subfolders) == video_label
                ),
                video_label,
            )
            finalized["per_video_durations"] = {video_label: finalized["duration_s"]}
            return finalized

        def _complete_finalize_future(future) -> None:
            finalized = future.result()
            video_label = str(finalized["video_label"])
            finalizing_videos.discard(video_label)
            finalized_videos.add(video_label)
            with ocr_lock:
                ocr_results.append(finalized)
            record_ocr_completed(video_label)
            LOGGER.log(
                OVERLAP_SCOPE,
                _format_overlap_complete_with_finalize(
                    video_label,
                    int(finalized.get("race_count", 0)),
                    extract_frames.format_duration(finalized.get("duration_s", 0.0)),
                    extract_frames.format_duration(finalized.get("finalize_duration_s", 0.0)),
                ),
                color_name="green",
            )

        def _drain_finalize_futures() -> None:
            completed_video_labels = []
            for video_label, future in list(finalize_futures.items()):
                if not future.done():
                    continue
                completed_video_labels.append(video_label)
                _complete_finalize_future(future)
            for video_label in completed_video_labels:
                finalize_futures.pop(video_label, None)

        def try_finalize_video(video_label: str) -> None:
            video_label = str(video_label)
            if video_label in finalized_videos or video_label in finalizing_videos:
                return
            if video_label not in video_extraction_complete:
                return
            expected_race_ids = set(video_expected_race_ids.get(video_label, set()))
            expected_revisions = dict(video_expected_race_revisions.get(video_label, {}))
            completed_revisions = dict(video_completed_race_revisions.get(video_label, {}))
            total = max(1, int(video_expected_races.get(video_label, len(expected_race_ids))))
            if not expected_race_ids:
                return
            if any(int(completed_revisions.get(race_id, 0)) < int(expected_revisions.get(race_id, 1)) for race_id in expected_race_ids):
                return

            finalizing_videos.add(video_label)
            LOGGER.log(
                OVERLAP_SCOPE,
                _format_overlap_finalize_start(video_label, total),
                color_name="cyan",
            )
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {
                    "event": "video_finalize_start",
                    "video_label": video_label,
                    "race_count": total,
                },
            )
            finalize_futures[video_label] = finalize_executor.submit(_finalize_video_result, video_label)

        def progress_collector():
            worker_exits = 0
            while worker_exits < overlap_ocr_consumers:
                event = ocr_progress_queue.get()
                event_type = event.get("event")
                if event_type == "worker_exit":
                    worker_exits += 1
                    continue
                video_label = str(event.get("video_label", "unknown"))
                if event_type == "race_start":
                    race_id = int(event.get("race_id") or 0)
                    with race_overlap_lock:
                        if queued_races_by_video.get(video_label, 0) > 0:
                            queued_races_by_video[video_label] -= 1
                        active_races_by_video[video_label] += 1
                    queued_for_video, active_for_video, queued_total, active_total = summarize_race_overlap_state(video_label)
                    LOGGER.log(
                        OVERLAP_SCOPE,
                        _format_overlap_race_event(
                            video_label,
                            "started",
                            race_id,
                            queued_for_video=queued_for_video,
                            queued_total=queued_total,
                        ),
                        color_name="cyan",
                    )
                    record_ocr_started(video_label)
                    continue

        progress_thread = threading.Thread(target=progress_collector, name="mk8-ocr-progress", daemon=True)
        progress_thread.start()

        def result_collector():
            worker_exits = 0
            while worker_exits < overlap_ocr_consumers or finalize_futures:
                try:
                    message = ocr_result_queue.get(timeout=0.5)
                except queue.Empty:
                    _drain_finalize_futures()
                    continue
                event_type = message.get("event")
                if event_type == "worker_exit":
                    worker_exits += 1
                    _drain_finalize_futures()
                    continue
                if event_type == "error":
                    ocr_errors.append(RuntimeError(f"{message.get('video_label')} race {message.get('race_id')}: {message.get('error')}"))
                    _drain_finalize_futures()
                    continue
                if event_type != "race_result":
                    _drain_finalize_futures()
                    continue

                video_label = str(message["video_label"])
                race_id = int(message.get("race_id") or 0)
                ocr_revision = int(message.get("ocr_revision") or 1)
                with race_overlap_lock:
                    if active_races_by_video.get(video_label, 0) > 0:
                        active_races_by_video[video_label] -= 1
                race_result = dict(message["race_result"])
                video_race_durations[video_label] += float(race_result.get("duration_s", 0.0))
                expected_revisions = video_expected_race_revisions[video_label]
                completed_revisions = video_completed_race_revisions[video_label]
                expected_revision = int(expected_revisions.get(race_id, 1))
                if ocr_revision >= expected_revision:
                    current_completed_revision = int(completed_revisions.get(race_id, 0))
                    if ocr_revision >= current_completed_revision:
                        video_race_results[video_label][race_id] = race_result
                        completed_revisions[race_id] = ocr_revision
                expected_race_ids = set(video_expected_race_ids.get(video_label, set()))
                if race_id not in expected_race_ids:
                    expected_race_ids.add(race_id)
                    video_expected_race_ids[video_label] = expected_race_ids
                completed = sum(
                    1
                    for expected_race_id in expected_race_ids
                    if int(completed_revisions.get(expected_race_id, 0)) >= int(expected_revisions.get(expected_race_id, 1))
                )
                total = max(1, int(video_expected_races.get(video_label, len(expected_race_ids) or completed)))
                queued_for_video, active_for_video, queued_total, active_total = summarize_race_overlap_state(video_label)
                race_summary = race_result.get("summary") or {}
                progress_event = {
                    "event": "progress",
                    "completed": completed,
                    "total": total,
                    "elapsed_s": sum(float(item.get("duration_s", 0.0)) for item in video_race_results[video_label].values()),
                    "video_label": video_label,
                    "race_id": race_id,
                    "track_name": race_summary.get("track_name"),
                    "race_score_players": race_summary.get("race_score_players"),
                    "total_score_players": race_summary.get("total_score_players"),
                    "pending": queued_for_video,
                    "active": active_for_video,
                    "global_pending": queued_total,
                    "global_active": active_total,
                }
                record_ocr_progress(video_label, progress_event)
                LOGGER.log(
                    OVERLAP_SCOPE,
                    _format_overlap_race_event(
                        video_label,
                        "finished",
                        race_id,
                        queued_for_video=queued_for_video,
                        queued_total=queued_total,
                    ),
                    color_name="green",
                )
                LOGGER.log(
                    OVERLAP_SCOPE_OCR,
                    _format_overlap_progress_line(
                        video_label,
                        _format_overlap_ocr_detail(progress_event, extract_frames.format_duration),
                    ),
                )

                try_finalize_video(video_label)
                _drain_finalize_futures()

        result_thread = threading.Thread(target=result_collector, name="mk8-ocr-results", daemon=True)
        result_thread.start()

        def enqueue_completed_race(race_payload: dict) -> None:
            with ocr_lock:
                if ocr_start_time_holder["value"] is None:
                    ocr_start_time_holder["value"] = time.time()
            video_label = str(race_payload["video_label"])
            race_number = int(race_payload["race_number"])
            ocr_revision = max(1, int(race_payload.get("ocr_revision", 1) or 1))
            grouped_item = extract_text.build_grouped_race_item(
                str(FRAMES_DIR),
                video_label,
                race_number,
            )
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {
                    "event": "race_queued",
                    "video_label": video_label,
                    "race_id": race_number,
                    "ocr_revision": ocr_revision,
                    "source": "extract_complete_callback",
                },
            )
            record_ocr_queued(video_label)
            with race_overlap_lock:
                queued_races_by_video[video_label] += 1
            video_expected_race_ids[video_label].add(race_number)
            previous_revision = int(video_expected_race_revisions[video_label].get(race_number, 0))
            video_expected_race_revisions[video_label][race_number] = max(previous_revision, ocr_revision)
            queued_for_video, _active_for_video, queued_total, _active_total = summarize_race_overlap_state(video_label)
            LOGGER.log(
                OVERLAP_SCOPE,
                _format_overlap_race_event(
                    video_label,
                    "queued",
                    race_number,
                    queued_for_video=queued_for_video,
                    queued_total=queued_total,
                ),
                color_name="cyan",
            )
            ocr_jobs.put(
                {
                    "video_label": video_label,
                    "race_id": race_number,
                    "ocr_revision": ocr_revision,
                    "grouped_item": grouped_item,
                }
            )

        def enqueue_completed_video(video_payload: dict) -> None:
            video_label = str(video_payload["video_label"])
            grouped_items = extract_text.build_grouped_race_images(
                str(FRAMES_DIR),
                selected_race_classes=[video_label],
            )
            video_extraction_complete.add(video_label)
            video_expected_races[video_label] = len(grouped_items)
            video_expected_race_ids[video_label] = {
                int(race_id_number)
                for (race_class, race_id_number), _images in grouped_items
                if str(race_class) == video_label
            }
            LOGGER.log(
                OVERLAP_SCOPE,
                _format_overlap_extraction_complete(video_label, len(grouped_items)),
                color_name="cyan",
            )
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {
                    "event": "video_extraction_complete",
                    "video_label": video_label,
                    "expected_races": len(grouped_items),
                },
            )
            try_finalize_video(video_label)

    elif use_multi_process_ocr:
        ctx = mp.get_context("spawn")
        ocr_jobs = ctx.Queue()
        ocr_result_queue = ctx.Queue()
        ocr_progress_queue = ctx.Queue()
        worker_processes = [
            ctx.Process(
                target=_overlap_ocr_process_worker,
                args=(ocr_jobs, ocr_result_queue, ocr_progress_queue),
                name=f"mk8-ocr-overlap-{index + 1}",
                daemon=True,
            )
            for index in range(overlap_ocr_consumers)
        ]
        for process in worker_processes:
            process.start()

        def progress_collector():
            worker_exits = 0
            while worker_exits < overlap_ocr_consumers:
                event = ocr_progress_queue.get()
                if event.get("event") == "worker_exit":
                    worker_exits += 1
                    continue
                video_label = str(event.get("video_label", "unknown"))
                if event.get("event") == "worker_start":
                    LOGGER.log(
                        OVERLAP_SCOPE,
                        _format_overlap_start_video(video_label),
                        color_name="cyan",
                    )
                    record_ocr_started(video_label)
                    continue
                record_ocr_progress(video_label, event)
                LOGGER.log(
                    OVERLAP_SCOPE_OCR,
                    _format_overlap_progress_line(
                        video_label,
                        _format_overlap_ocr_detail(event, extract_frames.format_duration),
                    ),
                )

        progress_thread = threading.Thread(target=progress_collector, name="mk8-ocr-progress", daemon=True)
        progress_thread.start()

        def result_collector():
            worker_exits = 0
            while worker_exits < overlap_ocr_consumers:
                message = ocr_result_queue.get()
                event_type = message.get("event")
                if event_type == "worker_exit":
                    worker_exits += 1
                    continue
                if event_type == "error":
                    ocr_errors.append(RuntimeError(f"{message.get('video_label')}: {message.get('error')}"))
                    continue
                if event_type != "result":
                    continue
                result = dict(message["result"])
                result["video_label"] = message["video_label"]
                result["video_name"] = message["video_name"]
                ocr_results.append(result)
                record_ocr_completed(str(message["video_label"]))
                LOGGER.log(
                    OVERLAP_SCOPE,
                    _format_overlap_complete(
                        str(message["video_label"]),
                        int(result.get("race_count", 0)),
                        extract_frames.format_duration(result.get("duration_s", 0.0)),
                    ),
                    color_name="green",
                )

        result_thread = threading.Thread(target=result_collector, name="mk8-ocr-results", daemon=True)
        result_thread.start()

        def enqueue_completed_video(video_payload: dict) -> None:
            with ocr_lock:
                if ocr_start_time_holder["value"] is None:
                    ocr_start_time_holder["value"] = time.time()
            LOGGER.log(
                OVERLAP_SCOPE,
                _format_overlap_queue_video(str(video_payload["video_label"])),
                color_name="cyan",
            )
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {
                    "event": "video_queued",
                    "video_label": str(video_payload["video_label"]),
                    "video_name": str(video_payload["video_name"]),
                },
            )
            record_ocr_queued(str(video_payload["video_label"]))
            ocr_jobs.put(
                {
                    "video_label": video_payload["video_label"],
                    "video_name": video_payload["video_name"],
                }
            )

    else:
        ocr_jobs = queue.Queue()
        overlap_progress_log = {}

        def progress_callback(event: dict, *, video_label: str) -> None:
            record_ocr_progress(video_label, event)
            callback = overlap_progress_log.get(video_label)
            if callback is None:
                callback = _make_overlap_ocr_progress_callback(video_label, extract_frames.format_duration, display_video_label)
                overlap_progress_log[video_label] = callback
            callback(event)

        def ocr_worker():
            while True:
                job = ocr_jobs.get()
                if job is None:
                    ocr_jobs.task_done()
                    return
                try:
                    with ocr_lock:
                        if ocr_start_time_holder["value"] is None:
                            ocr_start_time_holder["value"] = time.time()
                    LOGGER.log(
                        OVERLAP_SCOPE,
                        _format_overlap_start_video(str(job["video_label"])),
                        color_name="cyan",
                    )
                    record_ocr_started(str(job["video_label"]))
                    result = extract_text.process_images_in_folder(
                        str(FRAMES_DIR),
                        in_memory_frame_bundles=job["frame_bundle_cache"],
                        selected_race_classes=[job["video_label"]],
                        metadata_index_override=job["metadata_index"],
                        write_outputs=False,
                        emit_logs=False,
                        progress_callback=lambda event, video_label=str(job["video_label"]): progress_callback(event, video_label=video_label),
                    )
                    result["video_label"] = job["video_label"]
                    result["video_name"] = job["video_name"]
                    ocr_results.append(result)
                    record_ocr_completed(str(job["video_label"]))
                    LOGGER.log(
                        OVERLAP_SCOPE,
                        _format_overlap_complete(
                            str(job["video_label"]),
                            int(result.get("race_count", 0)),
                            extract_frames.format_duration(result.get("duration_s", 0.0)),
                        ),
                        color_name="green",
                    )
                except BaseException as exc:
                    ocr_errors.append(exc)
                finally:
                    ocr_jobs.task_done()

        worker_thread = threading.Thread(target=ocr_worker, name="mk8-ocr-overlap", daemon=True)
        worker_thread.start()

        def enqueue_completed_video(video_payload: dict) -> None:
            LOGGER.log(
                OVERLAP_SCOPE,
                _format_overlap_queue_video(str(video_payload["video_label"])),
                color_name="cyan",
            )
            append_ocr_trace_event(
                "scheduler_events.jsonl",
                {
                    "event": "video_queued",
                    "video_label": str(video_payload["video_label"]),
                    "video_name": str(video_payload["video_name"]),
                },
            )
            record_ocr_queued(str(video_payload["video_label"]))
            ocr_jobs.put(video_payload)

    extract_phase_start = time.time()
    extract_result = extract_frames.extract_frames(
        return_frame_cache=not use_multi_process_ocr,
        selected_videos=selected_video_names or None,
        include_subfolders=include_subfolders,
        per_video_complete_callback=enqueue_completed_video,
        per_race_complete_callback=enqueue_completed_race if use_race_overlap else None,
    )
    extract_summary = extract_result.get("summary", {})
    extract_summary["duration_s"] = time.time() - extract_phase_start

    if use_multi_process_ocr:
        for _ in range(overlap_ocr_consumers):
            ocr_jobs.put(None)
        for process in worker_processes:
            process.join()
        result_thread.join()
        if finalize_executor is not None:
            finalize_executor.shutdown(wait=True)
        progress_thread.join()
    else:
        ocr_jobs.put(None)
        worker_thread.join()
    sampler_stop.set()
    if sampler_thread is not None:
        sampler_thread.join(timeout=5.0)
    if ocr_errors:
        raise RuntimeError(f"Overlapped OCR failed: {ocr_errors[0]}")

    combined_frames = [item["df"] for item in ocr_results if isinstance(item.get("df"), pd.DataFrame) and not item["df"].empty]
    if combined_frames:
        combined_columns = list(combined_frames[0].columns)
        combined_records = []
        for frame in combined_frames:
            combined_records.extend(frame.to_dict("records"))
        combined_df = pd.DataFrame.from_records(combined_records, columns=combined_columns)
    else:
        combined_df = pd.DataFrame()
    if not combined_df.empty:
        combined_df = combined_df.sort_values(["RaceClass", "RaceIDNumber", "RacePosition"], kind="stable").reset_index(drop=True)

    total_ocr_duration_s = 0.0
    if ocr_start_time_holder["value"] is not None:
        total_ocr_duration_s = max(0.0, time.time() - float(ocr_start_time_holder["value"]))

    per_video_durations = {}
    per_video_summary = {}
    progress_peak_lines = []
    ocr_profiler_lines = [
        f"Overlapped per-{overlap_ocr_mode} OCR mode",
        f"OCR consumers: {overlap_ocr_consumers}",
        "Detailed OCR call profiling is omitted from the combined summary in overlap mode.",
    ]
    for result in ocr_results:
        per_video_durations.update(result.get("per_video_durations", {}))
        per_video_summary.update(result.get("per_video_summary", {}))
        for line in result.get("progress_peak_lines", []):
            if line not in progress_peak_lines:
                progress_peak_lines.append(line)

    if combined_df.empty:
        ocr_result = {"duration_s": total_ocr_duration_s, "output_excel_path": None, "race_count": 0, "per_video_durations": per_video_durations, "per_video_summary": per_video_summary}
    else:
        completion_payload = build_completion_payload(
            combined_df,
            str(FRAMES_DIR),
            ocr_start_time_holder["value"] or time.time(),
            progress_peak_lines,
            ocr_profiler_lines,
            per_video_durations,
            build_race_warning_messages,
            lambda count, singular, plural=None: singular if count == 1 else (plural or f"{singular}s"),
            extract_frames.format_duration,
        )
        LOGGER.summary_block("[OCR - Phase Complete]", completion_payload["lines"], color_name="green")
        ocr_result = {
            "duration_s": total_ocr_duration_s,
            "output_excel_path": completion_payload["output_excel_path"],
            "race_count": completion_payload["race_count"],
            "per_video_durations": completion_payload["per_video_durations"],
            "per_video_summary": completion_payload["per_video_summary"],
        }

    total_processing_seconds = LOGGER.elapsed_seconds()
    total_source_seconds = 0.0
    source_summaries = []
    for summary in extract_summary.get("per_video_summaries", []):
        source_length = float(summary.get("source_length_s", 0.0) or 0.0)
        total_source_seconds += source_length
        display_name = str(summary.get("video_name") or "")
        source_summaries.append(f"{display_name} ({extract_frames.format_duration(source_length)})")
    ratio = total_source_seconds / total_processing_seconds if total_processing_seconds > 0 else 0.0
    total_ocr_work_s = sum(float(value) for value in per_video_durations.values())
    total_corrupt_check_s = float(extract_summary.get('corrupt_check_duration_s', 0.0))
    total_repair_s = float(extract_summary.get('repair_duration_s', 0.0))
    total_repairs = int(extract_summary.get('repair_count', 0))
    cumulative_elapsed_s = sum(float(summary.get("processing_duration_s", 0.0)) for summary in extract_summary.get("per_video_summaries", []))
    overlap_time_saved_s = max(0.0, cumulative_elapsed_s - total_processing_seconds)
    likely_bottleneck = _summarize_pipeline_bottleneck(
        extract_duration_s=float(extract_summary.get("duration_s", 0.0)),
        ocr_duration_s=total_ocr_duration_s,
        total_processing_seconds=total_processing_seconds,
    )
    performance_lines = [
        "Run totals",
        *_format_metric_table([
            ("Source video length", extract_frames.format_duration(total_source_seconds)),
            ("Processing time", extract_frames.format_duration(total_processing_seconds)),
            ("Playback ratio", f"{ratio:.1f}x real-time"),
            ("Overlap time saved", extract_frames.format_duration(overlap_time_saved_s)),
            ("Likely bottleneck", likely_bottleneck),
        ]),
        "",
        "Phase timings",
        *_format_simple_table(
            ["Phase", "Duration", "Notes"],
            [
                ["Extract race and score screens", extract_frames.format_duration(extract_summary.get('duration_s', 0.0)), ""],
                ["OCR and workbook export", extract_frames.format_duration(total_ocr_duration_s), ""],
                ["Corrupt preflight checks", extract_frames.format_duration(total_corrupt_check_s), ""],
                ["Repair file creation", extract_frames.format_duration(total_repair_s), f"{total_repairs} {('video' if total_repairs == 1 else 'videos')}"],
                ["Mode", "", f"overlapped by {overlap_ocr_mode} ({overlap_ocr_consumers} GPU OCR consumer{'s' if overlap_ocr_consumers != 1 else ''})"],
            ],
            alignments=["left", "right", "left"],
        ),
    ]
    if diagnostics_enabled:
        queued_at = overlap_diag["queued_at"]
        started_at = overlap_diag["started_at"]
        first_progress_at = overlap_diag["first_progress_at"]
        completed_at = overlap_diag["completed_at"]
        queue_waits = []
        first_progress_waits = []
        wall_durations = []
        for video_label, queued_time in queued_at.items():
            started_time = started_at.get(video_label)
            first_progress_time = first_progress_at.get(video_label)
            completed_time = completed_at.get(video_label)
            if started_time is not None:
                queue_waits.append(float(started_time - queued_time))
            if started_time is not None and first_progress_time is not None:
                first_progress_waits.append(float(first_progress_time - started_time))
            if started_time is not None and completed_time is not None:
                wall_durations.append(float(completed_time - started_time))
        extract_duration_s = float(extract_summary.get("duration_s", 0.0))
        extract_samples = [item for item in overlap_diag["samples"] if float(item["time_s"]) <= extract_duration_s + 0.001]
        active_extract_samples = [item for item in extract_samples if int(item.get("active_ocr", 0)) > 0]
        performance_lines.extend(["", "Overlap diagnostics"])
        performance_lines.append(f"- Diagnostics enabled: yes")
        performance_lines.append(f"- Max simultaneously active OCR videos: {int(overlap_diag['max_active'])}")
        if queue_waits:
            performance_lines.append(
                f"- Avg queue wait before OCR start: {extract_frames.format_duration(sum(queue_waits) / len(queue_waits))}"
            )
        if first_progress_waits:
            performance_lines.append(
                f"- Avg OCR start to first progress: {extract_frames.format_duration(sum(first_progress_waits) / len(first_progress_waits))}"
            )
        if wall_durations:
            performance_lines.append(
                f"- Avg OCR wall time per video: {extract_frames.format_duration(sum(wall_durations) / len(wall_durations))}"
            )
        if extract_samples:
            avg_active_extract = sum(int(item.get("active_ocr", 0)) for item in extract_samples) / len(extract_samples)
            performance_lines.append(f"- Avg active OCR consumers during extract: {avg_active_extract:.2f}")
        if active_extract_samples:
            avg_cpu_extract = sum(float(item.get("cpu_percent") or 0.0) for item in active_extract_samples) / len(active_extract_samples)
            avg_ram_extract = sum(float(item.get("ram_used_gb") or 0.0) for item in active_extract_samples) / len(active_extract_samples)
            performance_lines.append(f"- Avg CPU during extract while OCR active: {avg_cpu_extract:.0f}%")
            performance_lines.append(f"- Avg RAM during extract while OCR active: {avg_ram_extract:.1f} GB")
    if extract_summary.get("per_video_summaries"):
        performance_lines.extend(["", "Per-video summary"])
        table_headers = ["Video", "Source", "Total", "Scan", "Score", "OCR", "Races", "Players", "Review"]
        table_alignments = ["left", "right", "right", "right", "right", "right", "right", "left", "left"]
        per_video_rows = []
        for summary in extract_summary["per_video_summaries"]:
            video_identity = str(summary.get("video_label") or build_video_identity(Path(summary["video_name"]), include_subfolders=include_subfolders))
            video_ocr_summary = per_video_summary.get(video_identity, {})
            player_summary = video_ocr_summary.get("player_count_summary", "n/a")
            per_video_rows.append([
                summary["video_name"],
                extract_frames.format_duration(summary["source_length_s"]),
                extract_frames.format_duration(summary["processing_duration_s"]),
                extract_frames.format_duration(summary["scan_duration_s"]),
                extract_frames.format_duration(summary["total_score_duration_s"]),
                extract_frames.format_duration(
                    total_ocr_duration_s * (float(per_video_durations.get(video_identity, 0.0)) / total_ocr_work_s)
                    if total_ocr_work_s > 0 else 0.0
                ),
                str(video_ocr_summary.get("race_count", 0)),
                player_summary,
                _review_summary_text(video_ocr_summary),
            ])
        formatted_table = _format_simple_table(
            table_headers,
            per_video_rows,
            alignments=table_alignments,
        )
        row_widths = _format_simple_table_widths(
            table_headers,
            per_video_rows,
        )
        performance_lines.extend(formatted_table[:2])
        for index, row_values in enumerate(per_video_rows):
            video_identity = str(
                extract_summary["per_video_summaries"][index].get("video_label")
                or build_video_identity(Path(extract_summary["per_video_summaries"][index]["video_name"]), include_subfolders=include_subfolders)
            )
            performance_lines.append(_format_colored_table_row(row_values, row_widths, video_identity, alignments=table_alignments))
    performance_lines.extend(["", "Resource peaks", *[f"- {line}" for line in LOGGER.peak_lines()]])
    LOGGER.summary_block("[Run - Performance Summary]", performance_lines, color_name="cyan")
    LOGGER.blank_lines(2)
    latest_results_xlsx = Path(ocr_result.get("output_excel_path") or find_latest_results_xlsx(OUTPUT_DIR))
    LOGGER.log("[Run - Output]", str(latest_results_xlsx), color_name="green")
    print(
        f"[{LOGGER.elapsed_label()}] "
        f"{LOGGER.color('[RUN - COMPLETED]', 'green')} "
        f"{LOGGER.color('[ ENJOY HAVE FUN ]', 'magenta')}"
        + LOGGER.color("[LET'S A GO!]", "yellow")
    )


def run_all(
    selected_video: str | list[str] | None = None,
    selection_mode: bool = False,
    *,
    include_subfolders: bool = False,
    debug: bool | None = None,
) -> None:
    LOGGER.reset()
    configure_headless_debug_outputs(debug)
    ensure_runtime_or_raise()
    from . import extract_frames, extract_text
    runtime_config = load_app_config()
    overlap_enabled = os.environ.get(
        "MK8_OVERLAP_OCR_BY_VIDEO",
        "1" if runtime_easyocr_gpu_enabled(runtime_config) else "0",
    ).lower() not in {"0", "false", "no"}

    mode_label = "Run selection" if selection_mode else "Run all"
    LOGGER.log("[Run - Phase Start]", mode_label, color_name="cyan")
    video_files = selected_input_video_files(selected_video=selected_video, include_subfolders=include_subfolders)
    overlap_active = overlap_enabled and runtime_easyocr_gpu_enabled(runtime_config) and len(video_files) > 1
    if ocr_trace_enabled():
        run_mode = (
            "selection_overlap_race"
            if selection_mode and overlap_active else
            "selection_sequential"
            if selection_mode else
            "all_overlap_race"
            if overlap_active else
            "all_sequential"
        )
        trace_label, trace_mode = ensure_ocr_trace_env(run_mode=run_mode)
        append_ocr_trace_event(
            "scheduler_events.jsonl",
            {
                "event": "run_start",
                "selection_mode": bool(selection_mode),
                "selected_video": selected_video,
                "include_subfolders": bool(include_subfolders),
                "video_count": len(video_files),
                "overlap_active": bool(overlap_active),
                "trace_label": trace_label,
                "trace_mode": trace_mode,
            },
        )
    if not video_files:
        target = selected_video or "Input_Videos"
        raise RuntimeError(
            f"No supported videos found for selection: {target} | "
            f"Searched under: {INPUT_DIR}"
        )
    source_summaries, total_source_seconds = _workflow_sorted_source_summaries(
        video_files,
        include_subfolders=include_subfolders,
    )
    LOGGER.summary_block(
        "[Run - Input Summary]",
        _format_input_summary_lines(source_summaries, total_source_seconds),
        color_name="cyan",
    )
    if selection_mode:
        cleared = clear_output_results_for_videos(video_files, include_subfolders=include_subfolders)
        cleanup_message = (
            f"Cleared prior exported artifacts for {len(video_files)} selected video(s)"
            if cleared else
            f"No prior exported artifacts found for {len(video_files)} selected video(s)"
        )
        LOGGER.log("[Run - Selection Cleanup]", cleanup_message, color_name="cyan")
    if overlap_active:
        _run_all_with_video_overlap(video_files, selection_mode=selection_mode, include_subfolders=include_subfolders)
        return
    selected_video_names = [
        str(path.relative_to(INPUT_DIR)).replace("\\", "/") if include_subfolders else path.name
        for path in video_files
    ]
    extract_result = extract_frames.extract_frames(
        return_frame_cache=False,
        selected_videos=selected_video_names or None,
        include_subfolders=include_subfolders,
    )
    LOGGER.blank_lines(2)
    selected_race_classes = (
        selected_race_classes_for_videos(video_files, include_subfolders=include_subfolders)
        if selection_mode else None
    )
    ocr_result = extract_text.process_images_in_folder(
        str(FRAMES_DIR),
        selected_race_classes=selected_race_classes,
    )
    total_processing_seconds = LOGGER.elapsed_seconds()
    ratio = total_source_seconds / total_processing_seconds if total_processing_seconds > 0 else 0.0
    extract_summary = extract_result.get("summary", {})
    per_video_summaries = extract_summary.get("per_video_summaries", [])
    ocr_per_video_work_durations = ocr_result.get("per_video_durations", {})
    ocr_per_video_summary = ocr_result.get("per_video_summary", {})
    total_ocr_duration_s = float(ocr_result.get("duration_s", 0.0))
    total_ocr_work_s = sum(float(value) for value in ocr_per_video_work_durations.values())
    total_corrupt_check_s = float(extract_summary.get('corrupt_check_duration_s', 0.0))
    total_repair_s = float(extract_summary.get('repair_duration_s', 0.0))
    total_repairs = int(extract_summary.get('repair_count', 0))
    cumulative_elapsed_s = sum(float(summary.get("processing_duration_s", 0.0)) for summary in per_video_summaries)
    overlap_time_saved_s = max(0.0, cumulative_elapsed_s - total_processing_seconds)
    likely_bottleneck = _summarize_pipeline_bottleneck(
        extract_duration_s=float(extract_summary.get('duration_s', 0.0)),
        ocr_duration_s=float(ocr_result.get('duration_s', 0.0)),
        total_processing_seconds=total_processing_seconds,
    )
    performance_lines = [
        "Run totals",
        *_format_metric_table([
            ("Source video length", extract_frames.format_duration(total_source_seconds)),
            ("Processing time", extract_frames.format_duration(total_processing_seconds)),
            ("Playback ratio", f"{ratio:.1f}x real-time"),
            ("Overlap time saved", extract_frames.format_duration(overlap_time_saved_s)),
            ("Likely bottleneck", likely_bottleneck),
        ]),
        "",
        "Phase timings",
        *_format_simple_table(
            ["Phase", "Duration", "Notes"],
            [
                ["Extract race and score screens", extract_frames.format_duration(extract_summary.get('duration_s', 0.0)), ""],
                ["OCR and workbook export", extract_frames.format_duration(ocr_result.get('duration_s', 0.0)), ""],
                ["Corrupt preflight checks", extract_frames.format_duration(total_corrupt_check_s), ""],
                ["Repair file creation", extract_frames.format_duration(total_repair_s), f"{total_repairs} {('video' if total_repairs == 1 else 'videos')}"],
            ],
            alignments=["left", "right", "left"],
        ),
    ]
    if per_video_summaries:
        performance_lines.extend(["", "Per-video summary"])
        table_headers = ["Video", "Source", "Total", "Scan", "Score", "OCR", "Races", "Players", "Review"]
        table_alignments = ["left", "right", "right", "right", "right", "right", "right", "left", "left"]
        per_video_rows = []
        for summary in per_video_summaries:
            video_identity = str(summary.get("video_label") or Path(summary["video_name"]).stem)
            video_ocr_summary = ocr_per_video_summary.get(video_identity, {})
            player_summary = video_ocr_summary.get("player_count_summary", "n/a")
            per_video_rows.append([
                summary["video_name"],
                extract_frames.format_duration(summary["source_length_s"]),
                extract_frames.format_duration(summary["processing_duration_s"]),
                extract_frames.format_duration(summary["scan_duration_s"]),
                extract_frames.format_duration(summary["total_score_duration_s"]),
                extract_frames.format_duration(
                    total_ocr_duration_s * (float(ocr_per_video_work_durations.get(video_identity, 0.0)) / total_ocr_work_s)
                    if total_ocr_work_s > 0 else 0.0
                ),
                str(video_ocr_summary.get("race_count", 0)),
                player_summary,
                _review_summary_text(video_ocr_summary),
            ])
        formatted_table = _format_simple_table(
            table_headers,
            per_video_rows,
            alignments=table_alignments,
        )
        row_widths = _format_simple_table_widths(
            table_headers,
            per_video_rows,
        )
        performance_lines.extend(formatted_table[:2])
        for index, row_values in enumerate(per_video_rows):
            performance_lines.append(
                _format_colored_table_row(
                    row_values,
                    row_widths,
                    str(per_video_summaries[index].get("video_label") or Path(per_video_summaries[index]["video_name"]).stem),
                    alignments=table_alignments,
                )
            )
    performance_lines.extend(["", "Resource peaks", *[f"- {line}" for line in LOGGER.peak_lines()]])
    LOGGER.summary_block(
        "[Run - Performance Summary]",
        performance_lines,
        color_name="cyan",
    )
    LOGGER.blank_lines(2)
    latest_results_xlsx = Path(ocr_result.get("output_excel_path") or find_latest_results_xlsx(OUTPUT_DIR))
    LOGGER.log("[Run - Output]", str(latest_results_xlsx), color_name="green")
    print(
        f"[{LOGGER.elapsed_label()}] "
        f"{LOGGER.color('[RUN - COMPLETED]', 'green')} "
        f"{LOGGER.color('[ ENJOY HAVE FUN ]', 'magenta')}"
        + LOGGER.color("[LET'S A GO!]", "yellow")
    )


def write_profile_report(profile: cProfile.Profile, output_path: Path, limit: int = 80) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Mario Kart 8 LAN Play Video Processor\n")
        handle.write("Whole-process profile report\n\n")
        stats = pstats.Stats(profile, stream=handle)
        stats.sort_stats("cumtime")
        handle.write("Top functions by cumulative time\n")
        stats.print_stats(limit)
        handle.write("\n")
        stats.sort_stats("tottime")
        handle.write("Top functions by self time\n")
        stats.print_stats(limit)
        handle.write("\nCallers for selected heavy functions\n")
        for pattern in (
            "process_race_group",
            "extract_scoreboard_observation",
            "analyze_score_window_task",
            "match_template",
            "_run_easyocr_single_roi",
            "_run_easyocr_player_names_batched",
        ):
            handle.write(f"\nCallers matching: {pattern}\n")
            stats.print_callers(pattern)


def run_profiled_all(
    selected_video: str | list[str] | None = None,
    *,
    include_subfolders: bool = False,
    selection_mode: bool = False,
    debug: bool | None = None,
) -> None:
    LOGGER.reset()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_all(
            selected_video=selected_video,
            include_subfolders=include_subfolders,
            selection_mode=selection_mode,
            debug=debug,
        )
    finally:
        profiler.disable()
        write_profile_report(profiler, PROFILE_OUTPUT)
        LOGGER.log("[Run - Profile Report]", str(PROFILE_OUTPUT), color_name="cyan")


def gui_runtime_available() -> tuple[bool, str]:
    if tk is None or messagebox is None or filedialog is None:
        return False, "Tkinter is unavailable"
    try:
        from PIL import ImageTk  # noqa: F401
    except Exception as exc:
        return False, f"Pillow ImageTk is unavailable ({exc})"
    return True, "GUI runtime available"


def print_runtime_status() -> int:
    ffmpeg_issues = check_runtime(APP_CONFIG, require_ffmpeg=True)
    gpu_runtime = detect_gpu_runtime(APP_CONFIG)
    easyocr_runtime = detect_easyocr_runtime(APP_CONFIG)
    gui_available, gui_reason = gui_runtime_available()
    easyocr_available = importlib.util.find_spec("easyocr") is not None

    print(f"Python executable: {sys.executable}")
    print(f"Child script Python: {resolve_project_python()}")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Frames folder: {FRAMES_DIR}")
    print(f"Latest results file: {find_latest_results_xlsx(OUTPUT_DIR)}")
    print(f"EasyOCR: {'OK' if easyocr_available else 'MISSING'}")
    print(f"FFmpeg: {'OK' if not ffmpeg_issues else 'MISSING'}")
    print(f"GUI runtime: {'OK' if gui_available else 'UNAVAILABLE'}")
    print(f"GUI detail: {gui_reason}")
    print(
        f"GPU mode: {gpu_runtime['mode']} "
        f"({'ENABLED' if gpu_runtime['enabled'] else 'DISABLED'}, backend={gpu_runtime['backend']}, "
        f"cuda_devices={gpu_runtime['device_count']}, opencl={gpu_runtime['opencl_in_use']})"
    )
    print(f"GPU detail: {gpu_runtime['reason']}")
    print(
        f"EasyOCR mode: {easyocr_runtime['mode']} "
        f"({'ENABLED' if easyocr_runtime['enabled'] else 'DISABLED'}, backend={easyocr_runtime['backend']}, "
        f"cuda_devices={easyocr_runtime['device_count']})"
    )
    print(f"EasyOCR detail: {easyocr_runtime['reason']}")
    print(f"OCR workers: {APP_CONFIG.ocr_workers}")
    print(f"Effective OCR workers: {1 if easyocr_runtime['enabled'] else APP_CONFIG.ocr_workers}")
    print(f"Logical CPU threads: {os.cpu_count() or 1}")
    overlap_mode_effective = runtime_effective_overlap_ocr_mode(APP_CONFIG)
    overlap_default_enabled = runtime_easyocr_gpu_enabled(APP_CONFIG)
    if overlap_mode_effective == APP_CONFIG.overlap_ocr_mode:
        print(f"Overlap OCR mode: {APP_CONFIG.overlap_ocr_mode}")
    else:
        print(f"Overlap OCR mode: {APP_CONFIG.overlap_ocr_mode} -> {overlap_mode_effective}")
    print(f"Overlap OCR active by default: {'yes' if overlap_default_enabled else 'no'}")
    print(f"Overlap OCR consumers: {APP_CONFIG.overlap_ocr_consumers}")
    print(f"Score analysis workers: {APP_CONFIG.score_analysis_workers}")
    print(f"Parallel video total score workers: {APP_CONFIG.parallel_video_score_workers}")
    print(f"Initial scan workers: {APP_CONFIG.pass1_scan_workers}")
    print(f"OCR consensus frames: {APP_CONFIG.ocr_consensus_frames}")
    print(f"Initial scan segment overlap frames: {APP_CONFIG.pass1_segment_overlap_frames}")
    print(f"Initial scan minimum segment frames: {APP_CONFIG.pass1_min_segment_frames}")
    print(f"Write debug CSV: {APP_CONFIG.write_debug_csv}")
    print(f"Write debug score images: {APP_CONFIG.write_debug_score_images}")
    print(f"Write debug linking Excel: {APP_CONFIG.write_debug_linking_excel}")
    print(f"Export image format: {EXPORT_IMAGE_FORMAT}")

    issues = []
    issues.extend(ffmpeg_issues)
    if not easyocr_available:
        issues.append("EasyOCR is not installed in the local environment.")
    if issues:
        print("\nRuntime issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    return 0


def exit_application(root_window) -> None:
    root_window.destroy()


GUI_THEME = {
    "window_bg": "#06060f",
    "panel_bg": "#0c1424",
    "panel_border": "#243247",
    "hero_overlay": "#08101fcc",
    "title_fg": "#ffd34d",
    "subtitle_fg": "#d7dde8",
    "muted_fg": "#9aa7bb",
    "divider_gold": "#ffd34d",
    "divider_red": "#d73737",
    "status_green": "#43A047",
}


def _bind_button_hover(button, *, normal_bg: str, hover_bg: str, normal_border: str, hover_border: str):
    def on_enter(_event):
        button.configure(bg=hover_bg, highlightbackground=hover_border, highlightcolor=hover_border)

    def on_leave(_event):
        button.configure(bg=normal_bg, highlightbackground=normal_border, highlightcolor=normal_border)

    button.bind("<Enter>", on_enter)
    button.bind("<Leave>", on_leave)


def _create_gui_button(parent, *, text: str, command, bg: str, fg: str, active_bg: str, border: str,
                       hover_bg: str | None = None, hover_border: str | None = None, font_size: int = 13):
    button = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground=fg,
        relief=tk.FLAT,
        bd=0,
        highlightthickness=1,
        highlightbackground=border,
        highlightcolor=border,
        cursor="hand2",
        font=("TkDefaultFont", font_size, "bold"),
        padx=10,
        pady=6,
        wraplength=220,
    )
    _bind_button_hover(
        button,
        normal_bg=bg,
        hover_bg=hover_bg or active_bg,
        normal_border=border,
        hover_border=hover_border or border,
    )
    return button


def _create_step_card(parent, *, step_number: str, title: str, description: str, accent_bg: str, accent_border: str,
                      number_bg: str, number_fg: str, header_bg: str):
    card = tk.Frame(parent, bg=GUI_THEME["panel_bg"], highlightthickness=1, highlightbackground=accent_border, bd=0)
    card.grid_columnconfigure(0, weight=1)

    header = tk.Frame(card, bg=header_bg)
    header.grid(row=0, column=0, sticky="ew")
    header.grid_columnconfigure(1, weight=1)

    number_label = tk.Label(
        header,
        text=step_number,
        bg=number_bg,
        fg=number_fg,
        font=("TkDefaultFont", 11, "bold"),
        padx=7,
        pady=3,
    )
    number_label.grid(row=0, column=0, padx=(12, 10), pady=(10, 4), sticky="w")

    title_label = tk.Label(
        header,
        text=title,
        bg=header_bg,
        fg="white",
        font=("TkDefaultFont", 15, "bold"),
        anchor="w",
        justify="left",
    )
    title_label.grid(row=0, column=1, padx=(0, 12), pady=(10, 4), sticky="w")

    description_label = tk.Label(
        header,
        text=description,
        bg=header_bg,
        fg=GUI_THEME["muted_fg"],
        font=("TkDefaultFont", 12),
        anchor="w",
        justify="left",
        wraplength=460,
    )
    description_label.grid(row=1, column=0, columnspan=2, padx=12, pady=(1, 10), sticky="w")

    body = tk.Frame(card, bg=GUI_THEME["panel_bg"])
    body.grid(row=1, column=0, sticky="nsew", padx=14, pady=12)
    body.grid_columnconfigure(0, weight=1)
    card.grid_rowconfigure(1, weight=1)

    return card, body


def _create_compact_card(parent, *, title: str, accent_border: str):
    card = tk.Frame(parent, bg=GUI_THEME["panel_bg"], highlightthickness=1, highlightbackground=accent_border, bd=0)
    card.grid_columnconfigure(0, weight=1)

    body = tk.Frame(card, bg=GUI_THEME["panel_bg"])
    body.grid(row=0, column=0, sticky="nsew", padx=14, pady=10)
    body.grid_columnconfigure(0, weight=1)

    title_label = tk.Label(
        body,
        text=title,
        bg=GUI_THEME["panel_bg"],
        fg="white",
        font=("TkDefaultFont", 14, "bold"),
        anchor="w",
        justify="left",
    )
    title_label.grid(row=0, column=0, sticky="w")

    return card, body


def _create_gui_toggle(parent, *, text: str, variable, bg: str, fg: str, selectcolor: str, active_bg: str, command=None):
    toggle = tk.Checkbutton(
        parent,
        text=text,
        variable=variable,
        command=command,
        onvalue=True,
        offvalue=False,
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground=fg,
        selectcolor=selectcolor,
        relief=tk.FLAT,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
        font=("TkDefaultFont", 10, "bold"),
        padx=2,
        pady=2,
    )
    return toggle


def _tile_hero_background(canvas, image):
    canvas.delete("bg_tile")
    width = max(canvas.winfo_width(), 1)
    height = max(canvas.winfo_height(), 1)
    tile_width = max(image.width(), 1)
    tile_height = max(image.height(), 1)
    x_tiles = max(1, (width + tile_width - 1) // tile_width)
    y_tiles = max(1, (height + tile_height - 1) // tile_height)
    for y_index in range(y_tiles):
        for x_index in range(x_tiles):
            canvas.create_image(x_index * tile_width, y_index * tile_height, image=image, anchor="nw", tags="bg_tile")


def _render_smooth_gradient_bar(canvas, *, image_factory, image_store: dict, offset: int):
    width = max(canvas.winfo_width(), 1)
    height = max(canvas.winfo_height(), 1)
    gradient_image = image_factory(width, height, offset)
    image_store["image"] = gradient_image
    canvas.delete("rainbow")
    canvas.create_image(0, 0, image=gradient_image, anchor="nw", tags="rainbow")


def _start_rainbow_bar_animation(root, canvas, *, image_factory, image_store: dict, start_offset: int = 0, delay_ms: int = 45):
    state = {"offset": start_offset}

    def tick():
        if not canvas.winfo_exists():
            return
        _render_smooth_gradient_bar(canvas, image_factory=image_factory, image_store=image_store, offset=state["offset"])
        state["offset"] = (state["offset"] + 3) % 360
        root.after(delay_ms, tick)

    canvas.bind(
        "<Configure>",
        lambda _event: _render_smooth_gradient_bar(canvas, image_factory=image_factory, image_store=image_store, offset=state["offset"]),
    )
    tick()


def _create_tinted_tile_image(base_image: Image.Image, *, tint_color: tuple[int, int, int], tint_alpha: int):
    tiled_base = base_image.convert("RGBA")
    tint_layer = Image.new("RGBA", tiled_base.size, (*tint_color, tint_alpha))
    return Image.alpha_composite(tiled_base, tint_layer)


def _fit_shell_width(viewport_width: int) -> int:
    return max(980, min(1240, viewport_width - 40))


def launch_gui() -> int:
    if tk is None or messagebox is None or filedialog is None:
        print("Tkinter is not available. Use headless mode, for example: python -m mk8_local_play.main --all", file=sys.stderr)
        return 1
    try:
        from PIL import ImageTk
    except Exception as exc:
        print(f"GUI image support is unavailable ({exc}). Use headless mode, for example: python -m mk8_local_play.main --all", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("Mario Kart 8 Race Analysis")
    root.configure(bg=GUI_THEME["window_bg"])
    try:
        current_scaling = float(root.tk.call("tk", "scaling"))
        root.tk.call("tk", "scaling", max(0.9, min(1.0, current_scaling * 0.95)))
    except Exception:
        pass
    root.geometry("1260x860")
    root.minsize(1040, 760)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    include_subfolders_var = tk.BooleanVar(value=False)
    execution_mode_var = tk.StringVar(value=APP_CONFIG.execution_mode.upper())
    easyocr_mode_var = tk.StringVar(value=APP_CONFIG.easyocr_gpu_mode.upper())

    def persist_runtime_mode_settings(*_args):
        global APP_CONFIG
        execution_mode = execution_mode_var.get().strip().lower()
        easyocr_mode = easyocr_mode_var.get().strip().lower()
        os.environ["MK8_EXECUTION_MODE"] = execution_mode
        os.environ["MK8_EASYOCR_GPU_MODE"] = easyocr_mode
        update_app_config_values({
            "execution_mode": execution_mode,
            "easyocr_gpu_mode": easyocr_mode,
        })
        APP_CONFIG = load_app_config()

    def gui_run_extract():
        try:
            run_extract(include_subfolders=include_subfolders_var.get())
            show_info("Success", "Video analyzed and races found.")
        except Exception as exc:
            show_error("Error", str(exc))

    def gui_run_selection():
        try:
            run_all(selection_mode=True, include_subfolders=include_subfolders_var.get())
            show_info("Success", "Selection run completed.")
        except Exception as exc:
            show_error("Error", str(exc))

    def gui_run_ocr():
        try:
            run_ocr(include_subfolders=include_subfolders_var.get())
            show_info("Success", "Races exported to Excel.")
        except Exception as exc:
            show_error("Error", str(exc))

    mario_kart_image_path = resolve_asset_file("gui", "bg.jpg")
    base_background_image = Image.open(mario_kart_image_path)
    outer_background_image = ImageTk.PhotoImage(base_background_image)
    hero_background_source = _create_tinted_tile_image(base_background_image, tint_color=(6, 10, 18), tint_alpha=150)
    hero_tile_image = ImageTk.PhotoImage(hero_background_source)

    def create_rainbow_gradient_image(width: int, height: int, offset: int):
        gradient = Image.new("RGB", (width, height))
        pixels = gradient.load()
        for x_index in range(width):
            hue = (((x_index * 1.6) + offset) % 360) / 360.0
            red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            rgb = (int(red * 255), int(green * 255), int(blue * 255))
            for y_index in range(height):
                pixels[x_index, y_index] = rgb
        return ImageTk.PhotoImage(gradient)

    stripe_top_image = {}
    stripe_bottom_image = {}

    background_canvas = tk.Canvas(root, bg=GUI_THEME["window_bg"], bd=0, highlightthickness=0)
    background_canvas.grid(row=0, column=0, sticky="nsew")

    shell = tk.Frame(background_canvas, bg=GUI_THEME["window_bg"], padx=24, pady=20)
    shell.grid_columnconfigure(0, weight=1)
    shell.grid_rowconfigure(1, weight=1)
    shell_window_id = background_canvas.create_window(0, 0, anchor="n", window=shell)

    app_frame = tk.Frame(shell, bg=GUI_THEME["panel_bg"], highlightthickness=1, highlightbackground=GUI_THEME["panel_border"], bd=0)
    app_frame.grid(row=0, column=0, sticky="nsew")
    app_frame.grid_columnconfigure(0, weight=1)
    app_frame.grid_rowconfigure(2, weight=1)

    stripe_top = tk.Canvas(app_frame, height=6, bg=GUI_THEME["panel_bg"], bd=0, highlightthickness=0)
    stripe_top.grid(row=0, column=0, sticky="ew")

    hero_frame = tk.Frame(app_frame, bg=GUI_THEME["panel_bg"], height=96)
    hero_frame.grid(row=1, column=0, sticky="ew")
    hero_frame.grid_columnconfigure(0, weight=1)
    hero_frame.grid_rowconfigure(0, weight=1)
    hero_frame.grid_propagate(False)

    hero_canvas = tk.Canvas(hero_frame, bg=GUI_THEME["window_bg"], bd=0, highlightthickness=0)
    hero_canvas.grid(row=0, column=0, sticky="nsew")

    hero_overlay = tk.Frame(hero_canvas, bg="#08101f")
    hero_overlay.grid_columnconfigure(0, weight=1)

    title_row = tk.Frame(hero_overlay, bg="#08101f")
    title_row.grid(row=0, column=0, sticky="ew", padx=22, pady=(10, 2))
    title_row.grid_columnconfigure(1, weight=1)

    title_block = tk.Frame(title_row, bg="#08101f")
    title_block.grid(row=0, column=0, sticky="w")

    title_label = tk.Label(
        title_block,
        text="MARIO KART 8",
        bg="#08101f",
        fg=GUI_THEME["title_fg"],
        font=("TkDefaultFont", 21, "bold"),
    )
    title_label.grid(row=0, column=0, sticky="w")

    subtitle_label = tk.Label(
        title_block,
        text="DELUXE · LAN TOURNAMENT · VIDEO ANALYSER",
        bg="#08101f",
        fg=GUI_THEME["subtitle_fg"],
        font=("TkDefaultFont", 13, "bold"),
    )
    subtitle_label.grid(row=1, column=0, sticky="w", pady=(4, 0))

    exit_button = _create_gui_button(
        title_row,
        text="Exit",
        command=lambda: exit_application(root),
        bg="#7a1010",
        fg="#ffffff",
        active_bg="#9d1818",
        border="#a53a3a",
        hover_bg="#b71f1f",
        hover_border="#d85b5b",
        font_size=11,
    )
    exit_button.grid(row=0, column=1, sticky="e")

    divider = tk.Frame(hero_overlay, bg=GUI_THEME["divider_gold"], height=2)
    divider.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 2))
    divider.grid_propagate(False)

    hero_copy = tk.Frame(hero_overlay, bg="#08101f")
    hero_copy.grid(row=2, column=0, sticky="ew", padx=22, pady=(1, 4))
    hero_copy.grid_columnconfigure(0, weight=1)

    hero_summary = tk.Label(
        hero_copy,
        text="Prepare videos, scan race results, and export tournament workbooks from one cross-platform desktop window.",
        bg="#08101f",
        fg=GUI_THEME["subtitle_fg"],
        justify="left",
        anchor="w",
        wraplength=760,
        font=("TkDefaultFont", 12),
    )
    hero_summary.grid(row=0, column=0, sticky="w")

    hero_window_id = hero_canvas.create_window(0, 0, anchor="nw", window=hero_overlay)

    body_frame = tk.Frame(app_frame, bg=GUI_THEME["panel_bg"], padx=20, pady=12)
    body_frame.grid(row=2, column=0, sticky="nsew")
    body_frame.grid_columnconfigure(0, weight=1)
    body_frame.grid_rowconfigure(0, weight=1)

    steps_frame = tk.Frame(body_frame, bg=GUI_THEME["panel_bg"])
    steps_frame.grid(row=0, column=0, sticky="nsew")
    steps_frame.grid_columnconfigure(0, weight=1)
    steps_frame.grid_columnconfigure(1, weight=1)
    steps_frame.grid_columnconfigure(2, weight=1)
    steps_frame.grid_rowconfigure(1, weight=1)

    step1_card, step1_body = _create_step_card(
        steps_frame,
        step_number="STEP 1",
        title="Input Videos",
        description="Open the input folder or combine source clips",
        accent_bg="#a07800",
        accent_border="#826815",
        number_bg="#4f4310",
        number_fg="#ffe88f",
        header_bg="#17170e",
    )
    step1_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
    step1_body.grid_columnconfigure(0, weight=1)
    _create_gui_button(step1_body, text="Open Video Folder", command=open_videos_folder, bg="#8c6a00", fg="#fff1b3", active_bg="#a97f00", border="#c59d2a", hover_bg="#c39200", hover_border="#e2bf58").grid(row=0, column=0, sticky="ew")
    _create_gui_button(step1_body, text="Combine Video Clips", command=merge_videos, bg="#8c6a00", fg="#fff1b3", active_bg="#a97f00", border="#c59d2a", hover_bg="#c39200", hover_border="#e2bf58").grid(row=1, column=0, sticky="ew", pady=(8, 0))
    merge_note = tk.Label(
        step1_body,
        text="Use this first to prepare the videos you want to process.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=260,
        font=("TkDefaultFont", 12),
    )
    merge_note.grid(row=2, column=0, sticky="w", pady=(8, 0))

    settings_card, settings_body = _create_compact_card(
        steps_frame,
        title="Global Settings",
        accent_border="#8a6436",
    )
    settings_card.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    settings_body.grid_columnconfigure(1, weight=1)
    settings_body.grid_columnconfigure(3, weight=1)
    settings_body.grid_columnconfigure(5, weight=1)

    settings_intro = tk.Label(
        settings_body,
        text="Persistent runtime settings for all GUI actions.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=560,
        font=("TkDefaultFont", 11),
    )
    settings_intro.grid(row=0, column=1, columnspan=5, sticky="w", padx=(14, 0))

    settings_subfolders_label = tk.Label(
        settings_body,
        text="Input Scope",
        bg=GUI_THEME["panel_bg"],
        fg="#fff1b3",
        anchor="w",
        justify="left",
        font=("TkDefaultFont", 13, "bold"),
    )
    settings_subfolders_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

    settings_subfolders_toggle = _create_gui_toggle(
        settings_body,
        text="Also Look In Subfolders",
        variable=include_subfolders_var,
        bg=GUI_THEME["panel_bg"],
        fg="#fff1b3",
        selectcolor="#4f4310",
        active_bg=GUI_THEME["panel_bg"],
    )
    settings_subfolders_toggle.grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(8, 0))

    step2_card, step2_body = _create_step_card(
        steps_frame,
        step_number="STEP 2",
        title="Extract Race Frames from Video",
        description="Find and save the race and score screenshots",
        accent_bg="#154d9e",
        accent_border="#335b91",
        number_bg="#132845",
        number_fg="#b9dcff",
        header_bg="#101826",
    )
    step2_card.grid(row=1, column=1, sticky="nsew", padx=8, pady=(0, 8))
    step2_body.grid_columnconfigure(0, weight=1)
    _create_gui_button(step2_body, text="Extract Race Frames", command=gui_run_extract, bg="#103b79", fg="#cbe4ff", active_bg="#18559f", border="#4473ae", hover_bg="#2166bc", hover_border="#6d9de0").grid(row=0, column=0, sticky="ew")
    _create_gui_button(step2_body, text="Open Extracted Frames", command=open_frames_folder, bg="#103b79", fg="#cbe4ff", active_bg="#18559f", border="#4473ae", hover_bg="#2166bc", hover_border="#6d9de0").grid(row=1, column=0, sticky="ew", pady=(8, 0))

    step2_note = tk.Label(
        step2_body,
        text="Creates the saved race-frame bundles used by OCR and review.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=260,
        font=("TkDefaultFont", 12),
    )
    step2_note.grid(row=2, column=0, sticky="w", pady=(8, 0))

    extract_mode_label = tk.Label(
        settings_body,
        text="Extraction GPU Mode",
        bg=GUI_THEME["panel_bg"],
        fg="#cbe4ff",
        anchor="w",
        justify="left",
        font=("TkDefaultFont", 13, "bold"),
    )
    extract_mode_label.grid(row=1, column=2, sticky="w", padx=(20, 0), pady=(8, 0))

    extract_mode_menu = tk.OptionMenu(settings_body, execution_mode_var, "AUTO", "GPU", "CPU", command=persist_runtime_mode_settings)
    extract_mode_menu.config(bg="#132845", fg="#cbe4ff", activebackground="#18559f", activeforeground="#ffffff", highlightthickness=0, bd=0)
    extract_mode_menu["menu"].config(bg="#132845", fg="#cbe4ff", activebackground="#2166bc", activeforeground="#ffffff")
    extract_mode_menu.grid(row=1, column=3, sticky="w", padx=(10, 0), pady=(8, 0))

    step3_card, step3_body = _create_step_card(
        steps_frame,
        step_number="STEP 3",
        title="OCR on Extracted Race Frames and Export",
        description="Read saved screenshots and build the workbook",
        accent_bg="#1f6a34",
        accent_border="#3a7f53",
        number_bg="#17331f",
        number_fg="#c7f2d1",
        header_bg="#101d16",
    )
    step3_card.grid(row=1, column=2, sticky="nsew", padx=(8, 0), pady=(0, 8))
    step3_body.grid_columnconfigure(0, weight=1)
    _create_gui_button(step3_body, text="OCR and Export", command=gui_run_ocr, bg="#1a5c28", fg="#d8f7df", active_bg="#27763a", border="#4a8a5b", hover_bg="#2e8a46", hover_border="#75b286").grid(row=0, column=0, sticky="ew")
    _create_gui_button(step3_body, text="Open Excel Scores", command=open_excel_scores, bg="#1a5c28", fg="#d8f7df", active_bg="#27763a", border="#4a8a5b", hover_bg="#2e8a46", hover_border="#75b286").grid(row=1, column=0, sticky="ew", pady=(8, 0))

    ocr_note = tk.Label(
        step3_body,
        text="Uses the extracted frames from Step 2 and writes the final workbook.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=260,
        font=("TkDefaultFont", 12),
    )
    ocr_note.grid(row=2, column=0, sticky="w", pady=(8, 0))

    easyocr_mode_label = tk.Label(
        settings_body,
        text="EasyOCR Mode",
        bg=GUI_THEME["panel_bg"],
        fg="#d8f7df",
        anchor="w",
        justify="left",
        font=("TkDefaultFont", 13, "bold"),
    )
    easyocr_mode_label.grid(row=1, column=4, sticky="w", padx=(20, 0), pady=(8, 0))

    easyocr_mode_menu = tk.OptionMenu(settings_body, easyocr_mode_var, "AUTO", "GPU", "CPU", command=persist_runtime_mode_settings)
    easyocr_mode_menu.config(bg="#17331f", fg="#d8f7df", activebackground="#27763a", activeforeground="#ffffff", highlightthickness=0, bd=0)
    easyocr_mode_menu["menu"].config(bg="#17331f", fg="#d8f7df", activebackground="#2e8a46", activeforeground="#ffffff")
    easyocr_mode_menu.grid(row=1, column=5, sticky="w", padx=(10, 0), pady=(8, 0))

    settings_note = tk.Label(
        settings_body,
        text="Subfolders affects all run buttons. Extraction is fastest on CPU here; OCR Auto uses CUDA when available.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=1040,
        font=("TkDefaultFont", 11),
    )
    settings_note.grid(row=2, column=0, columnspan=6, sticky="w", pady=(4, 0))

    selection_card, selection_body = _create_step_card(
        steps_frame,
        step_number="STEP 2 + STEP 3",
        title="Full Run",
        description="Run extraction, OCR, and export in one go",
        accent_bg="#7b1fa2",
        accent_border="#7d4ca0",
        number_bg="#30163f",
        number_fg="#e8c7ff",
        header_bg="#181024",
    )
    selection_card.grid(row=2, column=1, columnspan=2, sticky="ew")
    selection_body.grid_columnconfigure(0, weight=1)

    selection_intro = tk.Label(
        selection_body,
        text="This combines Step 2 and Step 3 for the currently selected videos.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["subtitle_fg"],
        anchor="w",
        justify="left",
        wraplength=560,
        font=("TkDefaultFont", 12),
    )
    selection_intro.grid(row=0, column=0, sticky="w")

    _create_gui_button(
        selection_body,
        text="Full Run",
        command=gui_run_selection,
        bg="#5c2682",
        fg="#f0ddff",
        active_bg="#7430a4",
        border="#9260bb",
        hover_bg="#8740bd",
        hover_border="#bc8be0",
    ).grid(row=1, column=0, sticky="ew", pady=(10, 0))

    cleanup_card, cleanup_body = _create_compact_card(
        steps_frame,
        title="Cleanup",
        accent_border="#ab4747",
    )
    cleanup_card.grid(row=2, column=0, sticky="ew", padx=(0, 8))
    cleanup_body.grid_columnconfigure(0, weight=1)

    cleanup_note = tk.Label(
        cleanup_body,
        text="Clear saved screenshots or generated outputs when you want a fresh rerun.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=260,
        font=("TkDefaultFont", 11),
    )
    cleanup_note.grid(row=1, column=0, sticky="w", pady=(6, 0))

    _create_gui_button(cleanup_body, text="Delete Found Race Screenshots", command=clear_all_races_found, bg="#7a1010", fg="#ffd7d7", active_bg="#9d1818", border="#ab4747", hover_bg="#b71f1f", hover_border="#d85b5b", font_size=12).grid(row=2, column=0, sticky="ew", pady=(8, 0))
    _create_gui_button(cleanup_body, text="Clear Output Folder", command=clear_output_results_gui, bg="#7a1010", fg="#ffd7d7", active_bg="#9d1818", border="#ab4747", hover_bg="#b71f1f", hover_border="#d85b5b", font_size=12).grid(row=3, column=0, sticky="ew", pady=(8, 0))

    bottom_bar = tk.Frame(body_frame, bg=GUI_THEME["panel_bg"])
    bottom_bar.grid(row=1, column=0, sticky="ew", pady=(12, 0))
    bottom_bar.grid_columnconfigure(0, weight=1)

    status_frame = tk.Frame(bottom_bar, bg=GUI_THEME["panel_bg"])
    status_frame.grid(row=0, column=0, sticky="w")

    status_dot = tk.Canvas(status_frame, width=14, height=14, bg=GUI_THEME["panel_bg"], highlightthickness=0, bd=0)
    status_dot.create_oval(3, 3, 11, 11, fill=GUI_THEME["status_green"], outline=GUI_THEME["status_green"])
    status_dot.grid(row=0, column=0, padx=(0, 8))

    status_label = tk.Label(
        status_frame,
        text="Ready",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        font=("TkDefaultFont", 10, "bold"),
    )
    status_label.grid(row=0, column=1, sticky="w")

    stripe_bottom = tk.Canvas(app_frame, height=6, bg=GUI_THEME["panel_bg"], bd=0, highlightthickness=0)
    stripe_bottom.grid(row=3, column=0, sticky="ew")

    def sync_hero_layout(_event=None):
        _tile_hero_background(hero_canvas, hero_tile_image)
        canvas_width = max(hero_canvas.winfo_width(), 1)
        canvas_height = max(hero_canvas.winfo_height(), 1)
        hero_canvas.coords(hero_window_id, 0, 0)
        hero_canvas.itemconfigure(hero_window_id, width=canvas_width, height=canvas_height)

    def sync_root_background(_event=None):
        _tile_hero_background(background_canvas, outer_background_image)
        canvas_width = max(background_canvas.winfo_width(), 1)
        canvas_height = max(background_canvas.winfo_height(), 1)
        shell_width = _fit_shell_width(canvas_width)
        background_canvas.coords(shell_window_id, canvas_width // 2, 24)
        background_canvas.itemconfigure(shell_window_id, width=shell_width)

    hero_canvas.bind("<Configure>", sync_hero_layout)
    background_canvas.bind("<Configure>", sync_root_background)
    root.after(10, sync_hero_layout)
    root.after(10, sync_root_background)
    _start_rainbow_bar_animation(
        root,
        stripe_top,
        image_factory=create_rainbow_gradient_image,
        image_store=stripe_top_image,
        start_offset=0,
        delay_ms=20,
    )
    _start_rainbow_bar_animation(
        root,
        stripe_bottom,
        image_factory=create_rainbow_gradient_image,
        image_store=stripe_bottom_image,
        start_offset=180,
        delay_ms=20,
    )
    root._mk8_outer_background_image = outer_background_image
    root._mk8_gui_background_image = hero_tile_image
    root._mk8_gui_top_rainbow_image = stripe_top_image
    root._mk8_gui_bottom_rainbow_image = stripe_bottom_image

    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mario Kart 8 local play video processing")
    parser.add_argument("--check", action="store_true", help="Print runtime/dependency status")
    parser.add_argument("--clear-output-results", action="store_true", help="Clear all files under Output_Results after interactive confirmation")
    parser.add_argument("--extract", action="store_true", help="Run frame extraction only")
    parser.add_argument("--scan-test", action="store_true", help="Benchmark extraction/scan only without OCR")
    parser.add_argument("--ocr", action="store_true", help="Run OCR/export only")
    parser.add_argument("--all", action="store_true", help="Run extraction and OCR/export")
    parser.add_argument("--selection", action="store_true", help="Run extraction and OCR only for the videos currently selected in Input_Videos")
    parser.add_argument("--subfolders", action="store_true", help="Include supported videos from subfolders under Input_Videos during headless runs")
    parser.add_argument("--profile", action="store_true", help="Write a whole-process performance profile during --all")
    parser.add_argument("--debug", action="store_true", help="Enable debug CSV/images/linking outputs for this headless run")
    parser.add_argument("--video", help="Process only a specific video filename, for example Test_3_Races.mkv")
    parser.add_argument(
        "--videos",
        nargs="+",
        help="Process multiple explicit video paths together, relative to Input_Videos when using --subfolders",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_videos = args.videos if args.videos else args.video
    if args.check:
        return print_runtime_status()
    if args.clear_output_results:
        deleted = clear_output_results(require_confirmation=True)
        if deleted:
            print(f"Cleared: {OUTPUT_DIR}")
        else:
            print("Cancelled.")
        return 0
    if args.extract:
        run_extract(selected_video=selected_videos, include_subfolders=args.subfolders, debug=args.debug)
        return 0
    if args.scan_test:
        run_extract(selected_video=selected_videos, include_subfolders=args.subfolders, debug=args.debug)
        return 0
    if args.ocr:
        run_ocr(
            selected_video=selected_videos,
            include_subfolders=args.subfolders,
            selection_mode=args.selection,
            debug=args.debug,
        )
        return 0
    if args.all:
        if args.profile:
            run_profiled_all(
                selected_video=selected_videos,
                include_subfolders=args.subfolders,
                debug=args.debug,
            )
        else:
            run_all(selected_video=selected_videos, include_subfolders=args.subfolders, debug=args.debug)
        return 0
    if args.selection:
        if args.profile:
            run_profiled_all(
                selected_video=selected_videos,
                include_subfolders=args.subfolders,
                selection_mode=True,
                debug=args.debug,
            )
        else:
            run_all(selected_video=selected_videos, selection_mode=True, include_subfolders=args.subfolders, debug=args.debug)
        return 0
    return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
