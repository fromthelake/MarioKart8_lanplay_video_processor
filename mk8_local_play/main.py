import argparse
import cProfile
import glob
import importlib.util
import os
import pstats
import re
import shutil
import subprocess
import sys
import cv2
import colorsys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None

from PIL import Image

from .app_runtime import check_runtime, detect_gpu_runtime, load_app_config, open_path
from .console_logging import LOGGER
from .data_paths import resolve_asset_file
from .extract_common import EXPORT_IMAGE_FORMAT, remove_tree_contents
from .project_paths import PROJECT_ROOT


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


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mkv", ".mov", ".avi", ".webm"}


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
    )


def selected_input_video_files(selected_video: str | None = None, *, include_subfolders: bool = False) -> list[Path]:
    all_video_files = discover_input_video_files(include_subfolders=include_subfolders)
    if not selected_video:
        return all_video_files
    selected_name = Path(selected_video).name.lower()
    exact_matches = [path for path in all_video_files if path.name.lower() == selected_name]
    if exact_matches:
        return exact_matches
    selected_relative = selected_video.replace("\\", "/").lower()
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
                    shutil.rmtree(child)
                else:
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
        annotated_files = []
        for pattern in ("annotated_*.png", "annotated_*.jpg", "annotated_*.jpeg"):
            annotated_files.extend(glob.glob(str(DEBUG_SCORE_FRAMES_DIR / pattern)))
        for file in annotated_files:
            try:
                os.remove(file)
                deleted_anything = True
            except Exception as exc:
                show_error("Error", f"Unable to delete file {file}: {exc}")
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


def run_extract(selected_video: str | None = None, *, include_subfolders: bool = False) -> None:
    ensure_runtime_or_raise()
    extra_args = []
    if selected_video:
        extra_args.extend(["--video", selected_video])
    if include_subfolders:
        extra_args.append("--subfolders")
    run_python_module(EXTRACT_MODULE, extra_args=extra_args)


def run_ocr(
    selected_video: str | None = None,
    *,
    include_subfolders: bool = False,
    selection_mode: bool = False,
) -> None:
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
    elif selected_video:
        selected_identifier = (
            build_video_identity(Path(selected_video), include_subfolders=True)
            if include_subfolders else Path(selected_video).stem
        )
        extra_args.extend(["--video", selected_identifier])
    run_python_module(OCR_MODULE, extra_args=extra_args)


def run_all(selected_video: str | None = None, selection_mode: bool = False, *, include_subfolders: bool = False) -> None:
    ensure_runtime_or_raise()
    from . import extract_frames, extract_text

    mode_label = "Run selection" if selection_mode else "Run all"
    LOGGER.log("[Run - Phase Start]", mode_label, color_name="cyan")
    video_files = selected_input_video_files(selected_video=selected_video, include_subfolders=include_subfolders)
    if not video_files:
        target = selected_video or "Input_Videos"
        raise RuntimeError(f"No supported videos found for selection: {target}")
    source_summaries = []
    total_source_seconds = 0.0
    for video_path in video_files:
        if not video_path.is_file():
            continue
        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            fps = capture.get(cv2.CAP_PROP_FPS) or 1
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            source_length = frame_count / max(fps, 1)
            total_source_seconds += source_length
            display_name = (
                str(video_path.relative_to(INPUT_DIR)).replace("\\", "/")
                if include_subfolders else video_path.name
            )
            source_summaries.append(f"{display_name} ({extract_frames.format_duration(source_length)})")
        capture.release()
    LOGGER.log("[Run - Input Summary]", f"Videos: {len(source_summaries)} | Total source length: {extract_frames.format_duration(total_source_seconds)}", color_name="cyan")
    for index, summary in enumerate(source_summaries, start=1):
        LOGGER.log("[Run - Input Summary]", f"{index}. {summary}")
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
    performance_lines = [
        f"Total processing time: {extract_frames.format_duration(total_processing_seconds)}",
        f"Total source video length: {extract_frames.format_duration(total_source_seconds)}",
        f"Source-length / processing-time ratio: {ratio:.1f}x real-time",
        "",
        "Phase durations",
        f"- Extract race and score screens: {extract_frames.format_duration(extract_summary.get('duration_s', 0.0))}",
        f"- OCR and workbook export: {extract_frames.format_duration(ocr_result.get('duration_s', 0.0))}",
        f"- Corrupt preflight checks: {extract_frames.format_duration(total_corrupt_check_s)}",
        f"- Repair file creation: {extract_frames.format_duration(total_repair_s)} ({total_repairs} {('video' if total_repairs == 1 else 'videos')})",
    ]
    if per_video_summaries:
        performance_lines.extend(["", "Per-video durations"])
        for summary in per_video_summaries:
            video_stem = Path(summary["video_name"]).stem
            video_ocr_summary = ocr_per_video_summary.get(video_stem, {})
            performance_lines.append(
                    f"- {summary['video_name']} | Source: {extract_frames.format_duration(summary['source_length_s'])} | "
                    f"Processing: {extract_frames.format_duration(summary['processing_duration_s'])}"
                )
            performance_lines.append(f"  - Scan: {extract_frames.format_duration(summary['scan_duration_s'])}")
            corrupt_check_duration_s = float(summary.get('corrupt_check_duration_s', 0.0))
            corrupt_check_status = summary.get('corrupt_check_status', 'skipped')
            if corrupt_check_duration_s > 0:
                performance_lines.append(
                    f"  - Corrupt preflight: {extract_frames.format_duration(corrupt_check_duration_s)} ({corrupt_check_status})"
                )
            else:
                performance_lines.append(f"  - Corrupt preflight: skipped ({corrupt_check_status})")
            repair_duration_s = float(summary.get('repair_duration_s', 0.0))
            if summary.get('repair_created'):
                performance_lines.append(
                    f"  - Repair file creation: {extract_frames.format_duration(repair_duration_s)}"
                )
            else:
                performance_lines.append("  - Repair file creation: not needed")
            performance_lines.append(f"  - Total score screen: {extract_frames.format_duration(summary['total_score_duration_s'])}")
            ocr_work_s = float(ocr_per_video_work_durations.get(video_stem, 0.0))
            if total_ocr_work_s > 0:
                ocr_duration_s = total_ocr_duration_s * (ocr_work_s / total_ocr_work_s)
            else:
                ocr_duration_s = 0.0
            performance_lines.append(f"  - OCR: {extract_frames.format_duration(ocr_duration_s)}")
            if video_ocr_summary:
                performance_lines.append(
                    f"  - Races found: {video_ocr_summary.get('race_count', 0)} | "
                    f"Players: {video_ocr_summary.get('dominant_players', 0)}/12"
                )
                review_race_count = int(video_ocr_summary.get("review_race_count", 0))
                review_row_count = int(video_ocr_summary.get("review_row_count", 0))
                if review_race_count > 0 or review_row_count > 0:
                    performance_lines.append(
                        f"  - Needs review: {review_race_count} {('race' if review_race_count == 1 else 'races')} | "
                        f"{review_row_count} flagged rows"
                    )
                else:
                    performance_lines.append("  - Needs review: none")
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


def run_profiled_all(selected_video: str | None = None) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run_all(selected_video=selected_video)
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
    print(f"OCR workers: {APP_CONFIG.ocr_workers}")
    print(f"Score analysis workers: {APP_CONFIG.score_analysis_workers}")
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
        padx=14,
        pady=10,
        wraplength=300,
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
        font=("TkDefaultFont", 12, "bold"),
        padx=8,
        pady=4,
    )
    number_label.grid(row=0, column=0, padx=(14, 10), pady=(12, 6), sticky="w")

    title_label = tk.Label(
        header,
        text=title,
        bg=header_bg,
        fg="white",
        font=("TkDefaultFont", 16, "bold"),
        anchor="w",
        justify="left",
    )
    title_label.grid(row=0, column=1, padx=(0, 14), pady=(12, 6), sticky="w")

    description_label = tk.Label(
        header,
        text=description,
        bg=header_bg,
        fg=GUI_THEME["muted_fg"],
        font=("TkDefaultFont", 14),
        anchor="w",
        justify="left",
        wraplength=460,
    )
    description_label.grid(row=1, column=0, columnspan=2, padx=14, pady=(2, 12), sticky="w")

    body = tk.Frame(card, bg=GUI_THEME["panel_bg"])
    body.grid(row=1, column=0, sticky="nsew", padx=16, pady=14)
    body.grid_columnconfigure(0, weight=1)
    card.grid_rowconfigure(1, weight=1)

    return card, body


def _create_gui_toggle(parent, *, text: str, variable, bg: str, fg: str, selectcolor: str, active_bg: str):
    toggle = tk.Checkbutton(
        parent,
        text=text,
        variable=variable,
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
    return max(860, min(1080, viewport_width - 56))


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
    root.tk.call("tk", "scaling", 1.0)
    root.geometry("1120x940")
    root.minsize(980, 820)
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)

    include_subfolders_var = tk.BooleanVar(value=False)

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

    shell = tk.Frame(background_canvas, bg=GUI_THEME["window_bg"], padx=28, pady=24)
    shell.grid_columnconfigure(0, weight=1)
    shell.grid_rowconfigure(1, weight=1)
    shell_window_id = background_canvas.create_window(0, 0, anchor="n", window=shell)

    app_frame = tk.Frame(shell, bg=GUI_THEME["panel_bg"], highlightthickness=1, highlightbackground=GUI_THEME["panel_border"], bd=0)
    app_frame.grid(row=0, column=0, sticky="nsew")
    app_frame.grid_columnconfigure(0, weight=1)
    app_frame.grid_rowconfigure(2, weight=1)

    stripe_top = tk.Canvas(app_frame, height=6, bg=GUI_THEME["panel_bg"], bd=0, highlightthickness=0)
    stripe_top.grid(row=0, column=0, sticky="ew")

    hero_frame = tk.Frame(app_frame, bg=GUI_THEME["panel_bg"], height=124)
    hero_frame.grid(row=1, column=0, sticky="ew")
    hero_frame.grid_columnconfigure(0, weight=1)
    hero_frame.grid_rowconfigure(0, weight=1)
    hero_frame.grid_propagate(False)

    hero_canvas = tk.Canvas(hero_frame, bg=GUI_THEME["window_bg"], bd=0, highlightthickness=0)
    hero_canvas.grid(row=0, column=0, sticky="nsew")

    hero_overlay = tk.Frame(hero_canvas, bg="#08101f")
    hero_overlay.grid_columnconfigure(0, weight=1)

    title_row = tk.Frame(hero_overlay, bg="#08101f")
    title_row.grid(row=0, column=0, sticky="ew", padx=28, pady=(12, 2))
    title_row.grid_columnconfigure(1, weight=1)

    title_block = tk.Frame(title_row, bg="#08101f")
    title_block.grid(row=0, column=0, sticky="w")

    title_label = tk.Label(
        title_block,
        text="MARIO KART 8",
        bg="#08101f",
        fg=GUI_THEME["title_fg"],
        font=("TkDefaultFont", 24, "bold"),
    )
    title_label.grid(row=0, column=0, sticky="w")

    subtitle_label = tk.Label(
        title_block,
        text="DELUXE · LAN TOURNAMENT · VIDEO ANALYSER",
        bg="#08101f",
        fg=GUI_THEME["subtitle_fg"],
        font=("TkDefaultFont", 16, "bold"),
    )
    subtitle_label.grid(row=1, column=0, sticky="w", pady=(6, 0))

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
    divider.grid(row=1, column=0, sticky="ew", padx=28, pady=(0, 2))
    divider.grid_propagate(False)

    hero_copy = tk.Frame(hero_overlay, bg="#08101f")
    hero_copy.grid(row=2, column=0, sticky="ew", padx=28, pady=(2, 4))
    hero_copy.grid_columnconfigure(0, weight=1)

    hero_summary = tk.Label(
        hero_copy,
        text="Prepare videos, scan race results, and export tournament workbooks from one cross-platform desktop window.",
        bg="#08101f",
        fg=GUI_THEME["subtitle_fg"],
        justify="left",
        anchor="w",
        wraplength=760,
        font=("TkDefaultFont", 14),
    )
    hero_summary.grid(row=0, column=0, sticky="w")

    hero_window_id = hero_canvas.create_window(0, 0, anchor="nw", window=hero_overlay)

    body_frame = tk.Frame(app_frame, bg=GUI_THEME["panel_bg"], padx=24, pady=16)
    body_frame.grid(row=2, column=0, sticky="nsew")
    body_frame.grid_columnconfigure(0, weight=1)
    body_frame.grid_rowconfigure(0, weight=1)

    steps_frame = tk.Frame(body_frame, bg=GUI_THEME["panel_bg"])
    steps_frame.grid(row=0, column=0, sticky="nsew")
    steps_frame.grid_columnconfigure(0, weight=8)
    steps_frame.grid_columnconfigure(1, weight=3)
    steps_frame.grid_rowconfigure(1, weight=1)
    steps_frame.grid_rowconfigure(2, weight=1)

    step1_card, step1_body = _create_step_card(
        steps_frame,
        step_number="STEP 1",
        title="Choose Your Videos",
        description="Get your recordings ready",
        accent_bg="#a07800",
        accent_border="#826815",
        number_bg="#4f4310",
        number_fg="#ffe88f",
        header_bg="#17170e",
    )
    step1_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
    step1_body.grid_columnconfigure(0, weight=1)
    step1_body.grid_columnconfigure(1, weight=1)
    _create_gui_button(step1_body, text="Open Video Folder", command=open_videos_folder, bg="#8c6a00", fg="#fff1b3", active_bg="#a97f00", border="#c59d2a", hover_bg="#c39200", hover_border="#e2bf58").grid(row=0, column=0, padx=(0, 8), pady=(0, 8), sticky="ew")
    _create_gui_button(step1_body, text="Combine Video Clips", command=merge_videos, bg="#8c6a00", fg="#fff1b3", active_bg="#a97f00", border="#c59d2a", hover_bg="#c39200", hover_border="#e2bf58").grid(row=0, column=1, padx=(8, 0), pady=(0, 8), sticky="ew")
    merge_note = tk.Label(
        step1_body,
        text="Optional: join multiple clips into one video when they belong to the same set of races.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=700,
        font=("TkDefaultFont", 13),
    )
    merge_note.grid(row=1, column=0, columnspan=2, sticky="w")

    step1_options = tk.Frame(step1_body, bg=GUI_THEME["panel_bg"])
    step1_options.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
    step1_options.grid_columnconfigure(1, weight=1)

    subfolders_toggle = _create_gui_toggle(
        step1_options,
        text="Also Look In Subfolders",
        variable=include_subfolders_var,
        bg=GUI_THEME["panel_bg"],
        fg="#fff1b3",
        selectcolor="#4f4310",
        active_bg=GUI_THEME["panel_bg"],
    )
    subfolders_toggle.grid(row=0, column=0, sticky="w")

    subfolders_note = tk.Label(
        step1_options,
        text="Turn this on if your videos are stored inside folders within the main Input_Videos folder.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=520,
        font=("TkDefaultFont", 13),
    )
    subfolders_note.grid(row=0, column=1, sticky="w", padx=(12, 0))

    step2_card, step2_body = _create_step_card(
        steps_frame,
        step_number="STEP 2",
        title="Find The Race Screens",
        description="Prepare the race images",
        accent_bg="#154d9e",
        accent_border="#335b91",
        number_bg="#132845",
        number_fg="#b9dcff",
        header_bg="#101826",
    )
    step2_card.grid(row=1, column=0, sticky="nsew", pady=(0, 12), padx=(0, 10))
    step2_body.grid_columnconfigure(0, weight=1)
    step2_body.grid_columnconfigure(1, weight=1)
    _create_gui_button(step2_body, text="Find Races In Videos", command=gui_run_extract, bg="#103b79", fg="#cbe4ff", active_bg="#18559f", border="#4473ae", hover_bg="#2166bc", hover_border="#6d9de0").grid(row=0, column=0, padx=(0, 8), sticky="ew")
    _create_gui_button(step2_body, text="View Races Found", command=open_frames_folder, bg="#103b79", fg="#cbe4ff", active_bg="#18559f", border="#4473ae", hover_bg="#2166bc", hover_border="#6d9de0").grid(row=0, column=1, padx=(8, 0), sticky="ew")

    step2_note = tk.Label(
        step2_body,
        text="This looks through your videos and saves the race result screens.\nYou can check the screenshots first before creating the Excel file.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=720,
        font=("TkDefaultFont", 13),
    )
    step2_note.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 2))

    step3_card, step3_body = _create_step_card(
        steps_frame,
        step_number="STEP 3",
        title="Create The Excel File",
        description="Read the saved race screens",
        accent_bg="#1f6a34",
        accent_border="#3a7f53",
        number_bg="#17331f",
        number_fg="#c7f2d1",
        header_bg="#101d16",
    )
    step3_card.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
    step3_body.grid_columnconfigure(0, weight=1)
    step3_body.grid_columnconfigure(1, weight=1)
    _create_gui_button(step3_body, text="Create Excel Results", command=gui_run_ocr, bg="#1a5c28", fg="#d8f7df", active_bg="#27763a", border="#4a8a5b", hover_bg="#2e8a46", hover_border="#75b286").grid(row=0, column=0, padx=(0, 8), sticky="ew")
    _create_gui_button(step3_body, text="Open Excel Scores", command=open_excel_scores, bg="#1a5c28", fg="#d8f7df", active_bg="#27763a", border="#4a8a5b", hover_bg="#2e8a46", hover_border="#75b286").grid(row=0, column=1, padx=(8, 0), sticky="ew")

    ocr_note = tk.Label(
        step3_body,
        text="This reads the race screens that were already found and turns them into the Excel results file.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=760,
        font=("TkDefaultFont", 13),
    )
    ocr_note.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

    selection_card, selection_body = _create_step_card(
        steps_frame,
        step_number="STEP 2 + STEP 3",
        title="Full Run",
        description="Find races and create Excel in one go",
        accent_bg="#7b1fa2",
        accent_border="#7d4ca0",
        number_bg="#30163f",
        number_fg="#e8c7ff",
        header_bg="#181024",
    )
    selection_card.grid(row=1, column=1, rowspan=2, sticky="nsew")
    selection_body.grid_columnconfigure(0, weight=1)

    selection_intro = tk.Label(
        selection_body,
        text="Use this when you want to do everything in one go, but only for the videos you have currently selected.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["subtitle_fg"],
        anchor="w",
        justify="left",
        wraplength=380,
        font=("TkDefaultFont", 14),
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
    ).grid(row=1, column=0, sticky="ew", pady=(14, 0))

    selection_detail = tk.Label(
        selection_body,
        text="This finds the race screens and creates the Excel file for the selected videos only.",
        bg=GUI_THEME["panel_bg"],
        fg=GUI_THEME["muted_fg"],
        anchor="w",
        justify="left",
        wraplength=320,
        font=("TkDefaultFont", 13),
    )
    selection_detail.grid(row=2, column=0, sticky="w", pady=(10, 0))

    bottom_bar = tk.Frame(body_frame, bg=GUI_THEME["panel_bg"])
    bottom_bar.grid(row=1, column=0, sticky="ew", pady=(18, 0))
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

    danger_frame = tk.Frame(bottom_bar, bg=GUI_THEME["panel_bg"])
    danger_frame.grid(row=0, column=1, sticky="e")
    _create_gui_button(danger_frame, text="Delete Found Race Screenshots", command=clear_all_races_found, bg="#7a1010", fg="#ffd7d7", active_bg="#9d1818", border="#ab4747", hover_bg="#b71f1f", hover_border="#d85b5b").grid(row=0, column=0, padx=(0, 8), sticky="ew")
    _create_gui_button(danger_frame, text="Clear Output Folder", command=clear_output_results_gui, bg="#7a1010", fg="#ffd7d7", active_bg="#9d1818", border="#ab4747", hover_bg="#b71f1f", hover_border="#d85b5b").grid(row=0, column=1, sticky="ew")

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
    parser.add_argument("--video", help="Process only a specific video filename, for example Test_3_Races.mkv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
        run_extract(selected_video=args.video, include_subfolders=args.subfolders)
        return 0
    if args.scan_test:
        run_extract(selected_video=args.video, include_subfolders=args.subfolders)
        return 0
    if args.ocr:
        run_ocr(
            selected_video=args.video,
            include_subfolders=args.subfolders,
            selection_mode=args.selection,
        )
        return 0
    if args.all:
        if args.profile:
            run_profiled_all(selected_video=args.video)
        else:
            run_all(selected_video=args.video, include_subfolders=args.subfolders)
        return 0
    if args.selection:
        run_all(selected_video=args.video, selection_mode=True, include_subfolders=args.subfolders)
        return 0
    return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
