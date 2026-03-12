import argparse
import cProfile
import glob
import os
import pstats
import subprocess
import sys
import cv2
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
from .project_paths import PROJECT_ROOT


APP_CONFIG = load_app_config()
SCRIPT_DIR = PROJECT_ROOT
INPUT_DIR = SCRIPT_DIR / "Input_Videos"
OUTPUT_DIR = SCRIPT_DIR / "Output_Results"
FRAMES_DIR = OUTPUT_DIR / "Frames"
EXTRACT_MODULE = "mk8_local_play.extract_frames"
OCR_MODULE = "mk8_local_play.extract_text"
PROFILE_OUTPUT = OUTPUT_DIR / "Debug" / "performance_profile.txt"


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def selected_input_video_files(selected_video: str | None = None) -> list[Path]:
    all_video_files = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES
    ) if INPUT_DIR.exists() else []
    if not selected_video:
        return all_video_files
    selected_name = Path(selected_video).name.lower()
    exact_matches = [path for path in all_video_files if path.name.lower() == selected_name]
    if exact_matches:
        return exact_matches
    selected_stem = Path(selected_video).stem.lower()
    return [path for path in all_video_files if path.stem.lower() == selected_stem]


def selected_race_classes_for_videos(video_files: list[Path]) -> list[str]:
    return [path.stem for path in video_files]


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


def run_python_module(module_name: str, extra_args: list[str] | None = None) -> None:
    # Always prefer the repo-local virtualenv for child scripts so extraction and OCR
    # run with the same dependencies a user installed during setup.
    command = [resolve_project_python(), "-m", module_name]
    if extra_args:
        command.extend(extra_args)
    subprocess.run(command, check=True)


def ensure_runtime_or_raise(require_tesseract: bool = False, require_ffmpeg: bool = False) -> None:
    issues = check_runtime(APP_CONFIG, require_tesseract=require_tesseract, require_ffmpeg=require_ffmpeg)
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
    if FRAMES_DIR.exists():
        png_files = glob.glob(str(FRAMES_DIR / "*.png"))
        if png_files:
            for file in png_files:
                try:
                    os.remove(file)
                except Exception as exc:
                    show_error("Error", f"Unable to delete file {file}: {exc}")
                    return
            show_info("Success", "All .png files have been deleted.")
        else:
            show_info("Info", "No .png files found to delete.")
    else:
        show_warning("Warning", "The frames folder does not exist.")


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


def run_extract(selected_video: str | None = None) -> None:
    ensure_runtime_or_raise()
    extra_args = ["--video", selected_video] if selected_video else None
    run_python_module(EXTRACT_MODULE, extra_args=extra_args)


def run_ocr(selected_video: str | None = None) -> None:
    ensure_runtime_or_raise(require_tesseract=True)
    extra_args = ["--video", Path(selected_video).stem] if selected_video else None
    run_python_module(OCR_MODULE, extra_args=extra_args)


def run_all(selected_video: str | None = None, selection_mode: bool = False) -> None:
    ensure_runtime_or_raise(require_tesseract=True)
    from . import extract_frames, extract_text

    mode_label = "Run selection" if selection_mode else "Run all"
    LOGGER.log("[Run - Phase Start]", mode_label, color_name="cyan")
    video_files = selected_input_video_files(selected_video=selected_video)
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
            source_summaries.append(f"{video_path.name} ({extract_frames.format_duration(source_length)})")
        capture.release()
    LOGGER.log("[Run - Input Summary]", f"Videos: {len(source_summaries)} | Total source length: {extract_frames.format_duration(total_source_seconds)}", color_name="cyan")
    for index, summary in enumerate(source_summaries, start=1):
        LOGGER.log("[Run - Input Summary]", f"{index}. {summary}")
    selected_video_names = [path.name for path in video_files]
    extract_result = extract_frames.extract_frames(return_frame_cache=True, selected_videos=selected_video_names or None)
    LOGGER.blank_lines(2)
    frame_bundle_cache = extract_result["frame_bundle_cache"]
    extract_text.configure_tesseract(extract_text.pytesseract, extract_text.APP_CONFIG)
    selected_race_classes = selected_race_classes_for_videos(video_files) if selection_mode else None
    ocr_result = extract_text.process_images_in_folder(
        str(FRAMES_DIR),
        in_memory_frame_bundles=frame_bundle_cache,
        selected_race_classes=selected_race_classes,
    )
    frame_bundle_cache.clear()
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
            "run_tesseract_image_to_data",
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
    tesseract_issues = check_runtime(APP_CONFIG, require_tesseract=True)
    gpu_runtime = detect_gpu_runtime(APP_CONFIG)
    gui_available, gui_reason = gui_runtime_available()

    print(f"Python executable: {sys.executable}")
    print(f"Child script Python: {resolve_project_python()}")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Frames folder: {FRAMES_DIR}")
    print(f"Latest results file: {find_latest_results_xlsx(OUTPUT_DIR)}")
    print(f"Tesseract: {'OK' if not tesseract_issues else 'MISSING'}")
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

    issues = []
    issues.extend(ffmpeg_issues)
    issues.extend(tesseract_issues)
    if issues:
        print("\nRuntime issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    return 0


def exit_application(root_window) -> None:
    root_window.destroy()


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

    mario_kart_image_path = resolve_asset_file("gui", "mariokart8_gui_background_50.jpg")
    mario_kart_image = Image.open(mario_kart_image_path)
    mario_kart_image = mario_kart_image.resize((746, 420), Image.LANCZOS)
    mario_kart_image = ImageTk.PhotoImage(mario_kart_image)

    root.geometry("746x420")

    image_label = tk.Label(root, image=mario_kart_image)
    image_label.place(x=0, y=0)

    top_spacing = 0.005
    button_height = (1 - top_spacing) / 8

    open_videos_button = tk.Button(root, text="Step 1 - Ensure Videos are in Input Folder", command=open_videos_folder, font=("Helvetica", 16))
    open_videos_button.config(bg="#ffcc00", fg="#000000")
    open_videos_button.place(relx=0.5, rely=top_spacing + button_height * 0.5, anchor=tk.CENTER)

    merge_videos_button = tk.Button(root, text="Merge Videos", command=merge_videos, font=("Helvetica", 16))
    merge_videos_button.config(bg="#d3d3d3", fg="#000000")
    merge_videos_button.place(relx=0.5, rely=top_spacing + button_height * 1.4, anchor=tk.CENTER)

    merge_videos_note = tk.Label(root, text="(optional only needed for multiple clips which should be treated as a single Race Poule)", font=("Helvetica", 10), bg="#d3d3d3", fg="#000000")
    merge_videos_note.place(relx=0.5, rely=top_spacing + button_height * 2.0, anchor=tk.CENTER)

    analyze_button = tk.Button(root, text="Step 2 - Analyse Videos and Find Races", command=select_video, font=("Helvetica", 16))
    analyze_button.config(bg="#ffcc00", fg="#000000")
    analyze_button.place(relx=0.5, rely=top_spacing + button_height * 2.7, anchor=tk.CENTER)

    view_races_button = tk.Button(root, text="View Races Found", command=open_frames_folder, font=("Helvetica", 16))
    view_races_button.config(bg="#d3d3d3", fg="#000000")
    view_races_button.place(relx=0.5, rely=top_spacing + button_height * 3.6, anchor=tk.CENTER)

    clear_races_button = tk.Button(root, text="Delete All Races Found", command=clear_all_races_found, font=("Helvetica", 16))
    clear_races_button.config(bg="#ff4444", fg="#ffffff")
    clear_races_button.place(relx=0.5, rely=top_spacing + button_height * 4.5, anchor=tk.CENTER)

    export_button = tk.Button(root, text="Step 3 - Export Found Races into Excel", command=export_to_excel, font=("Helvetica", 16))
    export_button.config(bg="#ffcc00", fg="#000000")
    export_button.place(relx=0.5, rely=top_spacing + button_height * 5.4, anchor=tk.CENTER)

    open_excel_button = tk.Button(root, text="Open Excel Scores", command=open_excel_scores, font=("Helvetica", 16))
    open_excel_button.config(bg="#d3d3d3", fg="#000000")
    open_excel_button.place(relx=0.5, rely=top_spacing + button_height * 6.3, anchor=tk.CENTER)

    exit_button = tk.Button(root, text="Exit", command=lambda: exit_application(root), font=("Helvetica", 16))
    exit_button.config(bg="#ff4444", fg="#ffffff")
    exit_button.place(relx=0.5, rely=top_spacing + button_height * 7.2, anchor=tk.CENTER)

    root.mainloop()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mario Kart 8 local play video processing")
    parser.add_argument("--check", action="store_true", help="Print runtime/dependency status")
    parser.add_argument("--extract", action="store_true", help="Run frame extraction only")
    parser.add_argument("--scan-test", action="store_true", help="Benchmark extraction/scan only without OCR")
    parser.add_argument("--ocr", action="store_true", help="Run OCR/export only")
    parser.add_argument("--all", action="store_true", help="Run extraction and OCR/export")
    parser.add_argument("--selection", action="store_true", help="Run extraction and OCR only for the videos currently selected in Input_Videos")
    parser.add_argument("--profile", action="store_true", help="Write a whole-process performance profile during --all")
    parser.add_argument("--video", help="Process only a specific video filename, for example Test_3_Races.mkv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        return print_runtime_status()
    if args.extract:
        run_extract(selected_video=args.video)
        return 0
    if args.scan_test:
        run_extract(selected_video=args.video)
        return 0
    if args.ocr:
        run_ocr(selected_video=args.video)
        return 0
    if args.all:
        if args.profile:
            run_profiled_all(selected_video=args.video)
        else:
            run_all(selected_video=args.video)
        return 0
    if args.selection:
        run_all(selected_video=args.video, selection_mode=True)
        return 0
    return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
