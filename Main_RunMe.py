import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None
    filedialog = None
    messagebox = None

from PIL import Image, ImageTk

from app_runtime import check_runtime, load_app_config, open_path


APP_CONFIG = load_app_config()
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "Input_Videos"
OUTPUT_DIR = SCRIPT_DIR / "Output_Results"
FRAMES_DIR = OUTPUT_DIR / "Frames"
RESULTS_XLSX = OUTPUT_DIR / "Tournament_Results.xlsx"
EXTRACT_SCRIPT = SCRIPT_DIR / "Extract_Frames_From_Video.py"
OCR_SCRIPT = SCRIPT_DIR / "Extract_Text_From_Frames.py"


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


def run_python_script(script_path: Path) -> None:
    subprocess.run([sys.executable, str(script_path)], check=True)


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
    if RESULTS_XLSX.exists():
        try:
            open_path(RESULTS_XLSX)
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


def run_extract() -> None:
    ensure_runtime_or_raise()
    run_python_script(EXTRACT_SCRIPT)


def run_ocr() -> None:
    ensure_runtime_or_raise(require_tesseract=True)
    run_python_script(OCR_SCRIPT)


def run_all() -> None:
    run_extract()
    run_ocr()


def print_runtime_status() -> int:
    ffmpeg_issues = check_runtime(APP_CONFIG, require_ffmpeg=True)
    tesseract_issues = check_runtime(APP_CONFIG, require_tesseract=True)

    print(f"Python executable: {sys.executable}")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Frames folder: {FRAMES_DIR}")
    print(f"Results file: {RESULTS_XLSX}")
    print(f"Tesseract: {'OK' if not tesseract_issues else 'MISSING'}")
    print(f"FFmpeg: {'OK' if not ffmpeg_issues else 'MISSING'}")
    print(f"OCR workers: {APP_CONFIG.ocr_workers}")
    print(f"Score analysis workers: {APP_CONFIG.score_analysis_workers}")
    print(f"Pass-1 scan workers: {APP_CONFIG.pass1_scan_workers}")
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
        print("Tkinter is not available. Use headless mode, for example: python Main_RunMe.py --all", file=sys.stderr)
        return 1

    root = tk.Tk()
    root.title("Mario Kart 8 Race Analysis")

    mario_kart_image_path = SCRIPT_DIR / "GUI" / "mariokart8_GUI_background_50.jpg"
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
    parser.add_argument("--ocr", action="store_true", help="Run OCR/export only")
    parser.add_argument("--all", action="store_true", help="Run extraction and OCR/export")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        return print_runtime_status()
    if args.extract:
        run_extract()
        return 0
    if args.ocr:
        run_ocr()
        return 0
    if args.all:
        run_all()
        return 0
    return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
