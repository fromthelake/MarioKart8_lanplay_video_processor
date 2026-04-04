from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path

from mk8_local_play.extract_common import build_video_identity
from mk8_local_play.project_paths import PROJECT_ROOT


VARIANTS = (
    ("baseline", False, False, "Baseline"),
    ("transition_only", True, False, "Transition-only"),
    ("stable_hint", True, True, "Transition + stable-hint"),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline, transition-only, and transition+stable-hint on one fixed video set, "
            "preserving outputs, traces, and frame bundles for human review."
        )
    )
    parser.add_argument("--selection-file", required=True, help="Text file with one relative video path per line")
    parser.add_argument("--label", default="three_mode_fast_path_review", help="Output folder label under .codex_tmp")
    parser.add_argument("--subfolders", action="store_true", help="Resolve relative paths under Input_Videos with subfolder support")
    parser.add_argument("--trace-ocr", action="store_true", help="Enable OCR identity tracing and preserve it per mode")
    return parser.parse_args()


def _load_videos(selection_file: Path) -> list[str]:
    return [
        line.strip()
        for line in selection_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _video_labels(videos: list[str], *, subfolders: bool) -> list[str]:
    input_root = PROJECT_ROOT / "Input_Videos"
    labels = []
    for video in videos:
        video_path = input_root / video if subfolders else Path(video)
        labels.append(build_video_identity(video_path, input_root=input_root, include_subfolders=subfolders))
    return labels


def _clean_selected_frame_dirs(video_labels: list[str]) -> None:
    frames_root = PROJECT_ROOT / "Output_Results" / "Frames"
    for label in video_labels:
        target = frames_root / label
        if target.exists():
            shutil.rmtree(target)


def _run_variant(
    videos: list[str],
    *,
    subfolders: bool,
    mode_label: str,
    fast_path_enabled: bool,
    stable_hint_enabled: bool,
    trace_ocr: bool,
) -> str:
    env = os.environ.copy()
    env["MK8_TRACE_TOTAL_SCORE_FRAMES"] = "1"
    env["MK8_TOTAL_SCORE_TIMING_FAST_PATH"] = "1" if fast_path_enabled else "0"
    env["MK8_TOTAL_SCORE_TRANSITION_PRIMARY"] = "1" if fast_path_enabled else "0"
    env["MK8_TOTAL_SCORE_STABLE_HINT"] = "1" if stable_hint_enabled else "0"
    if trace_ocr:
        env["MK8_TRACE_OCR_LINKING"] = "1"
        env["MK8_OCR_TRACE_LABEL"] = "three_mode_fast_path_review"
        env["MK8_OCR_TRACE_MODE"] = mode_label

    command = [str(PROJECT_ROOT / ".venv" / "Scripts" / "mk8-local-play.exe"), "--selection"]
    if subfolders:
        command.append("--subfolders")
    command.extend(["--videos", *videos])
    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout + ("\n" + completed.stderr if completed.stderr else "")


def _latest_output_files():
    out_dir = PROJECT_ROOT / "Output_Results"
    csvs = sorted(out_dir.glob("*_Tournament_Results.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    finals = sorted(out_dir.glob("*_Final_Standings.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    xlsxs = sorted(out_dir.glob("*_Tournament_Results.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {
        "results_csv": csvs[0] if csvs else None,
        "final_csv": finals[0] if finals else None,
        "results_xlsx": xlsxs[0] if xlsxs else None,
    }


def _copy_if_exists(src: Path, dst: Path):
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _copy_selected_frame_dirs(video_labels: list[str], target_frames_dir: Path) -> None:
    frames_root = PROJECT_ROOT / "Output_Results" / "Frames"
    target_frames_dir.mkdir(parents=True, exist_ok=True)
    for label in video_labels:
        src = frames_root / label
        if src.exists():
            _copy_if_exists(src, target_frames_dir / label)


def _extract_value(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _write_manifest(
    target_dir: Path,
    *,
    mode_label: str,
    mode_description: str,
    videos: list[str],
    video_labels: list[str],
    log_text: str,
    outputs: dict[str, Path | None],
) -> None:
    manifest = target_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Mode", mode_label])
        writer.writerow(["Description", mode_description])
        writer.writerow(["TotalProcessingTime", _extract_value(r"(?:Total processing time|Processing time)\s+([0-9:]+)", log_text)])
        writer.writerow(["ExtractTime", _extract_value(r"Extract race and score screens\s+([0-9:]+)", log_text)])
        writer.writerow(["OcrTime", _extract_value(r"OCR and workbook export\s+([0-9:]+)", log_text)])
        writer.writerow([])
        writer.writerow(["Videos"])
        writer.writerow(["RelativePath", "VideoLabel"])
        for video, label in zip(videos, video_labels):
            writer.writerow([video, label])
        writer.writerow([])
        writer.writerow(["Outputs"])
        for key, value in outputs.items():
            writer.writerow([key, str(value) if value else ""])
    (target_dir / "run.log").write_text(log_text, encoding="utf-8")


def main():
    args = parse_args()
    selection_file = Path(args.selection_file)
    videos = _load_videos(selection_file)
    video_labels = _video_labels(videos, subfolders=args.subfolders)
    target_root = PROJECT_ROOT / ".codex_tmp" / args.label
    target_root.mkdir(parents=True, exist_ok=True)

    for mode_label, fast_path_enabled, stable_hint_enabled, mode_description in VARIANTS:
        _clean_selected_frame_dirs(video_labels)
        trace_path = PROJECT_ROOT / "Output_Results" / "Debug" / "total_score_frame_trace.csv"
        if trace_path.exists():
            trace_path.unlink()

        log_text = _run_variant(
            videos,
            subfolders=args.subfolders,
            mode_label=mode_label,
            fast_path_enabled=fast_path_enabled,
            stable_hint_enabled=stable_hint_enabled,
            trace_ocr=args.trace_ocr,
        )
        outputs = _latest_output_files()
        target_dir = target_root / mode_label
        target_dir.mkdir(parents=True, exist_ok=True)
        _copy_selected_frame_dirs(video_labels, target_dir / "Frames")
        _copy_if_exists(PROJECT_ROOT / "Output_Results" / "Debug" / "total_score_frame_trace.csv", target_dir / "total_score_frame_trace.csv")
        if args.trace_ocr:
            _copy_if_exists(
                PROJECT_ROOT / "Output_Results" / "Debug" / "OCR_Tracing" / "three_mode_fast_path_review" / mode_label,
                target_dir / "OCR_Tracing",
            )
        for key, path in outputs.items():
            if path:
                _copy_if_exists(path, target_dir / path.name)
        _write_manifest(
            target_dir,
            mode_label=mode_label,
            mode_description=mode_description,
            videos=videos,
            video_labels=video_labels,
            log_text=log_text,
            outputs=outputs,
        )

    print(target_root)


if __name__ == "__main__":
    main()
