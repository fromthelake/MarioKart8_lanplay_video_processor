from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

from mk8_local_play.project_paths import PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline and transition-only selection on a fixed video set, "
            "preserving frame bundles, trace CSV, and outputs for review."
        )
    )
    parser.add_argument("--selection-file", required=True, help="Text file with one relative video path per line")
    parser.add_argument(
        "--label",
        default="fast_path_review_pair",
        help="Output folder label under .codex_tmp",
    )
    parser.add_argument("--subfolders", action="store_true", help="Resolve videos relative to Input_Videos with subfolder support")
    parser.add_argument("--trace-ocr", action="store_true", help="Enable OCR trace capture and copy the trace folder per variant")
    return parser.parse_args()


def _load_videos(selection_file: Path):
    return [
        line.strip()
        for line in selection_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_variant(videos: list[str], *, subfolders: bool, mode_label: str, transition_primary: bool, trace_ocr: bool):
    env = os.environ.copy()
    env["MK8_TRACE_TOTAL_SCORE_FRAMES"] = "1"
    env["MK8_TOTAL_SCORE_TIMING_FAST_PATH"] = "1" if transition_primary else "0"
    env["MK8_TOTAL_SCORE_TRANSITION_PRIMARY"] = "1" if transition_primary else "0"
    env["MK8_TOTAL_SCORE_STABLE_HINT"] = "0"
    if trace_ocr:
        env["MK8_TRACE_OCR_LINKING"] = "1"
        env["MK8_OCR_TRACE_LABEL"] = "fast_path_review_pair"
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


def _copy_if_exists(src: Path, dst: Path):
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    elif src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


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


def _write_manifest(target_dir: Path, *, mode_label: str, videos: list[str], log_text: str, outputs: dict[str, Path | None]):
    manifest = target_dir / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mode", mode_label])
        writer.writerow([])
        writer.writerow(["Videos"])
        for video in videos:
            writer.writerow([video])
        writer.writerow([])
        writer.writerow(["Outputs"])
        for key, value in outputs.items():
            writer.writerow([key, str(value) if value else ""])
    (target_dir / "run.log").write_text(log_text, encoding="utf-8")


def main():
    args = parse_args()
    selection_file = Path(args.selection_file)
    videos = _load_videos(selection_file)
    target_root = PROJECT_ROOT / ".codex_tmp" / args.label
    target_root.mkdir(parents=True, exist_ok=True)

    for mode_label, transition_primary in [("baseline", False), ("transition_only", True)]:
        log_text = _run_variant(
            videos,
            subfolders=args.subfolders,
            mode_label=mode_label,
            transition_primary=transition_primary,
            trace_ocr=args.trace_ocr,
        )
        outputs = _latest_output_files()
        target_dir = target_root / mode_label
        target_dir.mkdir(parents=True, exist_ok=True)
        _copy_if_exists(PROJECT_ROOT / "Output_Results" / "Frames", target_dir / "Frames")
        _copy_if_exists(PROJECT_ROOT / "Output_Results" / "Debug" / "total_score_frame_trace.csv", target_dir / "total_score_frame_trace.csv")
        if args.trace_ocr:
            _copy_if_exists(
                PROJECT_ROOT / "Output_Results" / "Debug" / "OCR_Tracing" / "fast_path_review_pair" / mode_label,
                target_dir / "OCR_Tracing",
            )
        for key, path in outputs.items():
            if path:
                _copy_if_exists(path, target_dir / path.name)
        _write_manifest(target_dir, mode_label=mode_label, videos=videos, log_text=log_text, outputs=outputs)

    print(target_root)


if __name__ == "__main__":
    main()
