import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mk8_local_play.project_paths import PROJECT_ROOT
from tools.analyze_total_score_trace import main as analyze_trace_main
from tools.select_longest_input_videos import collect_longest_videos, write_ranked_output


def parse_args():
    parser = argparse.ArgumentParser(description="Run Total Score trace study and analyze the results.")
    parser.add_argument("--top", type=int, default=30, help="Number of longest videos to trace")
    parser.add_argument("--subfolders", action="store_true", help="Include subfolders using app include rules")
    parser.add_argument(
        "--selection-file",
        default=str(PROJECT_ROOT / ".codex_tmp" / "total_score_trace_study_videos.txt"),
        help="Where to write the selected video list",
    )
    parser.add_argument(
        "--trace",
        default=str(PROJECT_ROOT / "Output_Results" / "Debug" / "total_score_frame_trace.csv"),
        help="Trace CSV output path",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / ".codex_tmp" / "total_score_trace_analysis.md"),
        help="Markdown analysis output path",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / ".codex_tmp" / "total_score_trace_analysis.csv"),
        help="CSV analysis output path",
    )
    parser.add_argument("--skip-run", action="store_true", help="Only analyze an existing trace file")
    return parser.parse_args()


def write_selection(selection_path: Path, top_n: int, include_subfolders: bool):
    ranked = collect_longest_videos(
        PROJECT_ROOT / "Input_Videos",
        include_subfolders=include_subfolders,
        top_n=top_n,
    )
    write_ranked_output(
        selection_path,
        ranked,
        input_root=PROJECT_ROOT / "Input_Videos",
        include_subfolders=include_subfolders,
    )
    return selection_path


def run_trace(selection_path: Path, trace_path: Path):
    videos = [
        line.strip()
        for line in selection_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    env = os.environ.copy()
    env["MK8_TRACE_TOTAL_SCORE_FRAMES"] = "1"
    command = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "mk8-local-play.exe"),
        "--selection",
        "--subfolders",
        "--videos",
        *videos,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=env)


def analyze(trace_path: Path, report_path: Path, csv_path: Path):
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "analyze_total_score_trace.py",
            "--trace",
            str(trace_path),
            "--report",
            str(report_path),
            "--csv",
            str(csv_path),
        ]
        analyze_trace_main()
    finally:
        sys.argv = original_argv


def main():
    args = parse_args()
    selection_path = Path(args.selection_file)
    trace_path = Path(args.trace)
    report_path = Path(args.report)
    csv_path = Path(args.csv)

    write_selection(selection_path, args.top, args.subfolders)
    if not args.skip_run:
        run_trace(selection_path, trace_path)
    analyze(trace_path, report_path, csv_path)
    print(selection_path)
    print(report_path)
    print(csv_path)


if __name__ == "__main__":
    main()
