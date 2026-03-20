#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark OCR worker counts on a scoped MK8 run while preserving the OCR timing profile "
            "from each run in separate log files."
        )
    )
    parser.add_argument(
        "--video",
        default="Demo_CaptureCard_Race.mp4",
        help="Scoped source video to run through `mk8_local_play.main --selection`.",
    )
    parser.add_argument(
        "--workers",
        default="1,2,4,8",
        help="Comma-separated OCR worker counts to test.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Hard timeout per run. Runs exceeding this are terminated and marked as timed out.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".codex_tmp") / "ocr_worker_benchmarks",
        help="Directory for benchmark logs and summary files.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(".venv") / "Scripts" / "python.exe",
        help="Python interpreter to use for benchmark runs.",
    )
    return parser.parse_args()


def parse_worker_counts(raw_value: str) -> list[int]:
    values: list[int] = []
    for part in str(raw_value).split(","):
        token = part.strip()
        if not token:
            continue
        worker_count = int(token)
        if worker_count <= 0:
            raise ValueError(f"Worker count must be positive: {token}")
        values.append(worker_count)
    if not values:
        raise ValueError("No worker counts supplied.")
    return values


def sanitize_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))


def build_env(worker_count: int) -> dict[str, str]:
    env = os.environ.copy()
    env["MK8_OCR_WORKERS"] = str(worker_count)
    env["MK8_WRITE_DEBUG_CSV"] = "0"
    env["MK8_WRITE_DEBUG_SCORE_IMAGES"] = "0"
    env["MK8_WRITE_DEBUG_LINKING_EXCEL"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def summarize_output(stdout: str, stderr: str) -> list[str]:
    lines = []
    combined = []
    if stdout:
        combined.extend(stdout.splitlines())
    if stderr:
        combined.extend(stderr.splitlines())

    for line in combined:
        stripped = line.strip()
        if not stripped:
            continue
        if "OCR engine calls:" in stripped:
            lines.append(stripped)
            continue
        if stripped.startswith("- player_name_") or stripped.startswith("- track_"):
            lines.append(stripped)
            continue
        if stripped.startswith("- detect_") or stripped.startswith("- extract_") or stripped.startswith("- build_"):
            lines.append(stripped)
            continue
        if stripped.startswith("Duration:") or stripped.startswith("Races processed:"):
            lines.append(stripped)
            continue
    return lines


def main() -> int:
    args = parse_args()
    worker_counts = parse_worker_counts(args.workers)
    project_root = Path(__file__).resolve().parents[1]
    python_path = args.python
    if not python_path.is_absolute():
        python_path = project_root / python_path
    if not python_path.exists():
        raise SystemExit(f"Python interpreter not found: {python_path}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str]] = []
    video_label = sanitize_name(Path(args.video).stem)

    for worker_count in worker_counts:
        command = [
            str(python_path),
            "-m",
            "mk8_local_play.main",
            "--selection",
            "--video",
            args.video,
        ]
        env = build_env(worker_count)
        start_time = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
            )
            elapsed = time.perf_counter() - start_time
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            exit_code = completed.returncode
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            elapsed = time.perf_counter() - start_time
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            exit_code = 124
            timed_out = True

        log_path = output_dir / f"{video_label}_workers_{worker_count}.log"
        summary_path = output_dir / f"{video_label}_workers_{worker_count}.summary.txt"
        log_path.write_text(
            "\n".join(
                [
                    f"workers={worker_count}",
                    f"elapsed_seconds={elapsed:.2f}",
                    f"exit_code={exit_code}",
                    "",
                    "=== STDOUT ===",
                    stdout.rstrip(),
                    "",
                    "=== STDERR ===",
                    stderr.rstrip(),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        summary_lines = summarize_output(stdout, stderr)
        summary_path.write_text("\n".join(summary_lines) + ("\n" if summary_lines else ""), encoding="utf-8")

        summary_rows.append(
            {
                "workers": str(worker_count),
                "elapsed_seconds": f"{elapsed:.2f}",
                "exit_code": str(exit_code),
                "timed_out": str(timed_out),
                "log": str(log_path),
                "summary": str(summary_path),
            }
        )

    summary_csv_path = output_dir / f"{video_label}_summary.csv"
    summary_csv_lines = ["workers,elapsed_seconds,exit_code,timed_out,log,summary"]
    for row in summary_rows:
        summary_csv_lines.append(
            ",".join(
                [
                    row["workers"],
                    row["elapsed_seconds"],
                    row["exit_code"],
                    row["timed_out"],
                    f'"{row["log"]}"',
                    f'"{row["summary"]}"',
                ]
            )
        )
    summary_csv_path.write_text("\n".join(summary_csv_lines) + "\n", encoding="utf-8")

    print(f"Benchmark summary written to {summary_csv_path}")
    for row in summary_rows:
        print(
            f"workers={row['workers']} | elapsed={row['elapsed_seconds']}s | "
            f"exit={row['exit_code']} | timed_out={row['timed_out']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
