import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path

from mk8_local_play.project_paths import PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Total Score timing fast path on a fixed video selection.")
    parser.add_argument("--subfolders", action="store_true", help="Resolve videos relative to Input_Videos with subfolder support")
    parser.add_argument("--videos", nargs="+", required=True, help="Video paths to pass to --videos")
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / ".codex_tmp" / "total_score_fast_path_benchmark.md"),
        help="Markdown report output path",
    )
    return parser.parse_args()


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def _extract_value(pattern: str, text: str):
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _extract_paths(text: str):
    results_csv = _extract_value(r"- Results CSV\s+([A-Z]:\\[^\r\n]+\.csv)", text)
    final_csv = _extract_value(r"- Final standings CSV\s+([A-Z]:\\[^\r\n]+\.csv)", text)
    return Path(results_csv) if results_csv else None, Path(final_csv) if final_csv else None


def run_variant(videos, *, subfolders: bool, enabled: bool):
    env = os.environ.copy()
    env["MK8_TOTAL_SCORE_TIMING_FAST_PATH"] = "1" if enabled else "0"
    command = [
        str(PROJECT_ROOT / ".venv" / "Scripts" / "mk8-local-play.exe"),
        "--selection",
    ]
    if subfolders:
        command.append("--subfolders")
    command.extend(["--videos", *videos])
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    output = completed.stdout
    results_csv, final_csv = _extract_paths(output)
    return {
        "enabled": enabled,
        "stdout": output,
        "processing_time": _extract_value(r"(?:Total processing time|Processing time)\s+([0-9:]+)", output),
        "extract_time": _extract_value(r"Extract race and score screens\s+([0-9:]+)", output),
        "ocr_time": _extract_value(r"OCR and workbook export\s+([0-9:]+)", output),
        "results_csv": str(results_csv) if results_csv else "",
        "final_csv": str(final_csv) if final_csv else "",
        "results_hash": _sha256(results_csv) if results_csv and results_csv.exists() else "",
        "final_hash": _sha256(final_csv) if final_csv and final_csv.exists() else "",
    }


def write_report(path: Path, videos, baseline, fast_path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Total Score Fast Path Benchmark")
    lines.append("")
    lines.append("## Selection")
    lines.append("")
    for video in videos:
        lines.append(f"- `{video}`")
    lines.append("")
    lines.append("| Variant | Total | Extract | OCR | Results Hash | Final Hash |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- |")
    for label, result in [("fast-path off", baseline), ("fast-path on", fast_path)]:
        lines.append(
            f"| {label} | {result['processing_time']} | {result['extract_time']} | {result['ocr_time']} | "
            f"`{result['results_hash']}` | `{result['final_hash']}` |"
        )
    lines.append("")
    hash_match = (
        baseline["results_hash"] == fast_path["results_hash"]
        and baseline["final_hash"] == fast_path["final_hash"]
    )
    lines.append(f"- Hash match: `{'yes' if hash_match else 'no'}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    baseline = run_variant(args.videos, subfolders=args.subfolders, enabled=False)
    fast_path = run_variant(args.videos, subfolders=args.subfolders, enabled=True)
    write_report(Path(args.report), args.videos, baseline, fast_path)
    print(Path(args.report))


if __name__ == "__main__":
    main()
