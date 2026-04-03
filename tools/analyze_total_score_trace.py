import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from mk8_local_play.project_paths import PROJECT_ROOT


TRACE_COLUMNS = [
    "VideoLabel",
    "Video",
    "Race",
    "Score Layout",
    "Candidate Frame",
    "Candidate Time",
    "Detail Start Frame",
    "Detail End Frame",
    "Score Hit Frame",
    "Score Hit Time",
    "Score Hit Minus Candidate",
    "Race Anchor Frame",
    "Race Anchor Time",
    "Race Anchor Minus Candidate",
    "Actual Race Anchor Frame",
    "Actual Race Anchor Time",
    "Actual Race Minus Requested",
    "Transition Frame",
    "Transition Time",
    "Transition Minus Race Anchor",
    "Points Anchor Frame",
    "Points Anchor Time",
    "Points Anchor Minus Transition",
    "Actual Points Anchor Frame",
    "Actual Points Anchor Time",
    "Actual Points Minus Requested",
    "Stable Total Frame",
    "Stable Total Time",
    "Total Anchor Frame",
    "Total Anchor Time",
    "Total Anchor Minus Transition",
    "Actual Total Anchor Frame",
    "Actual Total Anchor Time",
    "Actual Total Minus Requested",
    "Visible Players",
    "Used Total Fallback",
    "Ignored Candidate",
    "Ignore Label",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Total Score frame trace output.")
    parser.add_argument(
        "--trace",
        default=str(PROJECT_ROOT / "Output_Results" / "Debug" / "total_score_frame_trace.csv"),
        help="Path to total_score_frame_trace.csv",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / ".codex_tmp" / "total_score_trace_analysis.md"),
        help="Markdown report output path",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / ".codex_tmp" / "total_score_trace_analysis.csv"),
        help="CSV summary output path",
    )
    return parser.parse_args()


def _parse_int(value):
    text = str(value or "").strip()
    if not text:
        return None
    return int(text)


def _percentile(sorted_values, percentile):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    lower_value = float(sorted_values[lower])
    upper_value = float(sorted_values[upper])
    return lower_value + (upper_value - lower_value) * (rank - lower)


def _summarize(values):
    clean = [int(value) for value in values if value is not None]
    clean.sort()
    if not clean:
        return {}
    return {
        "count": len(clean),
        "min": clean[0],
        "p50": _percentile(clean, 0.50),
        "p90": _percentile(clean, 0.90),
        "p95": _percentile(clean, 0.95),
        "p99": _percentile(clean, 0.99),
        "max": clean[-1],
        "mean": statistics.fmean(clean),
    }


def _fmt(value, digits=1):
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def load_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def build_analysis(rows):
    deltas = {
        "score_hit_minus_candidate": [],
        "race_anchor_minus_candidate": [],
        "transition_minus_race_anchor": [],
        "points_anchor_minus_transition": [],
        "total_anchor_minus_transition": [],
        "actual_race_minus_requested": [],
        "actual_points_minus_requested": [],
        "actual_total_minus_requested": [],
    }
    per_layout = defaultdict(lambda: defaultdict(list))
    per_fps_guess = defaultdict(lambda: defaultdict(list))
    fallback_count = 0
    ignored_count = 0
    videos = set()

    for row in rows:
        videos.add(str(row.get("VideoLabel") or ""))
        fallback_count += int(_parse_int(row.get("Used Total Fallback")) or 0)
        ignored_count += int(_parse_int(row.get("Ignored Candidate")) or 0)
        mapping = {
            "score_hit_minus_candidate": _parse_int(row.get("Score Hit Minus Candidate")),
            "race_anchor_minus_candidate": _parse_int(row.get("Race Anchor Minus Candidate")),
            "transition_minus_race_anchor": _parse_int(row.get("Transition Minus Race Anchor")),
            "points_anchor_minus_transition": _parse_int(row.get("Points Anchor Minus Transition")),
            "total_anchor_minus_transition": _parse_int(row.get("Total Anchor Minus Transition")),
            "actual_race_minus_requested": _parse_int(row.get("Actual Race Minus Requested")),
            "actual_points_minus_requested": _parse_int(row.get("Actual Points Minus Requested")),
            "actual_total_minus_requested": _parse_int(row.get("Actual Total Minus Requested")),
        }
        for key, value in mapping.items():
            if value is not None:
                deltas[key].append(value)
        layout = str(row.get("Score Layout") or "")
        for key, value in mapping.items():
            if value is not None:
                per_layout[layout][key].append(value)

        candidate_frame = _parse_int(row.get("Candidate Frame"))
        candidate_time = str(row.get("Candidate Time") or "")
        fps_guess = ""
        if candidate_frame and candidate_time:
            # derive rough fps bucket from frame->time, enough to separate 30/60
            parts = candidate_time.split(":")
            if len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                if seconds > 0:
                    fps_estimate = round(candidate_frame / seconds)
                    fps_guess = "60" if fps_estimate >= 45 else "30"
        if fps_guess:
            for key, value in mapping.items():
                if value is not None:
                    per_fps_guess[fps_guess][key].append(value)

    summaries = {name: _summarize(values) for name, values in deltas.items()}
    layout_summaries = {
        layout: {name: _summarize(values) for name, values in metric_map.items()}
        for layout, metric_map in per_layout.items()
    }
    fps_summaries = {
        fps_key: {name: _summarize(values) for name, values in metric_map.items()}
        for fps_key, metric_map in per_fps_guess.items()
    }

    return {
        "row_count": len(rows),
        "video_count": len(videos),
        "fallback_count": fallback_count,
        "ignored_count": ignored_count,
        "summaries": summaries,
        "layout_summaries": layout_summaries,
        "fps_summaries": fps_summaries,
    }


def write_csv(path: Path, analysis):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Count", "Min", "P50", "P90", "P95", "P99", "Max", "Mean"])
        for metric, summary in analysis["summaries"].items():
            writer.writerow([
                metric,
                summary.get("count", 0),
                summary.get("min", ""),
                _fmt(summary.get("p50")),
                _fmt(summary.get("p90")),
                _fmt(summary.get("p95")),
                _fmt(summary.get("p99")),
                summary.get("max", ""),
                _fmt(summary.get("mean")),
            ])


def write_report(path: Path, analysis):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Total Score Trace Analysis")
    lines.append("")
    lines.append(f"- Rows: `{analysis['row_count']}`")
    lines.append(f"- Videos: `{analysis['video_count']}`")
    lines.append(f"- Fallback totals used: `{analysis['fallback_count']}`")
    lines.append(f"- Ignored candidates: `{analysis['ignored_count']}`")
    lines.append("")
    lines.append("## Overall Deltas")
    lines.append("")
    lines.append("| Metric | Count | Min | P50 | P90 | P95 | P99 | Max | Mean |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for metric, summary in analysis["summaries"].items():
        lines.append(
            f"| {metric} | {summary.get('count', 0)} | {summary.get('min', '')} | "
            f"{_fmt(summary.get('p50'))} | {_fmt(summary.get('p90'))} | {_fmt(summary.get('p95'))} | "
            f"{_fmt(summary.get('p99'))} | {summary.get('max', '')} | {_fmt(summary.get('mean'))} |"
        )
    lines.append("")
    lines.append("## By Layout")
    lines.append("")
    for layout, metric_map in sorted(analysis["layout_summaries"].items()):
        lines.append(f"### `{layout or 'unknown'}`")
        lines.append("")
        lines.append("| Metric | Count | Min | P50 | P95 | Max | Mean |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for metric, summary in metric_map.items():
            if not summary:
                continue
            lines.append(
                f"| {metric} | {summary.get('count', 0)} | {summary.get('min', '')} | "
                f"{_fmt(summary.get('p50'))} | {_fmt(summary.get('p95'))} | "
                f"{summary.get('max', '')} | {_fmt(summary.get('mean'))} |"
            )
        lines.append("")
    if analysis["fps_summaries"]:
        lines.append("## By FPS Bucket")
        lines.append("")
        for fps_key, metric_map in sorted(analysis["fps_summaries"].items()):
            lines.append(f"### `{fps_key} fps`")
            lines.append("")
            lines.append("| Metric | Count | Min | P50 | P95 | Max | Mean |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for metric, summary in metric_map.items():
                if not summary:
                    continue
                lines.append(
                    f"| {metric} | {summary.get('count', 0)} | {summary.get('min', '')} | "
                    f"{_fmt(summary.get('p50'))} | {_fmt(summary.get('p95'))} | "
                    f"{summary.get('max', '')} | {_fmt(summary.get('mean'))} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    trace_path = Path(args.trace)
    rows = load_rows(trace_path)
    analysis = build_analysis(rows)
    write_csv(Path(args.csv), analysis)
    write_report(Path(args.report), analysis)
    print(Path(args.report))
    print(Path(args.csv))


if __name__ == "__main__":
    main()
