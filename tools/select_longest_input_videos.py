import argparse
import csv
from pathlib import Path

from mk8_local_play.extract_common import load_videos_from_folder, relative_video_path
from mk8_local_play.extract_frames import build_workflow_video_plan, format_duration
from mk8_local_play.project_paths import PROJECT_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="List the longest input videos by source duration.")
    parser.add_argument("--top", type=int, default=30, help="Number of videos to list")
    parser.add_argument("--subfolders", action="store_true", help="Include subfolders using app include rules")
    parser.add_argument("--output", help="Optional text or CSV output path for the ranked list")
    return parser.parse_args()


def collect_longest_videos(input_root: Path, *, include_subfolders: bool, top_n: int):
    video_paths = load_videos_from_folder(str(input_root), include_subfolders=include_subfolders)
    workflow_plan, _total_source_seconds = build_workflow_video_plan(
        video_paths,
        str(input_root),
        include_subfolders=include_subfolders,
    )
    return sorted(
        workflow_plan,
        key=lambda item: (float(item.get("source_length_s", 0.0) or 0.0), str(item.get("source_display_name", "")).lower()),
        reverse=True,
    )[: max(1, int(top_n))]


def write_ranked_output(output_path: Path, ranked, *, input_root: Path, include_subfolders: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Rank", "Duration", "Frames", "FPS", "Video"])
            for index, entry in enumerate(ranked, start=1):
                video_path = Path(str(entry["video_path"]))
                display_name = relative_video_path(video_path, input_root) if include_subfolders else video_path.name
                writer.writerow(
                    [
                        index,
                        format_duration(float(entry.get("source_length_s", 0.0) or 0.0)),
                        int(entry.get("nominal_total_frames", 0) or 0),
                        f"{float(entry.get('nominal_fps', 0.0) or 0.0):.2f}",
                        display_name,
                    ]
                )
        return

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in ranked:
            video_path = Path(str(entry["video_path"]))
            display_name = relative_video_path(video_path, input_root) if include_subfolders else video_path.name
            handle.write(f"{display_name}\n")


def main():
    args = parse_args()
    input_root = PROJECT_ROOT / "Input_Videos"
    ranked = collect_longest_videos(
        input_root,
        include_subfolders=args.subfolders,
        top_n=args.top,
    )

    print("Rank | Duration | Frames    | FPS   | Video")
    print("---- | -------- | --------- | ----- | -----")
    for index, entry in enumerate(ranked, start=1):
        video_path = Path(str(entry["video_path"]))
        display_name = relative_video_path(video_path, input_root) if args.subfolders else video_path.name
        print(
            f"{index:>4} | "
            f"{format_duration(float(entry.get('source_length_s', 0.0) or 0.0)):>8} | "
            f"{int(entry.get('nominal_total_frames', 0) or 0):>9,} | "
            f"{float(entry.get('nominal_fps', 0.0) or 0.0):>5.2f} | "
            f"{display_name}"
        )

    if args.output:
        write_ranked_output(
            Path(args.output),
            ranked,
            input_root=input_root,
            include_subfolders=args.subfolders,
        )


if __name__ == "__main__":
    main()
