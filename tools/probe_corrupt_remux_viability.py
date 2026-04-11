import argparse
import json
from pathlib import Path

import cv2

from mk8_local_play import extract_video_io
from mk8_local_play.project_paths import PROJECT_ROOT


DEFAULT_VIDEO = (
    PROJECT_ROOT
    / "Input_Videos"
    / "corrupt"
    / "corrupt_Kampioen_2026-03-27 21-50-56.mkv"
)


def nominal_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the sampled OpenCV corrupt preflight before and after a light ffmpeg remux for one suspect video."
    )
    parser.add_argument(
        "--video",
        default=str(DEFAULT_VIDEO),
        help="Path to the corrupt video fixture to probe.",
    )
    parser.add_argument(
        "--keep-remux",
        action="store_true",
        help="Keep the remuxed probe file on disk instead of deleting it after comparison.",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    debug_dir = PROJECT_ROOT / "Output_Results" / "Debug" / "remux_probe" / video_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)
    remux_path = debug_dir / f"{video_path.stem}__remux_probe{video_path.suffix}"
    report_path = debug_dir / "preflight_compare.json"

    compare_result = extract_video_io.compare_preflight_before_after_remux(
        str(video_path),
        nominal_frame_count(video_path),
        remuxed_path=remux_path,
        keep_remuxed_file=args.keep_remux,
        video_identity=video_path.stem,
    )

    report_path.write_text(json.dumps(compare_result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), **compare_result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
