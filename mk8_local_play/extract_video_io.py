import os
import subprocess
import cv2
import time
import threading
import queue
import shutil
from pathlib import Path

from .console_logging import LOGGER
from .project_paths import PROJECT_ROOT


INPUT_VIDEOS_DIR = PROJECT_ROOT / "Input_Videos"
CORRUPT_VIDEOS_DIR = INPUT_VIDEOS_DIR / "corrupt"
CORRUPT_CHECK_SAMPLE_COUNT = 60
CORRUPT_CHECK_EDGE_SAMPLE_COUNT = 10
CORRUPT_CHECK_EDGE_FRACTION = 0.02
CORRUPT_CHECK_READ_TIMEOUT_S = 5.0
CORRUPT_CHECK_FRAME_SLACK = 5
VIDEO_OPERATION_TIMEOUT_DIVISOR = 50.0
VIDEO_OPERATION_TIMEOUT_MIN_S = 30.0
VIDEO_OPERATION_TIMEOUT_MAX_S = 300.0
VIDEO_REPAIR_PROGRESS_INTERVAL_S = 2.0


def add_timing(stats, key, start_time):
    """Accumulate elapsed time for a named timing bucket."""
    stats[key] += time.perf_counter() - start_time


def video_operation_timeout_s(duration_s: float | None) -> float:
    if duration_s is None or duration_s <= 0:
        return VIDEO_OPERATION_TIMEOUT_MIN_S
    return min(
        VIDEO_OPERATION_TIMEOUT_MAX_S,
        max(VIDEO_OPERATION_TIMEOUT_MIN_S, float(duration_s) / VIDEO_OPERATION_TIMEOUT_DIVISOR),
    )


def seek_to_frame(capture, frame_number, stats):
    """Seek to a frame and record the seek cost."""
    start_time = time.perf_counter()
    result = capture.set(1, frame_number)
    add_timing(stats, "seek_time_s", start_time)
    stats["seek_calls"] += 1
    return result


def read_video_frame(capture, stats):
    """Read and decode one frame while tracking I/O cost."""
    start_time = time.perf_counter()
    ret, frame = capture.read()
    add_timing(stats, "read_time_s", start_time)
    stats["read_calls"] += 1
    return ret, frame


def read_video_frame_with_timeout(capture, stats, timeout_s):
    """Read and decode one frame, but stop waiting if the backend stalls too long."""
    result = {}

    def _worker():
        try:
            result["value"] = capture.read()
        except Exception as exc:  # pragma: no cover - defensive bridge from native backend
            result["error"] = exc

    start_time = time.perf_counter()
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout_s)
    add_timing(stats, "read_time_s", start_time)
    stats["read_calls"] += 1
    if worker.is_alive():
        stats["read_timeouts"] = stats.get("read_timeouts", 0) + 1
        return False, None, True
    if "error" in result:
        raise result["error"]
    ret, frame = result.get("value", (False, None))
    return ret, frame, False


def grab_video_frame(capture, stats):
    """Advance one frame without full decode when image data is not needed."""
    start_time = time.perf_counter()
    ret = capture.grab()
    add_timing(stats, "grab_time_s", start_time)
    stats["grab_calls"] += 1
    return ret


def advance_frames_by_grab(capture, frames_to_advance, stats):
    """Skip forward cheaply inside scan loops."""
    for _ in range(max(0, frames_to_advance)):
        if not grab_video_frame(capture, stats):
            return False
    return True


def actual_frame_after_read(capture):
    """Best-effort actual decoded frame index after a successful read()."""
    return max(0, int(capture.get(1)) - 1)


def log_exported_frame(metadata_writer, video_path, race_number, kind, requested_frame, actual_frame, fps, frame_to_timecode):
    """Record requested and actual decoded frames for each exported screenshot."""
    if metadata_writer is None:
        return
    metadata_writer.writerow(
        [
            os.path.basename(video_path),
            f"{race_number:03}",
            kind,
            int(requested_frame),
            frame_to_timecode(requested_frame, fps),
            int(actual_frame),
            frame_to_timecode(actual_frame, fps),
        ]
    )


def _record_corrupt_check_duration(stats: dict | None, start_time: float) -> None:
    if stats is not None:
        stats["corrupt_check_duration_s"] = stats.get("corrupt_check_duration_s", 0.0) + (time.perf_counter() - start_time)


def _distribute_samples(start_frame: int, end_frame: int, sample_target: int) -> set[int]:
    if sample_target <= 0 or end_frame < start_frame:
        return set()
    frame_span = end_frame - start_frame + 1
    if frame_span <= sample_target:
        return set(range(start_frame, end_frame + 1))
    if sample_target == 1:
        return {start_frame}
    return {
        int(round(start_frame + (index * (frame_span - 1)) / (sample_target - 1)))
        for index in range(sample_target)
    }


def _sample_probe_frames(total_frames: int, sample_count: int = CORRUPT_CHECK_SAMPLE_COUNT) -> list[int]:
    if total_frames <= 0:
        return []
    last_frame = max(0, total_frames - 1)
    target_count = max(1, int(sample_count))
    edge_target = max(0, min(CORRUPT_CHECK_EDGE_SAMPLE_COUNT, target_count))
    edge_window = max(1, int(round(total_frames * CORRUPT_CHECK_EDGE_FRACTION)))

    front_end = min(last_frame, max(edge_target - 1, edge_window - 1))
    back_start = max(0, min(last_frame, total_frames - max(edge_target, edge_window)))

    probe_frames = set()
    probe_frames.update(_distribute_samples(0, front_end, edge_target))
    probe_frames.update(_distribute_samples(back_start, last_frame, edge_target))

    remaining_target = max(0, target_count - len(probe_frames))
    middle_start = min(last_frame, front_end + 1)
    middle_end = max(0, back_start - 1)
    probe_frames.update(_distribute_samples(middle_start, middle_end, remaining_target))

    if len(probe_frames) < min(total_frames, target_count):
        probe_frames.update(_distribute_samples(0, last_frame, target_count - len(probe_frames)))

    return sorted(frame for frame in probe_frames if 0 <= frame <= last_frame)


def _timed_probe_read(capture, frame_number: int, timeout_s: float) -> tuple[bool, object, bool, int]:
    result = {}

    def _worker():
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
            result["value"] = capture.read()
        except Exception as exc:  # pragma: no cover - native backend bridge
            result["error"] = exc

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        return False, None, True, int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    if "error" in result:
        raise result["error"]
    ret, frame = result.get("value", (False, None))
    actual_frame = max(0, int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0) - 1)
    return ret, frame, False, actual_frame


def sample_video_readability(video_path: str, nominal_total_frames: int, stats: dict | None = None) -> dict:
    """Cheap corruption preflight using sampled OpenCV seeks/reads across the file."""
    video_name = os.path.basename(video_path)
    LOGGER.log("[Frame Count Scan]", f"sampling readable frames with OpenCV for {video_name}", color_name="cyan")
    overall_start = time.perf_counter()
    result = {
        "status": "checked",
        "is_suspect": False,
        "reason": "all sampled frames read successfully",
        "probe_count": 0,
        "failed_frame": None,
        "actual_frame": None,
    }
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        _record_corrupt_check_duration(stats, overall_start)
        return {**result, "status": "inconclusive", "is_suspect": True, "reason": "opencv could not open file"}

    total_frames = nominal_total_frames or int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe_frames = _sample_probe_frames(total_frames)
    result["probe_count"] = len(probe_frames)
    for frame_number in probe_frames:
        try:
            ret, frame, timed_out, actual_frame = _timed_probe_read(capture, frame_number, CORRUPT_CHECK_READ_TIMEOUT_S)
        except Exception as exc:
            capture.release()
            _record_corrupt_check_duration(stats, overall_start)
            return {
                **result,
                "status": "inconclusive",
                "is_suspect": True,
                "reason": f"probe raised {type(exc).__name__}",
                "failed_frame": int(frame_number),
            }
        if timed_out:
            capture.release()
            _record_corrupt_check_duration(stats, overall_start)
            return {
                **result,
                "status": "inconclusive",
                "is_suspect": True,
                "reason": f"probe timed out at frame {frame_number}",
                "failed_frame": int(frame_number),
                "actual_frame": int(actual_frame),
            }
        if not ret or frame is None:
            capture.release()
            _record_corrupt_check_duration(stats, overall_start)
            return {
                **result,
                "status": "suspect",
                "is_suspect": True,
                "reason": f"probe read failed at frame {frame_number}",
                "failed_frame": int(frame_number),
                "actual_frame": int(actual_frame),
            }
        if abs(int(actual_frame) - int(frame_number)) > CORRUPT_CHECK_FRAME_SLACK:
            capture.release()
            _record_corrupt_check_duration(stats, overall_start)
            return {
                **result,
                "status": "suspect",
                "is_suspect": True,
                "reason": f"probe landed at frame {actual_frame} instead of {frame_number}",
                "failed_frame": int(frame_number),
                "actual_frame": int(actual_frame),
            }
    capture.release()
    LOGGER.log("[Frame Count Scan]", f"sample probe passed for {video_name}: {len(probe_frames)} frames checked", color_name="green")
    _record_corrupt_check_duration(stats, overall_start)
    return result


def _rename_with_collision_suffix(source_path: Path, target_path: Path) -> Path:
    candidate = target_path
    counter = 1
    while candidate.exists():
        candidate = target_path.with_name(f"{target_path.stem}_{counter}{target_path.suffix}")
        counter += 1
    source_path.replace(candidate)
    return candidate


def _repair_output_suffix(_video_path: Path) -> str:
    return ".mp4"


def has_corrupt_archive(video_path: str) -> bool:
    source_path = Path(video_path)
    if not CORRUPT_VIDEOS_DIR.exists():
        return False
    return any(candidate.is_file() for candidate in CORRUPT_VIDEOS_DIR.glob(f"corrupt_{source_path.stem}.*"))




def _ffmpeg_repair_command(source_path: Path, repaired_path: Path) -> list[str]:
    suffix = repaired_path.suffix.lower()
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-fflags",
        "+genpts+discardcorrupt",
        "-err_detect",
        "ignore_err",
        "-i",
        str(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-fps_mode",
        "cfr",
        "-y",
    ]
    if suffix == ".webm":
        command.extend(["-c:v", "libvpx-vp9", "-row-mt", "1"])
    else:
        command.extend([
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"
        ])
    command.append(str(repaired_path))
    return command


def _format_media_clock(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def _run_ffmpeg_repair_command(command: list[str], source_name: str, timeout_s: float, source_duration_s: float | None = None) -> None:
    progress_command = command[:1] + ["-progress", "pipe:1", "-nostats"] + command[1:]
    process = subprocess.Popen(
        progress_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    progress_lines: queue.Queue[str] = queue.Queue()

    def _reader() -> None:
        if process.stdout is None:
            return
        try:
            for line in process.stdout:
                progress_lines.put(line.rstrip())
        finally:
            process.stdout.close()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    start_time = time.perf_counter()
    last_progress_log = start_time
    encoded_seconds = 0.0

    while True:
        try:
            line = progress_lines.get(timeout=0.2)
            if "=" in line:
                key, value = line.split("=", 1)
                if key == "out_time_ms":
                    try:
                        encoded_seconds = max(encoded_seconds, int(value) / 1_000_000.0)
                    except ValueError:
                        pass
                elif key == "out_time_us":
                    try:
                        encoded_seconds = max(encoded_seconds, int(value) / 1_000_000.0)
                    except ValueError:
                        pass
        except queue.Empty:
            pass

        elapsed = time.perf_counter() - start_time
        if elapsed >= timeout_s:
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=5)
            except Exception:
                pass
            raise subprocess.TimeoutExpired(progress_command, timeout_s)

        if process.poll() is None and time.perf_counter() - last_progress_log >= VIDEO_REPAIR_PROGRESS_INTERVAL_S:
            detail = ""
            if encoded_seconds > 0 and source_duration_s and source_duration_s > 0:
                progress_pct = min(100.0, max(0.0, (encoded_seconds / source_duration_s) * 100.0))
                detail = (
                    f" | {progress_pct:.0f}%"
                    f" | encoded {_format_media_clock(encoded_seconds)} / {_format_media_clock(source_duration_s)}"
                )
            elif encoded_seconds > 0:
                detail = f" | encoded {_format_media_clock(encoded_seconds)}"
            LOGGER.log(
                "[Video Repair]",
                f"Still repairing {source_name} | elapsed {elapsed:.1f}s{detail}",
                color_name="yellow",
            )
            last_progress_log = time.perf_counter()

        if process.poll() is not None:
            reader.join(timeout=1.0)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, progress_command)
            return


def repair_video_if_needed(video_path: str, nominal_total_frames: int, preflight_result: dict | None, duration_s: float | None = None, stats: dict | None = None) -> str:
    """Repair a video with ffmpeg when sampled OpenCV preflight marks the source as suspect."""
    if not preflight_result or not preflight_result.get("is_suspect"):
        return video_path

    source_path = Path(video_path)
    if not shutil.which("ffmpeg"):
        LOGGER.log(
            "[Video Repair]",
            f"Skipping repair for {source_path.name}: ffmpeg not found on PATH",
            color_name="yellow",
        )
        return video_path

    repair_suffix = _repair_output_suffix(source_path)
    repairing_path = source_path.with_name(f"{source_path.stem}__repairing{repair_suffix}")
    archived_path = CORRUPT_VIDEOS_DIR / f"corrupt_{source_path.name}"

    try:
        if repairing_path.exists():
            repairing_path.unlink()
    except Exception:
        pass

    repair_reason = preflight_result.get("reason") or f"sample probe flagged file with nominal {nominal_total_frames} frames"
    LOGGER.log(
        "[Video Repair]",
        f"Repairing {source_path.name}: {repair_reason}",
        color_name="yellow",
    )

    repair_start = time.perf_counter()
    try:
        _run_ffmpeg_repair_command(
            _ffmpeg_repair_command(source_path, repairing_path),
            source_path.name,
            video_operation_timeout_s(duration_s),
            source_duration_s=duration_s,
        )
    except subprocess.TimeoutExpired:
        try:
            if repairing_path.exists():
                repairing_path.unlink()
        except Exception:
            pass
        if stats is not None:
            stats["repair_duration_s"] = stats.get("repair_duration_s", 0.0) + (time.perf_counter() - repair_start)
        LOGGER.log(
            "[Video Repair]",
            f"Repair timed out after {video_operation_timeout_s(duration_s):.0f}s for {source_path.name}",
            color_name="yellow",
        )
        return video_path
    except Exception as exc:
        try:
            if repairing_path.exists():
                repairing_path.unlink()
        except Exception:
            pass
        if stats is not None:
            stats["repair_duration_s"] = stats.get("repair_duration_s", 0.0) + (time.perf_counter() - repair_start)
        LOGGER.log(
            "[Video Repair]",
            f"Repair failed for {source_path.name}: {exc}",
            color_name="red",
        )
        return video_path

    repair_elapsed_s = time.perf_counter() - repair_start
    if stats is not None:
        stats["repair_duration_s"] = stats.get("repair_duration_s", 0.0) + repair_elapsed_s
        stats["repair_created"] = 1
    CORRUPT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    archived_actual_path = _rename_with_collision_suffix(source_path, archived_path)
    final_path = source_path.with_suffix(repair_suffix)
    if final_path.exists():
        final_path.unlink()
    final_path = _rename_with_collision_suffix(repairing_path, final_path)

    LOGGER.log(
        "[Video Repair]",
        (
            f"Repair succeeded in {repair_elapsed_s:.1f}s | "
            f"archived original as {archived_actual_path.name} | using repaired file {final_path.name}"
        ),
        color_name="green",
    )
    return str(final_path)



