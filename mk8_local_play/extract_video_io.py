import os
import subprocess
import cv2
import time
import threading
import queue
import shutil
import sys
from pathlib import Path
from contextlib import contextmanager

from .console_logging import LOGGER
from .project_paths import PROJECT_ROOT


INPUT_VIDEOS_DIR = PROJECT_ROOT / "Input_Videos"
CORRUPT_VIDEOS_DIR = INPUT_VIDEOS_DIR / "corrupt"
CORRUPT_CHECK_SAMPLE_COUNT = 60
CORRUPT_CHECK_EDGE_SAMPLE_COUNT = 10
CORRUPT_CHECK_EDGE_FRACTION = 0.02
CORRUPT_CHECK_READ_TIMEOUT_S = 5.0
CORRUPT_CHECK_FRAME_SLACK = 5
CORRUPT_CHECK_HEAD_FRAME_TOLERANCE = 2
CORRUPT_CHECK_TAIL_FRAME_TOLERANCE = 2
VIDEO_REPAIR_PROGRESS_INTERVAL_S = 2.0
VIDEO_REPAIR_STALL_TIMEOUT_S = 45.0
VIDEO_REPAIR_PROGRESS_EPSILON_S = 0.25
VIDEO_REMUX_STALL_TIMEOUT_S = 30.0
_STDERR_REDIRECT_LOCK = threading.Lock()


def add_timing(stats, key, start_time):
    """Accumulate elapsed time for a named timing bucket."""
    stats[key] += time.perf_counter() - start_time


def increment_counter(stats, key, amount=1):
    """Accumulate a named integer counter inside the shared stats dict."""
    stats[key] += int(amount)


def seek_to_frame(capture, frame_number, stats):
    """Seek to a frame and record the seek cost."""
    start_time = time.perf_counter()
    result = capture.set(1, frame_number)
    add_timing(stats, "seek_time_s", start_time)
    stats["seek_calls"] += 1
    return result


def position_capture_for_read(capture, frame_number, stats, *, max_forward_grab_frames=0):
    """Move capture to the next frame to read, preferring cheap forward grabs for short jumps."""
    target_frame = int(frame_number)
    try:
        current_next_frame = int(capture.get(1) or 0)
    except Exception:  # pragma: no cover - native backend safety
        current_next_frame = None

    if current_next_frame is not None:
        if current_next_frame == target_frame:
            return True
        if (
            max_forward_grab_frames > 0
            and 0 <= (target_frame - current_next_frame) <= int(max_forward_grab_frames)
        ):
            return advance_frames_by_grab(capture, target_frame - current_next_frame, stats)

    return seek_to_frame(capture, target_frame, stats)


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


@contextmanager
def suppress_native_stderr():
    """Temporarily silence native stderr noise from OpenCV/FFmpeg video backends."""
    with _STDERR_REDIRECT_LOCK:
        stderr_fd = None
        saved_stderr_fd = None
        devnull_file = None
        try:
            stderr_fd = sys.stderr.fileno()
            saved_stderr_fd = os.dup(stderr_fd)
            devnull_file = open(os.devnull, "w", encoding="utf-8", errors="ignore")
            os.dup2(devnull_file.fileno(), stderr_fd)
            yield
        except (OSError, ValueError, AttributeError):
            yield
        finally:
            if saved_stderr_fd is not None and stderr_fd is not None:
                try:
                    os.dup2(saved_stderr_fd, stderr_fd)
                except OSError:
                    pass
            if saved_stderr_fd is not None:
                try:
                    os.close(saved_stderr_fd)
                except OSError:
                    pass
            if devnull_file is not None:
                devnull_file.close()


def log_exported_frame(
    metadata_writer,
    video_path,
    race_number,
    kind,
    requested_frame,
    actual_frame,
    fps,
    frame_to_timecode,
    *,
    video_label=None,
    video_source_path=None,
    score_layout_id="",
    bundle_path="",
    anchor_path="",
):
    """Record requested and actual decoded frames for each exported screenshot."""
    if metadata_writer is None:
        return
    metadata_writer.writerow(
        [
            str(video_label or Path(video_source_path or os.path.basename(video_path)).stem),
            str(video_source_path or os.path.basename(video_path)),
            f"{race_number:03}",
            kind,
            int(requested_frame),
            frame_to_timecode(requested_frame, fps),
            int(actual_frame),
            frame_to_timecode(actual_frame, fps),
            str(score_layout_id or ""),
            str(bundle_path or ""),
            str(anchor_path or ""),
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


def _tail_probe_clamp(total_frames: int, failed_frame: int, actual_frame: int | None) -> int | None:
    """Return a usable frame count when only the nominal tail overstates decodable frames."""
    if total_frames <= 0:
        return None
    tail_start = max(0, int(total_frames) - max(1, int(CORRUPT_CHECK_TAIL_FRAME_TOLERANCE)))
    if int(failed_frame) < tail_start:
        return None
    actual = int(actual_frame) if actual_frame is not None else int(failed_frame) - 1
    usable_total_frames = max(0, min(int(total_frames), actual + 1))
    if usable_total_frames >= int(total_frames):
        return None
    return usable_total_frames or None


def _head_probe_clamp(total_frames: int, failed_frame: int) -> tuple[int, int] | None:
    """Return a readable frame window when only the nominal head is unreadable."""
    if total_frames <= 0:
        return None
    head_end = max(0, min(int(total_frames) - 1, max(1, int(CORRUPT_CHECK_HEAD_FRAME_TOLERANCE)) - 1))
    if int(failed_frame) > head_end:
        return None
    start_frame = min(int(total_frames) - 1, int(failed_frame) + 1)
    if start_frame <= 0:
        return None
    return start_frame, int(total_frames)


def _find_last_readable_frame(capture, low_frame: int, high_frame: int, timeout_s: float) -> int | None:
    """Binary-search the last decodable frame in a bounded interval."""
    lower = int(low_frame)
    upper = int(high_frame)
    last_good = None
    while lower <= upper:
        mid = (lower + upper) // 2
        ret, frame, timed_out, actual_frame = _timed_probe_read(capture, mid, timeout_s)
        if timed_out or not ret or frame is None:
            upper = mid - 1
            continue
        if abs(int(actual_frame) - int(mid)) > CORRUPT_CHECK_FRAME_SLACK:
            upper = mid - 1
            continue
        last_good = int(mid)
        lower = mid + 1
    return last_good


def _find_first_readable_frame(capture, low_frame: int, high_frame: int, timeout_s: float) -> int | None:
    """Binary-search the first decodable frame in a bounded interval."""
    lower = int(low_frame)
    upper = int(high_frame)
    first_good = None
    while lower <= upper:
        mid = (lower + upper) // 2
        ret, frame, timed_out, actual_frame = _timed_probe_read(capture, mid, timeout_s)
        if timed_out or not ret or frame is None:
            lower = mid + 1
            continue
        if abs(int(actual_frame) - int(mid)) > CORRUPT_CHECK_FRAME_SLACK:
            lower = mid + 1
            continue
        first_good = int(mid)
        upper = mid - 1
    return first_good


def read_nominal_frame_count(video_path: str | os.PathLike[str]) -> int:
    """Read the container-reported frame count via OpenCV."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return 0
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()


def sample_video_readability(video_path: str, nominal_total_frames: int, stats: dict | None = None, *, video_identity: object | None = None) -> dict:
    """Cheap corruption preflight using sampled OpenCV seeks/reads across the file."""
    video_name = os.path.basename(video_path)
    display_video_name = LOGGER.video_value(video_name, video_identity) if video_identity is not None else video_name
    LOGGER.log("[Frame Count Scan]", f"sampling readable frames with OpenCV for {display_video_name}", color_name="cyan")
    overall_start = time.perf_counter()
    result = {
        "status": "checked",
        "is_suspect": False,
        "reason": "all sampled frames read successfully",
        "probe_count": 0,
        "failed_frame": None,
        "actual_frame": None,
        "usable_start_frame": 0,
        "usable_total_frames": int(nominal_total_frames) if nominal_total_frames else None,
    }
    with suppress_native_stderr():
        capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        _record_corrupt_check_duration(stats, overall_start)
        return {**result, "status": "inconclusive", "is_suspect": True, "reason": "opencv could not open file"}

    total_frames = nominal_total_frames or int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    probe_frames = _sample_probe_frames(total_frames)
    result["probe_count"] = len(probe_frames)
    last_successful_probe_frame = None
    for frame_number in probe_frames:
        try:
            with suppress_native_stderr():
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
            head_window = _head_probe_clamp(total_frames, frame_number)
            if head_window is not None:
                first_readable_frame = _find_first_readable_frame(
                    capture,
                    head_window[0],
                    max(head_window[0], min(total_frames - 1, head_window[1] - 1)),
                    CORRUPT_CHECK_READ_TIMEOUT_S,
                )
                if first_readable_frame is not None:
                    usable_total_frames = max(0, int(total_frames) - int(first_readable_frame))
                    capture.release()
                    _record_corrupt_check_duration(stats, overall_start)
                    return {
                        **result,
                        "status": "head_clamped",
                        "is_suspect": False,
                        "reason": (
                            f"head probe timed out at frame {frame_number}; "
                            f"first readable frame {first_readable_frame}; using {usable_total_frames} readable frames"
                        ),
                        "failed_frame": int(frame_number),
                        "actual_frame": int(actual_frame),
                        "usable_start_frame": int(first_readable_frame),
                        "usable_total_frames": int(usable_total_frames),
                    }
            search_start = 0 if last_successful_probe_frame is None else int(last_successful_probe_frame)
            last_readable_frame = _find_last_readable_frame(
                capture,
                search_start,
                max(search_start, int(frame_number) - 1),
                CORRUPT_CHECK_READ_TIMEOUT_S,
            )
            usable_total_frames = (
                max(0, int(last_readable_frame) + 1)
                if last_readable_frame is not None and int(frame_number) >= max(0, int(total_frames) - max(1, int(CORRUPT_CHECK_TAIL_FRAME_TOLERANCE)))
                else None
            )
            if usable_total_frames is not None:
                capture.release()
                _record_corrupt_check_duration(stats, overall_start)
                return {
                    **result,
                    "status": "tail_clamped",
                    "is_suspect": False,
                    "reason": (
                        f"tail probe timed out at frame {frame_number}; "
                        f"last readable frame {last_readable_frame}; using {usable_total_frames} readable frames"
                    ),
                    "failed_frame": int(frame_number),
                    "actual_frame": int(actual_frame),
                    "usable_start_frame": 0,
                    "usable_total_frames": int(usable_total_frames),
                }
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
            head_window = _head_probe_clamp(total_frames, frame_number)
            if head_window is not None:
                first_readable_frame = _find_first_readable_frame(
                    capture,
                    head_window[0],
                    max(head_window[0], min(total_frames - 1, head_window[1] - 1)),
                    CORRUPT_CHECK_READ_TIMEOUT_S,
                )
                if first_readable_frame is not None:
                    usable_total_frames = max(0, int(total_frames) - int(first_readable_frame))
                    capture.release()
                    _record_corrupt_check_duration(stats, overall_start)
                    return {
                        **result,
                        "status": "head_clamped",
                        "is_suspect": False,
                        "reason": (
                            f"head probe read failed at frame {frame_number}; "
                            f"first readable frame {first_readable_frame}; using {usable_total_frames} readable frames"
                        ),
                        "failed_frame": int(frame_number),
                        "actual_frame": int(actual_frame),
                        "usable_start_frame": int(first_readable_frame),
                        "usable_total_frames": int(usable_total_frames),
                    }
            search_start = 0 if last_successful_probe_frame is None else int(last_successful_probe_frame)
            last_readable_frame = _find_last_readable_frame(
                capture,
                search_start,
                max(search_start, int(frame_number) - 1),
                CORRUPT_CHECK_READ_TIMEOUT_S,
            )
            usable_total_frames = (
                max(0, int(last_readable_frame) + 1)
                if last_readable_frame is not None and int(frame_number) >= max(0, int(total_frames) - max(1, int(CORRUPT_CHECK_TAIL_FRAME_TOLERANCE)))
                else None
            )
            if usable_total_frames is not None:
                capture.release()
                _record_corrupt_check_duration(stats, overall_start)
                return {
                    **result,
                    "status": "tail_clamped",
                    "is_suspect": False,
                    "reason": (
                        f"tail probe read failed at frame {frame_number}; "
                        f"last readable frame {last_readable_frame}; using {usable_total_frames} readable frames"
                    ),
                    "failed_frame": int(frame_number),
                    "actual_frame": int(actual_frame),
                    "usable_start_frame": 0,
                    "usable_total_frames": int(usable_total_frames),
                }
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
            head_window = _head_probe_clamp(total_frames, frame_number)
            if head_window is not None:
                first_readable_frame = _find_first_readable_frame(
                    capture,
                    head_window[0],
                    max(head_window[0], min(total_frames - 1, head_window[1] - 1)),
                    CORRUPT_CHECK_READ_TIMEOUT_S,
                )
                if first_readable_frame is not None:
                    usable_total_frames = max(0, int(total_frames) - int(first_readable_frame))
                    capture.release()
                    _record_corrupt_check_duration(stats, overall_start)
                    return {
                        **result,
                        "status": "head_clamped",
                        "is_suspect": False,
                        "reason": (
                            f"head probe landed at frame {actual_frame} instead of {frame_number}; "
                            f"first readable frame {first_readable_frame}; using {usable_total_frames} readable frames"
                        ),
                        "failed_frame": int(frame_number),
                        "actual_frame": int(actual_frame),
                        "usable_start_frame": int(first_readable_frame),
                        "usable_total_frames": int(usable_total_frames),
                    }
            search_start = 0 if last_successful_probe_frame is None else int(last_successful_probe_frame)
            last_readable_frame = _find_last_readable_frame(
                capture,
                search_start,
                max(search_start, int(frame_number) - 1),
                CORRUPT_CHECK_READ_TIMEOUT_S,
            )
            usable_total_frames = (
                max(0, int(last_readable_frame) + 1)
                if last_readable_frame is not None and int(frame_number) >= max(0, int(total_frames) - max(1, int(CORRUPT_CHECK_TAIL_FRAME_TOLERANCE)))
                else None
            )
            if usable_total_frames is not None:
                capture.release()
                _record_corrupt_check_duration(stats, overall_start)
                return {
                    **result,
                    "status": "tail_clamped",
                    "is_suspect": False,
                    "reason": (
                        f"tail probe landed at frame {actual_frame} instead of {frame_number}; "
                        f"last readable frame {last_readable_frame}; "
                        f"using {usable_total_frames} readable frames"
                    ),
                    "failed_frame": int(frame_number),
                    "actual_frame": int(actual_frame),
                    "usable_start_frame": 0,
                    "usable_total_frames": int(usable_total_frames),
                }
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
        last_successful_probe_frame = int(frame_number)
    capture.release()
    LOGGER.log("[Frame Count Scan]", f"sample probe passed for {display_video_name}: {len(probe_frames)} frames checked", color_name="green")
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


def _promote_repaired_output(source_path: Path, archived_path: Path, working_path: Path, final_suffix: str) -> tuple[Path, Path]:
    """Archive the original source and promote a validated working file into place."""
    CORRUPT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    archived_actual_path = _rename_with_collision_suffix(source_path, archived_path)
    final_path = source_path.with_suffix(final_suffix)
    if final_path.exists():
        final_path.unlink()
    final_path = _rename_with_collision_suffix(working_path, final_path)
    return archived_actual_path, final_path


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


def _ffmpeg_remux_command(source_path: Path, remuxed_path: Path) -> list[str]:
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
        "0",
        "-c",
        "copy",
        "-y",
    ]
    if remuxed_path.suffix.lower() == ".mp4":
        command.extend(["-movflags", "+faststart"])
    command.append(str(remuxed_path))
    return command


def _format_media_clock(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"


def _update_repair_progress_state(
    encoded_seconds: float,
    previous_progress_seconds: float,
    stalled_elapsed_s: float,
    elapsed_since_last_check_s: float,
    *,
    epsilon_s: float = VIDEO_REPAIR_PROGRESS_EPSILON_S,
) -> tuple[float, float]:
    if encoded_seconds > previous_progress_seconds + float(epsilon_s):
        return 0.0, float(encoded_seconds)
    return float(stalled_elapsed_s + elapsed_since_last_check_s), float(previous_progress_seconds)


def _run_ffmpeg_repair_command(command: list[str], source_name: str, stall_timeout_s: float, source_duration_s: float | None = None) -> None:
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
    previous_progress_seconds = 0.0
    stalled_progress_elapsed_s = 0.0

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

        if process.poll() is None and time.perf_counter() - last_progress_log >= VIDEO_REPAIR_PROGRESS_INTERVAL_S:
            elapsed = time.perf_counter() - start_time
            elapsed_since_last_check_s = max(0.0, time.perf_counter() - last_progress_log)
            stalled_progress_elapsed_s, previous_progress_seconds = _update_repair_progress_state(
                encoded_seconds,
                previous_progress_seconds,
                stalled_progress_elapsed_s,
                elapsed_since_last_check_s,
            )
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
            if stalled_progress_elapsed_s >= max(1.0, float(stall_timeout_s)):
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
                raise subprocess.TimeoutExpired(progress_command, elapsed)

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
    remuxing_path = source_path.with_name(f"{source_path.stem}__remuxing.mp4")
    archived_path = CORRUPT_VIDEOS_DIR / f"corrupt_{source_path.name}"

    try:
        if repairing_path.exists():
            repairing_path.unlink()
    except Exception:
        pass
    try:
        if remuxing_path.exists():
            remuxing_path.unlink()
    except Exception:
        pass

    repair_reason = preflight_result.get("reason") or f"sample probe flagged file with nominal {nominal_total_frames} frames"
    LOGGER.log(
        "[Video Repair]",
        f"Repairing {source_path.name}: {repair_reason}",
        color_name="yellow",
    )

    remux_start = time.perf_counter()
    try:
        _run_ffmpeg_repair_command(
            _ffmpeg_remux_command(source_path, remuxing_path),
            source_path.name,
            VIDEO_REMUX_STALL_TIMEOUT_S,
            source_duration_s=duration_s,
        )
        remux_frame_count = read_nominal_frame_count(remuxing_path)
        remux_preflight = sample_video_readability(
            str(remuxing_path),
            remux_frame_count,
            stats=stats,
            video_identity=source_path.stem,
        )
        if remux_frame_count > 0 and not remux_preflight.get("is_suspect"):
            if stats is not None:
                stats["repair_duration_s"] = stats.get("repair_duration_s", 0.0) + (time.perf_counter() - remux_start)
                stats["repair_created"] = 1
                stats["remux_created"] = 1
                stats["repair_mode_remux"] = 1
            archived_actual_path, final_path = _promote_repaired_output(source_path, archived_path, remuxing_path, ".mp4")
            LOGGER.log(
                "[Video Repair]",
                (
                    f"Remux succeeded | archived original as {archived_actual_path.name} | "
                    f"using repaired file {final_path.name}"
                ),
                color_name="green",
            )
            return str(final_path)
        LOGGER.log(
            "[Video Repair]",
            (
                f"Remux did not clear preflight for {source_path.name}: "
                f"{remux_preflight.get('reason', 'still suspect')}"
            ),
            color_name="yellow",
        )
    except subprocess.TimeoutExpired:
        LOGGER.log(
            "[Video Repair]",
            f"Remux stalled for {source_path.name}; falling back to transcode",
            color_name="yellow",
        )
    except Exception as exc:
        LOGGER.log(
            "[Video Repair]",
            f"Remux failed for {source_path.name}: {exc}; falling back to transcode",
            color_name="yellow",
        )
    finally:
        try:
            if remuxing_path.exists():
                remuxing_path.unlink()
        except Exception:
            pass

    repair_start = time.perf_counter()
    try:
        _run_ffmpeg_repair_command(
            _ffmpeg_repair_command(source_path, repairing_path),
            source_path.name,
            VIDEO_REPAIR_STALL_TIMEOUT_S,
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
            (
                f"Repair stalled after {VIDEO_REPAIR_STALL_TIMEOUT_S:.0f}s "
                f"without ffmpeg progress for {source_path.name}"
            ),
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
        stats["repair_mode_transcode"] = 1
    archived_actual_path, final_path = _promote_repaired_output(source_path, archived_path, repairing_path, repair_suffix)

    LOGGER.log(
        "[Video Repair]",
        (
            f"Repair succeeded in {repair_elapsed_s:.1f}s | "
            f"archived original as {archived_actual_path.name} | using repaired file {final_path.name}"
        ),
        color_name="green",
    )
    return str(final_path)


def compare_preflight_before_after_remux(
    video_path: str,
    nominal_total_frames: int,
    *,
    remuxed_path: str | os.PathLike[str] | None = None,
    keep_remuxed_file: bool = False,
    stats: dict | None = None,
    video_identity: object | None = None,
) -> dict:
    """Compare the sampled OpenCV preflight before and after a light ffmpeg remux."""
    source_path = Path(video_path)
    source_result = sample_video_readability(
        video_path,
        nominal_total_frames,
        stats=stats,
        video_identity=video_identity,
    )
    result = {
        "video_path": str(source_path),
        "source_preflight": source_result,
        "remux_attempted": False,
        "remux_created": False,
        "remux_path": "",
        "remux_preflight": None,
        "improved": False,
        "comparison_note": "source preflight did not require remux",
    }
    if not source_result.get("is_suspect"):
        return result

    if not shutil.which("ffmpeg"):
        result["comparison_note"] = "ffmpeg not found on PATH"
        return result

    remux_path = Path(remuxed_path) if remuxed_path else source_path.with_name(f"{source_path.stem}__remux_probe{source_path.suffix}")
    result["remux_attempted"] = True
    result["remux_path"] = str(remux_path)
    try:
        if remux_path.exists():
            remux_path.unlink()
    except Exception:
        pass

    try:
        _run_ffmpeg_repair_command(
            _ffmpeg_remux_command(source_path, remux_path),
            source_path.name,
            VIDEO_REMUX_STALL_TIMEOUT_S,
            source_duration_s=None,
        )
    except Exception as exc:
        result["comparison_note"] = f"remux failed: {type(exc).__name__}"
        return result

    result["remux_created"] = remux_path.exists()
    if not result["remux_created"]:
        result["comparison_note"] = "remux command completed without creating output"
        return result

    remux_result = sample_video_readability(
        str(remux_path),
        read_nominal_frame_count(remux_path),
        stats=stats,
        video_identity=video_identity,
    )
    result["remux_preflight"] = remux_result
    result["improved"] = bool(source_result.get("is_suspect") and not remux_result.get("is_suspect"))
    result["comparison_note"] = (
        "remux cleared the OpenCV preflight"
        if result["improved"]
        else "remux did not clear the OpenCV preflight"
    )

    if not keep_remuxed_file:
        try:
            remux_path.unlink(missing_ok=True)
            result["remux_created"] = False
        except Exception:
            pass
    return result



