import os
import time


def add_timing(stats, key, start_time):
    """Accumulate elapsed time for a named timing bucket."""
    stats[key] += time.perf_counter() - start_time


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
