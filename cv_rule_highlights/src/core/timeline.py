"""Single conversion between synchronized and source-video frame numbers."""


def source_frame(sync_frame: int, frame_zero: int) -> int:
    return int(sync_frame) - int(frame_zero)


def checked_source_frame(sync_frame: int, frame_zero: int, total_frames: int) -> int:
    result = source_frame(sync_frame, frame_zero)
    if not 0 <= result < total_frames:
        raise ValueError(
            f"sync frame {sync_frame} maps to source frame {result}, "
            f"outside [0, {total_frames})"
        )
    return result


def source_interval(start: int, end: int, frame_zero: int) -> tuple[int, int]:
    if end <= start:
        raise ValueError("Frame intervals must be nonempty and end-exclusive")
    return source_frame(start, frame_zero), source_frame(end, frame_zero)
