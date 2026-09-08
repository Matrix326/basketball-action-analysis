"""Shared temporal helpers for the standalone 3D skeleton renderers."""

from __future__ import annotations

from typing import Any

import numpy as np


BALL_COLOR = (1.0, 0.55, 0.0)


def frame_value(values: dict, frame_number: int, default: Any = None) -> Any:
    return values.get(str(frame_number), values.get(frame_number, default))


def ball_trail_segments(
    balls_3d: dict,
    frame_number: int,
    trail_frames: int,
    max_gap_frames: int,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Build finite ball-trail segments without drawing across long gaps."""
    samples: list[tuple[int, np.ndarray]] = []
    first_frame = max(0, frame_number - max(1, trail_frames) + 1)
    for candidate_frame in range(first_frame, frame_number + 1):
        value = frame_value(balls_3d, candidate_frame)
        if value is None:
            continue
        point = np.asarray(value, dtype=np.float32)
        if point.shape == (3,) and np.isfinite(point).all():
            samples.append((candidate_frame, point))

    segments: list[list[np.ndarray]] = []
    previous_frame: int | None = None
    for sample_frame, point in samples:
        if previous_frame is None or sample_frame - previous_frame > max_gap_frames + 1:
            segments.append([point])
        else:
            segments[-1].append(point)
        previous_frame = sample_frame

    current_value = frame_value(balls_3d, frame_number)
    current = None if current_value is None else np.asarray(current_value, dtype=np.float32)
    if current is not None and (current.shape != (3,) or not np.isfinite(current).all()):
        current = None
    return [np.stack(segment) for segment in segments], current


def draw_ball_trajectory_3d(
    ax,
    balls_3d: dict,
    balls_3d_predicted: dict,
    frame_number: int,
    trail_frames: int,
    max_gap_frames: int,
) -> None:
    segments, current = ball_trail_segments(
        balls_3d, frame_number, trail_frames, max_gap_frames
    )
    for segment in segments:
        if len(segment) >= 2:
            ax.plot(
                segment[:, 0], segment[:, 1], segment[:, 2],
                color=BALL_COLOR, linewidth=3.0, alpha=0.85,
            )
    if current is None:
        return
    predicted = bool(frame_value(balls_3d_predicted, frame_number, False))
    ax.scatter(
        current[0], current[1], current[2],
        c=[BALL_COLOR], s=36.0, marker="o", depthshade=True,
        edgecolors="white" if predicted else BALL_COLOR,
        linewidths=1.2 if predicted else 0.0,
    )
    ax.text(current[0], current[1], current[2], " BALL", color=BALL_COLOR, fontsize=8)
