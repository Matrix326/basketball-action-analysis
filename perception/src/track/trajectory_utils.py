"""Shared trajectory cleanup and metric court rendering helpers."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def repair_isolated_jumps(
    points: Iterable[tuple[float, float]],
    frames: Iterable[int],
    *,
    jump_distance_threshold: float,
    speed_ratio_threshold: float,
    frame_rate: float,
    lookback_frames: int,
) -> tuple[list[tuple[float, float]], list[int]]:
    """Replace a one-frame excursion with time-linear interpolation."""
    values = np.asarray(list(points), dtype=np.float64)
    frame_values = np.asarray(list(frames), dtype=np.int64)
    if len(values) < 3:
        return [tuple(point) for point in values], []

    repaired: set[int] = set()
    for _ in range(2):
        changed = False
        for index in range(1, len(values) - 1):
            first_frame = int(frame_values[index - 1])
            current_frame = int(frame_values[index])
            last_frame = int(frame_values[index + 1])
            if not first_frame < current_frame < last_frame:
                continue

            local_speeds = []
            left = max(0, index - max(2, int(lookback_frames)))
            right = min(len(values) - 1, index + max(2, int(lookback_frames)))
            for segment in range(left, right):
                if segment in (index - 1, index):
                    continue
                gap = max(1, int(frame_values[segment + 1] - frame_values[segment]))
                local_speeds.append(
                    float(np.linalg.norm(values[segment + 1] - values[segment]))
                    * float(frame_rate)
                    / gap
                )
            reference_speed = float(np.median(local_speeds)) if local_speeds else 0.0

            span = last_frame - first_frame
            fraction = (current_frame - first_frame) / span
            expected = values[index - 1] + fraction * (values[index + 1] - values[index - 1])
            residual = float(np.linalg.norm(values[index] - expected))
            adjacent_gap = max(current_frame - first_frame, last_frame - current_frame)
            adaptive_limit = max(
                0.35,
                reference_speed
                * adjacent_gap
                / max(float(frame_rate), 1.0)
                * max(float(speed_ratio_threshold), 1.0),
            )
            if jump_distance_threshold > 0:
                adaptive_limit = min(float(jump_distance_threshold), adaptive_limit)

            bridge_speed = (
                float(np.linalg.norm(values[index + 1] - values[index - 1]))
                * float(frame_rate)
                / span
            )
            plausible_bridge = bridge_speed <= max(8.0, 2.5 * reference_speed)
            if residual > adaptive_limit and plausible_bridge:
                values[index] = expected
                repaired.add(index)
                changed = True
        if not changed:
            break

    return [tuple(map(float, point)) for point in values], sorted(repaired)


def prepare_court_canvas(
    background: np.ndarray | None,
    output_width: int,
    output_height: int,
    court_width_m: float,
    court_length_m: float,
    scale_ratio: float,
) -> tuple[np.ndarray, float, int, int]:
    """Fit a metric court into the output without distorting its aspect ratio."""
    max_scale = min(output_width / court_width_m, output_height / court_length_m)
    scale = min(float(scale_ratio), max_scale) if scale_ratio > 0 else max_scale
    court_width_px = max(1, int(round(court_width_m * scale)))
    court_height_px = max(1, int(round(court_length_m * scale)))
    offset_x = (output_width - court_width_px) // 2
    offset_y = (output_height - court_height_px) // 2

    fill = 32 if background is not None else 200
    canvas = np.full((output_height, output_width, 3), fill, dtype=np.uint8)
    if background is not None:
        resized = cv2.resize(background, (court_width_px, court_height_px))
        canvas[
            offset_y : offset_y + court_height_px,
            offset_x : offset_x + court_width_px,
        ] = resized
    return canvas, scale, offset_x, offset_y


def ground_to_court_pixel(
    x: float,
    y: float,
    *,
    court_width_m: float,
    scale: float,
    offset_x: int,
    offset_y: int,
) -> tuple[int, int]:
    return (
        int(round(offset_x + (court_width_m - float(x)) * scale)),
        int(round(offset_y + float(y) * scale)),
    )
