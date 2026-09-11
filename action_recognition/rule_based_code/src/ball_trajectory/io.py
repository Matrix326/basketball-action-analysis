"""Loading of pipeline ball observations and interface-JSON serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class BallObservations:
    """Raw per-frame 3D ball triangulations (post outlier rejection)."""

    frame_indices: np.ndarray          # int64 (N,), sorted strictly increasing
    positions: np.ndarray              # float64 (N, 3), calibrated world metres, z up
    views: np.ndarray                  # int (N,), triangulation view count (>=2); 0 if unknown
    outlier_mask: np.ndarray           # bool (N,), True where rejected as outlier
    measured_positions: np.ndarray     # float64 (N, 3), raw measurement (pre-rejection)


@dataclass
class TrajectoryInput:
    """Everything the post-processor needs from a pipeline run."""

    observations: BallObservations
    fps: float
    skeleton: Optional[dict[int, dict[int, list[list[float]]]]] = None
    #   frame -> player_id -> (17, 3) keypoint positions, NaN where missing
    video_info: Optional[dict[str, Any]] = None


def load_poses_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_frame_int(key: str) -> int:
    return int(float(key))


def build_input(data: dict, fps: Optional[float] = None) -> TrajectoryInput:
    """Extract raw ball measurements from a poses_3d.json dict.

    Prefers the raw ``ball_measurements`` field (persisted by the pipeline
    before the online temporal filter). Falls back to ``balls_3d`` where
    ``balls_3d_predicted`` is falsy — those predicted fills are gaps here.
    """
    measurements = data.get("ball_measurements")
    if measurements:
        frames: list[int] = []
        positions: list[list[float]] = []
        views: list[int] = []
        for key, record in measurements.items():
            frame = _to_frame_int(key)
            pos = record.get("position")
            if pos is None or len(pos) != 3:
                continue
            frames.append(frame)
            positions.append(pos)
            views.append(int(record.get("views", 0)))
        observed_flags = np.ones(len(frames), dtype=bool)
    else:
        balls_3d = data.get("balls_3d", {})
        predicted = data.get("balls_3d_predicted", {})
        frames, positions, views = [], [], []
        for key, pos in balls_3d.items():
            frame = _to_frame_int(key)
            if predicted.get(key, False):
                continue  # online-filter fill, not a real observation
            if pos is None or len(pos) != 3:
                continue
            frames.append(frame)
            positions.append(pos)
            views.append(0)
        observed_flags = np.ones(len(frames), dtype=bool)

    order = np.argsort(frames) if frames else np.array([], dtype=np.int64)
    frame_arr = np.asarray(frames, dtype=np.int64)[order] if frames else np.array([], dtype=np.int64)
    pos_arr = (
        np.asarray(positions, dtype=np.float64)[order]
        if positions
        else np.zeros((0, 3), dtype=np.float64)
    )
    views_arr = np.asarray(views, dtype=np.int64)[order] if views else np.zeros(0, dtype=np.int64)
    obs_flags = observed_flags[order] if len(observed_flags) else np.zeros(0, dtype=bool)

    if fps is None:
        video_info = data.get("video_info", {})
        candidate = next(iter(video_info.values()), {})
        fps = float(candidate.get("fps", 30.0) or 30.0)
    fps = float(fps or 30.0)

    skeleton = None
    poses_3d = data.get("poses_3d")
    if isinstance(poses_3d, dict):
        skeleton = {
            _to_frame_int(key): {
                int(pid): np.asarray(kpts, dtype=np.float64).reshape(17, 3)
                for pid, kpts in players.items()
            }
            for key, players in poses_3d.items()
        }

    return TrajectoryInput(
        observations=BallObservations(
            frame_indices=frame_arr,
            positions=pos_arr,
            views=views_arr,
            outlier_mask=~obs_flags if len(obs_flags) else np.zeros(0, dtype=bool),
            measured_positions=pos_arr.copy(),
        ),
        fps=fps,
        skeleton=skeleton,
        video_info=data.get("video_info"),
    )


def _sanitize(value: Any) -> Any:
    """Recursively convert NaN to None for clean JSON output."""
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and np.isnan(item):
            return None
        return item
    return value


def save_interface(output: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize(output), handle, ensure_ascii=False, indent=1, allow_nan=False)
