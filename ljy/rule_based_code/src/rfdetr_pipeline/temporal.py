"""Temporal smoothing and short-gap filling for 3D outputs."""

from __future__ import annotations

from typing import Optional

import numpy as np

class Pose3DSmoother:
    def __init__(self, alpha: float, max_missing: int) -> None:
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.max_missing = max_missing
        self.previous: dict[int, np.ndarray] = {}
        self.missing: dict[int, int] = {}

    def update(self, track_id: int, pose: np.ndarray) -> np.ndarray:
        result = pose.copy()
        previous = self.previous.get(track_id)
        if previous is not None:
            current_valid = np.isfinite(result).all(axis=1)
            previous_valid = np.isfinite(previous).all(axis=1)
            both = current_valid & previous_valid
            result[both] = self.alpha * result[both] + (1.0 - self.alpha) * previous[both]
            fill = ~current_valid & previous_valid & (self.missing.get(track_id, 0) < self.max_missing)
            result[fill] = previous[fill]
        self.previous[track_id] = result.copy()
        self.missing[track_id] = 0
        return result

    def fill_missing(self, track_id: int, translation_xy: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        previous = self.previous.get(track_id)
        if previous is None:
            return None
        missing = self.missing.get(track_id, 0) + 1
        self.missing[track_id] = missing
        if missing > self.max_missing:
            return None
        result = previous.copy()
        if translation_xy is not None:
            valid = np.isfinite(result).all(axis=1)
            result[valid, :2] += np.asarray(translation_xy, dtype=np.float32)[:2]
            self.previous[track_id] = result.copy()
        return result

    def last_pose(self, track_id: int) -> Optional[np.ndarray]:
        pose = self.previous.get(track_id)
        return pose.copy() if pose is not None else None

class Ball3DTemporalFilter:
    """Reject isolated 3D ball spikes and bridge short triangulation gaps."""

    def __init__(self, alpha: float, max_missing: int, max_jump_m: float) -> None:
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.max_missing = max(0, int(max_missing))
        self.max_jump_m = float(max_jump_m)
        self.position: Optional[np.ndarray] = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self.missing = 0

    def update(self, measurement: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], bool]:
        predicted = None if self.position is None else self.position + self.velocity
        valid_measurement = measurement is not None and np.isfinite(measurement).all()
        if valid_measurement and predicted is not None:
            if float(np.linalg.norm(np.asarray(measurement) - predicted)) > self.max_jump_m:
                valid_measurement = False

        if valid_measurement:
            measurement = np.asarray(measurement, dtype=np.float32)
            if predicted is None:
                filtered = measurement
            else:
                filtered = self.alpha * measurement + (1.0 - self.alpha) * predicted
                measured_velocity = filtered - self.position
                self.velocity = 0.65 * self.velocity + 0.35 * measured_velocity
            self.position = filtered.astype(np.float32)
            self.missing = 0
            return self.position.copy(), False

        self.missing += 1
        if predicted is None:
            return None, False
        if self.missing > self.max_missing:
            self.position = None
            self.velocity.fill(0.0)
            return None, False
        self.velocity *= 0.92
        self.position = predicted.astype(np.float32)
        return self.position.copy(), True


