"""Temporal smoothing and short-gap filling for 3D outputs."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, cast

import numpy as np


def refine_pose_bone_lengths(
    poses_3d: dict,
    quality: dict,
    connections: list[tuple[int, int]],
    *,
    strength: float = 0.5,
    iterations: int = 6,
    min_reference_samples: int = 20,
    max_reference_reprojection_error: float = 20.0,
) -> dict[str, float | int]:
    """Stabilize bone lengths learned from reliable multi-view reconstructions."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or not connections:
        return {"reference_bones": 0, "adjusted_poses": 0, "adjusted_keypoints": 0}

    samples: defaultdict[tuple[object, int, int], list[float]] = defaultdict(list)
    for frame, tracks in poses_3d.items():
        frame_quality = quality.get(frame, {})
        for track_id, value in tracks.items():
            record = frame_quality.get(track_id, {})
            reprojection = record.get("mean_reprojection_error_px")
            if (
                record.get("predicted_3d", False)
                or reprojection is None
                or float(reprojection) > max_reference_reprojection_error
            ):
                continue
            pose = np.asarray(value, dtype=np.float64)
            if pose.shape != (17, 3):
                continue
            for first, second in connections:
                if np.isfinite(pose[[first, second]]).all():
                    length = float(np.linalg.norm(pose[first] - pose[second]))
                    if 0.01 <= length <= 1.25:
                        samples[(track_id, first, second)].append(length)

    minimum = max(3, int(min_reference_samples))
    targets = {
        key: float(np.median(values))
        for key, values in samples.items()
        if len(values) >= minimum
    }
    if not targets:
        return {"reference_bones": 0, "adjusted_poses": 0, "adjusted_keypoints": 0}

    displacements: list[float] = []
    adjusted_poses = 0
    adjusted_keypoints = 0
    for tracks in poses_3d.values():
        for track_id, value in tracks.items():
            original = np.asarray(value, dtype=np.float64)
            if original.shape != (17, 3):
                continue
            refined = original.copy()
            anchor_indices = [
                index for index in (11, 12) if np.isfinite(refined[index]).all()
            ]
            anchor = (
                np.mean(refined[anchor_indices], axis=0) if anchor_indices else None
            )
            for _ in range(max(1, int(iterations))):
                for first, second in connections:
                    target = targets.get((track_id, first, second))
                    if (
                        target is None
                        or not np.isfinite(refined[[first, second]]).all()
                    ):
                        continue
                    delta = refined[second] - refined[first]
                    length = float(np.linalg.norm(delta))
                    if length < 1e-6:
                        continue
                    correction = 0.5 * strength * (length - target) * delta / length
                    refined[first] += correction
                    refined[second] -= correction

            if anchor is not None:
                current_anchor = np.mean(refined[anchor_indices], axis=0)
                valid = np.isfinite(refined).all(axis=1)
                refined[valid] += anchor - current_anchor

            valid = np.isfinite(original).all(axis=1) & np.isfinite(refined).all(axis=1)
            delta = refined[valid] - original[valid]
            distances = np.linalg.norm(delta, axis=1)
            too_far = distances > 0.25
            if np.any(too_far):
                delta[too_far] *= (0.25 / distances[too_far])[:, None]
                refined[valid] = original[valid] + delta
                distances = np.linalg.norm(delta, axis=1)
            changed = distances > 1e-6
            if np.any(changed):
                adjusted_poses += 1
                adjusted_keypoints += int(np.count_nonzero(changed))
                displacements.extend(distances[changed].tolist())
                tracks[track_id] = refined.astype(np.float32).tolist()

    displacement_values = np.asarray(displacements, dtype=np.float64)
    return {
        "reference_bones": len(targets),
        "adjusted_poses": adjusted_poses,
        "adjusted_keypoints": adjusted_keypoints,
        "median_adjustment_m": (
            round(float(np.median(displacement_values)), 4)
            if len(displacement_values)
            else 0.0
        ),
        "p90_adjustment_m": (
            round(float(np.percentile(displacement_values, 90)), 4)
            if len(displacement_values)
            else 0.0
        ),
    }


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
            result[both] = (
                self.alpha * result[both] + (1.0 - self.alpha) * previous[both]
            )
            fill = (
                ~current_valid
                & previous_valid
                & (self.missing.get(track_id, 0) < self.max_missing)
            )
            result[fill] = previous[fill]
        self.previous[track_id] = result.copy()
        self.missing[track_id] = 0
        return result

    def fill_missing(
        self, track_id: int, translation_xy: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
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

    def update(
        self, measurement: Optional[np.ndarray]
    ) -> tuple[Optional[np.ndarray], bool]:
        predicted = None if self.position is None else self.position + self.velocity
        valid_measurement = measurement is not None and np.isfinite(measurement).all()
        if valid_measurement and predicted is not None:
            if (
                float(np.linalg.norm(np.asarray(measurement) - predicted))
                > self.max_jump_m
            ):
                valid_measurement = False

        if valid_measurement:
            measurement = np.asarray(measurement, dtype=np.float32)
            if predicted is None:
                filtered = measurement
            else:
                filtered = self.alpha * measurement + (1.0 - self.alpha) * predicted
                measured_velocity = filtered - cast(np.ndarray, self.position)
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
