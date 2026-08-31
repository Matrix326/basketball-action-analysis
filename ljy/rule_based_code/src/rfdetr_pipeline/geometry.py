"""Calibrated multi-view projection, matching and triangulation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from config import Config
from basketball_repro.detection_runtime import BallDetection

@dataclass
class CameraParams:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack((self.R, self.t.reshape(3, 1)))

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t.reshape(3)

class MultiViewGeometry:
    def __init__(self, config: Config) -> None:
        with open(config.get("camera.intrinsics_path"), "r", encoding="utf-8") as handle:
            intrinsics = json.load(handle)
        with open(config.get("camera.extrinsics_path"), "r", encoding="utf-8") as handle:
            extrinsics = json.load(handle)
        self.view_to_camera = config.view_to_camera
        self.cameras: dict[str, CameraParams] = {}
        for camera_name, intr in intrinsics.items():
            ext = extrinsics[camera_name]
            self.cameras[camera_name] = CameraParams(
                K=np.asarray(intr.get("K_undistorted", intr.get("K_original")), dtype=np.float64),
                R=np.asarray(ext.get("R_w2c", ext.get("R")), dtype=np.float64),
                t=np.asarray(ext.get("t_w2c", ext.get("t")), dtype=np.float64),
            )
        self.keypoint_threshold = float(config.get("pose.triangulation_keypoint_threshold", 0.20))
        self.max_reprojection_error = float(config.get("pose.max_reprojection_error_px", 30.0))
        self.ball_max_reprojection_error = float(
            config.get("ball.max_reprojection_error_px", 100.0)
        )
        self.court_world_bounds = tuple(
            float(value)
            for value in config.get("camera.court_world_bounds", [-2.0, 17.0, -4.0, 18.0])
        )
        self.ball_max_height = float(config.get("ball.max_height_m", 10.0))

    def camera(self, view: str) -> CameraParams:
        name = self.view_to_camera.get(view, view)
        if name not in self.cameras:
            raise KeyError(f"No calibration for view {view!r} (camera {name!r})")
        return self.cameras[name]

    def ground_point(self, view: str, pixel: Iterable[float]) -> Optional[np.ndarray]:
        cam = self.camera(view)
        uv = np.asarray(list(pixel)[:2], dtype=np.float64)
        ray = cam.R.T @ np.linalg.inv(cam.K) @ np.array([uv[0], uv[1], 1.0])
        if abs(ray[2]) < 1e-9:
            return None
        scale = -cam.center[2] / ray[2]
        if scale <= 0:
            return None
        point = cam.center + scale * ray
        return point.astype(np.float32) if np.isfinite(point).all() else None

    def _triangulate(
        self,
        samples: list[tuple[str, np.ndarray, float]],
        max_reprojection_error: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        if len(samples) < 2:
            return None
        rows: list[np.ndarray] = []
        for view, pixel, confidence in samples:
            p = self.camera(view).P
            weight = math.sqrt(max(float(confidence), 1e-6))
            rows.extend((weight * (pixel[0] * p[2] - p[0]), weight * (pixel[1] * p[2] - p[1])))
        try:
            _, _, vt = np.linalg.svd(np.asarray(rows))
            homogeneous = vt[-1]
            if abs(homogeneous[3]) < 1e-10:
                return None
            point = homogeneous[:3] / homogeneous[3]
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(point).all():
            return None
        errors = []
        for view, pixel, _ in samples:
            projected = self.camera(view).P @ np.r_[point, 1.0]
            if abs(projected[2]) < 1e-9:
                return None
            errors.append(float(np.linalg.norm(projected[:2] / projected[2] - pixel[:2])))
        error_limit = (
            self.max_reprojection_error
            if max_reprojection_error is None
            else float(max_reprojection_error)
        )
        if np.median(errors) > error_limit:
            return None
        return point.astype(np.float32)

    def _valid_ball_point(self, point: Optional[np.ndarray]) -> bool:
        if point is None or not np.isfinite(point).all():
            return False
        xmin, xmax, ymin, ymax = self.court_world_bounds
        return bool(
            xmin <= point[0] <= xmax
            and ymin <= point[1] <= ymax
            and -0.75 <= point[2] <= self.ball_max_height
        )

    def select_ball_detections(
        self,
        candidates_by_view: dict[str, list[BallDetection]],
        predicted_3d: Optional[np.ndarray] = None,
    ) -> dict[str, BallDetection]:
        """Choose the candidate set that agrees across calibrated views."""
        candidates = [
            (view, detection)
            for view, detections in candidates_by_view.items()
            for detection in detections
        ]
        best: dict[str, BallDetection] = {}
        best_score: tuple[int, float, float, float] = (-1, float("-inf"), float("-inf"), float("-inf"))
        predicted = None if predicted_3d is None else np.asarray(predicted_3d, dtype=np.float64)
        for first_index in range(len(candidates)):
            first_view, first_detection = candidates[first_index]
            for second_view, second_detection in candidates[first_index + 1 :]:
                if first_view == second_view:
                    continue
                hypothesis = self._triangulate(
                    [
                        (first_view, np.asarray(first_detection.center_xy), float(first_detection.confidence or 0.1)),
                        (second_view, np.asarray(second_detection.center_xy), float(second_detection.confidence or 0.1)),
                    ],
                    self.ball_max_reprojection_error,
                )
                if not self._valid_ball_point(hypothesis):
                    continue

                selected: dict[str, BallDetection] = {}
                errors: list[float] = []
                for view, detections in candidates_by_view.items():
                    projection = self.camera(view).P @ np.r_[hypothesis, 1.0]
                    if abs(projection[2]) < 1e-9:
                        continue
                    pixel = projection[:2] / projection[2]
                    ranked = sorted(
                        (
                            (float(np.linalg.norm(np.asarray(detection.center_xy) - pixel)), detection)
                            for detection in detections
                        ),
                        key=lambda item: item[0],
                    )
                    if ranked and ranked[0][0] <= self.ball_max_reprojection_error:
                        error, detection = ranked[0]
                        selected[view] = detection
                        errors.append(error)
                if len(selected) < 2:
                    continue

                refined = self._triangulate(
                    [
                        (view, np.asarray(detection.center_xy), float(detection.confidence or 0.1))
                        for view, detection in selected.items()
                    ],
                    self.ball_max_reprojection_error,
                )
                if not self._valid_ball_point(refined):
                    continue
                temporal_distance = 0.0 if predicted is None else float(np.linalg.norm(refined - predicted))
                quality = sum(
                    float(detection.confidence or 0.0)
                    + 0.95 * float(detection.orange_score)
                    + 0.25 * min(1.0, float(detection.size) / 80.0)
                    for detection in selected.values()
                )
                score = (
                    len(selected),
                    -temporal_distance,
                    quality,
                    -float(np.median(errors)),
                )
                if score > best_score:
                    best, best_score = selected, score
        return best

    def triangulate_pose(self, observations: Iterable["PoseObservation"]) -> tuple[np.ndarray, np.ndarray]:
        observations = list(observations)
        pose = np.full((17, 3), np.nan, dtype=np.float32)
        errors = np.full((17,), np.nan, dtype=np.float32)
        for keypoint_index in range(17):
            samples = [
                (obs.view, obs.keypoints_xy[keypoint_index], float(obs.keypoints_conf[keypoint_index]))
                for obs in observations
                if obs.keypoints_conf[keypoint_index] >= self.keypoint_threshold
                and np.isfinite(obs.keypoints_xy[keypoint_index]).all()
            ]
            point = self._triangulate(samples)
            if point is None or point[2] < -0.75 or point[2] > 4.0:
                continue
            pose[keypoint_index] = point
            reprojections = []
            for view, pixel, _ in samples:
                projection = self.camera(view).P @ np.r_[point, 1.0]
                reprojections.append(np.linalg.norm(projection[:2] / projection[2] - pixel))
            errors[keypoint_index] = float(np.mean(reprojections))
        return pose, errors

    def triangulate_ball(self, detections: dict[str, BallDetection]) -> Optional[np.ndarray]:
        samples = [(view, np.asarray(det.center_xy), float(det.confidence or 0.1)) for view, det in detections.items()]
        if len(samples) <= 2:
            point = self._triangulate(samples, self.ball_max_reprojection_error)
            return point if self._valid_ball_point(point) else None

        best_inliers: list[tuple[str, np.ndarray, float]] = []
        best_score = (-1, -1.0, float("-inf"))
        for first in range(len(samples)):
            for second in range(first + 1, len(samples)):
                point = self._triangulate(
                    [samples[first], samples[second]], self.ball_max_reprojection_error
                )
                if point is None:
                    continue
                errors = []
                for view, pixel, _ in samples:
                    projected = self.camera(view).P @ np.r_[point, 1.0]
                    error = float("inf") if abs(projected[2]) < 1e-9 else float(
                        np.linalg.norm(projected[:2] / projected[2] - pixel[:2])
                    )
                    errors.append(error)
                inliers = [
                    sample
                    for sample, error in zip(samples, errors)
                    if error <= self.ball_max_reprojection_error
                ]
                if len(inliers) < 2:
                    continue
                inlier_errors = [
                    error for error in errors if error <= self.ball_max_reprojection_error
                ]
                score = (
                    len(inliers),
                    sum(sample[2] for sample in inliers),
                    -float(np.median(inlier_errors)),
                )
                if score > best_score:
                    best_inliers, best_score = inliers, score
        point = self._triangulate(best_inliers, self.ball_max_reprojection_error)
        return point if self._valid_ball_point(point) else None

