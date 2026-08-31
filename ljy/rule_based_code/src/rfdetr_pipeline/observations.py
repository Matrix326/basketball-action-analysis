"""Per-view detections, pose observations and mask quality checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from basketball_repro.detection_runtime import BallDetection, PlayerDetection

@dataclass
class PoseObservation:
    view: str
    detection: PlayerDetection
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    ground_position: Optional[np.ndarray]
    appearance: Optional[np.ndarray]
    overlap_ratio: float = 0.0
    mask_only: bool = False
    face_embedding: Optional[np.ndarray] = None
    mask_shape: Optional[np.ndarray] = None

    @property
    def quality(self) -> float:
        pose_quality = float(np.mean(self.keypoints_conf[self.keypoints_conf > 0])) if np.any(self.keypoints_conf > 0) else 0.0
        return 0.5 * float(self.detection.confidence or 0.0) + 0.5 * pose_quality

@dataclass
class FrameExtraction:
    players: list[PlayerDetection]
    observations: list[PoseObservation]
    balls: list[BallDetection]

def _bbox_overlap_ratios(detections: list[PlayerDetection]) -> list[float]:
    """Intersection over the smaller box, used only to guard ReID updates."""
    ratios = np.zeros(len(detections), dtype=np.float32)
    for first_index, first in enumerate(detections):
        first_box = np.asarray(first.bbox_xyxy, dtype=np.float32)
        first_area = max(float((first_box[2] - first_box[0]) * (first_box[3] - first_box[1])), 1e-6)
        for second_index in range(first_index + 1, len(detections)):
            second_box = np.asarray(detections[second_index].bbox_xyxy, dtype=np.float32)
            width = max(0.0, float(min(first_box[2], second_box[2]) - max(first_box[0], second_box[0])))
            height = max(0.0, float(min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])))
            second_area = max(float((second_box[2] - second_box[0]) * (second_box[3] - second_box[1])), 1e-6)
            ratio = width * height / min(first_area, second_area)
            ratios[first_index] = max(ratios[first_index], ratio)
            ratios[second_index] = max(ratios[second_index], ratio)
    return ratios.astype(float).tolist()

def _mask_shape_feature(detection: PlayerDetection) -> Optional[np.ndarray]:
    """Compact, translation/scale-normalized silhouette descriptor."""
    mask = np.asarray(detection.mask, dtype=np.uint8)
    x1, y1, x2, y2 = np.round(detection.bbox_xyxy).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = mask[y1:y2, x1:x2]
    if int(np.count_nonzero(crop)) < 20:
        return None
    silhouette = cv2.resize(crop.astype(np.float32), (16, 32), interpolation=cv2.INTER_AREA)
    vertical = silhouette.mean(axis=1)
    horizontal = silhouette.mean(axis=0)
    aspect = np.array([(x2 - x1) / max(float(y2 - y1), 1.0)], dtype=np.float32)
    feature = np.concatenate((silhouette.reshape(-1), vertical, horizontal, aspect))
    norm = float(np.linalg.norm(feature))
    return feature / norm if norm > 0 else None

def _refined_mask_contours(
    detection: PlayerDetection,
    *,
    kernel_size: int = 5,
    threshold: float = 0.45,
) -> list[np.ndarray]:
    """Smooth an instance-mask edge locally without changing its geometry data."""
    binary = np.asarray(detection.mask, dtype=np.uint8)
    padding = max(3, kernel_size)
    x1, y1, x2, y2 = np.round(detection.bbox_xyxy).astype(int)
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2 = min(binary.shape[1], x2 + padding)
    y2 = min(binary.shape[0], y2 + padding)
    if x2 <= x1 or y2 <= y1:
        return []
    crop = binary[y1:y2, x1:x2] * 255

    kernel_size = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crop = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel)
    crop = cv2.GaussianBlur(crop, (kernel_size, kernel_size), 0)
    smooth = (crop >= int(round(255.0 * threshold))).astype(np.uint8)
    contours, _ = cv2.findContours(smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    minimum_area = max(12.0, float(detection.mask_pixels) * 0.001)
    offset = np.array([[[x1, y1]]], dtype=np.int32)
    return [contour + offset for contour in contours if cv2.contourArea(contour) >= minimum_area]

def _has_human_pose_evidence(
    detection: PlayerDetection,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    *,
    keypoint_threshold: float,
    min_valid_keypoints: int,
    min_mean_confidence: float,
    min_torso_keypoints: int,
    min_vertical_span_ratio: float,
) -> bool:
    """Reject person masks that do not contain a plausible current-frame pose."""
    xy = np.asarray(keypoints_xy, dtype=np.float32)
    confidence = np.asarray(keypoints_conf, dtype=np.float32)
    if xy.shape != (17, 2) or confidence.shape != (17,):
        return False

    box = np.asarray(detection.bbox_xyxy, dtype=np.float32)
    width = max(float(box[2] - box[0]), 1.0)
    height = max(float(box[3] - box[1]), 1.0)
    margin_x = 0.15 * width
    margin_y = 0.15 * height
    finite = np.isfinite(xy).all(axis=1) & np.isfinite(confidence)
    inside = (
        (xy[:, 0] >= box[0] - margin_x)
        & (xy[:, 0] <= box[2] + margin_x)
        & (xy[:, 1] >= box[1] - margin_y)
        & (xy[:, 1] <= box[3] + margin_y)
    )
    valid = finite & inside & (confidence >= float(keypoint_threshold))
    if int(np.count_nonzero(valid)) < int(min_valid_keypoints):
        return False
    if float(np.mean(confidence[valid])) < float(min_mean_confidence):
        return False

    if min_torso_keypoints > 0:
        # Optional strict mode for datasets where torso occlusion is rare.
        torso = np.array([5, 6, 11, 12], dtype=np.int32)
        if int(np.count_nonzero(valid[torso])) < int(min_torso_keypoints):
            return False
        if not np.any(valid[[5, 6]]) or not np.any(valid[[11, 12, 13, 14, 15, 16]]):
            return False

    vertical_span = float(np.ptp(xy[valid, 1]))
    return vertical_span >= max(0.0, float(min_vertical_span_ratio) * height)

@dataclass
class ObservationGroup:
    observations: dict[str, PoseObservation] = field(default_factory=dict)

    @property
    def ground_position(self) -> Optional[np.ndarray]:
        clean = [
            obs.ground_position for obs in self.observations.values()
            if obs.ground_position is not None and obs.overlap_ratio < 0.15
        ]
        values = clean or [obs.ground_position for obs in self.observations.values() if obs.ground_position is not None]
        return np.median(np.stack(values), axis=0) if values else None

    @property
    def appearance(self) -> Optional[np.ndarray]:
        # Overlap crops can contain another jersey and poison an identity EMA.
        # Use a clean view when available; if every view is occluded, position
        # continuity carries the ID and the appearance gallery is left intact.
        values = [
            obs.appearance for obs in self.observations.values()
            if obs.appearance is not None and obs.overlap_ratio < 0.15
        ]
        if not values:
            return None
        value = np.mean(np.stack(values), axis=0)
        norm = np.linalg.norm(value)
        return value / norm if norm > 0 else value

    @property
    def identity_appearance(self) -> Optional[np.ndarray]:
        """Clothing evidence for matching, including overlap masks but never gallery updates."""
        values = [obs.appearance for obs in self.observations.values() if obs.appearance is not None]
        if not values:
            return None
        value = np.mean(np.stack(values), axis=0)
        return value / max(float(np.linalg.norm(value)), 1e-8)

    @property
    def face_embedding(self) -> Optional[np.ndarray]:
        values = [obs.face_embedding for obs in self.observations.values() if obs.face_embedding is not None]
        if not values:
            return None
        value = np.mean(np.stack(values), axis=0)
        return value / max(float(np.linalg.norm(value)), 1e-8)

    @property
    def quality(self) -> float:
        return float(np.mean([obs.quality for obs in self.observations.values()]))

    @property
    def is_occluded(self) -> bool:
        return any(obs.overlap_ratio >= 0.15 for obs in self.observations.values())

