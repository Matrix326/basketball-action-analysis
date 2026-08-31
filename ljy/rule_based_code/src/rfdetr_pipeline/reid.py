"""Appearance encoding, cross-view association and global player tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision import models

from config import Config
from basketball_repro.detection_runtime import PlayerDetection
from .observations import ObservationGroup, PoseObservation, _bbox_overlap_ratios
from .pose import _device

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class AppearanceEncoder:
    """Mask-aware deep + colour embedding for temporal and cross-view ReID."""

    def __init__(self, config: Config) -> None:
        self.enabled = bool(config.get("reid.use_appearance_embeddings", True))
        self.use_deep_embeddings = bool(config.get("reid.use_deep_appearance_embeddings", True))
        self.device = torch.device(_device(config.get("reid.appearance_device", "auto")))
        self.input_size = max(96, int(config.get("reid.appearance_input_size", 160)))
        self.use_fp16 = bool(config.get("reid.appearance_fp16", True)) and self.device.type == "cuda"
        self.deep_refresh_interval = max(
            1,
            int(config.get("reid.appearance_deep_refresh_interval_packets", 1)),
        )
        self.packet_index = 0
        self.model: Optional[torch.nn.Module] = None
        if not self.enabled or not self.use_deep_embeddings:
            return

        checkpoint = Path(
            config.get(
                "reid.appearance_checkpoint",
                PROJECT_ROOT / "models/reid/mobilenet_v2-b0353104.pth",
            )
        )
        if not checkpoint.exists():
            print(f"[warn] appearance checkpoint missing; using HSV features only: {checkpoint}")
            return
        network = models.mobilenet_v2(weights=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        dtype = torch.float16 if self.use_fp16 else torch.float32
        self.model = network.features.eval().to(self.device, dtype=dtype)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=dtype).view(1, 3, 1, 1)

    @staticmethod
    def _colour_feature(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use full-body and clothing-only HSV histograms without background."""
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask_u8 = (mask.astype(np.uint8) * 255)
        clothing_mask = mask_u8.copy()
        clothing_mask[: int(round(clothing_mask.shape[0] * 0.15))] = 0
        clothing_mask[int(round(clothing_mask.shape[0] * 0.68)) :] = 0
        features: list[np.ndarray] = []
        for current_mask in (mask_u8, clothing_mask):
            if not np.any(current_mask):
                histogram = np.zeros((16, 4), dtype=np.float32)
            else:
                histogram = cv2.calcHist([hsv], [0, 1], current_mask, [16, 4], [0, 180, 0, 256])
            flattened = histogram.reshape(-1).astype(np.float32)
            flattened /= max(float(np.linalg.norm(flattened)), 1e-8)
            features.append(flattened)
        result = np.concatenate(features)
        return result / max(float(np.linalg.norm(result)), 1e-8)

    def encode(self, frame_bgr: np.ndarray, detections: List[PlayerDetection]) -> List[Optional[np.ndarray]]:
        return self.encode_many([(frame_bgr, detections)])[0]

    def encode_many(
        self,
        items: list[tuple[np.ndarray, List[PlayerDetection]]],
    ) -> list[List[Optional[np.ndarray]]]:
        """Batch appearance crops from all views/timestamps in a detector packet."""
        counts = [len(detections) for _, detections in items]
        if not self.enabled:
            return [[None] * count for count in counts]
        run_deep = self.model is not None and self.packet_index % self.deep_refresh_interval == 0
        self.packet_index += 1
        resized_crops: list[np.ndarray] = []
        colour_features: list[np.ndarray] = []
        for frame_bgr, detections in items:
            for det in detections:
                x1, y1, x2, y2 = np.round(det.bbox_xyxy).astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    colour_features.append(np.zeros(128, dtype=np.float32))
                    if run_deep:
                        resized_crops.append(np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8))
                    continue
                crop = frame_bgr[y1:y2, x1:x2].copy()
                mask = det.mask[y1:y2, x1:x2].astype(bool)
                colour_features.append(self._colour_feature(crop, mask))
                clothing_mask = mask.copy()
                clothing_mask[: int(round(clothing_mask.shape[0] * 0.12))] = False
                clothing_mask[int(round(clothing_mask.shape[0] * 0.75)) :] = False
                crop[~clothing_mask] = 114
                if run_deep:
                    resized = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
                    resized_crops.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        if self.model is None:
            flat_output: list[Optional[np.ndarray]] = list(colour_features)
        elif not run_deep:
            zeros = np.zeros(1280, dtype=np.float32)
            flat_output = [np.concatenate((zeros, colour)).astype(np.float32) for colour in colour_features]
        elif not resized_crops:
            flat_output = []
        else:
            with torch.inference_mode():
                dtype = torch.float16 if self.use_fp16 else torch.float32
                batch = torch.from_numpy(np.stack(resized_crops)).to(self.device, dtype=dtype)
                batch = batch.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
                batch = (batch / 255.0 - self.mean) / self.std
                features = self.model(batch).mean(dim=(-2, -1))
                features = torch.nn.functional.normalize(features, dim=1)
            flat_output = []
            for deep, colour in zip(features.detach().cpu().numpy().astype(np.float32), colour_features):
                combined = np.concatenate((deep, colour)).astype(np.float32)
                flat_output.append(combined / max(float(np.linalg.norm(combined)), 1e-8))

        grouped: list[List[Optional[np.ndarray]]] = []
        cursor = 0
        for count in counts:
            grouped.append(flat_output[cursor : cursor + count])
            cursor += count
        return grouped

class FaceEncoder:
    """Local InsightFace detector + ArcFace embedding, assigned by RTMPose head points."""

    def __init__(self, config: Config) -> None:
        self.enabled = bool(config.get("reid.use_face_embeddings", True))
        self.app: Any = None
        self.min_score = float(config.get("reid.face_min_score", 0.55))
        self.min_size = float(config.get("reid.face_min_size_px", 22.0))
        self.refresh_interval = max(1, int(config.get("reid.face_refresh_interval_packets", 3)))
        self.overlap_refresh_interval = max(
            1,
            int(config.get("reid.face_overlap_refresh_interval_packets", self.refresh_interval)),
        )
        self.packet_index = 0
        if not self.enabled:
            return
        root = Path(config.get("reid.insightface_root", PROJECT_ROOT / "models/insightface"))
        name = str(config.get("model.insightface_name", "buffalo_l"))
        model_dir = root / "models" / name
        required = (model_dir / "det_10g.onnx", model_dir / "w600k_r50.onnx")
        if not all(path.exists() for path in required):
            raise FileNotFoundError(f"Local InsightFace models are incomplete: {model_dir}")
        from insightface.app import FaceAnalysis

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name=name,
            root=str(root),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        det_size = int(config.get("reid.face_det_size", 1024))
        self.app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(det_size, det_size))

    def encode_many(
        self,
        items: list[tuple[np.ndarray, List[PlayerDetection], list[tuple[np.ndarray, np.ndarray]]]],
    ) -> list[List[Optional[np.ndarray]]]:
        outputs: list[List[Optional[np.ndarray]]] = []
        periodic_refresh = self.packet_index % self.refresh_interval == 0
        overlap_refresh = self.packet_index % self.overlap_refresh_interval == 0
        self.packet_index += 1
        for frame, detections, poses in items:
            assigned: List[Optional[np.ndarray]] = [None] * len(detections)
            if self.app is None or not detections:
                outputs.append(assigned)
                continue
            # Refresh the face gallery periodically and more often around
            # overlaps, while rate-limiting persistent overlap sequences.
            has_overlap = any(value >= 0.15 for value in _bbox_overlap_ratios(detections))
            if not periodic_refresh and not (has_overlap and overlap_refresh):
                outputs.append(assigned)
                continue
            faces = [
                face
                for face in self.app.get(frame)
                if float(face.det_score) >= self.min_score
                and min(float(face.bbox[2] - face.bbox[0]), float(face.bbox[3] - face.bbox[1])) >= self.min_size
            ]
            if not faces:
                outputs.append(assigned)
                continue
            costs = np.full((len(detections), len(faces)), 1e6, dtype=np.float64)
            for det_index, (detection, (xy, scores)) in enumerate(zip(detections, poses)):
                bbox = np.asarray(detection.bbox_xyxy, dtype=np.float32)
                head_points = xy[:5][(scores[:5] >= 0.20) & np.isfinite(xy[:5]).all(axis=1)]
                anchor = np.mean(head_points, axis=0) if len(head_points) else np.array(
                    [(bbox[0] + bbox[2]) * 0.5, bbox[1] + (bbox[3] - bbox[1]) * 0.16], dtype=np.float32
                )
                height = max(float(bbox[3] - bbox[1]), 1.0)
                for face_index, face in enumerate(faces):
                    face_bbox = np.asarray(face.bbox, dtype=np.float32)
                    center = (face_bbox[:2] + face_bbox[2:]) * 0.5
                    if not (bbox[0] <= center[0] <= bbox[2] and bbox[1] <= center[1] <= bbox[1] + 0.55 * height):
                        continue
                    distance = float(np.linalg.norm(center - anchor)) / height
                    if distance <= 0.28:
                        costs[det_index, face_index] = distance
            if costs.size:
                rows, columns = linear_sum_assignment(costs)
                for row, column in zip(rows, columns):
                    if costs[row, column] >= 1e5:
                        continue
                    embedding = np.asarray(faces[column].embedding, dtype=np.float32)
                    assigned[int(row)] = embedding / max(float(np.linalg.norm(embedding)), 1e-8)
            outputs.append(assigned)
        return outputs

def _cosine_distance(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> float:
    if first is None or second is None:
        return 0.5
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(1.0 - np.dot(first, second) / denom) if denom > 0 else 0.5

def _has_deep_appearance(value: Optional[np.ndarray]) -> bool:
    """Combined MobileNet+HSV vectors reserve their final 128 values for colour."""
    return value is not None and value.size > 128 and float(np.linalg.norm(value[:-128])) > 0.1

def _appearance_distance(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> float:
    if first is None or second is None:
        return 0.5
    if first.size == second.size and first.size > 128:
        if not (_has_deep_appearance(first) and _has_deep_appearance(second)):
            return _cosine_distance(first[-128:], second[-128:])
    return _cosine_distance(first, second)

def _optional_cosine_distance(
    first: Optional[np.ndarray], second: Optional[np.ndarray]
) -> Optional[float]:
    if first is None or second is None:
        return None
    return _cosine_distance(first, second)

def _face_evidence_cost(distance: Optional[float], weight: float) -> float:
    """Reward a strong ArcFace match; do not punish a low-resolution uncertain face."""
    if distance is None:
        return 0.0
    if distance <= 0.55:
        return -float(weight) * (0.55 - distance)
    if distance >= 0.75:
        return float(weight) * (distance - 0.75)
    return 0.0

class CrossViewFuser:
    def __init__(
        self,
        max_ground_distance: float = 2.0,
        appearance_weight: float = 0.8,
        face_weight: float = 1.2,
        mask_weight: float = 0.2,
    ) -> None:
        self.max_ground_distance = max_ground_distance
        self.appearance_weight = appearance_weight
        self.face_weight = face_weight
        self.mask_weight = mask_weight

    def fuse(self, observations_by_view: dict[str, list[PoseObservation]]) -> list[ObservationGroup]:
        views = sorted(observations_by_view, key=lambda view: len(observations_by_view[view]), reverse=True)
        if not views:
            return []
        groups = [ObservationGroup({views[0]: obs}) for obs in observations_by_view[views[0]]]
        for view in views[1:]:
            observations = observations_by_view[view]
            if not groups:
                groups = [ObservationGroup({view: obs}) for obs in observations]
                continue
            costs = np.full((len(groups), len(observations)), 1e6, dtype=np.float64)
            for group_index, group in enumerate(groups):
                group_ground = group.ground_position
                for obs_index, obs in enumerate(observations):
                    if group_ground is None or obs.ground_position is None:
                        continue
                    distance = float(np.linalg.norm(group_ground[:2] - obs.ground_position[:2]))
                    if distance <= self.max_ground_distance:
                        face_distance = _optional_cosine_distance(group.face_embedding, obs.face_embedding)
                        if face_distance is not None and face_distance > 0.85:
                            continue
                        mask_values = [
                            item.mask_shape for item in group.observations.values() if item.mask_shape is not None
                        ]
                        mask_reference = np.mean(np.stack(mask_values), axis=0) if mask_values else None
                        mask_distance = _optional_cosine_distance(mask_reference, obs.mask_shape)
                        costs[group_index, obs_index] = (
                            distance / max(self.max_ground_distance, 1e-6)
                            + self.appearance_weight * _appearance_distance(
                                group.identity_appearance, obs.appearance
                            )
                            + _face_evidence_cost(face_distance, self.face_weight)
                            + self.mask_weight * (mask_distance if mask_distance is not None else 0.0)
                        )
            used: set[int] = set()
            if costs.size:
                rows, columns = linear_sum_assignment(costs)
                for row, column in zip(rows, columns):
                    if costs[row, column] < 1e5:
                        groups[row].observations[view] = observations[column]
                        used.add(int(column))
            groups.extend(ObservationGroup({view: obs}) for index, obs in enumerate(observations) if index not in used)
        return groups

@dataclass
class GlobalPlayerTrack:
    track_id: int
    ground_position: np.ndarray
    previous_position: np.ndarray
    appearance: Optional[np.ndarray]
    face: Optional[np.ndarray] = None
    lost: int = 0
    hits: int = 1
    appearance_history: list[np.ndarray] = field(default_factory=list)
    face_history: list[np.ndarray] = field(default_factory=list)
    view_mask_history: dict[str, list[np.ndarray]] = field(default_factory=dict)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    view_centers: dict[str, np.ndarray] = field(default_factory=dict)
    view_velocities: dict[str, np.ndarray] = field(default_factory=dict)
    view_scales: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.appearance is not None and not self.appearance_history:
            self.appearance_history.append(self.appearance.copy())
        if self.face is not None and not self.face_history:
            self.face_history.append(self.face.copy())

    @property
    def reference_appearance(self) -> Optional[np.ndarray]:
        if not self.appearance_history:
            return self.appearance
        reference = np.mean(np.stack(self.appearance_history), axis=0)
        return reference / max(float(np.linalg.norm(reference)), 1e-8)

    @property
    def reference_face(self) -> Optional[np.ndarray]:
        if not self.face_history:
            return self.face
        reference = np.mean(np.stack(self.face_history), axis=0)
        return reference / max(float(np.linalg.norm(reference)), 1e-8)

    def reference_mask(self, view: str) -> Optional[np.ndarray]:
        history = self.view_mask_history.get(view, [])
        if not history:
            return None
        reference = np.mean(np.stack(history), axis=0)
        return reference / max(float(np.linalg.norm(reference)), 1e-8)

    def mask_distance(self, group: ObservationGroup) -> Optional[float]:
        distances = [
            _cosine_distance(self.reference_mask(view), observation.mask_shape)
            for view, observation in group.observations.items()
            if self.reference_mask(view) is not None and observation.mask_shape is not None
        ]
        return float(np.median(distances)) if distances else None

    def predicted_position(self) -> np.ndarray:
        steps = min(self.lost + 1, 3)
        return self.ground_position + self.velocity * steps

    def image_motion_cost(self, group: ObservationGroup) -> Optional[float]:
        costs = []
        steps = min(self.lost + 1, 3)
        for view, observation in group.observations.items():
            if view not in self.view_centers or not hasattr(observation.detection, "center_xy"):
                continue
            predicted = self.view_centers[view] + self.view_velocities.get(view, np.zeros(2)) * steps
            center = np.asarray(observation.detection.center_xy, dtype=np.float32)
            costs.append(float(np.linalg.norm(predicted - center)) / max(self.view_scales.get(view, 1.0), 1.0))
        return float(np.median(costs)) if costs else None

    def _update_view_motion(self, group: ObservationGroup) -> None:
        for view, observation in group.observations.items():
            if not hasattr(observation.detection, "center_xy") or not hasattr(observation.detection, "bbox_xyxy"):
                continue
            center = np.asarray(observation.detection.center_xy, dtype=np.float32)
            bbox = np.asarray(observation.detection.bbox_xyxy, dtype=np.float32)
            scale = max(float(bbox[3] - bbox[1]), 1.0)
            previous = self.view_centers.get(view)
            if previous is not None:
                measured = center - previous
                old_velocity = self.view_velocities.get(view, np.zeros(2, dtype=np.float32))
                self.view_velocities[view] = (0.7 * old_velocity + 0.3 * measured).astype(np.float32)
            else:
                self.view_velocities[view] = np.zeros(2, dtype=np.float32)
            self.view_centers[view] = center
            self.view_scales[view] = scale

    def update(self, group: ObservationGroup, *, position_alpha: float) -> None:
        position = group.ground_position
        if position is not None:
            old_position = self.ground_position.copy()
            updated_position = (
                float(position_alpha) * position + (1.0 - float(position_alpha)) * self.ground_position
            ).astype(np.float32)
            measured_velocity = updated_position - old_position
            if float(np.linalg.norm(measured_velocity[:2])) <= 0.75:
                self.velocity = (0.7 * self.velocity + 0.3 * measured_velocity).astype(np.float32)
            else:
                self.velocity *= 0.5
            self.previous_position = old_position
            self.ground_position = updated_position
        appearance = group.appearance
        if appearance is not None and (appearance.size <= 128 or _has_deep_appearance(appearance)):
            reference = self.reference_appearance
            if reference is None or _cosine_distance(reference, appearance) <= 0.40:
                self.appearance_history.append(appearance.copy())
                self.appearance_history = self.appearance_history[-100:]
                self.appearance = self.reference_appearance
        face = group.face_embedding
        if face is not None:
            reference_face = self.reference_face
            if reference_face is None or _cosine_distance(reference_face, face) <= 0.50:
                self.face_history.append(face.copy())
                self.face_history = self.face_history[-50:]
                self.face = self.reference_face
        # Segmentation becomes unreliable where boxes overlap. Never write an
        # occluded silhouette into the persistent mask gallery.
        for view, observation in group.observations.items():
            if observation.mask_shape is None or observation.overlap_ratio >= 0.15:
                continue
            reference_mask = self.reference_mask(view)
            if reference_mask is None or _cosine_distance(reference_mask, observation.mask_shape) <= 0.45:
                history = self.view_mask_history.setdefault(view, [])
                history.append(observation.mask_shape.copy())
                self.view_mask_history[view] = history[-30:]
        self._update_view_motion(group)
        self.lost = 0
        self.hits += 1

class GlobalPlayerTracker:
    def __init__(
        self,
        num_players: int,
        max_distance: float,
        max_lost: int,
        position_alpha: float = 0.55,
        appearance_weight: float = 0.9,
        image_motion_weight: float = 0.35,
        face_weight: float = 1.4,
        mask_weight: float = 0.55,
        new_track_min_views: int = 1,
        new_track_min_confidence: float = 0.0,
        max_assignment_cost: float = 3.0,
    ) -> None:
        self.num_players = num_players
        self.max_distance = max_distance
        self.max_lost = max_lost
        self.position_alpha = float(max(0.0, min(1.0, position_alpha)))
        self.appearance_weight = appearance_weight
        self.image_motion_weight = image_motion_weight
        self.face_weight = face_weight
        self.mask_weight = mask_weight
        self.new_track_min_views = max(1, int(new_track_min_views))
        self.new_track_min_confidence = float(new_track_min_confidence)
        self.max_assignment_cost = float(max_assignment_cost)
        self.next_id = 1
        self.tracks: dict[int, GlobalPlayerTrack] = {}

    def update(self, groups: list[ObservationGroup]) -> list[tuple[GlobalPlayerTrack, ObservationGroup]]:
        # IDs represent a fixed roster. Keep dormant tracks available for a
        # conservative appearance-gated revival instead of permanently losing
        # an ID after a long bench/occlusion interval.
        candidates = list(self.tracks.values())
        costs = np.full((len(candidates), len(groups)), 1e6, dtype=np.float64)
        for track_index, track in enumerate(candidates):
            for group_index, group in enumerate(groups):
                position = group.ground_position
                if position is None:
                    continue
                distance = float(np.linalg.norm(track.predicted_position()[:2] - position[:2]))
                appearance_distance = None
                if track.reference_appearance is not None and group.identity_appearance is not None:
                    appearance_distance = _appearance_distance(
                        track.reference_appearance, group.identity_appearance
                    )
                face_distance = _optional_cosine_distance(track.reference_face, group.face_embedding)
                mask_distance = track.mask_distance(group)
                image_motion = track.image_motion_cost(group)
                image_cost = self.image_motion_weight * image_motion if image_motion is not None else 0.0
                # Face is the strongest identity cue. A definite ArcFace
                # contradiction is never allowed to be overruled by position.
                if group.is_occluded and face_distance is not None and face_distance > 0.70:
                    continue
                evidence = [
                    value
                    for value in (
                        None if face_distance is None else face_distance <= 0.55,
                        None if appearance_distance is None else appearance_distance <= 0.42,
                        None if mask_distance is None else mask_distance <= 0.42,
                    )
                    if value is not None
                ]
                positives = sum(evidence)
                # When silhouettes overlap, two independent identity cues are
                # required. If only one exists, accept only a very tight
                # per-camera continuation; otherwise keep the detection
                # temporarily unassigned instead of risking an ID switch.
                if group.is_occluded:
                    if len(evidence) >= 2 and positives < 2:
                        continue
                    if len(evidence) < 2 and not (
                        positives == 1
                        and track.lost == 0
                        and image_motion is not None
                        and image_motion <= 0.55
                    ):
                        continue
                elif (
                    track.lost == 0
                    and image_motion is not None
                    and image_motion > 1.20
                    and appearance_distance is not None
                    and appearance_distance > 0.35
                ):
                    continue
                if track.lost <= self.max_lost:
                    gate = self.max_distance * (1.0 + min(track.lost, 15) / 15.0)
                    if distance <= gate:
                        costs[track_index, group_index] = (
                            distance / max(self.max_distance, 1e-6)
                            + self.appearance_weight * (
                                appearance_distance if appearance_distance is not None else 0.25
                            )
                            + _face_evidence_cost(face_distance, self.face_weight)
                            + self.mask_weight * (mask_distance if mask_distance is not None else 0.0)
                            + image_cost
                        )
                else:
                    revival_gate = self.max_distance * 4.0
                    revival_identity = (
                        (face_distance is not None and face_distance <= 0.50)
                        or (appearance_distance is not None and appearance_distance <= 0.35)
                    )
                    if distance <= revival_gate and revival_identity:
                        costs[track_index, group_index] = (
                            0.75
                            + 0.25 * distance / max(revival_gate, 1e-6)
                            + 1.5 * (appearance_distance if appearance_distance is not None else 0.25)
                            + _face_evidence_cost(face_distance, self.face_weight)
                        )

        matched_tracks: set[int] = set()
        matched_groups: set[int] = set()
        matches: list[tuple[GlobalPlayerTrack, ObservationGroup]] = []
        if costs.size:
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if costs[row, column] > self.max_assignment_cost:
                    continue
                track = candidates[row]
                track.update(groups[column], position_alpha=self.position_alpha)
                matches.append((track, groups[column]))
                matched_tracks.add(track.track_id)
                matched_groups.add(int(column))

        for track in self.tracks.values():
            if track.track_id not in matched_tracks:
                track.lost += 1

        unmatched = [
            (index, group) for index, group in enumerate(groups)
            if index not in matched_groups and group.ground_position is not None
        ]
        unmatched.sort(key=lambda item: (-len(item[1].observations), -item[1].quality, item[1].ground_position[0]))
        for index, group in unmatched:
            if self.next_id > self.num_players:
                break
            if len(group.observations) < self.new_track_min_views or group.quality < self.new_track_min_confidence:
                continue
            position = group.ground_position
            track = GlobalPlayerTrack(
                track_id=self.next_id,
                ground_position=position.copy(),
                previous_position=position.copy(),
                appearance=group.appearance,
                face=group.face_embedding,
            )
            for view, observation in group.observations.items():
                if observation.mask_shape is not None and observation.overlap_ratio < 0.15:
                    track.view_mask_history[view] = [observation.mask_shape.copy()]
            track._update_view_motion(group)
            self.tracks[track.track_id] = track
            self.next_id += 1
            matches.append((track, group))
        return sorted(matches, key=lambda pair: pair[0].track_id)

