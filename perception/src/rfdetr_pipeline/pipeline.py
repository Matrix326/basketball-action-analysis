"""End-to-end four-view RF-DETR, RTMPose and 3D reconstruction orchestration."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any, Iterable, Optional, cast

import cv2
import numpy as np
from tqdm import tqdm

from basketball_repro.detection_runtime import (
    AsyncVideoWriter,
    BallDetection,
    BallTracker,
    PlayerDetection,
    extract_detections,
    make_roi_mask,
    parse_points,
)
from config import Config

from .detector import RFDetrSegmenter, _temporal_batch_length
from .geometry import MultiViewGeometry
from .observations import (
    FrameExtraction,
    PoseObservation,
    _bbox_overlap_ratios,
    _has_human_pose_evidence,
    _mask_shape_feature,
    _refined_mask_contours,
)
from .pose import RTMPoseEstimator
from .reid import AppearanceEncoder, CrossViewFuser, FaceEncoder, GlobalPlayerTracker
from .temporal import Ball3DTemporalFilter, Pose3DSmoother, refine_pose_bone_lengths

COCO17_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _draw_skeleton(
    frame: np.ndarray,
    observation: PoseObservation,
    color: tuple[int, int, int],
    connections: list[tuple[int, int]],
    threshold: float,
) -> None:
    xy, scores = observation.keypoints_xy, observation.keypoints_conf
    for start, end in connections:
        if scores[start] >= threshold and scores[end] >= threshold:
            cv2.line(
                frame,
                tuple(np.round(xy[start]).astype(int)),
                tuple(np.round(xy[end]).astype(int)),
                color,
                3,
                cv2.LINE_AA,
            )
    for index in range(17):
        if scores[index] >= threshold:
            cv2.circle(
                frame,
                tuple(np.round(xy[index]).astype(int)),
                4,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )


class RFDetrPoseMultiViewPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.geometry = MultiViewGeometry(config)
        print("[load] RF-DETR-Seg 2XL")
        self.segmenter = RFDetrSegmenter(config)
        print("[load] RTMPose COCO-17")
        self.pose_estimator = RTMPoseEstimator(config)
        appearance_mode = (
            "deep + colour"
            if config.get("reid.use_deep_appearance_embeddings", True)
            else "colour-only"
        )
        print(f"[load] mask-aware appearance ReID ({appearance_mode})")
        self.appearance_encoder = AppearanceEncoder(config)
        face_mode = (
            "enabled" if config.get("reid.use_face_embeddings", True) else "disabled"
        )
        print(f"[load] local InsightFace ArcFace ReID ({face_mode})")
        self.face_encoder = FaceEncoder(config)
        self.fuser = CrossViewFuser(
            float(config.get("reid.cross_view_max_ground_distance", 2.0)),
            float(config.get("reid.cross_view_appearance_weight", 0.8)),
            float(config.get("reid.cross_view_face_weight", 1.2)),
            float(config.get("reid.cross_view_mask_weight", 0.2)),
        )
        self.tracker = GlobalPlayerTracker(
            int(config.get("reid.num_players", 6)),
            float(config.get("reid.temporal_max_ground_distance", 2.5)),
            int(config.get("reid.max_missing_frames", 45)),
            float(config.get("reid.ground_smoothing_alpha", 0.55)),
            float(config.get("reid.temporal_appearance_weight", 0.9)),
            float(config.get("reid.image_motion_weight", 0.35)),
            float(config.get("reid.temporal_face_weight", 1.4)),
            float(config.get("reid.temporal_mask_weight", 0.55)),
            int(config.get("reid.new_track_min_views", 2)),
            float(config.get("reid.new_track_min_confidence", 0.50)),
            float(config.get("reid.max_assignment_cost", 3.0)),
        )
        self.smoother = Pose3DSmoother(
            float(config.get("pose.smoothing_alpha", 0.65)),
            int(config.get("pose.max_keypoint_fill_frames", 3)),
        )
        self.max_player_candidates = max(
            int(config.get("reid.num_players", 6)),
            int(config.get("rfdetr.max_player_candidates_per_view", 8)),
        )
        self.refine_contours = bool(
            config.get("visualization.refine_mask_contours", True)
        )
        self.contour_smoothing_kernel = int(
            config.get("visualization.mask_smoothing_kernel", 5)
        )
        self.contour_smoothing_threshold = float(
            config.get("visualization.mask_smoothing_threshold", 0.45)
        )
        self.draw_unassigned_contours = bool(
            config.get("visualization.draw_unassigned_contours", False)
        )
        self.ball_exclusion_zones = config.get("ball.exclusion_zones", {})
        self.stage_seconds: defaultdict[str, float] = defaultdict(float)
        self.connections = config.skeleton_connections
        self.player_colors_bgr = config.player_colors_bgr
        if not self.player_colors_bgr:
            raise ValueError("player_colors must contain at least one RGB color")
        self.keypoint_threshold = float(config.get("pose.keypoint_threshold", 0.20))

    def _roi(self, view: str, width: int, height: int) -> Optional[np.ndarray]:
        value = self.config.get(f"rfdetr.roi_polygons.{view}", "")
        return parse_points(value, width, height, min_points=3) if value else None

    def _inside_court(self, ground: Optional[np.ndarray]) -> bool:
        if ground is None:
            return False
        bounds = self.config.get("camera.court_world_bounds", [-2.0, 17.0, -4.0, 18.0])
        return bool(
            bounds[0] <= ground[0] <= bounds[1] and bounds[2] <= ground[1] <= bounds[3]
        )

    def _extract_candidates(
        self,
        frame: np.ndarray,
        raw: Any,
        roi: Optional[np.ndarray],
    ) -> tuple[list[PlayerDetection], list[BallDetection]]:
        return extract_detections(
            frame,
            raw,
            person_class="person",
            ball_class="sports ball",
            threshold=float(self.config.get("rfdetr.person_threshold", 0.35)),
            ball_threshold=float(self.config.get("rfdetr.ball_threshold", 0.16)),
            roi_polygon=roi,
            roi_mask=make_roi_mask((frame.shape[0], frame.shape[1]), roi),
            roi_mode="bottom-center",
            min_mask_roi_ratio=float(
                self.config.get("rfdetr.min_mask_roi_ratio", 0.08)
            ),
            min_mask_pixels=int(self.config.get("rfdetr.min_mask_pixels", 100)),
            ball_min_size=float(self.config.get("rfdetr.ball_min_size", 5.0)),
            ball_max_size=float(self.config.get("rfdetr.ball_max_size", 100.0)),
            ball_min_aspect=float(self.config.get("rfdetr.ball_min_aspect", 0.35)),
            ball_max_aspect=float(self.config.get("rfdetr.ball_max_aspect", 2.8)),
            run_nms=False,
        )

    def _extract_batch(
        self,
        items: list[tuple[str, np.ndarray, Any, Optional[np.ndarray]]],
    ) -> list[FrameExtraction]:
        stage_started = time.perf_counter()
        raw_candidates = [
            self._extract_candidates(frame, raw, roi) for _, frame, raw, roi in items
        ]
        candidates: list[tuple[list[PlayerDetection], list[BallDetection]]] = []
        for (view, _, _, _), (players, balls) in zip(items, raw_candidates):
            # The calibrated mask-foot point is the court admission rule. Do
            # this before RTMPose/appearance/face inference so bench and
            # sideline people neither consume GPU work nor enter the ID pool.
            on_court = []
            for detection in players:
                ground = self.geometry.ground_point(view, detection.foot_xy)
                if (
                    ground is not None
                    and np.isfinite(ground).all()
                    and self._inside_court(ground)
                ):
                    on_court.append(detection)
            on_court.sort(
                key=lambda detection: float(detection.confidence or 0.0), reverse=True
            )
            on_court = on_court[: self.max_player_candidates]
            zones = self.ball_exclusion_zones.get(view, [])
            balls = [
                detection
                for detection in balls
                if all(
                    float(
                        np.hypot(
                            detection.center_xy[0] - float(zone[0]),
                            detection.center_xy[1] - float(zone[1]),
                        )
                    )
                    > float(zone[2])
                    for zone in zones
                )
            ]
            if self.refine_contours:
                for detection in on_court:
                    refined = _refined_mask_contours(
                        detection,
                        kernel_size=self.contour_smoothing_kernel,
                        threshold=self.contour_smoothing_threshold,
                    )
                    if refined:
                        detection.contours = refined
            candidates.append((on_court, balls))
        self.stage_seconds["detection_postprocess"] += (
            time.perf_counter() - stage_started
        )
        pose_inputs = [
            (frame, candidates[index][0])
            for index, (_, frame, _, _) in enumerate(items)
        ]
        stage_started = time.perf_counter()
        poses_by_item = self.pose_estimator.predict_many(pose_inputs)
        self.stage_seconds["rtmpose"] += time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        embeddings_by_item = self.appearance_encoder.encode_many(pose_inputs)
        self.stage_seconds["appearance_reid"] += time.perf_counter() - stage_started
        face_inputs = [
            (frame, candidates[index][0], poses_by_item[index])
            for index, (_, frame, _, _) in enumerate(items)
        ]
        stage_started = time.perf_counter()
        faces_by_item = self.face_encoder.encode_many(face_inputs)
        self.stage_seconds["face_reid"] += time.perf_counter() - stage_started
        extractions: list[FrameExtraction] = []
        min_valid = int(self.config.get("pose.min_valid_keypoints", 5))
        require_human_evidence = bool(
            self.config.get("pose.require_human_evidence", True)
        )
        for (view, _, _, _), (players, balls), poses, embeddings, faces in zip(
            items, candidates, poses_by_item, embeddings_by_item, faces_by_item
        ):
            overlaps = _bbox_overlap_ratios(players)
            observations: list[PoseObservation] = []
            visible_players: list[PlayerDetection] = []
            for detection, (xy, scores), appearance, face, overlap_ratio in zip(
                players, poses, embeddings, faces, overlaps
            ):
                mask_ground = self.geometry.ground_point(view, detection.foot_xy)
                valid_pose = _has_human_pose_evidence(
                    detection,
                    xy,
                    scores,
                    keypoint_threshold=self.keypoint_threshold,
                    min_valid_keypoints=min_valid,
                    min_mean_confidence=float(
                        self.config.get("pose.min_mean_keypoint_confidence", 0.35)
                    ),
                    min_torso_keypoints=int(
                        self.config.get("pose.min_torso_keypoints", 2)
                    ),
                    min_vertical_span_ratio=float(
                        self.config.get("pose.min_vertical_span_ratio", 0.18)
                    ),
                )
                if require_human_evidence and not valid_pose:
                    continue
                if mask_ground is not None:
                    visible_players.append(detection)
                ground = mask_ground
                if (
                    ground is None
                    or not np.isfinite(ground).all()
                    or not self._inside_court(ground)
                ):
                    continue
                observations.append(
                    PoseObservation(
                        view,
                        detection,
                        xy,
                        scores,
                        ground,
                        appearance,
                        overlap_ratio,
                        face_embedding=face,
                        mask_shape=_mask_shape_feature(detection),
                    )
                )
            extractions.append(FrameExtraction(visible_players, observations, balls))
        return extractions

    def _extract_frame(
        self,
        view: str,
        frame: np.ndarray,
        raw: Any,
        roi: Optional[np.ndarray],
    ) -> tuple[list[PoseObservation], list[BallDetection]]:
        extraction = self._extract_batch([(view, frame, raw, roi)])[0]
        return extraction.observations, extraction.balls

    def process(
        self,
        video_paths: dict[str, str],
        output_dir: str,
        start_frame: int,
        end_frame: Optional[int],
    ) -> dict[str, Any]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        captures: dict[str, cv2.VideoCapture] = {}
        writers: dict[str, Any] = {}
        video_info: dict[str, dict[str, Any]] = {}
        rois: dict[str, Optional[np.ndarray]] = {}
        frame_offsets = self.config.get("camera.frame_offsets", {})

        for view, path in video_paths.items():
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                print(f"[warn] cannot open {view}: {path}")
                continue
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            offset = int(frame_offsets.get(view, 0))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame + offset)
            captures[view] = capture
            rois[view] = self._roi(view, width, height)
            video_info[view] = {
                "path": path,
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": count,
                "frame_offset": offset,
            }
            writer_path = output / f"{view}_rfdetr_pose.mp4"
            writer = cv2.VideoWriter(
                str(writer_path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
            )
            if bool(self.config.get("visualization.async_video_writer", True)):
                writer = AsyncVideoWriter(
                    writer,
                    max_queue=int(
                        self.config.get("visualization.writer_queue_size", 96)
                    ),
                )
            writers[view] = writer

        if len(captures) < 2:
            raise RuntimeError(
                "At least two calibrated videos are required for multi-view 3D pose"
            )
        available = min(
            info["total_frames"] - start_frame - info["frame_offset"]
            for info in video_info.values()
        )
        requested = available if end_frame is None else max(0, end_frame - start_frame)
        frame_count = min(available, requested)

        ball_trackers = {
            view: BallTracker(
                max_distance=float(self.config.get("ball.max_distance", 520.0)),
                max_jump=float(self.config.get("ball.max_jump", 240.0)),
                switch_margin=float(self.config.get("ball.switch_margin", 0.25)),
                min_jump_quality=float(self.config.get("ball.min_jump_quality", 0.30)),
                max_missed=int(self.config.get("ball.max_missed", 12)),
                trail_length=int(self.config.get("ball.trail", 60)),
                stationary_switch_frames=int(
                    self.config.get("ball.stationary_switch_frames", 8)
                ),
            )
            for view in captures
        }
        poses_3d: dict[int, dict[int, list[list[float]]]] = {}
        poses_2d: dict[int, dict[int, dict[str, Any]]] = {}
        ground_positions_3d: dict[int, dict[int, list[float]]] = {}
        balls_2d: dict[int, dict[str, Any]] = {}
        balls_3d: dict[int, list[float]] = {}
        balls_3d_predicted: dict[int, bool] = {}
        quality: dict[int, dict[int, Any]] = {}
        ball_3d_filter = Ball3DTemporalFilter(
            float(self.config.get("ball.smoothing_alpha", 0.70)),
            int(self.config.get("ball.output_hold_frames", 6)),
            float(self.config.get("ball.max_3d_jump_m", 1.5)),
        )
        detection_frames = 0
        detector_calls_before = self.segmenter.inference_calls
        detector_slots_before = self.segmenter.inference_slots
        view_order = list(captures)
        temporal_batch_size = _temporal_batch_length(
            self.segmenter.backend_batch_size,
            len(view_order),
        )
        print(
            f"[batch] {temporal_batch_size} timestamps x {len(view_order)} views "
            f"-> detector batch capacity {self.segmenter.backend_batch_size}"
        )
        started = time.perf_counter()

        def batched_detector_frames() -> Iterable[
            tuple[int, dict[str, np.ndarray], list[FrameExtraction]]
        ]:
            """Batch adjacent timestamps while running inference on every source frame."""
            local_index = 0
            while local_index < frame_count:
                packets: list[tuple[int, dict[str, np.ndarray]]] = []
                while local_index < frame_count and len(packets) < temporal_batch_size:
                    frame_number = start_frame + local_index
                    frames: dict[str, np.ndarray] = {}
                    for view in view_order:
                        ok, frame = captures[view].read()
                        if not ok:
                            return
                        frames[view] = frame
                    packets.append((frame_number, frames))
                    local_index += 1

                flat_frames = [
                    frames[view] for _, frames in packets for view in view_order
                ]
                flat_rois = [rois[view] for _ in packets for view in view_order]
                stage_started = time.perf_counter()
                flat_outputs = self.segmenter.predict(flat_frames, flat_rois)
                self.stage_seconds["rfdetr"] += time.perf_counter() - stage_started
                flat_items = [
                    (view, frames[view], raw, rois[view])
                    for (_, frames), raw_group in zip(
                        packets,
                        [
                            flat_outputs[index : index + len(view_order)]
                            for index in range(0, len(flat_outputs), len(view_order))
                        ],
                    )
                    for view, raw in zip(view_order, raw_group)
                ]
                flat_extractions = self._extract_batch(flat_items)
                cursor = 0
                for frame_number, frames in packets:
                    next_cursor = cursor + len(view_order)
                    yield frame_number, frames, flat_extractions[cursor:next_cursor]
                    cursor = next_cursor

        progress = tqdm(
            batched_detector_frames(),
            total=frame_count,
            desc="RF-DETR 2XL + RTMPose multi-view",
        )
        try:
            for frame_number, frames, extractions in progress:
                views = view_order
                observations_by_view: dict[str, list[PoseObservation]] = {}
                players_by_view: dict[str, list[PlayerDetection]] = {}
                for view, extraction in zip(views, extractions):
                    players_by_view[view] = extraction.players
                    observations_by_view[view] = extraction.observations
                predicted_ball_3d = (
                    None
                    if ball_3d_filter.position is None
                    else ball_3d_filter.position + ball_3d_filter.velocity
                )
                selected_balls = self.geometry.select_ball_detections(
                    {
                        view: extraction.balls
                        for view, extraction in zip(views, extractions)
                    },
                    predicted_ball_3d,
                )
                measured_ball_3d = self.geometry.triangulate_ball(selected_balls)
                ball_3d, ball_predicted = ball_3d_filter.update(measured_ball_3d)
                if ball_predicted:
                    selected_balls = {}
                for view in views:
                    ball_trackers[view].update(
                        [selected_balls[view]] if view in selected_balls else []
                    )
                detection_frames += len(views)

                groups = self.fuser.fuse(observations_by_view)
                matches = self.tracker.update(groups)
                poses_3d[frame_number] = {}
                poses_2d[frame_number] = {}
                ground_positions_3d[frame_number] = {}
                quality[frame_number] = {}
                id_by_observation: dict[int, int] = {}
                for track, group in matches:
                    ground_positions_3d[frame_number][track.track_id] = (
                        track.ground_position.astype(float).tolist()
                    )
                    for observation in group.observations.values():
                        id_by_observation[id(observation)] = track.track_id
                    pose_3d, reprojection = self.geometry.triangulate_pose(
                        group.observations.values()
                    )
                    raw_valid_3d = int(
                        np.count_nonzero(np.isfinite(pose_3d).all(axis=1))
                    )
                    predicted_3d = False
                    if raw_valid_3d >= 5:
                        pose_3d = self.smoother.update(track.track_id, pose_3d)
                        poses_3d[frame_number][track.track_id] = pose_3d.tolist()
                    else:
                        filled_pose = self.smoother.fill_missing(
                            track.track_id,
                            track.ground_position[:2] - track.previous_position[:2],
                        )
                        if filled_pose is not None:
                            pose_3d = filled_pose
                            poses_3d[frame_number][track.track_id] = (
                                filled_pose.tolist()
                            )
                            predicted_3d = True
                    poses_2d[frame_number][track.track_id] = {}
                    for view, observation in group.observations.items():
                        poses_2d[frame_number][track.track_id][view] = {
                            "bbox": observation.detection.bbox_xyxy.astype(
                                float
                            ).tolist(),
                            "keypoints_xy": observation.keypoints_xy.astype(
                                float
                            ).tolist(),
                            "keypoints_conf": observation.keypoints_conf.astype(
                                float
                            ).tolist(),
                            "mask_foot_xy": list(
                                map(float, observation.detection.foot_xy)
                            ),
                            "ground_position": cast(
                                np.ndarray, observation.ground_position
                            )
                            .astype(float)
                            .tolist(),
                            "detection_confidence": float(
                                observation.detection.confidence or 0.0
                            ),
                        }
                    valid_errors = reprojection[np.isfinite(reprojection)]
                    quality[frame_number][track.track_id] = {
                        "views": sorted(group.observations),
                        "valid_3d_keypoints": int(
                            np.count_nonzero(np.isfinite(pose_3d).all(axis=1))
                        ),
                        "raw_valid_3d_keypoints": raw_valid_3d,
                        "predicted_3d": predicted_3d,
                        "predicted_track": False,
                        "mean_reprojection_error_px": float(np.mean(valid_errors))
                        if len(valid_errors)
                        else None,
                    }

                if selected_balls:
                    balls_2d[frame_number] = {
                        view: {
                            "center_xy": list(map(float, ball.center_xy)),
                            "bbox": ball.bbox_xyxy.astype(float).tolist(),
                            "confidence": float(ball.confidence or 0.0),
                        }
                        for view, ball in selected_balls.items()
                    }
                if ball_3d is not None:
                    balls_3d[frame_number] = ball_3d.astype(float).tolist()
                    balls_3d_predicted[frame_number] = ball_predicted

                for view, frame in frames.items():
                    canvas = frame.copy()
                    tracked_detections: set[int] = set()
                    for observation in observations_by_view[view]:
                        track_id = id_by_observation.get(id(observation))
                        if track_id is None:
                            continue
                        tracked_detections.add(id(observation.detection))
                        color = self.player_colors_bgr[
                            (track_id - 1) % len(self.player_colors_bgr)
                        ]
                        cv2.drawContours(
                            canvas,
                            observation.detection.contours,
                            -1,
                            color,
                            int(
                                self.config.get("visualization.person_contour_width", 2)
                            ),
                            cv2.LINE_AA,
                        )
                        _draw_skeleton(
                            canvas,
                            observation,
                            color,
                            self.connections,
                            self.keypoint_threshold,
                        )
                        x1, y1 = np.round(observation.detection.bbox_xyxy[:2]).astype(
                            int
                        )
                        cv2.putText(
                            canvas,
                            f"ID {track_id}",
                            (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                            cv2.LINE_AA,
                        )
                    if self.draw_unassigned_contours:
                        for detection in players_by_view[view]:
                            if id(detection) not in tracked_detections:
                                cv2.drawContours(
                                    canvas,
                                    detection.contours,
                                    -1,
                                    (225, 225, 225),
                                    int(
                                        self.config.get(
                                            "visualization.unassigned_contour_width", 2
                                        )
                                    ),
                                    cv2.LINE_AA,
                                )
                    trail = [
                        tuple(np.round(point).astype(int))
                        for point in ball_trackers[view].trail
                    ]
                    for first, second in zip(trail, trail[1:]):
                        cv2.line(canvas, first, second, (0, 196, 255), 3, cv2.LINE_AA)
                    selected = selected_balls.get(view)
                    if selected is not None:
                        cv2.drawContours(
                            canvas,
                            selected.contours,
                            -1,
                            (0, 196, 255),
                            int(self.config.get("visualization.ball_contour_width", 3)),
                            cv2.LINE_AA,
                        )
                        cv2.circle(
                            canvas,
                            tuple(np.round(selected.center_xy).astype(int)),
                            5,
                            (0, 196, 255),
                            -1,
                            cv2.LINE_AA,
                        )
                    cv2.putText(
                        canvas,
                        f"frame {frame_number} | RF-DETR-Seg 2XL + RTMPose",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 230, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    writers[view].write(canvas)
        finally:
            for capture in captures.values():
                capture.release()
            for writer in writers.values():
                writer.release()

        bone_refinement = refine_pose_bone_lengths(
            poses_3d,
            quality,
            self.connections,
            strength=float(self.config.get("pose.bone_refinement_strength", 0.5)),
            iterations=int(self.config.get("pose.bone_refinement_iterations", 6)),
            min_reference_samples=int(
                self.config.get("pose.bone_reference_min_samples", 20)
            ),
            max_reference_reprojection_error=float(
                self.config.get("pose.bone_reference_max_reprojection_error_px", 20.0)
            ),
        )
        result = {
            "schema_version": "2.0-rfdetr-rtmpose",
            "video_info": video_info,
            "models": {
                "detector": "RF-DETR-Seg 2XL",
                "detector_backend": self.segmenter.name,
                "pose": "RTMPose-M COCO-17",
                "keypoint_names": COCO17_NAMES,
            },
            "poses_3d": poses_3d,
            "poses_2d": poses_2d,
            "ground_positions_3d": ground_positions_3d,
            "balls_2d": balls_2d,
            "balls_3d": balls_3d,
            "balls_3d_predicted": balls_3d_predicted,
            "quality": quality,
            "postprocessing": {"bone_refinement": bone_refinement},
        }
        json_path = output / "poses_3d.json"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(
                result,
                handle,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=True,
            )
        tracks_dir = output / "tracks"
        tracks_dir.mkdir(exist_ok=True)
        player_tracks_path = tracks_dir / "player_tracks.jsonl"
        with open(player_tracks_path, "w", encoding="utf-8") as handle:
            for frame_number in sorted(ground_positions_3d):
                for track_id, point in sorted(
                    ground_positions_3d[frame_number].items()
                ):
                    handle.write(
                        json.dumps(
                            {
                                "frame_index": frame_number,
                                "track_id": track_id,
                                "ground_xyz": point,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        ball_tracks_path = tracks_dir / "ball_tracks.jsonl"
        with open(ball_tracks_path, "w", encoding="utf-8") as handle:
            for frame_number in sorted(balls_2d):
                handle.write(
                    json.dumps(
                        {
                            "frame_index": frame_number,
                            "views": balls_2d[frame_number],
                            "world_xyz": balls_3d.get(frame_number),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        elapsed = time.perf_counter() - started
        detector_invocations = self.segmenter.inference_calls - detector_calls_before
        detector_slots = self.segmenter.inference_slots - detector_slots_before
        metrics = {
            "processed_multiview_frames": len(poses_3d),
            "detector_frames": detection_frames,
            "detector_invocations": detector_invocations,
            "detector_batch_capacity": self.segmenter.backend_batch_size,
            "temporal_batch_size": temporal_batch_size,
            "detector_slot_utilization": round(detection_frames / detector_slots, 4)
            if detector_slots
            else None,
            "elapsed_seconds": round(elapsed, 3),
            "multiview_fps": round(len(poses_3d) / elapsed, 3) if elapsed else None,
            "poses_3d": sum(len(value) for value in poses_3d.values()),
            "poses_2d": sum(
                len(views) for frame in poses_2d.values() for views in frame.values()
            ),
            "ball_2d_frames": len(balls_2d),
            "ball_3d_frames": len(balls_3d),
            "ball_3d_observed_frames": sum(
                not predicted for predicted in balls_3d_predicted.values()
            ),
            "predicted_player_poses": sum(
                bool(record.get("predicted_track", False))
                for frame in quality.values()
                for record in frame.values()
            ),
            "bone_refinement": bone_refinement,
            "stage_seconds": {
                key: round(value, 3)
                for key, value in sorted(self.stage_seconds.items())
            },
            "output_json": str(json_path),
            "player_tracks": str(player_tracks_path),
            "ball_tracks": str(ball_tracks_path),
        }
        with open(output / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)
        print(f"[ok] {json_path}")
        print(
            f"[ok] {metrics['poses_3d']} 3D poses, {metrics['ball_3d_frames']} 3D ball frames"
        )
        return result
