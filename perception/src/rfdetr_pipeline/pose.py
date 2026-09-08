"""RTMPose model loading and batched pose inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import torch

from config import Config
from basketball_repro.detection_runtime import PlayerDetection

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _device(value: str) -> str:
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return value

def _trusted_torch_load(*args: Any, **kwargs: Any) -> Any:
    """Load local, user-configured checkpoints across the torch 2.6 change."""
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)

_ORIGINAL_TORCH_LOAD = torch.load

class RTMPoseEstimator:
    """Top-down COCO-17 estimator receiving RF-DETR person boxes."""

    def __init__(self, config: Config) -> None:
        mmpose_path = Path(
            config.get(
                "pose.mmpose_site_packages",
                PROJECT_ROOT / "third_party",
            )
        )
        # Prefer the vendored MMPose source while keeping installed binary
        # dependencies from the active Python environment.
        if mmpose_path.exists() and str(mmpose_path) not in sys.path:
            sys.path.append(str(mmpose_path))
        try:
            from mmpose.apis import init_model
            from mmengine.dataset import Compose, pseudo_collate
            from mmengine.registry import init_default_scope
        except ImportError as exc:
            raise RuntimeError(
                "RTMPose dependencies are unavailable. Install requirements.txt "
                "and check pose.mmpose_site_packages in the config."
            ) from exc

        pose_config = config.get("pose.config")
        checkpoint = config.get("pose.checkpoint")
        if not pose_config or not Path(pose_config).exists():
            raise FileNotFoundError(f"RTMPose config not found: {pose_config}")
        if not checkpoint or not Path(checkpoint).exists():
            raise FileNotFoundError(f"RTMPose checkpoint not found: {checkpoint}")

        device = _device(config.get("pose.device", "auto"))
        # MMPose/MMEngine versions predating torch 2.6 call torch.load without
        # weights_only=False.  The checkpoint path is explicit and local.
        torch.load = _trusted_torch_load
        try:
            self.model = init_model(
                pose_config,
                checkpoint,
                device=device,
                cfg_options={"model": {"test_cfg": {"output_heatmaps": False}}},
            )
        finally:
            torch.load = _ORIGINAL_TORCH_LOAD
        self.pipeline = Compose(self.model.cfg.test_dataloader.dataset.pipeline)
        self.pseudo_collate = pseudo_collate
        self.init_default_scope = init_default_scope

    def predict(self, frame_bgr: np.ndarray, detections: List[PlayerDetection]) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.predict_many([(frame_bgr, detections)])[0]

    def predict_many(
        self,
        items: list[tuple[np.ndarray, List[PlayerDetection]]],
    ) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        """Run all RF-DETR boxes in one RTMPose batch without mixing images."""
        counts = [len(detections) for _, detections in items]
        if not any(counts):
            return [[] for _ in items]

        scope = self.model.cfg.get("default_scope", "mmpose")
        if scope is not None:
            self.init_default_scope(scope)
        data_list = []
        for frame_bgr, detections in items:
            for detection in detections:
                data_info = {
                    "img": frame_bgr,
                    "bbox": np.asarray(detection.bbox_xyxy, dtype=np.float32)[None],
                    "bbox_score": np.ones(1, dtype=np.float32),
                }
                data_info.update(self.model.dataset_meta)
                data_list.append(self.pipeline(data_info))
        with torch.no_grad():
            results = self.model.test_step(self.pseudo_collate(data_list))

        poses: list[tuple[np.ndarray, np.ndarray]] = []
        for result in results:
            instances = result.pred_instances
            xy = np.asarray(instances.keypoints[0, :17], dtype=np.float32)
            scores = np.asarray(instances.keypoint_scores[0, :17], dtype=np.float32)
            if xy.shape != (17, 2) or scores.shape != (17,):
                raise RuntimeError(f"RTMPose must return COCO-17, got xy={xy.shape}, scores={scores.shape}")
            poses.append((xy, scores))
        if len(poses) != sum(counts):
            raise RuntimeError(f"RTMPose returned {len(poses)} poses for {sum(counts)} RF-DETR boxes")
        grouped: list[list[tuple[np.ndarray, np.ndarray]]] = []
        cursor = 0
        for count in counts:
            grouped.append(poses[cursor : cursor + count])
            cursor += count
        return grouped

