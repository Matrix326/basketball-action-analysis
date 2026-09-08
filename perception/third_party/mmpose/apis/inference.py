# Copyright (c) OpenMMLab. All rights reserved.
"""Minimal RTMPose model initialization API used by this project."""

from pathlib import Path
from typing import Optional, Union

import torch.nn as nn
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

import mmpose.datasets  # Register the inference transforms.
from mmpose.models.builder import build_pose_estimator


def init_model(
    config: Union[str, Path, Config],
    checkpoint: Optional[str] = None,
    device: str = "cuda:0",
    cfg_options: Optional[dict] = None,
) -> nn.Module:
    """Build a pose estimator and load checkpoint metadata."""
    if isinstance(config, (str, Path)):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError(
            f"config must be a filename or Config object, got {type(config)}"
        )

    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    elif "init_cfg" in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None

    scope = config.get("default_scope", "mmpose")
    if scope is not None:
        init_default_scope(scope)

    model = build_pose_estimator(config.model)
    dataset_meta = None
    if checkpoint is not None:
        checkpoint_data = load_checkpoint(model, checkpoint, map_location="cpu")
        dataset_meta = checkpoint_data.get("meta", {}).get("dataset_meta")
    if dataset_meta is None:
        raise RuntimeError("RTMPose checkpoint does not contain dataset_meta")

    model.dataset_meta = dataset_meta
    model.cfg = config
    model.to(device)
    model.eval()
    return model
