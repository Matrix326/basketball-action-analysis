#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .contrastive import ContrastiveModel  # noqa
from .custom_video_model_builder import *  # noqa
from .masked import MaskMViT  # noqa
from .video_model_builder import MViT, ResNet, SlowFast  # noqa
from .clip_video_model import *  # noqa: F403 - Preserve the upstream public exports.
from .clip_image_model import *  # noqa: F403 - Preserve the upstream public exports.
from .temporalclip_video_model import TemporalClipVideo as TemporalClipVideo

try:
    from .ptv_model_builder import (
        PTVCSN as PTVCSN,
        PTVX3D as PTVX3D,
        PTVR2plus1D as PTVR2plus1D,
        PTVResNet as PTVResNet,
        PTVSlowFast as PTVSlowFast,
    )  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
