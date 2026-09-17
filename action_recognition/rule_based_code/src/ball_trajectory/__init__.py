"""Offline ball-trajectory post-processing: kinematics gap filling and smoothing.

Consumes the multi-view pipeline's raw ball triangulations (poses_3d.json,
``ball_measurements`` preferred, ``balls_3d`` fallback) and produces a
continuous, physics-aware 3D ball trajectory as a JSON interface aligned with
the 3D skeleton frames, for downstream rule-based action recognition.

Pure numpy/scipy — no torch/onnxruntime dependency.
"""

from .io import (
    BallObservations,
    TrajectoryInput,
    build_input,
    load_poses_json,
    save_interface,
)
from .pipeline import BallTrajectoryPostProcessor

__all__ = [
    "BallTrajectoryPostProcessor",
    "BallObservations",
    "TrajectoryInput",
    "load_poses_json",
    "build_input",
    "save_interface",
]
