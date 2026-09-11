"""Offline ball-trajectory post-processing: kinematics gap filling and smoothing.

Consumes the multi-view pipeline's raw ball triangulations (poses_3d.json,
``ball_measurements`` preferred, ``balls_3d`` fallback) and produces a
continuous, physics-aware 3D ball trajectory as a JSON interface aligned with
the 3D skeleton frames, for downstream rule-based action recognition.

Pure numpy/scipy — no torch/onnxruntime dependency.
"""

from .pipeline import BallTrajectoryPostProcessor
from .io import BallObservations, TrajectoryInput, load_poses_json, build_input, save_interface

__all__ = [
    "BallTrajectoryPostProcessor",
    "BallObservations",
    "TrajectoryInput",
    "load_poses_json",
    "build_input",
    "save_interface",
]
