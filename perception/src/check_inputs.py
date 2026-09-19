#!/usr/bin/env python
"""Check configured videos, models and calibration without running inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    load_config,
)


def check_inputs(
    config_path, *, start_frame=None, end_frame=None, limit=None, views=None
):
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    config = load_config(str(path))
    views = list(views or config.video_paths)
    if len(set(views)) < 2 or set(views) - set(config.video_paths):
        raise ValueError("Select at least two distinct configured views")
    start = (
        int(config.get("trajectory.start_frame", 0))
        if start_frame is None
        else start_frame
    )
    fps = float(config.get("trajectory.fps", 30))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("trajectory.fps must be positive and finite")
    end = start + limit if limit is not None else end_frame
    if end is None:
        end = start + int(float(config.get("trajectory.process_seconds", 30)) * fps)
    if start < 0 or end <= start:
        raise ValueError("Expected 0 <= start-frame < end-frame, or a positive limit")

    files = [
        "camera.intrinsics_path",
        "camera.extrinsics_path",
        "assets.court_background",
        "pose.config",
        "pose.checkpoint",
    ]
    if config.get("reid.use_appearance_embeddings", True) and config.get(
        "reid.use_deep_appearance_embeddings", True
    ):
        files.append("reid.appearance_checkpoint")
    resolved = {}
    for key in files:
        value = config.get(key)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"{key}: {value}")
        resolved[key] = str(Path(value).resolve())
    backend = str(config.get("rfdetr.backend", "auto")).lower()
    options = {"tensorrt": "rfdetr.engine_path", "onnx": "rfdetr.onnx_path"}
    if backend not in {"auto", "hybrid", *options}:
        raise ValueError(f"Unsupported perception detector backend: {backend}")
    if backend == "auto":
        candidates = list(options.values())
    elif backend == "hybrid":
        # detector.py loads the person detector through ONNX for hybrid; the
        # TensorRT engine is not part of that path.
        candidates = [options["onnx"]]
    else:
        candidates = [options[backend]]
    available = [
        key for key in candidates if config.get(key) and Path(config.get(key)).is_file()
    ]
    if not available:
        raise FileNotFoundError(f"No model file for {backend}: {candidates}")
    resolved.update({key: str(Path(config.get(key)).resolve()) for key in available})
    if backend == "hybrid":
        # An empty path means the caller opted into the single-model fallback.
        # A path that is set but missing also degrades ball detection, but the
        # detector only warns, so fail fast here instead.
        ball_checkpoint = config.get("rfdetr.ball_checkpoint_path")
        if ball_checkpoint:
            if not Path(ball_checkpoint).is_file():
                raise FileNotFoundError(
                    f"rfdetr.ball_checkpoint_path: {ball_checkpoint}"
                )
            resolved["rfdetr.ball_checkpoint_path"] = str(
                Path(ball_checkpoint).resolve()
            )
    if config.get("reid.use_face_embeddings", True):
        face_root = (
            Path(config.get("reid.insightface_root"))
            / "models"
            / config.get("model.insightface_name", "buffalo_l")
        )
        for name in ("det_10g.onnx", "w600k_r50.onnx"):
            if not (face_root / name).is_file():
                raise FileNotFoundError(face_root / name)
            resolved[f"insightface.{name}"] = str((face_root / name).resolve())
    with open(resolved["camera.intrinsics_path"], encoding="utf-8") as handle:
        intrinsics = json.load(handle)
    with open(resolved["camera.extrinsics_path"], encoding="utf-8") as handle:
        extrinsics = json.load(handle)
    videos = {}
    for view in views:
        camera = config.view_to_camera.get(view, view)
        if camera not in intrinsics or camera not in extrinsics:
            raise ValueError(f"Missing calibration for {view}: {camera}")
        cap = cv2.VideoCapture(str(config.video_paths[view]))
        try:
            if not cap.isOpened():
                raise OSError(f"Cannot open {view}: {config.video_paths[view]}")
            actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            offset = int(config.get(f"camera.frame_offsets.{view}", 0))
            if not math.isfinite(actual_fps) or abs(actual_fps - fps) > 0.01:
                raise ValueError(
                    f"{view} FPS={actual_fps}, but trajectory.fps={fps}; synchronize the videos/config first"
                )
            if start + offset < 0 or end + offset > total:
                raise ValueError(
                    f"{view} source range [{start + offset}, {end + offset}) exceeds [0, {total})"
                )
            cap.set(cv2.CAP_PROP_POS_FRAMES, start + offset)
            ok, frame = cap.read()
            if not ok:
                raise OSError(f"Cannot decode {view} source frame {start + offset}")
            videos[view] = {
                "path": config.video_paths[view],
                "camera": camera,
                "fps": actual_fps,
                "total_frames": total,
                "frame_offset": offset,
                "width": frame.shape[1],
                "height": frame.shape[0],
            }
        finally:
            cap.release()
    return {
        "status": "passed",
        "project_root": config.get("project_root"),
        "frame_range": [start, end],
        "frame_range_convention": "start_inclusive_end_exclusive",
        "videos": videos,
        "files": resolved,
        "note": "File/video checks only; CUDA and engine compatibility require inference.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/config.yaml"))
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--views", nargs="+")
    args = parser.parse_args()
    try:
        report = check_inputs(
            args.config,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            limit=args.limit,
            views=args.views,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Input check failed: {error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
