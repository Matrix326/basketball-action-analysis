from __future__ import annotations

import json

import cv2
import numpy as np

from config import load_config
from src.track.traj_gen_3d import PlayerTrajectoryTracker3D


def test_trajectory_video_applies_offset_without_changing_json_frame_ids(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter.fourcc(*"FFV1"), 30, (160, 120)
    )
    assert writer.isOpened()
    for frame in range(12):
        writer.write(np.full((120, 160, 3), frame * 20, dtype=np.uint8))
    writer.release()
    data = {
        "poses_3d": {str(f): {"1": np.zeros((17, 3)).tolist()} for f in range(2, 8)},
        "ground_positions_3d": {str(f): {"1": [1.0, 1.0, 0.0]} for f in range(2, 8)},
    }
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps(data))
    config = load_config().override({"camera.frame_offsets.view1": 2})
    tracker = PlayerTrajectoryTracker3D(
        output_root_dir=str(tmp_path / "output"),
        input_video_path=str(source),
        poses_3d_json_path=str(poses),
        start_frame=2,
        process_seconds=6 / 30,
        fps=30,
        target_view="view1",
        app_config=config,
    )
    tracker.process()
    assert [record[0] for record in tracker.raw_trajectories[1]] == list(range(2, 8))
    cap = cv2.VideoCapture(tracker.config["OUTPUT_VIDEO_PATH"])
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    for frame in range(2, 8):
        ok, image = cap.read()
        assert ok
        assert abs(float(image[90:110, 120:150].mean()) - (frame + 2) * 20) < 8
    cap.release()
