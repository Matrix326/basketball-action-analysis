#!/usr/bin/env python
"""Verify a default full-pipeline output tree by decoding every video frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def verify(root: Path, start: int, end: int, fps: float) -> dict:
    if start < 0 or end <= start or fps <= 0:
        raise ValueError("Expected 0 <= start < end and positive FPS")
    data = json.loads((root / "poses/poses_3d.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "2.0-rfdetr-rtmpose"
    frames = set(range(start, end))
    assert {int(f) for f in data["poses_3d"]} == frames
    ids = set()
    missing_joints = 0
    for frame, players in data["poses_3d"].items():
        assert set(players) <= set(data["ground_positions_3d"][frame])
        ids.update(int(player) for player in data["ground_positions_3d"][frame])
        assert all(np.isfinite(point).all()
                   for point in data["ground_positions_3d"][frame].values())
        for player_id, values in players.items():
            ids.add(int(player_id))
            joints = np.asarray(values, dtype=float)
            assert joints.shape == (17, 3)
            missing_joints += int((~np.isfinite(joints).all(axis=1)).sum())
            assert np.isfinite(data["ground_positions_3d"][frame][player_id]).all()
    assert ids and min(ids) >= 1
    for field in ("poses_2d", "ground_positions_3d", "balls_2d", "balls_3d", "quality"):
        assert {int(f) for f in data[field]} <= frames
    for frame, players in data["poses_2d"].items():
        assert set(players) == set(data["ground_positions_3d"][frame])
        for observations in players.values():
            for view, observation in observations.items():
                assert view in data["video_info"]
                assert np.asarray(observation["keypoints_xy"]).shape == (17, 2)
                assert len(observation["keypoints_conf"]) == 17
                assert len(observation["bbox"]) == 4

    jsonl_counts = {}
    for filename, kind in (("player_tracks.jsonl", "player"), ("ball_tracks.jsonl", "ball")):
        count = 0
        with (root / "poses/tracks" / filename).open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                frame = str(record["frame_index"])
                assert int(frame) in frames
                if kind == "player":
                    point = data["ground_positions_3d"][frame][str(record["track_id"])]
                    np.testing.assert_allclose(record["ground_xyz"], point)
                else:
                    assert record["views"] == data["balls_2d"][frame]
                    assert record["world_xyz"] == data["balls_3d"].get(frame)
                count += 1
        expected = (sum(len(p) for p in data["ground_positions_3d"].values())
                    if kind == "player" else len(data["balls_2d"]))
        assert count == expected
        jsonl_counts[filename] = count

    expected_videos = {}
    for index, view in enumerate(data["video_info"], start=1):
        meta = data["video_info"][view]
        source_size = (meta["width"], meta["height"])
        expected_videos[f"poses/{view}_rfdetr_pose.mp4"] = source_size
        prefix = f"trajectory_pipeline/{index}/traj_gen"
        expected_videos[f"{prefix}/output_video_final.mp4"] = source_size
        expected_videos[f"{prefix}/topview_smooth.mp4"] = (800, 1400)
        for name in ("player_trajectory.json", "smooth_traj.json"):
            trajectories = json.loads((root / prefix / name).read_text(encoding="utf-8"))
            tracks = trajectories["final_merged_finished_trajectories"]
            assert tracks
            for player, points in tracks.items():
                assert int(player.removeprefix("player_")) in ids
                assert {int(f) for f in points} <= frames
                assert all(np.isfinite([p["x"], p["y"]]).all() for p in points.values())
    expected_videos["skeletons_3d/skeletons_3d_multi_view.mp4"] = (1600, 1200)
    assert {str(p.relative_to(root)) for p in root.rglob("*.mp4")} == set(expected_videos)

    videos = {}
    previews = []
    preview_names = {next(iter(expected_videos)),
                     "trajectory_pipeline/1/traj_gen/topview_smooth.mp4",
                     "skeletons_3d/skeletons_3d_multi_view.mp4"}
    for relative, size in expected_videos.items():
        cap = cv2.VideoCapture(str(root / relative))
        try:
            assert cap.isOpened(), relative
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            assert abs(actual_fps - fps) < 0.01, (relative, actual_fps)
            count = 0
            first_mean = last_mean = None
            for_index = (end - start) // 2
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                assert (frame.shape[1], frame.shape[0]) == size
                if first_mean is None:
                    first_mean = float(frame.mean())
                last_mean = float(frame.mean())
                if count == for_index and relative in preview_names:
                    assert frame.std() > 5, f"Blank preview: {relative}"
                    scale = min(640 / frame.shape[1], 480 / frame.shape[0])
                    resized = cv2.resize(frame, None, fx=scale, fy=scale)
                    tile = np.full((520, 640, 3), 245, dtype=np.uint8)
                    h, w = resized.shape[:2]
                    tile[40:40 + h, (640 - w) // 2:(640 - w) // 2 + w] = resized
                    cv2.putText(tile, relative.split("/")[-1], (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
                    previews.append(tile)
                count += 1
            assert count == end - start, (relative, count)
            videos[relative] = {"frames": count, "fps": actual_fps, "size": list(size),
                                "first_mean": first_mean, "last_mean": last_mean}
        finally:
            cap.release()

    gif_path = root / "skeletons_3d/skeletons_3d_multi_view.gif"
    with Image.open(gif_path) as gif:
        gif_frames = gif.n_frames
        assert gif_frames > 0
        for index in range(gif_frames):
            gif.seek(index)
            gif.load()
    assert len(previews) == 3
    assert cv2.imwrite(str(root / "verification_preview.jpg"), np.hstack(previews))
    report = {"status": "passed", "frame_range": [start, end], "track_ids": sorted(ids),
              "pose_records": sum(len(p) for p in data["poses_3d"].values()),
              "missing_3d_joints": missing_joints, "jsonl_records": jsonl_counts,
              "videos": videos, "gif_frames": gif_frames,
              "scope": "Default dimensions/FPS, smoothing and 3D enabled; artifact integrity, not accuracy."}
    (root / "verification.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30)
    args = parser.parse_args()
    report = verify(args.output_root.resolve(), args.start_frame, args.end_frame, args.fps)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
