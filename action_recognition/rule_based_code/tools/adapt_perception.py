#!/usr/bin/env python
"""Adapt a perception/poses_3d.json for the rule_based pipeline.

Perception's output stores the ball as `balls_3d` (online-filtered) with a
`balls_3d_predicted` flag, but the rule_based ball-trajectory stage consumes
`ball_measurements` (observed positions + view counts). This tool:

1. copies the perception poses_3d.json and adds `ball_measurements`;
2. writes a hoop_3d.json in the perception world frame (the hoop position is
   triangulated from our annotated rim pixels through the perception
   calibration, because perception itself emits no hoop).

Usage:
    python tools/adapt_perception.py \
        --poses <perception poses_3d.json> \
        --out-dir <dir for adapted inputs> \
        [--hoop-center x y z]      # override; default = triangulated value
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Rim centre pixels annotated on the undistorted videos (shared by both
# pipelines — same source videos). Triangulated through the perception
# calibration these give the hoop below.
HOOP_2D = {
    "view1": (790.0, 190.8),
    "view2": (1110.6, 104.7),
    "view3": (413.3, 174.7),
    "view4": (1192.0, 226.6),
}
VIEW_TO_CAM = {"view1": "A1", "view2": "A2", "view3": "B3", "view4": "B4"}
DEFAULT_HOOP = [7.642, 1.81, 3.045]


def triangulate_hoop(extrinsics_path: Path, intrinsics_path: Path) -> list[float]:
    ext = json.load(open(extrinsics_path))
    intr = json.load(open(intrinsics_path))
    A, b = [], []
    for view, cam in VIEW_TO_CAM.items():
        uv = HOOP_2D.get(view)
        if uv is None or cam not in ext or cam not in intr:
            continue
        K = np.array(intr[cam]["K_undistorted"], dtype=float)
        R = np.array(ext[cam]["R_w2c"], dtype=float)
        t = np.array(ext[cam]["t_w2c"], dtype=float)
        d_cam = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
        d = R.T @ d_cam
        d = d / np.linalg.norm(d)
        C = -R.T @ t
        P = np.eye(3) - np.outer(d, d)
        A.append(P)
        b.append(P @ C)
    if not A:
        return DEFAULT_HOOP
    X, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    return [float(v) for v in X]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True, help="perception poses_3d.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hoop-center", type=float, nargs=3, default=None)
    parser.add_argument("--extrinsics", default=None)
    parser.add_argument("--intrinsics", default=None)
    args = parser.parse_args()

    poses_path = Path(args.poses)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(poses_path, encoding="utf-8"))
    b3d = data.get("balls_3d", {}) or {}
    bpred = data.get("balls_3d_predicted", {}) or {}
    b2d = data.get("balls_2d", {}) or {}

    sibling_hoop = poses_path.parent / "hoop_3d.json"
    if data.get("ball_measurements"):
        # Already in the rule_based schema (our own pipeline output) — keep
        # the measurements, and reuse the sibling hoop_3d.json: it is already
        # expressed in this file's world frame.
        measurements = data["ball_measurements"]
        print(
            f"[info] ball_measurements already present ({len(measurements)}), kept as-is"
        )
        if args.hoop_center is None and sibling_hoop.exists():
            hoop = json.load(open(sibling_hoop, encoding="utf-8"))
            (out_dir / "hoop_3d.json").write_text(
                json.dumps(hoop, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            json.dump(data, open(out_dir / "poses_3d.json", "w", encoding="utf-8"))
            print(
                f"[ok] adapted poses -> {out_dir / 'poses_3d.json'}  (ball_measurements {len(measurements)})"
            )
            print(
                f"[ok] hoop -> {out_dir / 'hoop_3d.json'}  center={hoop.get('hoop_center')} (sibling)"
            )
            return
    else:
        measurements = _build_measurements(b3d, bpred, b2d)
        data["ball_measurements"] = measurements

    _write_outputs(data, measurements, args, out_dir, poses_path)


def _build_measurements(b3d, bpred, b2d) -> dict:
    measurements: dict[str, dict] = {}
    for frame_key, xyz in b3d.items():
        if bpred.get(frame_key, False):
            continue  # predicted hold-over, not an observation
        arr = np.asarray(xyz, dtype=float)
        if not np.isfinite(arr).all():
            continue
        views = len(b2d.get(frame_key, {}) or {})
        measurements[frame_key] = {
            "position": [float(v) for v in arr],
            "views": int(max(views, 2)),
        }
    return measurements


def _write_outputs(data, measurements, args, out_dir, poses_path) -> None:
    if args.hoop_center is not None:
        hoop_center = [float(v) for v in args.hoop_center]
    elif args.extrinsics and args.intrinsics:
        hoop_center = triangulate_hoop(Path(args.extrinsics), Path(args.intrinsics))
    else:
        hoop_center = DEFAULT_HOOP

    hoop = {
        "schema_version": "hoop-3d/v1",
        "hoop_center": hoop_center,
        "hoop_bottom": hoop_center,
        "height_m": 3.05,
        "frame_range": [0, 10**9],
        "candidates": [],
        "coordinate_system": "perception world metres (triangulated from rim pixels)",
        "validated": True,
        "note": "triangulated with the perception calibration",
    }

    poses_out = out_dir / "poses_3d.json"
    hoop_out = out_dir / "hoop_3d.json"
    json.dump(data, open(poses_out, "w", encoding="utf-8"))
    json.dump(hoop, open(hoop_out, "w", encoding="utf-8"))
    print(f"[ok] adapted poses -> {poses_out}  (ball_measurements {len(measurements)})")
    print(f"[ok] hoop -> {hoop_out}  center={hoop_center}")


if __name__ == "__main__":
    main()
