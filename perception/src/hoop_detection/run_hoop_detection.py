#!/usr/bin/env python
"""Detect basketball hoops in all four views and triangulate the 3D hoop.

Stage 1: YOLO hoop/ball detection on every frame of the requested range.
Stage 2: per-view candidate clustering (the hoop is static -> stable boxes).
Stage 3: cross-view triangulation of the per-view cluster bottoms using the
         existing calibration; consensus 3D point within the court bounds and
         hoop height (~3.05 m) is the hoop.

Output: output/rfdetr_multiview/poses/hoop_3d.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import load_config  # noqa: E402

if TYPE_CHECKING:
    from ultralytics.engine.results import Boxes, Results


def triangulate(
    P_list: list[np.ndarray], pixels: list[np.ndarray]
) -> np.ndarray | None:
    """DLT triangulation from >=2 views; returns 3D point or None."""
    if len(P_list) < 2 or len(P_list) != len(pixels):
        return None
    rows = []
    for P, (u, v) in zip(P_list, pixels):
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    try:
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
        homogeneous = vt[-1]
        if abs(homogeneous[3]) < 1e-10:
            return None
        point = homogeneous[:3] / homogeneous[3]
    except np.linalg.LinAlgError:
        return None
    return point if np.isfinite(point).all() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Hoop detection and 3D triangulation")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "config.yaml")
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="YOLO hoop weights (default: hoop_detection.weights in the config, "
        "else <project_root>/models/hoop_yolo.pt)",
    )
    parser.add_argument("--start-frame", type=int, default=900)
    parser.add_argument("--end-frame", type=int, default=1800)
    parser.add_argument("--hoop-conf", type=float, default=0.30)
    parser.add_argument(
        "--sample-every", type=int, default=5, help="process every Nth frame"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    from ultralytics import YOLO

    config = load_config(args.config)
    weights = (
        args.weights
        or config.get("hoop_detection.weights")
        or str(PROJECT_ROOT / "models" / "hoop_yolo.pt")
    )
    print(f"hoop weights: {weights}")
    model = YOLO(weights)
    print("classes:", model.names)

    views = list(config.video_paths)
    boxes_by_view: dict[str, list[list[float]]] = defaultdict(
        list
    )  # view -> [cx, y2, w, conf]

    for view in views:
        path = config.video_paths.get(view)
        cap = cv2.VideoCapture(str(path))
        offset = int(config.get(f"camera.frame_offsets.{view}", 0))
        print(f"[{view}] detecting frames {args.start_frame}-{args.end_frame} ...")
        for frame_number in range(
            args.start_frame, args.end_frame + 1, args.sample_every
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number + offset)
            ok, frame = cap.read()
            if not ok:
                break
            results = cast("list[Results]", model(frame, verbose=False))
            for box in cast("Boxes", results[0].boxes):
                if int(box.cls[0]) != 1 or float(box.conf[0]) < args.hoop_conf:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes_by_view[view].append(
                    [
                        float((x1 + x2) / 2.0),
                        float((y1 + y2) / 2.0),
                        float(x2 - x1),
                        float(box.conf[0]),
                    ]
                )
        cap.release()
        print(f"[{view}] {len(boxes_by_view[view])} hoop detections")

    # ---- stage 2: per-view clustering (k-means on bottom-centre, k=2 max) ----
    view_clusters: dict[str, list[list[float]]] = {}
    for view, boxes in boxes_by_view.items():
        if not boxes:
            view_clusters[view] = []
            continue
        arr = np.asarray(boxes)
        # 1D cluster on x (hoops are horizontally separated in the frame)
        xs = arr[:, 0]
        x_min, x_max = xs.min(), xs.max()
        if x_max - x_min < 200:  # single hoop in view
            clusters = [np.median(arr, axis=0).tolist()]
        else:
            # two clusters via median split
            mid = (x_min + x_max) / 2.0
            clusters = [
                np.median(arr[arr[:, 0] < mid], axis=0).tolist(),
                np.median(arr[arr[:, 0] >= mid], axis=0).tolist(),
            ]
        view_clusters[view] = [c for c in clusters if len(c)]
        print(
            f"[{view}] clusters: {[[round(v, 1) for v in c] for c in view_clusters[view]]}"
        )

    # ---- stage 3: cross-view triangulation ----
    intrinsics = json.load(open(config.get("camera.intrinsics_path"), encoding="utf-8"))
    extrinsics = json.load(open(config.get("camera.extrinsics_path"), encoding="utf-8"))
    P_by_view: dict[str, np.ndarray] = {}
    for view in views:
        name = config.view_to_camera.get(view, view)
        K = np.asarray(intrinsics[name]["K_undistorted"], dtype=np.float64)
        R = np.asarray(extrinsics[name]["R_w2c"], dtype=np.float64)
        t = np.asarray(extrinsics[name]["t_w2c"], dtype=np.float64).reshape(3, 1)
        P_by_view[view] = K @ np.hstack([R, t])

    candidates: list[dict] = []
    view_names = [v for v in views if view_clusters.get(v)]
    for first_idx, first_view in enumerate(view_names):
        for first_cluster in view_clusters[first_view]:
            for second_view in view_names[first_idx + 1 :]:
                for second_cluster in view_clusters[second_view]:
                    point = triangulate(
                        [P_by_view[first_view], P_by_view[second_view]],
                        [np.asarray(first_cluster[:2]), np.asarray(second_cluster[:2])],
                    )
                    if point is None:
                        continue
                    candidates.append(
                        {
                            "point": point.tolist(),
                            "views": [first_view, second_view],
                            "pixels": [first_cluster[:2], second_cluster[:2]],
                        }
                    )

    bounds = config.get("camera.court_world_bounds", [0.0, 15.0, 0.0, 14.0])
    plausible = [
        c
        for c in candidates
        if bounds[0] - 2 <= c["point"][0] <= bounds[1] + 2
        and bounds[2] - 2 <= c["point"][1] <= bounds[3] + 2
        and 2.5 <= c["point"][2] <= 3.6
    ]
    print(
        f"{len(candidates)} pairwise candidates, {len(plausible)} in court+height range"
    )

    # Consensus: the true hoop reprojects onto a detected cluster in every
    # view that can see it (B3 sees none). Count supporting views per point.
    support_px = 250.0
    supported: list[dict] = []
    for c in plausible:
        point = np.asarray(c["point"])
        supporting = set(c["views"])
        for view in view_names:
            projected = P_by_view[view] @ np.r_[point, 1.0]
            if abs(projected[2]) < 1e-9:
                continue
            px, py = projected[:2] / projected[2]
            for cluster in view_clusters.get(view, []):
                if np.hypot(px - cluster[0], py - cluster[1]) <= support_px:
                    supporting.add(view)
                    break
        if len(supporting) >= max(2, len(view_names) - 1):
            c["supporting_views"] = sorted(supporting)
            supported.append(c)
            print(
                "  consensus candidate:",
                [round(v, 2) for v in c["point"]],
                "views:",
                sorted(supporting),
            )
    if not supported:
        raise SystemExit("no consensus hoop 3D found")

    # Prefer the candidate the BALL actually approaches: during shots and
    # layups the ball flies to (and through) the real hoop, so the true hoop
    # has a much smaller min horizontal distance to the ball trajectory.
    # The ball trajectory is produced by the action module (ljy/rule_based_code);
    # point this at that output when it is available — it helps pick the true
    # hoop among candidates (shots fly toward it).
    ball_traj_path = Path(
        config.get("hoop_detection.ball_trajectory_path")
        or (Path(config.get("output.reid_3d_dir")) / "ball_trajectory.json")
    )
    ball_positions: list[list[float]] = []
    if ball_traj_path.exists():
        with open(ball_traj_path, encoding="utf-8") as handle:
            ball_traj = json.load(handle)
        ball_positions = [
            record["position"]
            for record in ball_traj.get("frames", {}).values()
            if record.get("position") is not None
        ]
    if ball_positions:
        ball_arr = np.asarray(ball_positions)
        for c in supported:
            point = np.asarray(c["point"])
            c["min_ball_distance_m"] = float(
                np.min(np.linalg.norm(ball_arr[:, :2] - point[:2], axis=1))
            )
        min_ball_d = min(c["min_ball_distance_m"] for c in supported)
        best = [c for c in supported if c["min_ball_distance_m"] <= min_ball_d + 0.5]
        # The hoop sits on a baseline (y = court bound): among equally close
        # candidates prefer those near a baseline (a shot arc passes the hoop
        # at its vertex, which is not necessarily directly below the rim).
        if len(best) > 1:
            y_min, y_max = bounds[2], bounds[3]
            best.sort(
                key=lambda c: min(
                    abs(c["point"][1] - y_min), abs(c["point"][1] - y_max)
                )
            )
            nearest_y = min(
                min(abs(c["point"][1] - y_min), abs(c["point"][1] - y_max))
                for c in best
            )
            best = [
                c
                for c in best
                if min(abs(c["point"][1] - y_min), abs(c["point"][1] - y_max))
                <= nearest_y + 0.5
            ]
        print(
            f"ball-based selection: min ball distance {min_ball_d:.2f} m "
            f"({len(best)}/{len(supported)} candidates)"
        )
    else:
        # fallback: the most-supported candidate group
        max_support = max(len(c.get("supporting_views", c["views"])) for c in supported)
        best = [
            c
            for c in supported
            if len(c.get("supporting_views", c["views"])) == max_support
        ]
    points = np.asarray([c["point"] for c in best])
    hoop = np.median(points, axis=0).tolist()

    output = {
        "schema_version": "hoop-3d/v1",
        "hoop_center": hoop,
        "hoop_bottom": [hoop[0], hoop[1], hoop[2]],
        "height_m": round(float(hoop[2]), 3),
        "frame_range": [args.start_frame, args.end_frame],
        "candidates": [
            {
                "point": [round(v, 3) for v in c["point"]],
                "views": c.get("supporting_views", c["views"]),
            }
            for c in supported
        ],
        "coordinate_system": "calibrated world metres, z up",
    }
    output_path = (
        Path(args.output)
        if args.output
        else (Path(config.get("output.reid_3d_dir")) / "hoop_3d.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=1)
    print(f"[ok] hoop 3D -> {output_path}")


if __name__ == "__main__":
    main()
