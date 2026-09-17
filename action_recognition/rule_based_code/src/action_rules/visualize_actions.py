#!/usr/bin/env python
"""Visualize actions on the source video: handler box + action labels +
shot result (MAKE/MISS) in the top-right corner.

Usage:
    python src/action_rules/visualize_actions.py \
        --config config/config.yaml --view view1 --handler-only \
        --start-frame 5400 --end-frame 6300 --scale 0.6 \
        --output /tmp/actions_demo.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402

COLORS = {
    "pass": (255, 255, 0),
    "dribble_start": (255, 165, 0),
    "shoot": (0, 255, 0),
    "layup": (0, 255, 0),
    "rebound": (255, 0, 255),
    "follow_up": (0, 200, 255),
    "block": (0, 0, 255),
    "catch": (255, 255, 255),
    "release": (128, 128, 128),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize actions on video")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "config.yaml")
    )
    parser.add_argument("--actions", default=None)
    parser.add_argument("--ball-trajectory", default=None)
    parser.add_argument("--poses-json", default=None)
    parser.add_argument("--view", default="view1")
    parser.add_argument("--all-views", action="store_true")
    parser.add_argument("--output", default="/tmp/actions_demo.mp4")
    parser.add_argument("--start-frame", type=int, default=900)
    parser.add_argument("--end-frame", type=int, default=1800)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--trail-frames", type=int, default=15)
    parser.add_argument(
        "--handler-only",
        action="store_true",
        help="draw only the handler box, not every person",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    poses_dir = Path(config.get("output.reid_3d_dir"))
    actions_path = Path(args.actions or poses_dir / "actions.json")
    ball_traj_path = Path(
        args.ball_trajectory
        or config.get("ball_trajectory.output_path", poses_dir / "ball_trajectory.json")
    )
    poses_path = Path(args.poses_json or poses_dir / "poses_3d.json")

    actions = json.load(open(actions_path, encoding="utf-8"))["actions"]
    poses = json.load(open(poses_path, encoding="utf-8"))
    p2d = poses.get("poses_2d", {})
    ball_traj = json.load(open(ball_traj_path, encoding="utf-8"))
    ball_pos = {
        int(k): np.asarray(v["position"], dtype=float)
        for k, v in ball_traj.get("frames", {}).items()
        if v.get("position") is not None
    }

    views = list(config.video_paths) if args.all_views else [args.view]
    fps = float(config.get("trajectory.fps", 30.0))
    frames = list(range(args.start_frame, args.end_frame + 1))

    # per-view readers
    caps = {}
    for v in views:
        cap = cv2.VideoCapture(str(config.video_paths[v]))
        caps[v] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    h0 = w0 = None
    for v in views:
        ok, frame = caps[v].read()
        if ok:
            h0, w0 = frame.shape[:2]
            break
    if h0 is None or w0 is None:
        raise SystemExit("cannot read video")
    out_w, out_h = int(w0 * args.scale), int(h0 * args.scale)
    out = cv2.VideoWriter(
        args.output, cv2.VideoWriter.fourcc(*"mp4v"), fps, (out_w, out_h)
    )
    print(
        f"output {out_w}x{out_h} @ {fps:.0f}fps, frames {args.start_frame}-{args.end_frame}"
    )

    # actions indexed by frame
    events_by_frame = {}
    for a in actions:
        for f in range(a["frame"], a.get("end_frame", a["frame"]) + 1):
            events_by_frame.setdefault(f, []).append(a)

    for fi, frame_no in enumerate(frames):
        tiles = []
        for v in views:
            ok, frame = caps[v].read()
            if not ok:
                frame = np.zeros((h0, w0, 3), dtype=np.uint8)
            img = frame.copy()
            # handler boxes from 2D pose detections
            for tid, rec in p2d.get(str(frame_no), {}).items():
                vrec = rec.get(v)
                if not vrec:
                    continue
                bb = vrec.get("bbox")
                if not bb:
                    continue
                x1, y1, x2, y2 = [int(x) for x in bb]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if not args.handler_only:
                    cv2.putText(
                        img,
                        f"P{tid}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            # ball
            if frame_no in ball_pos:
                p = ball_pos[frame_no]
                cv2.circle(img, (int(p[0]), int(p[1])), 6, (0, 255, 255), -1)
            # action labels
            for a in events_by_frame.get(frame_no, []):
                label = a["type"]
                color = COLORS.get(a["type"], (255, 255, 255))
                cv2.putText(
                    img, label, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3
                )
            # shot result top-right
            for a in events_by_frame.get(frame_no, []):
                if a["type"] in ("shoot", "layup") and a.get("result"):
                    res = a["result"]
                    txt = "MAKE!" if res == "make" else "MISS"
                    c = (0, 255, 0) if res == "make" else (0, 0, 255)
                    cv2.putText(
                        img, txt, (w0 - 260, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 4
                    )
            if args.scale != 1.0:
                img = cv2.resize(img, (out_w, out_h))
            tiles.append(img)
        if len(tiles) == 2:
            canvas = np.hstack(tiles)
        elif len(tiles) == 4:
            top = np.hstack(tiles[:2])
            bot = np.hstack(tiles[2:])
            canvas = np.vstack([top, bot])
        else:
            canvas = tiles[0]
        out.write(canvas)

    for cap in caps.values():
        cap.release()
    out.release()
    print(f"[ok] demo video -> {args.output}")


if __name__ == "__main__":
    main()
