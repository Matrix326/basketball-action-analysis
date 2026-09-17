"""Render the view1 ball track and rim geometry for visual debugging."""

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .data import Game


def render_diagnostic(game: Game, output: str, view: str = "view1", scale: float = 0.5):
    if view not in game.views:
        raise ValueError(f"Unknown view: {view}")
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    settings = game.views[view]
    cap = cv2.VideoCapture(settings["path"])
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {settings['path']}")
    width, height = settings["width"], settings["height"]
    size = (round(width * scale), round(height * scale))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), game.fps, size
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output: {output}")
    cx, cy, rw, rh = settings["rim"]
    trail = deque(maxlen=90)
    try:
        for frame in range(game.start, game.end):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode {view} at frame {frame}")
            ball = game.balls(frame).get(view)
            if ball:
                x, y = map(float, ball["center_xy"])
                trail.append((x, y, frame))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            sx, sy = size[0] / width, size[1] / height
            point = lambda x, y: (round(x * sx), round(y * sy))
            rim_center = point(cx, cy)
            rim_size = (max(1, round(rw * sx / 2)), max(1, round(rh * sy / 2)))
            cv2.ellipse(image, rim_center, rim_size, 0, 0, 360, (0, 255, 0), 2)
            cv2.line(image, point(cx - rw / 2, cy), point(cx + rw / 2, cy), (255, 255, 0), 2)
            cv2.line(image, point(cx - rw / 2, cy + rh), point(cx + rw / 2, cy + rh), (255, 120, 0), 2)
            nx1, ny1, nx2, ny2 = game.rules.net_roi
            cv2.polylines(image, [np.array([point(nx1, ny1), point(nx2, ny1), point(nx2 - 12, ny2), point(nx1 + 8, ny2)])], True, (0, 165, 255), 2)
            for i in range(1, len(trail)):
                a, b = trail[i - 1], trail[i]
                color = (0, 180, 255) if b[2] - a[2] <= 1 else (80, 80, 80)
                cv2.line(image, point(a[0], a[1]), point(b[0], b[1]), color, 2)
            if ball:
                x1, y1, x2, y2 = ball["bbox"]
                cv2.rectangle(image, point(x1, y1), point(x2, y2), (0, 0, 255), 2)
                cv2.circle(image, point(*ball["center_xy"]), 5, (0, 0, 255), -1)
                status = f"ball observed conf={ball['confidence']:.2f}"
            else:
                status = "ball missing/predicted"
            cv2.putText(image, f"sync_frame={frame}  {status}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f"rim center=({cx:.1f},{cy:.1f})  band y=[{cy:.1f},{cy + rh:.1f}]", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            writer.write(image)
    finally:
        cap.release()
        writer.release()
