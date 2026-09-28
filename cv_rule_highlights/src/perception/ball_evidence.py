"""Full-video rim-neighborhood scan for additional, explicitly weak proposals."""

from collections import deque
import json
from pathlib import Path

import cv2
import numpy as np

from ..core.timeline import source_frame


def scan_rims(game, output: Path) -> dict:
    """Scan every source frame; color/motion proposals never prove a basket."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sidecar = output / "rim_scan.jsonl"
    visits = []
    frames_read = {}
    with sidecar.open("w", encoding="utf-8") as stream:
        for view, settings in game.views.items():
            cap = cv2.VideoCapture(settings["path"])
            if not cap.isOpened():
                raise ValueError(f"Cannot open {view}: {settings['path']}")
            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    source_frame(game.start, settings["frame_zero"]))
            cx, cy, width, _ = settings["rim"]
            x1 = max(0, round(cx - 1.6 * width))
            x2 = min(settings["width"], round(cx + 1.6 * width))
            y1 = max(0, round(cy - 2.2 * width))
            y2 = min(settings["height"], round(cy + 2.3 * width))
            previous = None
            baseline = deque(maxlen=90)
            active = None
            count = 0
            try:
                for frame in range(game.start, game.end):
                    ok, image = cap.read()
                    if not ok:
                        break
                    count += 1
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    small = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    if previous is None:
                        previous = gray
                        continue
                    delta = cv2.absdiff(gray, previous)
                    previous = gray
                    motion = float(np.count_nonzero(delta > 18) / delta.size)
                    normal = float(np.median(baseline)) if baseline else .01
                    baseline.append(motion)
                    relative = motion / max(normal, .006)
                    # Local change only discovers a window. It is deliberately
                    # never inserted into observed balls or made-shot evidence.
                    trigger = motion >= .025 and relative >= 1.8
                    if trigger:
                        record = {
                            "schema_version": "rim-scan-1", "view": view,
                            "frame": frame, "motion_fraction": round(motion, 5),
                            "baseline_fraction": round(normal, 5),
                            "relative_motion": round(relative, 3),
                            "method": "rim_region_motion",
                        }
                        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                        if active is None:
                            active = {
                                "view": view, "start_frame": frame,
                                "end_frame": frame + 1, "peak_frame": frame,
                                "peak_relative_motion": relative,
                            }
                        else:
                            active["end_frame"] = frame + 1
                            if relative > active["peak_relative_motion"]:
                                active["peak_relative_motion"] = relative
                                active["peak_frame"] = frame
                    elif active and frame - active["end_frame"] >= round(.20 * game.fps):
                        if active["end_frame"] - active["start_frame"] >= 2:
                            visits.append(active)
                        active = None
                if active and active["end_frame"] - active["start_frame"] >= 2:
                    visits.append(active)
            finally:
                cap.release()
            frames_read[view] = count
    return {
        "schema_version": "rim-scan-1",
        "sidecar": str(sidecar),
        "frames_read": frames_read,
        "rims": {view: settings["rim"] for view, settings in game.views.items()},
        "visits": visits,
    }
