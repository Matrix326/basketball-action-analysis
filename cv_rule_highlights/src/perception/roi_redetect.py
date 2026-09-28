"""Frozen RF-DETR ball re-detection near every automatically proposed rim visit.

The full video is scanned by :mod:`ball_evidence`; this stage inspects every
frame of each proposed window. Detections remain candidates until they form a
short, physically plausible image track. No annotation is read here.
"""

from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from ..core.timeline import source_frame


def _windows(game, visits, view):
    intervals = []
    pad_before, pad_after = round(.5 * game.fps), round(.7 * game.fps)
    for visit in visits:
        if visit["view"] == view:
            intervals.append((max(game.start, visit["start_frame"] - pad_before),
                              min(game.end, visit["end_frame"] + pad_after)))
    # Ball arrivals already visible in the old cache are also checked, even
    # when the motion proposal misses the net or the ball is briefly occluded.
    rim = game.views[view]["rim"]
    for frame, x, y, _ in game.ball_samples(view):
        if abs(x - rim[0]) < 1.8 * rim[2] and abs(y - rim[1]) < 2.2 * rim[2]:
            frame = int(frame)
            intervals.append((max(game.start, frame - pad_before),
                              min(game.end, frame + pad_after)))
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        elif end > start:
            merged.append((start, end))
    return merged


def _track_candidates(rows, max_gap=2, max_speed=48):
    """Keep measured detections only when a local track spans three frames."""
    tracks = []
    for frame in sorted(rows):
        options = sorted(rows[frame], key=lambda row: row["confidence"], reverse=True)
        for item in options:
            center = np.asarray(item["center_xy"], dtype=float)
            matches = []
            for index, track in enumerate(tracks):
                previous_frame, previous = track[-1]
                gap = frame - previous_frame
                if not 0 < gap <= max_gap:
                    continue
                distance = float(np.linalg.norm(center - previous["center_xy"]))
                if distance <= max_speed * gap:
                    matches.append((distance / gap, index))
            if matches:
                _, index = min(matches)
                if tracks[index][-1][0] != frame:
                    tracks[index].append((frame, item))
            else:
                tracks.append([(frame, item)])
    selected = {}
    for track in tracks:
        if len(track) < 3 or track[-1][0] - track[0][0] < 2:
            continue
        for frame, item in track:
            if frame not in selected or item["confidence"] > selected[frame]["confidence"]:
                selected[frame] = item
    return selected


def redetect(game, rim_scan, perception_config, output, *, batch_size=8):
    """Write independent 2D ball evidence; return a summary and sidecar path."""
    project = Path(__file__).resolve().parents[3] / "perception"
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    if str(project / "src") not in sys.path:
        sys.path.insert(0, str(project / "src"))
    from config import load_config
    from rfdetr_pipeline.detector import RFDetrSegmenter
    from basketball_repro.inference_runtime import BALL_CLASS_ID

    model = RFDetrSegmenter(load_config(str(perception_config)))
    model.ball_threshold = .08
    model.ball_min_size = 3.0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sidecar = output / "ball_evidence.jsonl"
    rows = defaultdict(lambda: {"candidates": {}, "selected_observed_2d": {}})
    coverage = {}
    for view, settings in game.views.items():
        windows = _windows(game, rim_scan["visits"], view)
        print(f"[rim-roi] {view}: {len(windows)} windows, "
              f"{sum(end - start for start, end in windows)} frames", flush=True)
        coverage[view] = {"windows": len(windows), "frames_attempted": 0,
                          "frames_read": 0, "raw_candidates": 0,
                          "stable_observations": 0}
        cap = cv2.VideoCapture(settings["path"])
        if not cap.isOpened():
            raise ValueError(f"Cannot open source video for {view}")
        cx, cy, width, _ = settings["rim"]
        x1 = max(0, round(cx - 2.2 * width))
        x2 = min(settings["width"], round(cx + 2.2 * width))
        y1 = max(0, round(cy - 2.4 * width))
        y2 = min(settings["height"], round(cy + 3.0 * width))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid rim ROI for {view}")
        candidates = defaultdict(list)
        pending = []

        def flush():
            if not pending:
                return
            predictions = model.predict([crop for _, crop in pending], [None] * len(pending))
            for (frame, _), prediction in zip(pending, predictions):
                found = []
                for box, score, label in zip(prediction.xyxy, prediction.confidence,
                                             prediction.class_id):
                    if int(label) != BALL_CLASS_ID or float(score) < .08:
                        continue
                    box = np.asarray(box, dtype=float)
                    box[[0, 2]] += x1
                    box[[1, 3]] += y1
                    w, h = box[2] - box[0], box[3] - box[1]
                    if not (3 <= w <= max(35, .85 * width) and
                            3 <= h <= max(35, .85 * width) and .4 <= w / h <= 2.5):
                        continue
                    item = {"bbox": box.tolist(),
                            "center_xy": [float((box[0] + box[2]) / 2),
                                          float((box[1] + box[3]) / 2)],
                            "confidence": float(score), "source": "frozen_rfdetr_rim_roi"}
                    found.append(item)
                found.sort(key=lambda item: item["confidence"], reverse=True)
                candidates[frame] = found[:5]
                rows[frame]["candidates"][view] = found[:5]
                coverage[view]["raw_candidates"] += len(found[:5])
            pending.clear()

        try:
            for start, end in windows:
                local = source_frame(start, settings["frame_zero"])
                if local < 0:
                    start -= local
                    local = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, local)
                for frame in range(start, end):
                    coverage[view]["frames_attempted"] += 1
                    ok, image = cap.read()
                    if not ok:
                        break
                    coverage[view]["frames_read"] += 1
                    pending.append((frame, image[y1:y2, x1:x2]))
                    if len(pending) >= batch_size:
                        flush()
            flush()
        finally:
            cap.release()
        selected = _track_candidates(candidates)
        for frame, ball in selected.items():
            # Preserve the old selected observation. The ROI result fills gaps;
            # it cannot silently replace a different upstream association.
            if view not in game.balls(frame):
                rows[frame]["selected_observed_2d"][view] = ball
                coverage[view]["stable_observations"] += 1
        print(f"[rim-roi] {view}: {coverage[view]['frames_read']} read, "
              f"{coverage[view]['stable_observations']} gap observations", flush=True)
    with sidecar.open("w", encoding="utf-8") as stream:
        for frame in sorted(rows):
            stream.write(json.dumps({"schema_version": "ball-evidence-1", "frame": frame,
                                     **rows[frame]}, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
    return {"schema_version": "rim-roi-redetect-1", "sidecar": str(sidecar),
            "views": coverage, "model": model.name,
            "rims": {view: settings["rim"] for view, settings in game.views.items()},
            "frame_range": [game.start, game.end]}
