"""Generate original-frame rim crops for visual geometry and outcome inspection."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..core.config import load_run_config


def contact_sheet(config, segment_id, frames, output):
    spec = config["segments"][segment_id]
    handles = {
        view: cv2.VideoCapture(path)
        for view, path in spec["videos"].items()
    }
    width, height = 560, 560
    rows = []
    try:
        for frame in frames:
            row = []
            for view, cap in handles.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                ok, original = cap.read()
                tile = np.zeros((height, width, 3), np.uint8)
                if ok:
                    cx, cy, rw, rh = config["rims"][view]
                    x1 = max(0, round(cx - 2 * rw))
                    x2 = min(original.shape[1], round(cx + 2 * rw))
                    y1 = max(0, round(cy - 2 * rw))
                    y2 = min(original.shape[0], round(cy + 3 * rw))
                    crop = original[y1:y2, x1:x2].copy()
                    local_x, local_y = round(cx - x1), round(cy - y1)
                    cv2.ellipse(crop, (local_x, local_y),
                                (round(rw / 2), max(1, round(rh / 2))),
                                0, 0, 360, (0, 255, 0), 2)
                    cv2.line(crop, (round(local_x - rw / 2), local_y),
                             (round(local_x + rw / 2), local_y),
                             (255, 255, 0), 1)
                    scale = min(width / max(crop.shape[1], 1),
                                (height - 38) / max(crop.shape[0], 1))
                    resized = cv2.resize(crop,
                                         (round(crop.shape[1] * scale),
                                          round(crop.shape[0] * scale)),
                                         interpolation=cv2.INTER_NEAREST)
                    tile[38:38 + resized.shape[0], :resized.shape[1]] = resized
                cv2.putText(tile, f"{segment_id} {view} frame {frame}",
                            (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (255, 255, 255), 2)
                row.append(tile)
            rows.append(np.hstack(row))
    finally:
        for cap in handles.values():
            cap.release()
    if not rows:
        return None
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), np.vstack(rows)):
        raise RuntimeError(f"Cannot write rim audit image: {target}")
    return str(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--events", type=Path, help="Optional automatic event ledger")
    args = parser.parse_args()
    config = load_run_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    images = []
    for segment_id, spec in config["segments"].items():
        frames = [0, int(spec["frames"]) // 2, int(spec["frames"]) - 1]
        images.append(contact_sheet(config, segment_id, frames,
                                    args.output / f"{segment_id}_geometry.png"))
    if args.events is not None:
        events = json.loads(args.events.read_text(encoding="utf-8"))
        for event in events:
            if not (event["type"] == "shot"
                    and event["outcome"]["value"] == "made"
                    and event["outcome"]["status"] == "confirmed"):
                continue
            frame = event["outcome"]["evidence_frame"]
            if frame is None:
                continue
            segment_id = event["segment_id"]
            last = config["segments"][segment_id]["frames"] - 1
            frames = sorted({max(0, min(last, frame + offset))
                             for offset in (-4, 0, 4)})
            images.append(contact_sheet(config, segment_id, frames,
                                        args.output / "makes"
                                        / f"{event['event_id']}.png"))
    print(json.dumps({"images": len(images), "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
