"""The adapter for perception's 2.0-rfdetr-rtmpose output."""

from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Rules:
    ball_confidence: float = 0.16
    keypoint_confidence: float = 0.35
    hand_distance: float = 0.16  # fraction of image-space person height
    control_seconds: float = 0.10
    control_gap_seconds: float = 0.30
    observation_gap_seconds: float = 0.10
    passage_seconds: float = 0.40
    fusion_seconds: float = 0.20
    visit_gap_seconds: float = 0.45
    flight_seconds: float = 3.0
    min_make_views: int = 1
    trusted_views: tuple[str, ...] = ()
    result_view: str = "view1"
    net_roi: tuple[int, int, int, int] = (750, 195, 835, 280)
    net_motion_ratio: float = 1.8
    net_motion_peak: float = 8.0
    net_motion_seconds: float = 0.5
    pre_seconds: float = 2.5
    post_seconds: float = 1.5
    budget_seconds: float | None = None

    def __post_init__(self):
        numeric = [getattr(self, f.name) for f in fields(self) if f.name not in ("trusted_views", "result_view", "net_roi")]
        if any(value is not None and value <= 0 for value in numeric):
            raise ValueError("All rule thresholds must be positive")
        if self.min_make_views < 1:
            raise ValueError("A confirmed basket requires at least one view")


@dataclass
class Game:
    data: dict
    views: dict
    rules: Rules
    teams: dict
    inside_arc: np.ndarray | None
    fps: float
    start: int
    end: int
    poses_path: Path
    config_path: Path

    @classmethod
    def load(cls, config_path):
        config_path = Path(config_path).resolve()
        config = yaml.safe_load(config_path.read_text())

        def path(value):
            return (config_path.parent / value).resolve()

        poses_path = path(config["poses"])
        with poses_path.open() as stream:
            data = json.load(stream)
        if data["schema_version"] != "2.0-rfdetr-rtmpose":
            raise ValueError("Expected perception schema 2.0-rfdetr-rtmpose")
        fps_values = {float(v["fps"]) for v in data["video_info"].values()}
        if len(fps_values) != 1 or min(fps_values) <= 0:
            raise ValueError("Input views must have the same positive FPS")
        frames = sorted(int(f) for f in data["poses_2d"])
        if not frames:
            raise ValueError("No processed frames in perception output")
        views = {}
        for name, settings in config["views"].items():
            info = data["video_info"][name]
            rim = np.asarray(settings["rim"], dtype=float)
            if rim.shape != (4,) or not np.isfinite(rim).all() or min(rim[2:]) <= 0:
                raise ValueError(f"{name}.rim must be [cx, cy, width, height] in pixels")
            if not (0 <= rim[0] < info["width"] and 0 <= rim[1] < info["height"]):
                raise ValueError(f"{name}.rim lies outside the perception image")
            views[name] = {
                "rim": rim.tolist(),
                "path": str(path(settings["path"])) if "path" in settings else info["path"],
                "frame_zero": int(settings.get("frame_zero", -info["frame_offset"])),
                "width": info["width"],
                "height": info["height"],
            }
        if not views:
            raise ValueError("Configure at least one calibrated rim view")
        polygon = config.get("inside_arc_polygon")
        if polygon is not None:
            polygon = np.asarray(polygon, dtype=np.float32)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError("inside_arc_polygon needs at least three world XY points")
            if not np.isfinite(polygon).all():
                raise ValueError("inside_arc_polygon must be finite")
        return cls(
            data, views, Rules(**config.get("rules", {})),
            {str(k): v for k, v in config.get("teams", {}).items()}, polygon,
            fps_values.pop(), frames[0], frames[-1] + 1, poses_path, config_path,
        )

    def players(self, frame):
        return self.data["poses_2d"].get(str(frame), {})

    def balls(self, frame):
        return {
            view: ball for view, ball in self.data["balls_2d"].get(str(frame), {}).items()
            if view in self.views and ball["confidence"] >= self.rules.ball_confidence
            and np.isfinite(ball["center_xy"]).all()
            and np.isfinite(ball["bbox"]).all()
        }

    def ball_samples(self, view):
        result = []
        for key in sorted(self.data["balls_2d"], key=int):
            ball = self.balls(int(key)).get(view)
            if ball:
                x1, y1, x2, y2 = ball["bbox"]
                result.append([int(key), *ball["center_xy"], ((x2 - x1) + (y2 - y1)) / 4])
        return np.asarray(result, dtype=float).reshape(-1, 4)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
