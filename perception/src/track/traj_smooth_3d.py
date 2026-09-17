"""
轨迹平滑模块（与原项目 traj_smooth.py 接口一致）
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import load_config

try:
    from .trajectory_utils import (
        ground_to_court_pixel,
        prepare_court_canvas,
        repair_isolated_jumps,
    )
except ImportError:  # Direct script execution.
    from trajectory_utils import (
        ground_to_court_pixel,
        prepare_court_canvas,
        repair_isolated_jumps,
    )


class MergedAdaptiveJumpRemover:
    """
    合并轨迹的自适应跳变移除器（与原项目接口一致）
    """

    def __init__(
        self,
        input_json_path: str,
        output_json_path: str,
        vis_image_path: Optional[str] = None,
        jump_distance_threshold: float = 3.0,
        speed_ratio_threshold: float = 8.0,
        frame_rate: int = 30,
        lookback_frames: int = 15,
        moving_average_window: int = 20,
        gaussian_sigma: float = 1.0,
        scale_ratio: int = 50,
        court_background_path: Optional[str] = None,
        top_view_width: int = 800,
        top_view_height: int = 1400,
    ):
        self.input_json_path = input_json_path
        self.output_json_path = output_json_path
        self.vis_image_path = vis_image_path
        self.jump_distance_threshold = jump_distance_threshold
        self.speed_ratio_threshold = speed_ratio_threshold
        self.frame_rate = frame_rate
        self.lookback_frames = lookback_frames
        self.moving_average_window = moving_average_window
        self.gaussian_sigma = gaussian_sigma
        self.scale_ratio = scale_ratio
        self.court_background_path = court_background_path
        self.top_view_width = top_view_width
        self.top_view_height = top_view_height

    def calculate_average_speed(self, points, frames, idx):
        if idx < self.lookback_frames:
            return None
        total_dist, total_frames = 0.0, 0
        for i in range(idx - self.lookback_frames, idx):
            if i + 1 >= len(points):
                break
            dist = np.linalg.norm(np.array(points[i + 1]) - np.array(points[i]))
            frame_gap = max(1, frames[i + 1] - frames[i])
            total_dist += dist
            total_frames += frame_gap
        return (
            (total_dist / total_frames) * self.frame_rate if total_frames > 0 else None
        )

    def detect_and_remove_jump(self, points, frames, boxes=None, confs=None):
        points, _ = repair_isolated_jumps(
            points,
            frames,
            jump_distance_threshold=self.jump_distance_threshold,
            speed_ratio_threshold=self.speed_ratio_threshold,
            frame_rate=self.frame_rate,
            lookback_frames=self.lookback_frames,
        )
        frames = list(frames)
        boxes = list(boxes) if boxes else []
        confs = list(confs) if confs else []
        return points, frames, boxes, confs

    def _filter(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        n = len(points)
        if n < 3:
            return points

        xs = np.array([p[0] for p in points], dtype=np.float32)
        ys = np.array([p[1] for p in points], dtype=np.float32)

        if self.moving_average_window > 1 and n >= self.moving_average_window:
            half = self.moving_average_window // 2
            xs_src = xs.copy()
            ys_src = ys.copy()
            for i in range(n):
                left_idx = max(0, i - half)
                r = min(n, i + half + 1)
                xs[i] = xs_src[left_idx:r].mean()
                ys[i] = ys_src[left_idx:r].mean()

        if self.gaussian_sigma > 0:
            radius = int(3 * self.gaussian_sigma)
            xs_g, ys_g = np.zeros(n), np.zeros(n)
            for i in range(n):
                left_idx = max(0, i - radius)
                r = min(n, i + radius + 1)
                idx = np.arange(left_idx, r)
                w = np.exp(-((idx - i) ** 2) / (2 * self.gaussian_sigma**2))
                w /= w.sum()
                xs_g[i] = np.sum(xs[left_idx:r] * w)
                ys_g[i] = np.sum(ys[left_idx:r] * w)
            xs, ys = xs_g, ys_g

        return list(zip(xs.tolist(), ys.tolist()))

    def _is_frame_key(self, key: str) -> bool:
        try:
            int(key)
            return True
        except ValueError:
            return False

    def _extract_trajectory_data(self, traj: Dict) -> Tuple[Dict, Optional[str]]:
        frame_data = {}
        player_id = None

        for key, value in traj.items():
            if key == "player_id":
                player_id = value
            elif self._is_frame_key(key):
                frame_data[key] = value

        return frame_data, player_id

    def _reconstruct_trajectory(
        self, frame_data: Dict, player_id: Optional[str]
    ) -> Dict:
        result = dict(frame_data)
        if player_id is not None:
            result["player_id"] = player_id
        return result

    def _load_bg(self):
        background = None
        if self.court_background_path and os.path.exists(self.court_background_path):
            background = cv2.imread(self.court_background_path)
        return prepare_court_canvas(
            background,
            self.top_view_width,
            self.top_view_height,
            15.0,
            28.0,
            self.scale_ratio,
        )

    def _vis(self, traj):
        if not self.vis_image_path:
            return

        bg, court_scale, offset_x, offset_y = self._load_bg()

        for traj_name, data in traj.items():
            if "player_id" in data:
                continue
            pts = [
                ground_to_court_pixel(
                    v["x"],
                    v["y"],
                    court_width_m=15.0,
                    scale=court_scale,
                    offset_x=offset_x,
                    offset_y=offset_y,
                )
                for v in data.values()
                if isinstance(v, dict) and "x" in v and "y" in v
            ]
            if len(pts) < 2:
                continue
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(bg, [pts], False, (0, 255, 0), 2)
            if len(pts) > 0:
                cv2.putText(
                    bg,
                    traj_name,
                    tuple(pts[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        os.makedirs(os.path.dirname(self.vis_image_path), exist_ok=True)
        cv2.imwrite(self.vis_image_path, bg)

    def run(self) -> str:
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories = data.get("final_merged_finished_trajectories", {})
        if not trajectories:
            print("警告: 未找到final_merged_finished_trajectories字段")
            trajectories = data

        processed_trajectories = {}

        for name, traj in trajectories.items():
            frame_data, player_id = self._extract_trajectory_data(traj)

            if not frame_data:
                print(f"警告: 轨迹 '{name}' 没有帧数据，跳过")
                continue

            frames = sorted(map(int, frame_data.keys()))
            points = [
                (frame_data[str(f)]["x"], frame_data[str(f)]["y"]) for f in frames
            ]
            boxes = [frame_data[str(f)].get("box") for f in frames]
            confs = [frame_data[str(f)].get("confidence") for f in frames]

            points, frames, boxes, confs = self.detect_and_remove_jump(
                points, frames, boxes, confs
            )

            pixel_pts = [
                (x * self.scale_ratio, y * self.scale_ratio) for x, y in points
            ]
            pixel_pts = self._filter(pixel_pts)
            smooth_pts = [
                (x / self.scale_ratio, y / self.scale_ratio) for x, y in pixel_pts
            ]

            new_frame_data = {}
            for f, (x, y), b, c in zip(frames, smooth_pts, boxes, confs):
                original_data = frame_data.get(str(f), {})
                entry = {
                    **original_data,
                    "x": float(x),
                    "y": float(y),
                }
                if c is not None:
                    entry["confidence"] = float(c)
                if b is not None:
                    entry["box"] = b
                new_frame_data[str(f)] = entry

            processed_trajectories[name] = self._reconstruct_trajectory(
                new_frame_data, player_id
            )

        if "final_merged_finished_trajectories" in data:
            data["final_merged_finished_trajectories"] = processed_trajectories
        else:
            data = processed_trajectories

        self._vis(processed_trajectories)

        os.makedirs(os.path.dirname(self.output_json_path), exist_ok=True)
        with open(self.output_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[OK] Saved to {self.output_json_path}")
        return self.output_json_path


class AdaptiveJumpRemover:
    """
    批量轨迹平滑器（与原项目接口一致）
    """

    def __init__(
        self,
        traj_gen_paths_list: List[Optional[str]],
        output_json_name: str = "smooth_traj.json",
        jump_distance_threshold: float = 3.0,
        speed_ratio_threshold: float = 8.0,
        frame_rate: int = 30,
        lookback_frames: int = 15,
        moving_average_window: int = 20,
        gaussian_sigma: float = 1.0,
        scale_ratio: int = 50,
        court_background_path: Optional[str] = None,
    ):
        self.traj_gen_paths_list = traj_gen_paths_list
        self.output_json_name = output_json_name
        self.jump_distance_threshold = jump_distance_threshold
        self.speed_ratio_threshold = speed_ratio_threshold
        self.frame_rate = frame_rate
        self.lookback_frames = lookback_frames
        self.moving_average_window = moving_average_window
        self.gaussian_sigma = gaussian_sigma
        self.scale_ratio = scale_ratio
        self.court_background_path = court_background_path

    def process_batch(self) -> List[Optional[str]]:
        output_paths = []

        for traj_path in self.traj_gen_paths_list:
            if traj_path is None:
                output_paths.append(None)
                continue

            output_dir = os.path.dirname(traj_path)
            output_json_path = os.path.join(output_dir, self.output_json_name)
            vis_image_path = os.path.join(output_dir, "smooth_vis.png")

            smoother = MergedAdaptiveJumpRemover(
                input_json_path=traj_path,
                output_json_path=output_json_path,
                vis_image_path=vis_image_path,
                jump_distance_threshold=self.jump_distance_threshold,
                speed_ratio_threshold=self.speed_ratio_threshold,
                frame_rate=self.frame_rate,
                lookback_frames=self.lookback_frames,
                moving_average_window=self.moving_average_window,
                gaussian_sigma=self.gaussian_sigma,
                scale_ratio=self.scale_ratio,
                court_background_path=self.court_background_path,
            )

            _output_path = smoother.run()
            output_paths.append(output_dir)

        return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="轨迹平滑")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--input", type=str, default=None, help="输入轨迹JSON路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    _input = args.input or os.path.join(
        cfg.get("output.pipeline_dir", ""), "1", "traj_gen", "player_trajectory.json"
    )

    smoother = MergedAdaptiveJumpRemover(
        input_json_path=_input,
        output_json_path="./smooth.json",
        vis_image_path="./smooth.png",
        moving_average_window=cfg.get("smoothing.moving_average_window", 20),
        gaussian_sigma=cfg.get("smoothing.gaussian_sigma", 1.0),
    )
    final_smooth = smoother.run()
