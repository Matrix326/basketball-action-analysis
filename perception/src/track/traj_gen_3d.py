"""
基于3D骨架的轨迹生成模块
接口设计与原项目 traj_gen.py 保持一致
"""

import argparse
from collections import defaultdict
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import (
    gaussian_filter1d as gaussian_filter1d,
    uniform_filter1d as uniform_filter1d,
)
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import Config, load_config

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


class AdaptiveJumpRemover:
    """
    自适应跳变移除器（与原项目 traj_smooth.py 接口一致）
    """

    def __init__(
        self,
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
        points, removed_indices = repair_isolated_jumps(
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
        return points, frames, boxes, confs, removed_indices

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

    def process_trajectory(self, points, frames):
        filtered_points, filtered_frames, _, _, removed_indices = (
            self.detect_and_remove_jump(points, frames)
        )
        pixel_pts = [
            (x * self.scale_ratio, y * self.scale_ratio) for x, y in filtered_points
        ]
        smoothed_pixel_pts = self._filter(pixel_pts)
        smoothed_points = [
            (x / self.scale_ratio, y / self.scale_ratio) for x, y in smoothed_pixel_pts
        ]

        stats = {
            "original_points": len(points),
            "removed_jumps": len(removed_indices),
            "final_points": len(smoothed_points),
            "removal_rate": len(removed_indices) / len(points) * 100
            if len(points) > 0
            else 0,
        }
        return smoothed_points, filtered_frames, stats


class PlayerTrajectoryTracker3D:
    """
    基于3D骨架的轨迹追踪器（与原项目 PlayerTrajectoryTracker 接口一致）
    """

    def __init__(
        self,
        output_root_dir: str = "./",
        video_index: int = 1,
        input_video_path: Optional[str] = None,
        poses_3d_json_path: Optional[str] = None,
        court_background_path: Optional[str] = None,
        start_frame: Optional[int] = None,
        process_seconds: Optional[float] = None,
        fps: Optional[float] = None,
        court_total_x: Optional[float] = None,
        court_total_y: Optional[float] = None,
        scale_ratio: Optional[int] = None,
        target_view: Optional[str] = None,
        num_players: Optional[int] = None,
        generate_video: Optional[bool] = None,
        jump_distance_threshold: Optional[float] = None,
        speed_ratio_threshold: Optional[float] = None,
        lookback_frames: Optional[int] = None,
        moving_average_window: Optional[int] = None,
        gaussian_sigma: Optional[float] = None,
        config: Optional[Dict] = None,
        app_config: Optional[Config] = None,
    ):
        self.video_folder = str(video_index)
        self.output_root = os.path.join(output_root_dir, self.video_folder, "traj_gen")
        self.ensure_dir(self.output_root)

        _cfg = app_config or load_config()
        _videos = _cfg.video_paths
        _first_video = list(_videos.values())[0] if _videos else ""
        default_config: dict[str, Any] = {
            "INPUT_VIDEO_PATH": _first_video,
            "POSES_3D_JSON_PATH": os.path.join(
                _cfg.get("output.reid_3d_dir", ""), "poses_3d.json"
            ),
            "COURT_BACKGROUND_PATH": _cfg.get("assets.court_background", ""),
            "START_FRAME": _cfg.get("trajectory.start_frame", 0),
            "PROCESS_SECONDS": _cfg.get("trajectory.process_seconds", 30),
            "FPS": _cfg.get("trajectory.fps", 30),
            "COURT_TOTAL_X": _cfg.get("trajectory.court_total_x", 15.0),
            "COURT_TOTAL_Y": _cfg.get("trajectory.court_total_y", 28.0),
            "SCALE_RATIO": _cfg.get("trajectory.scale_ratio", 50),
            "TARGET_VIEW": _cfg.get("trajectory.target_view", "view1"),
            "NUM_PLAYERS": _cfg.get("reid.num_players", 6),
            "GENERATE_VIDEO": _cfg.get("trajectory.generate_video", True),
            "JUMP_DISTANCE_THRESHOLD": _cfg.get(
                "smoothing.jump_distance_threshold", 3.0
            ),
            "SPEED_RATIO_THRESHOLD": _cfg.get("smoothing.speed_ratio_threshold", 8.0),
            "LOOKBACK_FRAMES": _cfg.get("smoothing.lookback_frames", 15),
            "MOVING_AVERAGE_WINDOW": _cfg.get("smoothing.moving_average_window", 20),
            "GAUSSIAN_SIGMA": _cfg.get("smoothing.gaussian_sigma", 1.0),
        }
        self._app_config = _cfg
        self.player_colors_bgr = _cfg.player_colors_bgr
        self.skeleton_connections = _cfg.skeleton_connections

        self.config = default_config
        if config is not None:
            self.config.update(config)

        param_mapping = {
            "INPUT_VIDEO_PATH": input_video_path,
            "POSES_3D_JSON_PATH": poses_3d_json_path,
            "COURT_BACKGROUND_PATH": court_background_path,
            "START_FRAME": start_frame,
            "PROCESS_SECONDS": process_seconds,
            "FPS": fps,
            "COURT_TOTAL_X": court_total_x,
            "COURT_TOTAL_Y": court_total_y,
            "SCALE_RATIO": scale_ratio,
            "TARGET_VIEW": target_view,
            "NUM_PLAYERS": num_players,
            "GENERATE_VIDEO": generate_video,
            "JUMP_DISTANCE_THRESHOLD": jump_distance_threshold,
            "SPEED_RATIO_THRESHOLD": speed_ratio_threshold,
            "LOOKBACK_FRAMES": lookback_frames,
            "MOVING_AVERAGE_WINDOW": moving_average_window,
            "GAUSSIAN_SIGMA": gaussian_sigma,
        }
        for key, value in param_mapping.items():
            if value is not None:
                self.config[key] = value

        output_paths = {
            "TRACKING_INFO_JSON": os.path.join(self.output_root, "tracking_info.json"),
            "FINAL_TRAJECTORY_JSON": os.path.join(
                self.output_root, "player_trajectory.json"
            ),
            "OUTPUT_VIDEO_PATH": os.path.join(
                self.output_root, "output_video_final.mp4"
            ),
            "TOPVIEW_VIDEO_PATH": os.path.join(self.output_root, "topview_smooth.mp4"),
        }
        self.config.update(output_paths)

        self.smoother = AdaptiveJumpRemover(
            jump_distance_threshold=self.config["JUMP_DISTANCE_THRESHOLD"],
            speed_ratio_threshold=self.config["SPEED_RATIO_THRESHOLD"],
            frame_rate=self.config["FPS"],
            lookback_frames=self.config["LOOKBACK_FRAMES"],
            moving_average_window=self.config["MOVING_AVERAGE_WINDOW"],
            gaussian_sigma=self.config["GAUSSIAN_SIGMA"],
            scale_ratio=self.config["SCALE_RATIO"],
            court_background_path=self.config["COURT_BACKGROUND_PATH"],
        )

    @staticmethod
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def load_data(self):
        print(f"视频{self.video_folder}：加载3D骨架数据...")
        with open(self.config["POSES_3D_JSON_PATH"], "r") as f:
            data = json.load(f)
        self.poses_3d = data["poses_3d"]
        self.poses_2d = data.get("poses_2d", {})
        # Prefer mask/ankle-derived ground points and fall back to hip centers
        # when a producer does not provide ground_positions_3d.
        self.ground_positions_3d = data.get("ground_positions_3d", {})
        self.balls_3d = data.get("balls_3d", {})
        self.balls_3d_predicted = data.get("balls_3d_predicted", {})
        print(f"视频{self.video_folder}：加载完成，共 {len(self.poses_3d)} 帧")

    def extract_trajectories(self):
        print(f"视频{self.video_folder}：提取轨迹...")
        start_frame = self.config["START_FRAME"]
        end_frame = start_frame + int(
            self.config["PROCESS_SECONDS"] * self.config["FPS"]
        )

        raw_trajectories = defaultdict(list)

        for frame_num in tqdm(range(start_frame, end_frame), desc="提取轨迹"):
            frame_str = str(frame_num)
            if frame_str not in self.poses_3d:
                continue

            for track_id_str, kps_3d in self.poses_3d[frame_str].items():
                track_id = int(track_id_str)
                kps = np.array(kps_3d)
                ground = self.ground_positions_3d.get(frame_str, {}).get(track_id_str)
                if ground is not None and len(ground) >= 2:
                    x, y = float(ground[0]), float(ground[1])
                    raw_trajectories[track_id].append((frame_num, x, y))
                elif len(kps) >= 13:
                    hip = (kps[11] + kps[12]) / 2
                    x, y = hip[0], hip[1]
                    raw_trajectories[track_id].append((frame_num, x, y))

        self.raw_trajectories = dict(raw_trajectories)
        print(
            f"视频{self.video_folder}：提取完成，共 {len(self.raw_trajectories)} 条轨迹"
        )

    def smooth_trajectories(self):
        print(f"视频{self.video_folder}：轨迹平滑处理...")
        smoothed_trajectories = {}
        smoothing_stats = {}

        for track_id, traj_data in self.raw_trajectories.items():
            if len(traj_data) < 3:
                smoothed_trajectories[track_id] = traj_data
                smoothing_stats[track_id] = {
                    "original_points": len(traj_data),
                    "removed_jumps": 0,
                    "final_points": len(traj_data),
                    "removal_rate": 0,
                }
                continue

            frames = [t[0] for t in traj_data]
            points = [(t[1], t[2]) for t in traj_data]

            smoothed_points, smoothed_frames, stats = self.smoother.process_trajectory(
                points, frames
            )

            smoothed_trajectories[track_id] = [
                (frame, x, y) for frame, (x, y) in zip(smoothed_frames, smoothed_points)
            ]
            smoothing_stats[track_id] = stats

            print(
                f"  Player {track_id}: {stats['original_points']}点 -> 移除{stats['removed_jumps']}个跳变 -> {stats['final_points']}点"
            )

        self.smoothed_trajectories = smoothed_trajectories
        self.smoothing_stats = smoothing_stats

    def save_trajectory_json(self):
        output_path = self.config["FINAL_TRAJECTORY_JSON"]

        trajectory_data = {"final_merged_finished_trajectories": {}}

        for track_id, traj_data in self.smoothed_trajectories.items():
            traj_dict = {}
            for frame, x, y in traj_data:
                traj_dict[str(frame)] = {
                    "x": float(x),
                    "y": float(y),
                    "confidence": 1.0,
                }
            trajectory_data["final_merged_finished_trajectories"][
                f"player_{track_id}"
            ] = traj_dict

        with open(output_path, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        print(f"视频{self.video_folder}：轨迹数据已保存至 {output_path}")
        return output_path

    def generate_video(self):
        if not self.config["GENERATE_VIDEO"]:
            print(f"视频{self.video_folder}：未启用视频生成，跳过")
            return

        print(f"视频{self.video_folder}：生成视频...")

        cap = cv2.VideoCapture(self.config["INPUT_VIDEO_PATH"])
        fps = self.config["FPS"]
        w = int(cap.get(3))
        h = int(cap.get(4))

        start_frame = self.config["START_FRAME"]
        end_frame = start_frame + int(self.config["PROCESS_SECONDS"] * fps)
        # Pose JSON uses synchronized frame numbers; only video I/O is offset.
        frame_offset = int(
            self._app_config.get(
                f"camera.frame_offsets.{self.config['TARGET_VIEW']}", 0
            )
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_offset)

        court_bg = cv2.imread(self.config["COURT_BACKGROUND_PATH"])
        topview_w = self._app_config.get("trajectory.topview_width", 800)
        topview_height = self._app_config.get("trajectory.topview_height", 1400)
        court_width = float(self.config["COURT_TOTAL_X"])
        court_length = float(self.config["COURT_TOTAL_Y"])
        court_topview, court_scale, court_offset_x, court_offset_y = (
            prepare_court_canvas(
                court_bg,
                topview_w,
                topview_height,
                court_width,
                court_length,
                float(self.config["SCALE_RATIO"]),
            )
        )

        def ground_to_pixel(x, y):
            return ground_to_court_pixel(
                x,
                y,
                court_width_m=court_width,
                scale=court_scale,
                offset_x=court_offset_x,
                offset_y=court_offset_y,
            )

        output_video_path = self.config["OUTPUT_VIDEO_PATH"]
        topview_video_path = self.config["TOPVIEW_VIDEO_PATH"]
        writer_rgb = cv2.VideoWriter(
            output_video_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h)
        )
        writer_topview = cv2.VideoWriter(
            topview_video_path,
            cv2.VideoWriter.fourcc(*"mp4v"),
            fps,
            (topview_w, topview_height),
        )

        target_view = self.config["TARGET_VIEW"]
        num_players = self.config["NUM_PLAYERS"]
        ball_color = tuple(
            self._app_config.get("visualization.ball_color_bgr", [0, 165, 255])
        )
        ball_tail_frames = max(
            1,
            int(
                round(
                    float(self._app_config.get("trajectory.ball_trail_seconds", 2.0))
                    * fps
                )
            ),
        )
        ball_max_gap_frames = max(
            0, int(self._app_config.get("trajectory.ball_max_gap_frames", 6))
        )

        for frame_num in tqdm(range(start_frame, end_frame), desc="生成视频"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_str = str(frame_num)

            if frame_str in self.poses_2d:
                for track_id_str, views_data in self.poses_2d[frame_str].items():
                    track_id = int(track_id_str)
                    color = self.player_colors_bgr[
                        (track_id - 1) % len(self.player_colors_bgr)
                    ]

                    if target_view in views_data:
                        view_data = views_data[target_view]
                        bbox = view_data["bbox"]
                        kps_xy = np.array(view_data["keypoints_xy"])
                        kps_conf = np.array(view_data["keypoints_conf"])

                        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame,
                            f"P{track_id:02d}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            color,
                            2,
                        )

                        for j, (px, py) in enumerate(kps_xy):
                            if kps_conf[j] >= 0.3:
                                cv2.circle(frame, (int(px), int(py)), 4, color, -1)

                        for i, j in self.skeleton_connections:
                            if kps_conf[i] >= 0.3 and kps_conf[j] >= 0.3:
                                pt1 = (int(kps_xy[i, 0]), int(kps_xy[i, 1]))
                                pt2 = (int(kps_xy[j, 0]), int(kps_xy[j, 1]))
                                cv2.line(frame, pt1, pt2, color, 2)

            writer_rgb.write(frame)

            frame_topview = court_topview.copy()

            for track_id in range(1, num_players + 1):
                color = self.player_colors_bgr[
                    (track_id - 1) % len(self.player_colors_bgr)
                ]
                traj = self.smoothed_trajectories.get(track_id, [])

                history = [(f, x, y) for f, x, y in traj if f <= frame_num]
                if len(history) > 1:
                    history = history[-50:]
                    for k in range(len(history) - 1):
                        _, x1_h, y1_h = history[k]
                        _, x2_h, y2_h = history[k + 1]
                        px1, py1 = ground_to_pixel(x1_h, y1_h)
                        px2, py2 = ground_to_pixel(x2_h, y2_h)
                        if (
                            0 <= px1 < topview_w
                            and 0 <= py1 < topview_height
                            and 0 <= px2 < topview_w
                            and 0 <= py2 < topview_height
                        ):
                            alpha = (k + 1) / len(history)
                            thickness = int(1 + alpha * 2)
                            cv2.line(
                                frame_topview, (px1, py1), (px2, py2), color, thickness
                            )

                current_positions = [(f, x, y) for f, x, y in traj if f == frame_num]
                if current_positions:
                    _, cx, cy = current_positions[0]
                    px, py = ground_to_pixel(cx, cy)
                    if 0 <= px < topview_w and 0 <= py < topview_height:
                        cv2.circle(frame_topview, (px, py), 8, color, -1)
                        cv2.putText(
                            frame_topview,
                            f"{track_id}",
                            (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

            ball_history = []
            for ball_frame in range(
                max(start_frame, frame_num - ball_tail_frames + 1), frame_num + 1
            ):
                point = self.balls_3d.get(str(ball_frame))
                if (
                    point is not None
                    and len(point) >= 2
                    and np.isfinite(point[:2]).all()
                ):
                    ball_history.append((ball_frame, float(point[0]), float(point[1])))
            for index, (first, second) in enumerate(
                zip(ball_history, ball_history[1:])
            ):
                if second[0] - first[0] > ball_max_gap_frames + 1:
                    continue
                start_pixel = ground_to_pixel(first[1], first[2])
                end_pixel = ground_to_pixel(second[1], second[2])
                if all(
                    (
                        0 <= start_pixel[0] < topview_w,
                        0 <= start_pixel[1] < topview_height,
                        0 <= end_pixel[0] < topview_w,
                        0 <= end_pixel[1] < topview_height,
                    )
                ):
                    intensity = 0.25 + 0.75 * (index + 1) / max(
                        len(ball_history) - 1, 1
                    )
                    trail_color = tuple(
                        int(channel * intensity) for channel in ball_color
                    )
                    cv2.line(
                        frame_topview,
                        start_pixel,
                        end_pixel,
                        trail_color,
                        4,
                        cv2.LINE_AA,
                    )
            if ball_history and frame_num - ball_history[-1][0] <= ball_max_gap_frames:
                _, ball_x, ball_y = ball_history[-1]
                ball_pixel = ground_to_pixel(ball_x, ball_y)
                if (
                    0 <= ball_pixel[0] < topview_w
                    and 0 <= ball_pixel[1] < topview_height
                ):
                    cv2.circle(
                        frame_topview, ball_pixel, 9, (245, 245, 245), -1, cv2.LINE_AA
                    )
                    cv2.circle(
                        frame_topview, ball_pixel, 7, ball_color, -1, cv2.LINE_AA
                    )
                    cv2.putText(
                        frame_topview,
                        "BALL",
                        (ball_pixel[0] + 11, ball_pixel[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        ball_color,
                        2,
                        cv2.LINE_AA,
                    )

            writer_topview.write(frame_topview)

        cap.release()
        writer_rgb.release()
        writer_topview.release()

        print(f"视频{self.video_folder}：RGB视频已保存至 {output_video_path}")
        print(f"视频{self.video_folder}：Topview视频已保存至 {topview_video_path}")

    def process(self) -> str:
        print(f"\n=== 开始处理视频{self.video_folder} ===")

        self.load_data()
        self.extract_trajectories()
        self.smooth_trajectories()
        output_path = self.save_trajectory_json()

        if self.config["GENERATE_VIDEO"]:
            self.generate_video()

        print(f"\n=== 视频{self.video_folder}处理完成 ===")
        return output_path


def batch_process_videos(
    output_root_dir: str,
    video_configs: List[Dict],
    common_config: Optional[Dict] = None,
    app_config: Optional[Config] = None,
) -> List[Optional[str]]:
    """
    批量处理多段视频（与原项目接口一致）

    Args:
        output_root_dir: 总输出根路径
        video_configs: 每个视频的专属配置列表
        common_config: 所有视频共用的配置

    Returns:
        每个视频的输出文件路径列表
    """
    common_config = common_config or {}
    video_output_paths = []

    print("\n=== 开始批量处理视频 ===")

    for idx, video_config in enumerate(video_configs, start=1):
        print(f"\n==================== 开始处理第{idx}个视频 ====================")
        try:
            final_config = common_config.copy()
            final_config.update(video_config)

            tracker = PlayerTrajectoryTracker3D(
                output_root_dir=output_root_dir,
                video_index=idx,
                config=final_config,
                app_config=app_config,
            )

            output_path = tracker.process()
            video_output_paths.append(output_path)
            print(f"\n==================== 第{idx}个视频处理完成 ====================")

        except Exception as e:
            print(f"\n==================== 第{idx}个视频处理失败 ====================")
            print(f"错误信息：{e}")
            import traceback

            traceback.print_exc()
            video_output_paths.append(None)

    return video_output_paths


def main():
    parser = argparse.ArgumentParser(description="基于3D骨架的轨迹生成")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_root_dir = cfg.get("output.pipeline_dir")

    common_config = {
        "POSES_3D_JSON_PATH": os.path.join(
            cfg.get("output.reid_3d_dir", ""), "poses_3d.json"
        ),
        "COURT_BACKGROUND_PATH": cfg.get("assets.court_background", ""),
        "PROCESS_SECONDS": cfg.get("trajectory.process_seconds", 30),
        "FPS": cfg.get("trajectory.fps", 30),
        "SCALE_RATIO": cfg.get("trajectory.scale_ratio", 50),
        "TARGET_VIEW": cfg.get("trajectory.target_view", "view1"),
        "NUM_PLAYERS": cfg.get("reid.num_players", 6),
        "GENERATE_VIDEO": cfg.get("trajectory.generate_video", True),
        "MOVING_AVERAGE_WINDOW": cfg.get("smoothing.moving_average_window", 20),
        "GAUSSIAN_SIGMA": cfg.get("smoothing.gaussian_sigma", 1.0),
    }

    video_paths = cfg.video_paths
    video_configs = [
        {
            "INPUT_VIDEO_PATH": path,
            "TARGET_VIEW": view,
            "START_FRAME": cfg.get("trajectory.start_frame", 0),
        }
        for view, path in video_paths.items()
    ]

    video_output_paths = batch_process_videos(
        output_root_dir=output_root_dir,
        video_configs=video_configs,
        common_config=common_config,
        app_config=cfg,
    )

    print("\n=== 批量处理完成 ===")
    print("所有视频的输出路径列表：")
    for idx, path in enumerate(video_output_paths, start=1):
        if path:
            print(f"视频{idx}：{path}")
        else:
            print(f"视频{idx}：处理失败，无输出路径")


if __name__ == "__main__":
    main()
