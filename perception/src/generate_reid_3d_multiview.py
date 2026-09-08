#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成多视角3D骨架动画 - 使用新的ReID结果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import load_config
from src.skeleton_3d_utils import draw_ball_trajectory_3d, frame_value

_cfg = load_config()
SKELETON_CONNECTIONS = list(_cfg.skeleton_connections)
PLAYER_COLORS = list(_cfg.player_colors_rgb)


def draw_skeleton_3d(ax, keypoints_3d: np.ndarray, player_color: Tuple[int, int, int],
                     line_width: float = 3.0, point_size: float = 20.0):
    valid_mask = ~np.isnan(keypoints_3d).any(axis=1)
    
    for i in range(len(keypoints_3d)):
        if valid_mask[i]:
            pt = keypoints_3d[i]
            if np.linalg.norm(pt) < 0.1:
                valid_mask[i] = False
    
    line_color = tuple(c / 255.0 for c in player_color)
    
    for idx1, idx2 in SKELETON_CONNECTIONS:
        if not valid_mask[idx1] or not valid_mask[idx2]:
            continue
        
        pt1 = keypoints_3d[idx1]
        pt2 = keypoints_3d[idx2]
        
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 
                color=line_color, linewidth=line_width)
    
    valid_points = keypoints_3d[valid_mask]
    if len(valid_points):
        ax.scatter(
            valid_points[:, 0], valid_points[:, 1], valid_points[:, 2],
            c=[line_color], s=point_size, marker='o', depthshade=True,
        )


def create_multi_view_animation(poses_3d: Dict, output_dir: str, 
                                 player_colors: List[Tuple[int, int, int]],
                                 start_frame: int = 0, end_frame: int = None,
                                 balls_3d: Dict | None = None,
                                 balls_3d_predicted: Dict | None = None,
                                 source_fps: float = 30.0,
                                 animation_fps: float = 30.0,
                                 point_size: float = 9.0,
                                 line_width: float = 3.0,
                                 ball_max_gap_frames: int = 6,
                                 ball_trail_seconds: float = 2.0):
    all_frames = sorted([int(k) for k in poses_3d.keys()])
    
    if end_frame is None:
        end_frame = max(all_frames)
    
    frames_to_process = [f for f in all_frames if start_frame <= f <= end_frame]
    
    if not frames_to_process:
        print("没有找到帧数据")
        return
    
    all_kps = []
    for frame_num in frames_to_process:
        frame_key = str(frame_num)
        if frame_key in poses_3d:
            for gid, kps_list in poses_3d[frame_key].items():
                all_kps.append(np.array(kps_list))
    
    if not all_kps:
        print("没有找到骨架数据")
        return
        
    all_kps = np.concatenate(all_kps)
    valid_kps = all_kps[~np.isnan(all_kps).any(axis=1)]
    
    if len(valid_kps) == 0:
        print("没有有效的骨架点")
        return
    
    x_center = (valid_kps[:, 0].min() + valid_kps[:, 0].max()) / 2
    y_center = (valid_kps[:, 1].min() + valid_kps[:, 1].max()) / 2
    z_center = (valid_kps[:, 2].min() + valid_kps[:, 2].max()) / 2
    
    x_extent = 7.0
    y_extent = 7.5
    z_extent = 2.0
    
    x_range = [x_center - x_extent, x_center + x_extent]
    y_range = [y_center - y_extent, y_center + y_extent]
    z_range = [z_center - z_extent, z_center + z_extent]
    
    views = [
        (30, 45, "Front-Top"),
        (0, 0, "Front"),
        (90, 0, "Top"),
        (30, 135, "Side"),
    ]
    
    print(f"  多视角动画")
    print(f"  显示范围: X±{x_extent:.1f}m, Y±{y_extent:.1f}m, Z±{z_extent:.1f}m")
    print(f"  处理帧数: {len(frames_to_process)}")
    
    balls_3d = balls_3d or {}
    balls_3d_predicted = balls_3d_predicted or {}
    ball_trail_frames = max(1, int(round(ball_trail_seconds * source_fps)))
    os.makedirs(output_dir, exist_ok=True)
    mp4_path = f"{output_dir}/skeletons_3d_multi_view.mp4"
    writer = None
    
    for frame_idx, frame_num in enumerate(frames_to_process):
        if frame_idx % 50 == 0:
            print(f"    处理帧 {frame_idx}/{len(frames_to_process)}...")
        
        fig = plt.figure(figsize=(16, 12), facecolor='black')
        
        frame_poses = frame_value(poses_3d, frame_num, {}) or {}
        for v_idx, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(2, 2, v_idx + 1, projection='3d')
            
            ax.set_facecolor('black')
            
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)
            ax.set_box_aspect((
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                z_range[1] - z_range[0],
            ))
            
            ax.set_xlabel('X (m)', fontsize=10, color='white')
            ax.set_ylabel('Y (m)', fontsize=10, color='white')
            ax.set_zlabel('Z (m)', fontsize=10, color='white')
            ax.set_title(f'{title} - Frame {frame_num}', fontsize=12, color='white')
            
            ax.tick_params(axis='both', which='major', labelsize=8, colors='white')
            
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            
            ax.grid(True, alpha=0.2, color='gray')
            
            for gid, kps_list in frame_poses.items():
                kps_3d = np.array(kps_list)
                player_color = player_colors[(int(gid) - 1) % len(player_colors)]
                draw_skeleton_3d(
                    ax, kps_3d, player_color,
                    line_width=line_width, point_size=point_size,
                )
            draw_ball_trajectory_3d(
                ax, balls_3d, balls_3d_predicted, frame_num,
                ball_trail_frames, ball_max_gap_frames,
            )
            
            ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frame_bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
        if writer is None:
            height, width = frame_bgr.shape[:2]
            writer = cv2.VideoWriter(
                mp4_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                animation_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot create 3D skeleton video: {mp4_path}")
        writer.write(frame_bgr)
        
        plt.close(fig)
    
    if writer is not None:
        writer.release()
    print(f"  ✅ MP4保存完成！")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            pass
    if ffmpeg:
        gif_path = f"{output_dir}/skeletons_3d_multi_view.gif"
        gif_fps = min(float(animation_fps), 10.0)
        print(f"  保存GIF预览: {gif_path}")
        subprocess.run(
            [
                ffmpeg, "-y", "-loglevel", "error", "-i", mp4_path,
                "-vf",
                f"fps={gif_fps:g},scale=960:-1:flags=lanczos,split[s0][s1];"
                "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                "-loop", "0", gif_path,
            ],
            check=True,
        )
        print("  ✅ GIF预览生成完成！")
    else:
        print("  [warn] 未找到 FFmpeg，已保留完整 MP4，跳过 GIF 预览")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="生成多视角3D骨架动画")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    poses_path = os.path.join(cfg.get("output.reid_3d_dir", ""), "poses_3d.json")
    output_dir = cfg.get("output.skeleton_dir")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(poses_path) as f:
        data = json.load(f)
    
    poses_3d = data['poses_3d']
    balls_3d = data.get('balls_3d', {})
    balls_3d_predicted = data.get('balls_3d_predicted', {})
    source_fps = float(cfg.get("trajectory.fps", 30))
    point_radius = float(cfg.get("visualization.skeleton_3d.point_radius", 3))
    
    print("=" * 80)
    print("生成多视角3D骨架动画（使用新的ReID结果）")
    print("=" * 80)
    print("球员颜色:")
    for i, color in enumerate(PLAYER_COLORS):
        print(f"  Player {i+1:02d}: RGB{color}")
    print()
    
    start_frame = int(cfg.get("trajectory.start_frame", 0))
    create_multi_view_animation(
        poses_3d, output_dir, PLAYER_COLORS,
        start_frame=start_frame,
        end_frame=start_frame + int(cfg.get("trajectory.process_seconds", 30) * source_fps) - 1,
        balls_3d=balls_3d,
        balls_3d_predicted=balls_3d_predicted,
        source_fps=source_fps,
        animation_fps=float(cfg.get("visualization.skeleton_3d.animation_fps", source_fps)),
        point_size=max(4.0, point_radius * point_radius),
        line_width=float(cfg.get("visualization.skeleton_3d.line_width", 3)),
        ball_max_gap_frames=int(cfg.get("visualization.skeleton_3d.ball_max_gap_frames", 6)),
        ball_trail_seconds=float(cfg.get("visualization.skeleton_3d.ball_trail_seconds", 2.0)),
    )
    
    print("\n" + "=" * 80)
    print("✅ 动画生成完成！")
    print("=" * 80)
