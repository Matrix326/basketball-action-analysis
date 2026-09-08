"""
基于3D骨架的轨迹处理流水线
接口与原项目 pipeline.py 保持一致
"""

import os
import sys
from typing import Dict, List, Optional

from .traj_gen_3d import PlayerTrajectoryTracker3D, batch_process_videos
from .traj_smooth_3d import AdaptiveJumpRemover, MergedAdaptiveJumpRemover

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import Config, load_config


def run_trajectory_pipeline(
    output_root_dir: str,
    video_configs: List[Dict],
    common_config: Optional[Dict] = None,
    enable_smoothing: bool = True,
    app_config: Optional[Config] = None,
) -> Dict:
    """
    运行完整的轨迹处理流水线
    
    Args:
        output_root_dir: 输出根目录
        video_configs: 视频配置列表
        common_config: 公共配置
        enable_smoothing: 是否启用平滑
    
    Returns:
        包含所有输出路径的字典
    """
    common_config = common_config or {}
    
    print("\n" + "=" * 80)
    print("阶段1: 轨迹生成 (traj_gen)")
    print("=" * 80)
    
    traj_gen_outputs = batch_process_videos(
        output_root_dir=output_root_dir,
        video_configs=video_configs,
        common_config=common_config,
        app_config=app_config,
    )
    
    print("\n轨迹生成完成，输出路径：")
    for idx, path in enumerate(traj_gen_outputs, start=1):
        print(f"  视频{idx}: {path}")
    
    if enable_smoothing:
        print("\n" + "=" * 80)
        print("阶段2: 轨迹平滑 (traj_smooth)")
        print("=" * 80)
        
        smoother = AdaptiveJumpRemover(
            traj_gen_paths_list=traj_gen_outputs,
            output_json_name="smooth_traj.json",
            jump_distance_threshold=common_config.get("JUMP_DISTANCE_THRESHOLD", 3.0),
            speed_ratio_threshold=common_config.get("SPEED_RATIO_THRESHOLD", 8.0),
            frame_rate=common_config.get("FPS", 30),
            lookback_frames=common_config.get("LOOKBACK_FRAMES", 15),
            moving_average_window=common_config.get("MOVING_AVERAGE_WINDOW", 20),
            gaussian_sigma=common_config.get("GAUSSIAN_SIGMA", 1.0),
            scale_ratio=common_config.get("SCALE_RATIO", 50),
            court_background_path=common_config.get("COURT_BACKGROUND_PATH"),
        )
        
        smooth_outputs = smoother.process_batch()
        
        print("\n轨迹平滑完成，输出路径：")
        for idx, path in enumerate(smooth_outputs, start=1):
            print(f"  视频{idx}: {path}")
    else:
        smooth_outputs = traj_gen_outputs
    
    return {
        "traj_gen_outputs": traj_gen_outputs,
        "smooth_outputs": smooth_outputs,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="轨迹处理流水线")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    output_root_dir = cfg.get("output.pipeline_dir")
    
    common_config = {
        "POSES_3D_JSON_PATH": os.path.join(cfg.get("output.reid_3d_dir", ""), "poses_3d.json"),
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
    
    results = run_trajectory_pipeline(
        output_root_dir=output_root_dir,
        video_configs=video_configs,
        common_config=common_config,
        enable_smoothing=True,
        app_config=cfg,
    )
    
    print("\n" + "=" * 80)
    print("流水线执行完成！")
    print("=" * 80)
    print(f"轨迹生成输出: {results['traj_gen_outputs']}")
    print(f"轨迹平滑输出: {results['smooth_outputs']}")


if __name__ == "__main__":
    main()
