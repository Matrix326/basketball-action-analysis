#!/usr/bin/env python
"""One-command RF-DETR multi-view pose, trajectory and visualization pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from rfdetr_pose_multiview import RFDetrPoseMultiViewPipeline
from track.pipeline import run_trajectory_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete four-view RF-DETR pipeline")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--views", nargs="+", help="Process only these configured views")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=None,
        help="Write this run to an isolated output tree instead of the configured output directories.",
    )
    parser.add_argument("--skip-analysis", action="store_true", help="Reuse an existing poses_3d.json")
    parser.add_argument("--skip-smoothing", action="store_true")
    parser.add_argument("--skip-3d-animation", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_base is not None:
        output_base = args.output_base.resolve()
        config = config.override(
            {
                "output.base_dir": str(output_base),
                "output.reid_3d_dir": str(output_base / "poses"),
                "output.pipeline_dir": str(output_base / "trajectory_pipeline"),
                "output.skeleton_dir": str(output_base / "skeletons_3d"),
            }
        )
    start = args.start_frame if args.start_frame is not None else int(config.get("trajectory.start_frame", 0))
    end = args.end_frame
    if args.limit is not None:
        end = start + args.limit
    elif end is None:
        end = start + int(float(config.get("trajectory.process_seconds", 30)) * float(config.get("trajectory.fps", 30)))

    videos = {view: path for view, path in config.video_paths.items() if Path(path).exists()}
    if args.views:
        unknown = set(args.views) - set(config.video_paths)
        if unknown:
            raise SystemExit(f"Unknown views: {sorted(unknown)}")
        videos = {view: videos[view] for view in args.views if view in videos}
    if not args.skip_analysis:
        RFDetrPoseMultiViewPipeline(config).process(
            videos,
            config.get("output.reid_3d_dir"),
            start,
            end,
        )

    process_frames = max(0, end - start)
    fps = float(config.get("trajectory.fps", 30))
    common = {
        "POSES_3D_JSON_PATH": os.path.join(config.get("output.reid_3d_dir"), "poses_3d.json"),
        "COURT_BACKGROUND_PATH": config.get("assets.court_background", ""),
        "PROCESS_SECONDS": process_frames / fps,
        "FPS": fps,
        "SCALE_RATIO": config.get("trajectory.scale_ratio", 50),
        "NUM_PLAYERS": config.get("reid.num_players", 6),
        "GENERATE_VIDEO": config.get("trajectory.generate_video", True),
        "MOVING_AVERAGE_WINDOW": config.get("smoothing.moving_average_window", 20),
        "GAUSSIAN_SIGMA": config.get("smoothing.gaussian_sigma", 1.0),
    }
    video_configs = [
        {
            "INPUT_VIDEO_PATH": path,
            "TARGET_VIEW": view,
            "START_FRAME": start,
        }
        for view, path in videos.items()
    ]
    run_trajectory_pipeline(
        output_root_dir=config.get("output.pipeline_dir"),
        video_configs=video_configs,
        common_config=common,
        enable_smoothing=not args.skip_smoothing,
        app_config=config,
    )

    if not args.skip_3d_animation:
        from generate_reid_3d_multiview import create_multi_view_animation

        poses_path = Path(config.get("output.reid_3d_dir")) / "poses_3d.json"
        with poses_path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        create_multi_view_animation(
            result["poses_3d"],
            config.get("output.skeleton_dir"),
            [tuple(color) for color in config.get("player_colors", [])],
            start_frame=start,
            end_frame=end - 1,
            balls_3d=result.get("balls_3d", {}),
            balls_3d_predicted=result.get("balls_3d_predicted", {}),
            source_fps=fps,
            animation_fps=float(config.get("visualization.skeleton_3d.animation_fps", fps)),
            point_size=max(
                4.0,
                float(config.get("visualization.skeleton_3d.point_radius", 3)) ** 2,
            ),
            line_width=float(config.get("visualization.skeleton_3d.line_width", 3)),
            ball_max_gap_frames=int(
                config.get("visualization.skeleton_3d.ball_max_gap_frames", 6)
            ),
            ball_trail_seconds=float(
                config.get("visualization.skeleton_3d.ball_trail_seconds", 2.0)
            ),
        )


if __name__ == "__main__":
    main()
