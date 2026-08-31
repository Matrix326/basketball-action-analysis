#!/usr/bin/env python
"""Compatibility entry point for the RF-DETR multi-view pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
for path in (PROJECT_ROOT, SRC_ROOT, THIRD_PARTY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import load_config
from rfdetr_pipeline.pipeline import RFDetrPoseMultiViewPipeline

__all__ = ["RFDetrPoseMultiViewPipeline", "main"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RF-DETR-Seg 2XL + RTMPose COCO-17 multi-view pipeline"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Convenience limit relative to --start-frame")
    parser.add_argument("--output-dir", default=None, help="Override output.reid_3d_dir for this run")
    parser.add_argument("--views", nargs="+", help="Process only these configured views")
    args = parser.parse_args()

    config = load_config(args.config)
    videos = {view: path for view, path in config.video_paths.items() if Path(path).exists()}
    if args.views:
        unknown = set(args.views) - set(config.video_paths)
        if unknown:
            raise SystemExit(f"Unknown views: {sorted(unknown)}")
        videos = {view: videos[view] for view in args.views if view in videos}
    if len(videos) < 2:
        raise SystemExit(f"Need at least two existing videos; configured: {config.video_paths}")

    start = (
        args.start_frame
        if args.start_frame is not None
        else int(config.get("trajectory.start_frame", 0))
    )
    end = args.end_frame
    if args.limit is not None:
        end = start + args.limit
    elif end is None and config.get("trajectory.process_seconds") is not None:
        end = start + int(
            float(config.get("trajectory.process_seconds"))
            * float(config.get("trajectory.fps", 30))
        )

    output_dir = args.output_dir or config.get("output.reid_3d_dir")
    RFDetrPoseMultiViewPipeline(config).process(videos, output_dir, start, end)


if __name__ == "__main__":
    main()
