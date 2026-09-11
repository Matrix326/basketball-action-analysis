#!/usr/bin/env python
"""CLI for the ball-trajectory post-processor.

Reads a pipeline poses_3d.json (raw ``ball_measurements`` preferred, ``balls_3d``
fallback), applies kinematic gap filling + RTS smoothing + rule-based state
classification, and writes the interface JSON aligned with skeleton frames.

Usage:
    python src/ball_trajectory/run_ball_trajectory.py \
        --config config/config.yaml \
        [--poses-json output/rfdetr_multiview/poses/poses_3d.json] \
        [--output output/rfdetr_multiview/poses/ball_trajectory.json] \
        [--print-stats]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, SRC_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import load_config  # noqa: E402
from ball_trajectory import BallTrajectoryPostProcessor  # noqa: E402

def main() -> None:
    parser = argparse.ArgumentParser(description="Ball-trajectory post-processing")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Project config (reads ball_trajectory.* section)",
    )
    parser.add_argument(
        "--poses-json",
        default=None,
        help="Input poses_3d.json (default: config ball_trajectory.poses_json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output interface JSON (default: config ball_trajectory.output_path)",
    )
    parser.add_argument("--print-stats", action="store_true", help="Print stats after processing")
    args = parser.parse_args()

    config = load_config(args.config)

    poses_json = Path(
        args.poses_json
        or config.get("ball_trajectory.poses_json", str(PROJECT_ROOT / "output" / "rfdetr_multiview" / "poses" / "poses_3d.json"))
    )
    output_path = Path(
        args.output
        or config.get("ball_trajectory.output_path", str(poses_json.parent / "ball_trajectory.json"))
    )

    if not poses_json.exists():
        raise SystemExit(f"poses_3d.json not found: {poses_json}")

    processor = BallTrajectoryPostProcessor(config)
    result = processor.process(poses_json, output_path)

    if args.print_stats:
        print(json.dumps(result.get("stats", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

