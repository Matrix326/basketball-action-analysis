"""CLI: run the rule-based action recognition on the pipeline outputs.

Usage:
    python src/action_rules/run_action_rules.py \
        --config config/config.yaml \
        [--ball-trajectory path] [--poses-json path] [--hoop-3d path] [--output path]

Output: output/rfdetr_multiview/poses/actions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from src.action_rules.pipeline import ActionRuleEngine  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-based action recognition")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config" / "config.yaml")
    )
    parser.add_argument("--ball-trajectory", default=None)
    parser.add_argument("--poses-json", default=None)
    parser.add_argument("--hoop-3d", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    poses_dir = Path(config.get("output.reid_3d_dir"))

    ball_traj_path = Path(
        args.ball_trajectory
        or config.get("ball_trajectory.output_path", poses_dir / "ball_trajectory.json")
    )
    poses_path = Path(args.poses_json or poses_dir / "poses_3d.json")
    hoop_path = Path(args.hoop_3d or poses_dir / "hoop_3d.json")
    output_path = Path(args.output or poses_dir / "actions.json")

    engine = ActionRuleEngine(config)
    result = engine.process(ball_traj_path, poses_path, hoop_path, output_path)
    print(json.dumps(result.get("stats", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
