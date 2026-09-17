#!/usr/bin/env python
"""Action-recognition module: poses_3d.json -> actions.json (one command).

Pipeline: adapt perception output -> ball trajectory post-processing ->
rule engine -> actions.json (+ per-frame possession).

The input can be either:
  * perception (teammate module) output: schema "2.0-rfdetr-rtmpose" with
    `balls_3d` / `balls_3d_predicted` / `balls_2d` — the adapter builds
    `ball_measurements` and triangulates the hoop from the annotated rim
    pixels through the perception calibration;
  * our own pipeline output: has `ball_measurements` and a sibling
    `hoop_3d.json` — both are reused as-is.

Usage:
    python tools/run_action_pipeline.py \
        --poses <poses_3d.json> --output-dir <dir> \
        [--config config/config.yaml] \
        [--hoop-3d <hoop_3d.json>] \
        [--extrinsics <json>] [--intrinsics <json>]   # for hoop triangulation
        [--skip-ball-trajectory]      # reuse an existing ball_trajectory.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import load_config  # noqa: E402


def run(cmd: list[str | Path]) -> None:
    print("[run]", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--poses", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "config" / "config.yaml"))
    parser.add_argument("--hoop-3d", default=None)
    parser.add_argument("--extrinsics", default=None)
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--skip-ball-trajectory", action="store_true")
    parser.add_argument("--print-stats", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    poses = Path(args.poses)

    # 1) schema bridge: build ball_measurements + hoop in the file's frame
    adapt_cmd = [
        py,
        ROOT / "tools" / "adapt_perception.py",
        "--poses",
        poses,
        "--out-dir",
        out_dir,
    ]
    if args.hoop_3d:
        # explicit hoop: copy it next to the adapted poses
        hoop = json.load(open(args.hoop_3d, encoding="utf-8"))
        adapt_cmd += ["--hoop-center", *[str(v) for v in hoop["hoop_center"]]]
    if args.extrinsics:
        adapt_cmd += ["--extrinsics", args.extrinsics]
    if args.intrinsics:
        adapt_cmd += ["--intrinsics", args.intrinsics]
    run(adapt_cmd)

    _config = load_config(args.config)

    # 2) ball trajectory (ballistic segmentation + states)
    traj_path = out_dir / "ball_trajectory.json"
    if not args.skip_ball_trajectory:
        run(
            [
                py,
                ROOT / "src" / "ball_trajectory" / "run_ball_trajectory.py",
                "--config",
                args.config,
                "--poses-json",
                out_dir / "poses_3d.json",
                "--output",
                traj_path,
            ]
        )

    # 3) rule engine
    actions_path = out_dir / "actions.json"
    run(
        [
            py,
            ROOT / "src" / "action_rules" / "run_action_rules.py",
            "--config",
            args.config,
            "--ball-trajectory",
            traj_path,
            "--poses-json",
            out_dir / "poses_3d.json",
            "--hoop-3d",
            out_dir / "hoop_3d.json",
            "--output",
            actions_path,
        ]
    )

    if args.print_stats:
        result = json.load(open(actions_path, encoding="utf-8"))
        print(json.dumps(result.get("stats", {}), ensure_ascii=False, indent=2))
    print(f"[done] actions -> {actions_path}")


if __name__ == "__main__":
    main()
