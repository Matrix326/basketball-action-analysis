"""Run from the repository root: python -m highlights --help."""

import argparse
import json
from pathlib import Path

from .clips import plan_clips, preview, render
from .data import Game, save_json
from .detect import detect
from .visualize import render_diagnostic


def main():
    parser = argparse.ArgumentParser(description="Basketball shot detection and highlight editing")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("detect", "run", "preview", "visualize"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path, help="New output directory")
        if name == "preview":
            command.add_argument("--frame", type=int, help="Synchronized frame; default: first analyzed frame")
        elif name == "visualize":
            command.add_argument("--view", default="view1")
            command.add_argument("--scale", type=float, default=0.5)
        else:
            command.add_argument("--review", action="store_true", help="Include misses and unresolved candidates in clips")
        if name == "run":
            command.add_argument("--size", nargs=2, type=int, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    command = commands.add_parser("render")
    command.add_argument("--plan", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path, help="New output directory")
    command.add_argument("--size", nargs=2, type=int, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    if args.command == "render":
        plan = json.loads(args.plan.read_text())
        args.output.mkdir(parents=True)
        print(json.dumps(render(plan, args.output, args.size), ensure_ascii=False))
        return
    game = Game.load(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "preview":
        preview(game, args.output, game.start if args.frame is None else args.frame)
        return
    if args.command == "visualize":
        render_diagnostic(game, args.output / f"{args.view}_ball_rim_diagnostic.mp4", args.view, args.scale)
        return
    report = detect(game)
    plan = plan_clips(game, report, args.review)
    save_json(args.output / "events.json", report)
    save_json(args.output / "edit_decision_list.json", plan)
    summary = {"events": len(report["events"]), "confirmed_makes": sum(e["visual_outcome"] == "made" and e["confidence"] == "confirmed" for e in report["events"]), "planned_clips": len(plan["clips"])}
    if args.command == "run":
        summary["render"] = render(plan, args.output, args.size)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
