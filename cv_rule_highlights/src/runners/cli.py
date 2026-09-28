"""Run from the repository root: python -m cv_rule_highlights --help."""

import argparse
import json
from pathlib import Path

from ..editing.clips import plan_clips, preview, render
from ..core.data import Game, save_json
from ..perception.detect import detect
from ..perception.visualize import render_diagnostic


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
    command = commands.add_parser("auto-run", help="Run automatic multi-class highlights without event annotations")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--segments", nargs="+", help="Subset for debugging; default is every configured segment")
    command.add_argument("--reuse-perception", action="store_true")
    command.add_argument("--skip-rim-scan", action="store_true", help="Debug only: skip full-video rim motion proposals")
    command.add_argument("--skip-roi-redetect", action="store_true", help="Debug only: skip frozen-model rim ROI ball re-detection")
    command.add_argument("--skip-render", action="store_true", help="Generate event/stats/EDL outputs only")
    command.add_argument("--size", nargs=2, type=int, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    command = commands.add_parser("auto-stats", help="Recompute statistics from a completed automatic event ledger")
    command.add_argument("--events", required=True, type=Path)
    command.add_argument("--players", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("auto-plan", help="Rebuild the global EDL without GPU inference")
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--events", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("event-clips", help="Export confirmed reels by event type from an existing run")
    command.add_argument("--run", required=True, type=Path)
    command.add_argument("--size", nargs=2, type=int, default=(1280, 720), metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args()
    if args.command == "event-clips":
        from ..editing.event_clips import export_run_event_clips
        print(json.dumps(export_run_event_clips(args.run, tuple(args.size)), ensure_ascii=False))
        return
    if args.command == "auto-run":
        from .auto_run import auto_run
        result = auto_run(args.config, args.output, segments=args.segments,
                          reuse_perception=args.reuse_perception,
                          scan=not args.skip_rim_scan,
                          roi_redetect=not args.skip_roi_redetect,
                          do_render=not args.skip_render,
                          size=tuple(args.size))
        print(json.dumps(result, ensure_ascii=False))
        return
    if args.command == "auto-stats":
        from ..reporting.stats import compute_stats, write_stats
        events = json.loads(args.events.read_text(encoding="utf-8"))
        players = json.loads(args.players.read_text(encoding="utf-8"))
        if args.output.exists():
            raise FileExistsError(args.output)
        result = compute_stats(events, players)
        write_stats(result, args.output)
        print(json.dumps({"players": len(result["players"]),
                          "shot_attempts": result["all_players"]["shot_attempts_detected"]}))
        return
    if args.command == "auto-plan":
        from ..editing.auto_clips import combine_plans
        from ..core.config import load_run_config, write_game_config
        config = load_run_config(args.config)
        events = json.loads(args.events.read_text(encoding="utf-8"))
        groups = {}
        for event in events:
            groups.setdefault(event["segment_id"], []).append(event)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        games = {}
        for segment_id in groups:
            game_path = write_game_config(config, segment_id,
                                     config["segments"][segment_id], args.output)
            games[segment_id] = Game.load(game_path)
        plan = combine_plans(games, groups)
        save_json(args.output / "edit_decision_list.json", plan)
        print(json.dumps({"clips": len(plan["clips"]),
                          "eligible_events": plan["eligible_events"]}))
        return
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
