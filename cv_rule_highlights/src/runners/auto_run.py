"""Standalone no-annotation, no-training three-segment workflow."""

from collections import OrderedDict
from html import escape
import json
from pathlib import Path
import subprocess
import sys
import traceback


from ..reporting.audit import audit, write_audit, content_digest, code_digest
from ..editing.auto_clips import combine_plans
from ..events.auto_events import build_events
from ..perception.ball_evidence import scan_rims
from ..editing.clips import render
from ..core.data import Game, save_json
from ..editing.event_clips import export_event_clips
from ..editing.candidate_review import export_candidate_review
from ..core.config import load_run_config, write_game_config
from ..events.identity import IdentityRegistry
from ..reporting.stats import compute_stats, write_stats
from ..reporting.reports import write_reports


def _ensure_perception(config, segment_id, spec, reuse):
    poses = Path(spec["poses"])
    evidence = Path(spec["ball_evidence"]) if spec.get("ball_evidence") else None
    if poses.is_file() and (evidence is None or evidence.is_file()):
        return "reused"
    if reuse:
        raise FileNotFoundError(
            f"{segment_id}: validated perception cache is missing: "
            f"{poses if not poses.is_file() else evidence}"
        )
    project = Path(config["project_root"])
    output_base = poses.parent.parent
    if output_base.exists() and any(output_base.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite partial perception output: {output_base}"
        )
    command = [
        sys.executable, str(project / "perception/src/run_rfdetr_full_pipeline.py"),
        "--config", spec["perception_config"],
        "--start-frame", "0", "--end-frame", str(spec["frames"]),
        "--output-base", str(output_base),
        "--skip-trajectories", "--skip-3d-animation",
    ]
    subprocess.run(command, cwd=project, check=True)
    if not poses.is_file() or (evidence and not evidence.is_file()):
        raise RuntimeError(f"{segment_id}: perception did not produce required files")
    return "computed"


def _write_event_index(events, output):
    lines = ["<!doctype html><meta charset='utf-8'><title>自动事件索引</title>",
             "<h1>自动事件索引</h1><p>仅来自自动视觉证据。</p>",
             "<table border='1'><tr><th>事件</th><th>段</th><th>帧</th>"
             "<th>类型</th><th>状态</th><th>结果</th><th>人物</th></tr>"]
    for event in events:
        roles = ", ".join(
            f"{name}:{role.get('player_id') or '未知'}"
            for name, role in event["actor_roles"].items() if role
        )
        cells = [
            event["event_id"], event["segment_id"], event["anchor_frame"],
            event["type"], event["status"], event["outcome"]["value"], roles,
        ]
        lines.append("<tr>" + "".join(f"<td>{escape(str(cell))}</td>"
                                       for cell in cells) + "</tr>")
    lines.append("</table>")
    output.mkdir(parents=True, exist_ok=True)
    (output / "event_index.html").write_text("\n".join(lines), encoding="utf-8")


def auto_run(config_path, output, *, segments=None, reuse_perception=False,
             scan=True, roi_redetect=True, do_render=True, size=(1280, 720)):
    config = load_run_config(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output run directory already exists: {output}")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": "automatic-run-1", "config": str(Path(config_path).resolve()),
        "annotation_policy": "forbidden", "training": False,
        "config_digest": content_digest(config),
        "code_digest": code_digest(Path(__file__).resolve().parents[2]),
        "read_inputs": [str(Path(config_path).resolve())],
        "segments": {}, "stages": {}, "status": "running",
    }
    save_json(output / "run_manifest.json", manifest)
    selected = list(segments or config["segments"])
    unknown = set(selected) - set(config["segments"])
    if unknown:
        raise ValueError(f"Unknown segments: {sorted(unknown)}")
    games = OrderedDict()
    registry = IdentityRegistry()
    events_by_segment = {}
    audits = {}
    try:
        for segment_id in selected:
            spec = config["segments"][segment_id]
            state = manifest["segments"][segment_id] = {"status": "running"}
            try:
                state["perception"] = _ensure_perception(
                    config, segment_id, spec, reuse_perception
                )
                manifest["read_inputs"].extend(
                    [spec["poses"], *spec["videos"].values()]
                    + ([spec["ball_evidence"]] if spec.get("ball_evidence") else [])
                    + ([spec["perception_config"]] if spec.get("perception_config") else [])
                )
                game_path = write_game_config(config, segment_id, spec, output)
                game = Game.load(game_path)
                if abs(game.fps - float(config["fps"])) > 1e-6:
                    raise ValueError(f"{segment_id}: FPS mismatch")
                if game.end - game.start != int(spec["frames"]):
                    raise ValueError(f"{segment_id}: incomplete perception frames")
                games[segment_id] = game
                registry.add_game(game, segment_id, output / "identity")
                state["status"] = "perception_loaded"
            except Exception as exc:
                state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                games.pop(segment_id, None)
                save_json(output / "run_manifest.json", manifest)
                continue
            rim_scan = None
            if scan:
                cache = spec.get("rim_scan_cache")
                if cache and Path(cache).is_file():
                    rim_scan = json.loads(Path(cache).read_text(encoding="utf-8"))
                    if rim_scan.get("rims") != config["rims"]:
                        raise ValueError(f"{segment_id}: rim cache uses a different calibration; remove rim_scan_cache to recompute")
                    if any(int(count) != spec["frames"]
                           for count in rim_scan["frames_read"].values()):
                        raise ValueError(f"{segment_id}: incomplete rim scan cache")
                    state["rim_scan"] = "reused"
                    manifest["read_inputs"].append(cache)
                else:
                    rim_scan = scan_rims(game, output / "ball_scans" / segment_id)
                    save_json(output / "ball_scans" / segment_id / "summary.json", rim_scan)
                    state["rim_scan"] = "complete"
                if roi_redetect:
                    if game.ball_evidence is not None:
                        state["roi_redetect"] = "reused"
                    else:
                        from ..perception.roi_redetect import redetect
                        result = redetect(game, rim_scan, spec["perception_config"],
                                          output / "ball_scans" / segment_id)
                        game.attach_ball_evidence(result["sidecar"])
                        save_json(output / "ball_scans" / segment_id / "redetect_summary.json",
                                  result)
                        state["roi_redetect"] = "complete"
                        manifest["read_inputs"].append(result["sidecar"])
                else:
                    state["roi_redetect"] = "skipped_by_request"
            else:
                state["rim_scan"] = "skipped_by_request"
                state["roi_redetect"] = "skipped_by_request"
            events, segment_audit, possession, team_report = build_events(
                game, segment_id, registry, rim_scan
            )
            possession_path = output / "possession"
            possession_path.mkdir(exist_ok=True)
            save_json(possession_path / f"{segment_id}.json", possession)
            team_path = output / "identity"
            save_json(team_path / f"{segment_id}_teams.json", team_report)
            segment_audit["court_geometry"] = game.court
            segment_audit["team_status"] = team_report["status"]
            segment_audit["team_reason"] = team_report["reason"]
            events_by_segment[segment_id] = events
            audits[segment_id] = {
                **segment_audit,
                "rim_scan_frames": rim_scan["frames_read"] if rim_scan else None,
                "rim_scan_visits": len(rim_scan["visits"]) if rim_scan else None,
            }
            save_json(possession_path / f"{segment_id}_controls.json", game.control_segments)
            events_path = output / "events"
            events_path.mkdir(exist_ok=True)
            save_json(events_path / f"{segment_id}.json", events)
            state.update(status="complete", events=len(events))
            save_json(output / "run_manifest.json", manifest)
        if not games:
            raise RuntimeError("No segment has valid perception data")
        identity = registry.export()
        identity_path = output / "identity"
        save_json(identity_path / "players.json", identity["players"])
        save_json(identity_path / "track_to_player.json", identity["track_to_player"])
        manifest["stages"]["identity"] = "complete"
        all_events = [
            event for segment_id in games for event in events_by_segment[segment_id]
        ]
        save_json(output / "events" / "all_events.json", all_events)
        save_json(output / "events" / "uncertain_events.json", [
            event for event in all_events if event["status"] == "uncertain"
            or (event["type"] == "shot"
                and event["outcome"]["value"] == "unknown")
        ])
        manifest["stages"]["events"] = "complete"
        stats = compute_stats(all_events, identity["players"])
        write_stats(stats, output / "stats")
        manifest["stages"]["stats"] = "complete"
        global_plan = combine_plans(games, events_by_segment)
        candidates_plan = combine_plans(games, events_by_segment, candidates=True)
        personal = {
            player_id: combine_plans(games, events_by_segment, player_id=player_id)
            for player_id in identity["players"]
        }
        global_path = output / "confirmed" / "global"
        global_path.mkdir(parents=True, exist_ok=True)
        save_json(global_path / "edit_decision_list.json", global_plan)
        candidates_path = output / "candidates"
        candidates_path.mkdir(exist_ok=True)
        save_json(candidates_path / "edit_decision_list.json", candidates_plan)
        for player_id, plan in personal.items():
            path = output / "confirmed" / "players" / player_id
            path.mkdir(parents=True, exist_ok=True)
            save_json(path / "edit_decision_list.json", plan)
        manifest["stages"]["clip_plans"] = "complete"
        report = audit(all_events, events_by_segment,
                       {"global": global_plan, "players": personal}, stats,
                       identity, audits, config.get("baseline_events"))
        write_audit(report, output)
        _write_event_index(all_events, output / "reports")
        manifest["stages"]["audit"] = "complete"
        if do_render:
            render(global_plan, global_path, size)
            for player_id, plan in personal.items():
                if plan["clips"]:
                    render(plan, output / "confirmed" / "players" / player_id, size)
            render(candidates_plan, candidates_path, size)
            if candidates_plan["clips"]:
                (candidates_path / "highlights.mp4").rename(candidates_path / "candidates.mp4")
            manifest["stages"]["render"] = "complete"
        else:
            manifest["stages"]["render"] = "skipped_by_request"
        export_event_clips(games, events_by_segment, output, do_render=do_render, size=size)
        manifest["stages"]["event_type_clips"] = "complete" if do_render else "planned"
        export_candidate_review(games, events_by_segment, output, do_render=do_render, size=size)
        manifest["stages"]["candidate_review"] = "complete" if do_render else "planned"
        write_reports(all_events, stats, identity["players"], global_plan, personal,
                      candidates_plan, output)
        manifest["stages"]["reports"] = "complete"
        manifest["status"] = (
            "complete" if all(manifest["segments"][s]["status"] == "complete" for s in selected) and scan and roi_redetect
            and do_render else "partial"
        )
        manifest["output"] = str(output)
        manifest["event_count"] = len(all_events)
        manifest["confirmed_highlight_events"] = global_plan["eligible_events"]
        manifest["player_count"] = len(identity["players"])
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        save_json(output / "run_manifest.json", manifest)
