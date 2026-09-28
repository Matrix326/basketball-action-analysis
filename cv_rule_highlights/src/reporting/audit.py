"""Evidence and coverage reporting without any event answer files."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from ..editing.auto_clips import eligible
from ..core.data import save_json


def content_digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def code_digest(package_root):
    """Hash entry points and implementations, excluding tests and run artifacts."""
    root = Path(package_root)
    sources = sorted([*root.glob("*.py"), *(root / "src").rglob("*.py")])
    return content_digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources})


def _baseline_diff(baseline_path, current):
    if not baseline_path or not Path(baseline_path).is_file():
        return {"status": "unavailable"}
    before = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    old = [event for event in before["events"]
           if event["visual_outcome"] == "made" and event["confidence"] == "confirmed"]
    new = [event for event in current
           if event["segment_id"] == "segment_1" and event["type"] == "shot"
           and event["status"] == "confirmed"
           and event["outcome"]["value"] == "made"
           and event["outcome"]["status"] == "confirmed"]
    matched = set()
    pairs = []
    for earlier in old:
        best = min(
            (event for event in new if event["event_id"] not in matched),
            key=lambda event: abs(
                (event["outcome"]["evidence_frame"]
                 if event["outcome"]["evidence_frame"] is not None
                 else event["anchor_frame"])
                - (earlier["resolution_frame"]
                   if earlier["resolution_frame"] is not None
                   else earlier["anchor_frame"])
            ), default=None,
        )
        if best is not None:
            distance = abs(
                (best["outcome"]["evidence_frame"]
                 if best["outcome"]["evidence_frame"] is not None
                 else best["anchor_frame"])
                - (earlier["resolution_frame"]
                   if earlier["resolution_frame"] is not None
                   else earlier["anchor_frame"])
            )
            if distance <= 30:
                matched.add(best["event_id"])
                pairs.append({"before": earlier["id"], "after": best["event_id"],
                              "frame_delta": distance})
    return {
        "status": "compared_to_previous_automatic_output",
        "old_confirmed_makes": len(old), "new_confirmed_makes": len(new),
        "matched": pairs,
        "new_event_ids": [event["event_id"] for event in new
                          if event["event_id"] not in matched],
        "old_unmatched_ids": [event["id"] for event in old
                              if event["id"] not in {p["before"] for p in pairs}],
        "note": "Counts and timestamp matches are not precision or recall.",
    }


def audit(events, events_by_segment, plans, stats, identity, segment_audits,
          baseline_path=None):
    counts = Counter((event["type"], event["status"]) for event in events)
    outcomes = Counter(
        (event["outcome"]["value"], event["outcome"]["status"])
        for event in events if event["type"] == "shot"
    )
    global_plan = plans["global"]
    included = {event_id for clip in global_plan["clips"]
                for event_id in clip["event_ids"]}
    required = {event["event_id"] for event in events if eligible(event)}
    if required - included:
        raise AssertionError("Confirmed highlights absent from global EDL")
    for person, plan in plans["players"].items():
        ids = {event_id for clip in plan["clips"]
               for event_id in clip["event_ids"]}
        if ids != set(plan["selected_event_ids"]):
            raise AssertionError(f"Personal EDL incomplete for {person}")
    return {
        "schema_version": "automatic-audit-1",
        "events_by_type_status": {
            f"{kind}:{status}": count
            for (kind, status), count in sorted(counts.items())
        },
        "shot_outcomes": {
            f"{value}:{status}": count
            for (value, status), count in sorted(outcomes.items())
        },
        "segment_input_coverage": segment_audits,
        "anonymous_players": len(identity["players"]),
        "cross_segment_matched_tracks": sum(
            record["identity_status"] == "cross_segment_matched"
            for record in identity["track_to_player"].values()
        ),
        "unassigned_shots": stats["unassigned"]["shot_attempts_detected"],
        "confirmed_events_eligible": len(required),
        "confirmed_events_in_edl": len(included),
        "global_clips": len(global_plan["clips"]),
        "personal_clips": {
            person: len(plan["clips"]) for person, plan in plans["players"].items()
        },
        "digests": {
            "events": content_digest(events),
            "stats": content_digest(stats),
            "global_edl": content_digest(global_plan),
        },
        "baseline_diff": _baseline_diff(baseline_path, events),
        "claim": "No human event annotations or accuracy/recall estimates were used.",
    }


def write_audit(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "quality_report.json", result)
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    save_json(reports / "baseline_diff.json", result["baseline_diff"])
    lines = [
        "# 自动高光运行报告", "",
        f"匿名人物：{result['anonymous_players']}；确认高光事件：{result['confirmed_events_eligible']}；"
        f"全场片段：{result['global_clips']}。", "",
        "## 事件数量", "",
        *[f"- {name}: {count}" for name, count in result["events_by_type_status"].items()],
        "", "## 投篮结果", "",
        *[f"- {name}: {count}" for name, count in result["shot_outcomes"].items()],
        "", "数量变化不是无真值条件下的准确率或召回率。", "",
    ]
    (reports / "summary.md").write_text("\n".join(lines), encoding="utf-8")
