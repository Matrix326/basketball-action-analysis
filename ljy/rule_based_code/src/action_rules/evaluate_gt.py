#!/usr/bin/env python
"""Evaluate detected actions against the manual GT and dump the mismatches
as JSON (which segments are wrong and why).

The GT timecodes are HH:MM:SS:FF @ 30 fps. GT player IDs differ from our
track IDs — pass --id-map (e.g. /data/ljy23/project/ref_ours/id_map.json).

Usage:
    python src/action_rules/evaluate_gt.py \
        --gt <gt.json> --actions <actions.json> \
        --id-map <id_map.json> --start-frame 0 --end-frame 19739 \
        --tolerance 50 --output /tmp/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# GT action type -> our event types that match
GT_TO_OURS = {
    "Passing": ["pass"],
    "Passing-Bounce": ["pass"],
    "Passing-Overhead": ["pass"],
    "Dribbling": ["dribble_start"],
    "Shooting": ["shoot", "layup"],
    "Shooting-Jump": ["shoot", "layup"],
    "Shooting-Three-pointer": ["shoot"],
    "Layup": ["layup", "shoot", "follow_up"],
    "Rebound": ["rebound"],
    "Follow-up": ["follow_up", "shoot", "layup"],
    "Block": ["block"],
    "Defence-Block": ["block"],
    "Defence": [],
    "Defence-Contest": [],
    "Defence-Steal": [],
    "Steal": [],
    "Screen": [],
}
COMPARABLE = {k for k, v in GT_TO_OURS.items() if v}

EXTRA_TYPES = ("pass", "shoot", "layup", "rebound", "follow_up", "block")


def timecode_to_frame(tc: str, fps: int = 30) -> int:
    hh, mm, ss, ff = (int(x) for x in tc.split(":"))
    return ((hh * 60 + mm) * 60 + ss) * fps + ff


def event_matches(a: dict, etype: str, gt_ids: list[str], id_map: dict,
                  three_point: bool = False) -> bool:
    if a["type"] not in (etype if isinstance(etype, tuple) else (etype,)):
        return False
    if three_point and not a.get("three_point"):
        return False
    if id_map:
        tracks = {id_map[i] for i in gt_ids if i in id_map}
        # actor=None means a handoff/low pass without a clean releaser —
        # time+type match is enough for those.
        if tracks and a.get("actor_id") is not None and a.get("actor_id") not in tracks:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate actions vs GT")
    parser.add_argument("--gt", default="")
    parser.add_argument("--actions", default="")
    parser.add_argument("--id-map", default="")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=10**9)
    parser.add_argument("--tolerance", type=int, default=50)
    parser.add_argument("--output", default="/tmp/gt_eval.json")
    args = parser.parse_args()

    id_map: dict[str, int] = {}
    if args.id_map and Path(args.id_map).exists():
        with open(args.id_map, encoding="utf-8") as handle:
            id_map = {str(k): int(v) for k, v in json.load(handle).items()}

    with open(args.gt, encoding="utf-8") as handle:
        gt_actions = json.load(handle)["Actions"]
    with open(args.actions, encoding="utf-8") as handle:
        ours = json.load(handle)["actions"]

    ours_by_type: dict[str, list[dict]] = defaultdict(list)
    for a in ours:
        ours_by_type[a["type"]].append(a)

    comparable = matched = 0
    per_category: dict[str, Counter] = defaultdict(Counter)
    mismatches: list[dict] = []

    for gt in gt_actions:
        start = timecode_to_frame(gt["StartTime"])
        end = timecode_to_frame(gt["EndTime"])
        if end < args.start_frame or start > args.end_frame:
            continue
        action_name = gt["Action"]
        gt_ids = [str(i) for i in gt["ID"]]
        if action_name not in COMPARABLE:
            per_category[action_name]["non_comparable"] += 1
            continue
        if end < start:
            per_category[action_name]["bad_annotation"] += 1
            mismatches.append({
                "gt_action": action_name, "gt_start": start, "gt_end": end,
                "gt_id": gt["ID"], "reason": "bad_annotation (end < start)",
            })
            continue
        comparable += 1
        three_point = action_name == "Shooting-Three-pointer"
        expected = GT_TO_OURS[action_name]
        match, match_d = None, None
        for etype in expected:
            for a in ours_by_type.get(etype, []):
                d = abs(a["frame"] - start)
                if d > args.tolerance:
                    continue
                if not event_matches(a, etype, gt_ids, id_map, three_point):
                    continue
                if match is None or d < match_d:
                    match, match_d = a, d
            if match:
                break
        if match:
            matched += 1
            per_category[action_name]["matched"] += 1
            continue
        per_category[action_name]["missed"] += 1
        nearby = []
        for etype in expected:
            for a in ours_by_type.get(etype, []):
                d = abs(a["frame"] - start)
                if d <= 150:
                    nearby.append({
                        "type": a["type"], "frame": a["frame"],
                        "end_frame": a.get("end_frame"), "actor_id": a.get("actor_id"),
                        "offset": d, "three_point": a.get("three_point", False),
                    })
        nearby.sort(key=lambda x: x["offset"])
        reason = "no_event"
        if nearby:
            best = nearby[0]
            if best["offset"] > args.tolerance:
                reason = f"time_offset_{best['offset']} (> {args.tolerance})"
            elif not event_matches(
                {"type": best["type"], "actor_id": best["actor_id"], "three_point": best["three_point"]},
                best["type"], gt_ids, id_map, three_point,
            ):
                if three_point and not best["three_point"]:
                    reason = "not_three_point"
                else:
                    reason = "wrong_player"
            else:
                reason = "ambiguous"
        mismatches.append({
            "gt_action": action_name, "gt_start": start, "gt_end": end,
            "gt_id": gt["ID"], "reason": reason,
            "nearest_ours": nearby[:3],
        })

    used = set()
    for gt in gt_actions:
        start = timecode_to_frame(gt["StartTime"])
        end = timecode_to_frame(gt["EndTime"])
        action_name = gt["Action"]
        if action_name not in COMPARABLE:
            continue
        gt_ids = [str(i) for i in gt["ID"]]
        three_point = action_name == "Shooting-Three-pointer"
        for etype in GT_TO_OURS[action_name]:
            for a in ours_by_type.get(etype, []):
                if abs(a["frame"] - start) <= args.tolerance and event_matches(a, etype, gt_ids, id_map, three_point):
                    used.add(id(a))
    extras = []
    for a in ours:
        if a["type"] not in EXTRA_TYPES:
            continue
        if id(a) in used:
            continue
        extras.append({
            "type": a["type"], "frame": a["frame"], "end_frame": a.get("end_frame"),
            "actor_id": a.get("actor_id"),
            "result": a.get("result"), "three_point": a.get("three_point", False),
        })

    report = {
        "summary": {
            "gt_comparable": comparable,
            "matched": matched,
            "match_rate": round(matched / max(comparable, 1), 3),
            "tolerance_frames": args.tolerance,
            "by_category": {k: dict(v) for k, v in sorted(per_category.items())},
            "extra_detections": len(extras),
            "mismatches": len(mismatches),
        },
        "mismatches": mismatches,
        "extra_detections": extras,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"matched {matched}/{comparable} ({matched / max(comparable, 1):.0%}) "
          f"| mismatches {len(mismatches)} | extras {len(extras)}")
    print(f"[ok] report -> {args.output}")


if __name__ == "__main__":
    main()
