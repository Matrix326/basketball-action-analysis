#!/usr/bin/env python
"""Compare detected actions against the manual ground truth (1-3v3-action.json).

The GT IDs (from ref1) differ from our tracking IDs, so comparison is
type + time-window based: for every GT action in the requested frame range,
report whether we detected a matching action type within a tolerance window.

Usage:
    python src/action_rules/compare_ground_truth.py \
        --gt /data/ljy23/project/stal/1-3v3-action.json \
        --actions output/rfdetr_multiview/poses/actions.json \
        --start-frame 900 --end-frame 1800 --tolerance 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# GT action type -> our event types that match
GT_TO_OURS = {
    "Passing": ["pass"],
    "Passing-Bounce": ["pass"],
    "Dribbling": ["dribble_start"],
    "Shooting": ["shoot", "layup"],
    "Shooting-Jump": ["shoot", "layup"],
    "Shooting-Three-pointer": ["shoot"],
    "Layup": ["layup", "shoot"],
    "Rebound": ["rebound"],
    "Defence": [],
    "Defence-Contest": [],
    "Follow-up": [],
}

OURS_TO_GT = {
    "pass": ["Passing", "Passing-Bounce"],
    "dribble_start": ["Dribbling"],
    "shoot": ["Shooting", "Shooting-Jump", "Shooting-Three-pointer", "Layup"],
    "layup": ["Layup", "Shooting", "Shooting-Jump"],
    "rebound": ["Rebound"],
}


def timecode_to_frame(tc: str, fps: int = 30) -> int:
    hh, mm, ss, ff = (int(x) for x in tc.split(":"))
    return ((hh * 60 + mm) * 60 + ss) * fps + ff


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare actions to ground truth")
    parser.add_argument("--gt", default="/data/ljy23/project/stal/1-3v3-action.json")
    parser.add_argument("--actions", default=str(
        PROJECT_ROOT / "output" / "rfdetr_multiview" / "poses" / "actions.json"
    ))
    parser.add_argument("--start-frame", type=int, default=900)
    parser.add_argument("--end-frame", type=int, default=1800)
    parser.add_argument("--tolerance", type=int, default=40,
                        help="frame tolerance for matching a GT action")
    args = parser.parse_args()

    with open(args.gt, encoding="utf-8") as handle:
        gt_actions = json.load(handle)["Actions"]
    with open(args.actions, encoding="utf-8") as handle:
        ours = json.load(handle)["actions"]

    # index our events by type
    ours_by_type: dict[str, list[dict]] = {}
    for a in ours:
        ours_by_type.setdefault(a["type"], []).append(a)

    print(f"{'GT action':32s} {'GT frames':14s} {'GT ID':6s} | {'matched':30s}")
    print("-" * 100)
    matched_count = 0
    comparable = 0
    for gt in gt_actions:
        start = timecode_to_frame(gt["StartTime"])
        end = timecode_to_frame(gt["EndTime"])
        if end < args.start_frame or start > args.end_frame:
            continue
        comparable += 1
        gt_id = gt["ID"]
        expected_types = GT_TO_OURS.get(gt["Action"], [])
        match = None
        for etype in expected_types:
            for a in ours_by_type.get(etype, []):
                if abs(a["frame"] - start) <= args.tolerance:
                    match = a
                    break
            if match:
                break
        if match:
            matched_count += 1
        label = (f"{match['type']}@{match['frame']} actor={match.get('actor_id')}"
                 if match else "MISSED")
        print(f"{gt['Action']:32s} {start:6d}-{end:<7d} {str(gt_id):6s} | {label:30s}")

    print("-" * 100)
    print(f"GT actions in range: {comparable}, matched: {matched_count} "
          f"({matched_count / max(comparable, 1):.0%})  (tolerance {args.tolerance} frames)")


if __name__ == "__main__":
    main()
