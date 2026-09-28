"""Evaluate automatic events against the existing highlight intervals.

The annotations are selective highlight clips, so precision is reported only as
an exhaustive-label assumption; recall of annotated clips is the primary metric.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


CATEGORY_KEYWORDS = {
    "made": ("命中", "得分", "得手", "打进", "进球", "两分"),
    "rebound": ("篮板", "补篮"),
    "steal": ("抢断",),
    "block": ("盖帽",),
}

def timestamp(value):
    parts = value.split(":")
    if len(parts) != 3 or not all(re.fullmatch(r"\d+(?:\.\d+)?", p) for p in parts):
        raise ValueError(f"Unsupported timestamp: {value}")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def categories(description):
    return {kind for kind, words in CATEGORY_KEYWORDS.items()
            if any(word in description for word in words)}


def load_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    intervals = []
    for annotation_id, item in data.items():
        description = item["Description"]
        start, end = timestamp(item["StartTime"]), timestamp(item["EndTime"])
        labels = categories(description)
        for kind in labels:
            multiplicity = 3 if kind == "made" and "三次" in description else 1
            intervals.append({"id": annotation_id, "kind": kind, "start": start,
                              "end": end, "description": description,
                              "multiplicity": multiplicity})
    return intervals


def load_predictions(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {kind: [] for kind in CATEGORY_KEYWORDS}
    for event in events:
        if event["type"] == "shot" and event["status"] == "confirmed" and event["outcome"]["value"] == "made":
            result["made"].append(event)
        elif event["type"] in {"rebound", "steal", "block"} and event["status"] == "confirmed":
            result[event["type"]].append(event)
    return result


def match_category(predictions, intervals, fps):
    slots = []
    for interval in intervals:
        slots.extend({"annotation_id": interval["id"], "start": interval["start"],
                      "end": interval["end"], "description": interval["description"]}
                     for _ in range(interval["multiplicity"]))
    used = set()
    matches = []
    for event in sorted(predictions, key=lambda e: e["anchor_frame"]):
        second = event["anchor_frame"] / fps
        choices = [(index, slot) for index, slot in enumerate(slots)
                   if index not in used and slot["start"] <= second <= slot["end"]]
        if not choices:
            continue
        index, slot = min(choices, key=lambda pair: abs(second - (pair[1]["start"] + pair[1]["end"]) / 2))
        used.add(index)
        matches.append({"event_id": event["event_id"], "annotation_id": slot["annotation_id"],
                        "prediction_seconds": round(second, 3),
                        "annotation_interval": [slot["start"], slot["end"]],
                        "description": slot["description"]})
    tp = len(matches)
    fn = len(slots) - tp
    fp = len(predictions) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(slots) if slots else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall else None
    interval_recall = len({m["annotation_id"] for m in matches}) / len({s["annotation_id"] for s in slots}) if slots else None
    return {"annotation_slots": len(slots), "annotated_intervals": len({s["annotation_id"] for s in slots}),
            "predictions": len(predictions), "tp": tp, "fp": fp, "fn": fn,
            "precision_exhaustive_assumption": precision, "recall": recall, "f1": f1,
            "annotated_interval_recall": interval_recall, "matches": matches}


def evaluate(annotation_path, events_path, output, fps=30):
    annotations = load_annotations(annotation_path)
    predictions = load_predictions(events_path)
    by_kind = {}
    for kind in CATEGORY_KEYWORDS:
        by_kind[kind] = match_category(predictions[kind],
                                       [a for a in annotations if a["kind"] == kind], fps)
    total_slots = sum(x["annotation_slots"] for x in by_kind.values())
    total_pred = sum(x["predictions"] for x in by_kind.values())
    total_tp = sum(x["tp"] for x in by_kind.values())
    total_fp = total_pred - total_tp
    total_fn = total_slots - total_tp
    precision = total_tp / total_pred if total_pred else None
    recall = total_tp / total_slots if total_slots else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    report = {"schema_version": "highlight-annotation-evaluation-1",
              "annotation_file": str(Path(annotation_path).resolve()),
              "prediction_file": str(Path(events_path).resolve()), "fps": fps,
              "segment_mapping": "segment_1 -> 1-3v3.json",
              "label_semantics": "selective highlight intervals; not exhaustive event truth",
              "categories": by_kind,
              "micro_average": {"annotation_slots": total_slots, "predictions": total_pred,
                                "tp": total_tp, "fp": total_fp, "fn": total_fn,
                                "precision_exhaustive_assumption": precision,
                                "recall": recall, "f1": f1},
              "interpretation": {
                  "primary": "annotated_interval_recall",
                  "precision_warning": "Because the source annotations select highlights rather than enumerate every rebound or defensive action, unmatched automatic events are not proven false positives. Precision and F1 are conservative values under an exhaustive-label assumption.",
                  "made_multiplicity": "A description containing 连续三次 is counted as three scoring slots; all other scoring clips count as one.",
                  "no_threshold_tuning": True}}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(args.annotations, args.events, args.output)
    for kind, value in report["categories"].items():
        print(kind, value["tp"], value["fp"], value["fn"], value["precision_exhaustive_assumption"], value["recall"], value["f1"])
    print("micro", report["micro_average"])


if __name__ == "__main__":
    main()
