"""Recompute visual box scores from unique automatic atom events."""

import csv
from pathlib import Path

from ..core.data import save_json


METRICS = (
    "shot_attempts_detected", "made", "missed", "outcome_unknown",
    "made_inside_arc", "made_outside_arc", "zone_unknown",
    "rebounds_total", "rebounds_offensive", "rebounds_defensive",
    "rebounds_team_unknown", "steals", "blocks", "successful_drives",
    "assist_chains", "passes", "receptions",
)
ROLES = {
    "shot": ("shooter",),
    "rebound": ("rebounder",),
    "steal": ("stealer",),
    "block": ("blocker",),
    "successful_drive": ("driver",),
    "assist_chain": ("passer",),
    "pass": ("passer", "receiver"),
}


def _blank(player_id):
    result = {"player_id": player_id, **{key: 0 for key in METRICS}}
    result.update({
        "resolved_fg_pct": None, "outcome_coverage": None,
        "visual_points": None, "official_points": None,
        "event_ids": [], "unresolved_events": [], "uncertain_counts": {},
    })
    return result


def _increment(row, event, role=None):
    kind = event["type"]
    if kind == "shot":
        row["shot_attempts_detected"] += 1
        outcome = event["outcome"]["value"] if event["outcome"]["status"] == "confirmed" else "unknown"
        key = outcome if outcome in {"made", "missed"} else "outcome_unknown"
        row[key] += 1
        if outcome == "made":
            zone = event.get("shot_zone")
            if zone == "inside_arc":
                row["made_inside_arc"] += 1
            elif zone == "outside_arc":
                row["made_outside_arc"] += 1
            else:
                row["zone_unknown"] += 1
    elif kind == "rebound":
        row["rebounds_total"] += 1
        side = event.get("rebound_side", "team_unknown")
        row[f"rebounds_{side}" if side in {"offensive", "defensive"}
            else "rebounds_team_unknown"] += 1
    elif kind == "pass":
        row["passes" if role == "passer" else "receptions"] += 1
    else:
        metric = {"steal": "steals", "block": "blocks",
                  "successful_drive": "successful_drives",
                  "assist_chain": "assist_chains"}.get(kind)
        if metric:
            row[metric] += 1
    row["event_ids"].append(event["event_id"])


def _finish(row):
    row["event_ids"] = sorted(set(row["event_ids"]))
    resolved = row["made"] + row["missed"]
    attempts = row["shot_attempts_detected"]
    row["resolved_fg_pct"] = round(row["made"] / resolved, 4) if resolved else None
    row["outcome_coverage"] = round(resolved / attempts, 4) if attempts else None
    if attempts != resolved + row["outcome_unknown"]:
        raise AssertionError("Shot outcomes do not partition attempts")
    if row["rebounds_total"] != (
        row["rebounds_offensive"] + row["rebounds_defensive"]
        + row["rebounds_team_unknown"]
    ):
        raise AssertionError("Rebound subtypes do not partition rebounds")


def compute_stats(events, player_ids):
    by_id = {}
    for event in events:
        event_id = event["event_id"]
        if event_id in by_id:
            raise ValueError(f"Duplicate atomic event ID: {event_id}")
        by_id[event_id] = event
    players = {pid: _blank(pid) for pid in sorted(player_ids)}
    totals = _blank("all_players")
    unassigned = _blank("unassigned")
    uncertain = []
    for event in by_id.values():
        if event["status"] != "confirmed" or event["type"] not in ROLES:
            if event["status"] == "uncertain":
                uncertain.append(event["event_id"])
                kind = event["type"]
                totals["uncertain_counts"][kind] = totals["uncertain_counts"].get(kind, 0) + 1
                for role_name in ROLES.get(kind, ()):
                    actor = event["actor_roles"].get(role_name) or {}
                    pid = actor.get("player_id")
                    row = players.get(pid, unassigned)
                    row["uncertain_counts"][kind] = row["uncertain_counts"].get(kind, 0) + 1
            continue
        roles = ROLES[event["type"]]
        for role_name in roles:
            _increment(totals, event, role_name)
            actor = event["actor_roles"].get(role_name) or {}
            pid = actor.get("player_id")
            confirmed_role = (actor.get("role_status") == "confirmed"
                              and actor.get("identity_status") == "confirmed")
            if pid in players and confirmed_role:
                _increment(players[pid], event, role_name)
            else:
                _increment(unassigned, event, role_name)
                unassigned["unresolved_events"].append({
                    "event_id": event["event_id"], "role": role_name,
                    "candidate_player_id": pid,
                    "role_status": actor.get("role_status", "unknown"),
                })
    if isinstance(player_ids, dict):
        for pid, person in player_ids.items():
            players[pid]["tracked_frames"] = sum(t["observed_frames"] for t in person.get("tracks", []))
    for row in [totals, unassigned, *players.values()]:
        _finish(row)
    for metric in METRICS:
        if totals[metric] != unassigned[metric] + sum(
            row[metric] for row in players.values()
        ):
            raise AssertionError(f"Personal totals do not conserve {metric}")
    return {
        "schema_version": "visual-stats-1", "source": "automatic_events",
        "all_players": totals, "players": players,
        "unassigned": unassigned, "uncertain_event_ids": uncertain,
    }


def write_stats(stats, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "all_players.json", stats)
    save_json(output / "unassigned.json", stats["unassigned"])
    players_dir = output / "players"
    players_dir.mkdir(exist_ok=True)
    for pid, row in stats["players"].items():
        save_json(players_dir / f"{pid}.json", row)
    with (output / "all_players.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["player_id", *METRICS, "resolved_fg_pct", "outcome_coverage"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in stats["players"].values():
            writer.writerow({field: row.get(field) for field in fields})
