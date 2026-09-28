"""Unlimited confirmed and candidate clip plans for global and personal reels."""

from .shot_selection import clip_boundary, camera_shots


HIGHLIGHT_TYPES = {
    "shot", "rebound", "steal", "block", "successful_drive", "assist_chain",
}
POSITIVE_ROLES = {
    "shot": "shooter", "rebound": "rebounder", "steal": "stealer",
    "block": "blocker", "successful_drive": "driver",
    "assist_chain": "passer",
}


def eligible(event, candidates=False):
    if event["type"] not in HIGHLIGHT_TYPES:
        return False
    if candidates:
        return event["status"] == "uncertain" or (
            event["type"] == "shot" and event["outcome"]["value"] == "unknown"
        )
    if event["status"] != "confirmed":
        return False
    if event["type"] == "shot":
        return (event["outcome"]["value"] == "made"
                and event["outcome"]["status"] == "confirmed")
    return True


def plan_segment(game, segment_id, events, *, player_id=None, candidates=False, review=False):
    chosen = []
    for event in events:
        if not review and not eligible(event, candidates):
            continue
        if player_id is not None:
            role = event["actor_roles"].get(POSITIVE_ROLES[event["type"]]) or {}
            if (role.get("player_id") != player_id
                or role.get("role_status") != "confirmed"
                or role.get("identity_status") != "confirmed"):
                continue
        chosen.append(event)
    controls = getattr(game, 'control_segments', None)
    if controls is None and hasattr(game, 'rules'):
        from ..perception.detect import controls as detect_controls
        from ..events.event_rules import merge_dribble_controls
        controls = merge_dribble_controls(detect_controls(game), game.fps)
        game.control_segments = controls
    controls = controls or []
    intervals = []
    for event in chosen:
        boundary = clip_boundary(game, event, controls)
        if boundary['end'] > boundary['start']:
            intervals.append({'start': boundary['start'], 'end': boundary['end'],
                              'events': [event], 'boundaries': [boundary]})
    intervals.sort(key=lambda item: (item["start"], item["end"]))
    merged = []
    for interval in intervals:
        if not review and merged and interval["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
            merged[-1]["events"].extend(interval["events"])
            merged[-1]["boundaries"].extend(interval["boundaries"])
        else:
            merged.append(interval)
    clips = []
    for interval in merged:
        start, end = interval['start'], interval['end']
        shots, scores = camera_shots(game, start, end, interval['events'])
        source = shots[0]
        clips.append({
            "id": f"{segment_id}_clip_{len(clips) + 1:04d}",
            "segment_id": segment_id,
            "event_ids": [event["event_id"] for event in interval["events"]],
            "critical_frames": [
                event["outcome"].get("evidence_frame")
                if event["outcome"].get("evidence_frame") is not None
                else event["anchor_frame"]
                for event in interval["events"]
            ],
            "view": source["view"], "source": source["source"],
            "sync_start_frame": start, "sync_end_frame": end,
            "source_start_frame": source["source_start_frame"],
            "source_end_frame": source["source_start_frame"] + end - start,
            "shots": shots, "boundaries": interval["boundaries"],
            "view_scores": scores,
        })
    included = {eid for clip in clips for eid in clip["event_ids"]}
    missing = {event["event_id"] for event in chosen} - included
    if missing:
        raise ValueError(f"Selected events are absent from clip plan: {sorted(missing)}")
    for clip in clips:
        if any(not clip["sync_start_frame"] <= frame < clip["sync_end_frame"]
               for frame in clip["critical_frames"]):
            raise ValueError(f"Critical frame lies outside clip {clip['id']}")
    return clips


def combine_plans(games, events_by_segment, *, player_id=None, candidates=False):
    clips = []
    selected = []
    for segment_id, game in games.items():
        events = events_by_segment.get(segment_id, [])
        clips.extend(plan_segment(game, segment_id, events,
                                  player_id=player_id, candidates=candidates))
        selected.extend(event["event_id"] for event in events if eligible(event, candidates)
                        and (player_id is None or (
                            (event["actor_roles"].get(POSITIVE_ROLES[event["type"]]) or {})
                            .get("player_id") == player_id
                            and (event["actor_roles"].get(POSITIVE_ROLES[event["type"]]) or {})
                            .get("role_status") == "confirmed"
                            and (event["actor_roles"].get(POSITIVE_ROLES[event["type"]]) or {})
                            .get("identity_status") == "confirmed"
                        )))
    if len(clips) != len({clip["id"] for clip in clips}):
        raise AssertionError("Duplicate clip ID across segments")
    return {
        "schema_version": "automatic-edl-1", "fps": next(iter(games.values())).fps,
        "mode": "candidates" if candidates else "all_confirmed",
        "player_id": player_id, "budget_seconds": None, "audio": "preserve",
        "eligible_events": len(selected), "selected_event_ids": selected,
        "clips": clips,
    }
