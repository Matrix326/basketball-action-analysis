"""Deterministic ball-control timeline with explicit unknown and loose states."""

import numpy as np


def _wrist_contact(game, frame, track):
    balls = game.balls(frame)
    observations = game.players(frame).get(str(track), {})
    support = []
    for view, ball in balls.items():
        obs = observations.get(view)
        if not obs:
            continue
        xy = np.asarray(obs["keypoints_xy"], dtype=float)[[9, 10]]
        confidence = np.asarray(obs["keypoints_conf"], dtype=float)[[9, 10]]
        valid = (confidence >= game.rules.keypoint_confidence) & np.isfinite(xy).all(axis=1)
        height = obs["bbox"][3] - obs["bbox"][1]
        if valid.any() and height > 0:
            distance = float(np.linalg.norm(xy[valid] - ball["center_xy"], axis=1).min())
            support.append(distance / height)
    return bool(support and min(support) <= game.rules.hand_distance)


def build_possession(game, controls, shots, segment_id, teams=None):
    """Return compact half-open spans and stable possession IDs.

    An absent ball never creates a hand contact. A short unobserved interval
    inside a stable control span can retain the owner with state ``unknown``.
    Team possession remains unknown until independent team evidence exists.
    """
    owner_at = {}
    possession_at = {}
    team_at = {}
    pass_windows = set()
    teams = teams or {}
    previous = None
    possession_count = 0
    pid = None
    for item in controls:
        side = teams.get(str(item["player_id"]))
        linked = (previous is not None and side is not None
                  and side == teams.get(str(previous["player_id"]))
                  and 0 <= item["start_frame"] - previous["end_frame"]
                  <= round(1.2 * game.fps)
                  and not any(previous["end_frame"] <= shot["anchor_frame"]
                              < item["start_frame"] and
                              shot["outcome"]["value"] == "made"
                              for shot in shots))
        if not linked:
            possession_count += 1
            pid = f"{segment_id}-pos-{possession_count:05d}"
        if linked:
            for frame in range(previous["end_frame"], item["start_frame"]):
                possession_at[frame] = pid
                team_at[frame] = side
                pass_windows.add(frame)
        for frame in range(max(game.start, item["start_frame"]),
                           min(game.end, item["end_frame"])):
            owner_at[frame] = item["player_id"]
            possession_at[frame] = pid
            team_at[frame] = side
        previous = item
    flight = {}
    dead = set()
    for shot in shots:
        start = shot["anchor_frame"]
        end = min(game.end, shot["end_frame"])
        for frame in range(max(game.start, start), end):
            flight[frame] = shot["event_id"]
        if shot["outcome"]["value"] == "made":
            resolved = shot["outcome"].get("evidence_frame")
            if resolved is not None:
                dead.update(range(resolved + 1, min(game.end, resolved + round(game.fps))))
    spans = []
    for frame in range(game.start, game.end):
        owner = owner_at.get(frame)
        visible = bool(game.balls(frame))
        if frame in dead:
            state, reason = "dead_or_reset", "made_basket_reset"
            owner = None
        elif frame in flight:
            state, reason = "shot_flight", "shot_attempt"
            owner = None
        elif frame in pass_windows and visible:
            state, reason = "pass_flight", "same_team_control_transfer"
            owner = None
        elif owner is not None and visible:
            if _wrist_contact(game, frame, owner):
                state, reason = "held", "observed_wrist_ball_contact"
            else:
                state, reason = "dribble", "stable_control_without_contact"
        elif owner is not None:
            state, reason = "unknown", "short_ball_visibility_gap"
        elif visible:
            state, reason = "loose", "observed_ball_without_stable_controller"
        else:
            state, reason = "unknown", "ball_missing"
        pid = possession_at.get(frame) if state in {
            "held", "dribble", "pass_flight", "shot_flight"
        } else None
        side = team_at.get(frame) if pid is not None else None
        key = (state, owner, pid, side, reason)
        if spans and spans[-1]["_key"] == key:
            spans[-1]["end_frame"] = frame + 1
        else:
            spans.append({"start_frame": frame, "end_frame": frame + 1,
                          "state": state, "controller_track_id": owner,
                          "team_possession": side or "unknown", "possession_id": pid,
                          "reason": reason, "_key": key})
    for span in spans:
        del span["_key"]
    return {"schema_version": "possession-1", "segment_id": segment_id,
            "frame_range": [game.start, game.end], "spans": spans}


def possession_at(timeline, frame):
    # The event builder calls this only for a small number of anchors. A
    # compact span scan avoids storing another per-frame copy of the timeline.
    for span in timeline["spans"]:
        if span["start_frame"] <= frame < span["end_frame"]:
            return span
    return None
