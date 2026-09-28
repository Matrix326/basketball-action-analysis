"""Rule events derived solely from automatic perception and image evidence."""

from collections import defaultdict

import numpy as np

from ..perception.detect import detect
from .possession import build_possession, possession_at
from .team import infer_teams
from .event_rules import contact_views, trajectory_turn, drive_evidence, opposed, merge_dribble_controls, steal_evidence, rebound_acquisition


def _role(registry, segment_id, track_id, quality="probable", frame=None):
    player_id = registry.resolve(segment_id, track_id, frame)
    return {
        "player_id": player_id, "track_id": track_id,
        "identity_status": registry.status(segment_id, track_id)
        if player_id else "unknown",
        "role_status": quality if player_id else "unknown",
    }


def _event(segment_id, kind, ordinal, start, anchor, end, roles,
           status="confirmed", outcome=None, evidence=None, reasons=None):
    start, anchor, end = int(start), int(anchor), int(end)
    start = min(start, anchor)
    end = max(end, anchor + 1)
    return {
        "schema_version": "1.0-auto-events",
        "event_id": f"{segment_id}-{kind}-{ordinal:05d}",
        "segment_id": segment_id, "type": kind,
        "start_frame": start, "anchor_frame": anchor, "end_frame": end,
        "status": status,
        "outcome": outcome or {"value": "unknown", "status": "unknown",
                               "evidence_frame": None},
        "actor_roles": roles, "possession_id": None,
        "related_event_ids": [], "source": "automatic",
        "evidence": evidence or [], "uncertainty_reasons": reasons or [],
        "scoring_validity": "unknown", "awarded_points": None,
    }


def _track_xy(game, frame, track, view):
    obs = game.players(frame).get(str(track), {}).get(view)
    if obs is None:
        return None
    box = obs["bbox"]
    return np.asarray(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))


def _transfer_evidence(game, a, b):
    """Require a continuous observed ball path from A's hands to B's hands."""
    start = max(game.start, a["end_frame"] - round(.2 * game.fps))
    end = min(game.end, b["start_frame"] + round(.2 * game.fps))
    for view in game.views:
        seen = []
        for frame in range(start, end):
            ball = game.balls(frame).get(view)
            if ball is not None:
                seen.append((frame, ball["center_xy"]))
        if len(seen) < 3 or seen[-1][0] - seen[0][0] < 3:
            continue
        if max(np.diff([frame for frame, _ in seen])) > round(.2 * game.fps):
            continue
        first = np.asarray(seen[0][1], dtype=float)
        last = np.asarray(seen[-1][1], dtype=float)
        if np.linalg.norm(last - first) < 25:
            continue
        sender = game.players(seen[0][0]).get(str(a["player_id"]), {}).get(view)
        receiver = game.players(seen[-1][0]).get(str(b["player_id"]), {}).get(view)
        if not sender or not receiver:
            continue
        def near_hand(point, observation):
            xy = np.asarray(observation["keypoints_xy"], dtype=float)[[9, 10]]
            conf = np.asarray(observation["keypoints_conf"])[[9, 10]]
            valid = (conf >= game.rules.keypoint_confidence) & np.isfinite(xy).all(axis=1)
            height = observation["bbox"][3] - observation["bbox"][1]
            return bool(valid.any() and height > 0 and
                        np.linalg.norm(xy[valid] - point, axis=1).min() < .3 * height)
        if near_hand(first, sender) and near_hand(last, receiver):
            return {"view": view, "first_frame": seen[0][0],
                    "last_frame": seen[-1][0], "observed_frames": len(seen),
                    "method": "observed_hand_to_hand_ball_path"}
    return None


def _block_candidate(game, shot, registry, segment_id, teams):
    """Require defensive contact plus a time-aligned deflection in distinct views."""
    anchor = shot["anchor_frame"]
    shooter_role = shot["actor_roles"].get("shooter") or {}
    shooter = shooter_role.get("track_id")
    candidates = []
    for frame in range(max(game.start, anchor - 8), min(game.end, anchor + round(.8 * game.fps))):
        for track in game.players(frame):
            if str(track) == str(shooter):
                continue
            contacts = contact_views(game, frame, track)
            evidence = []
            for contact in contacts:
                view = contact["view"]
                ball = game.balls(frame)[view]
                rim = game.views[view]["rim"]
                if np.linalg.norm(np.asarray(ball["center_xy"]) - rim[:2]) < .7 * rim[2]:
                    continue
                obs = game.players(frame).get(str(shooter), {}).get(view)
                if obs is None:
                    continue
                shoulders = np.asarray(obs["keypoints_xy"])[[5, 6]]
                if ball["center_xy"][1] > float(np.mean(shoulders[:, 1])) + contact["ball_radius_px"]:
                    continue
                turn = trajectory_turn(game, frame, view)
                if turn:
                    evidence.append({**contact, **turn})
            if evidence:
                candidates.append((len(evidence), frame, track, evidence))
    if not candidates:
        return None
    count, frame, track, evidence = max(candidates, key=lambda item: (item[0], -abs(item[1] - anchor)))
    strong = count >= 2 and opposed(teams, shooter, track) and shooter_role.get("role_status") == "confirmed"
    made = shot["outcome"]["value"] == "made"
    status = "rejected" if made else "confirmed" if strong else "uncertain"
    return _event(segment_id, "block", 0, max(game.start, frame - 12), frame, min(game.end, frame + 16),
                  {"blocker": _role(registry, segment_id, track, "confirmed" if strong else "probable", frame),
                   "shooter": shooter_role}, status, evidence=evidence,
                  reasons=["SHOT_SCORED_AFTER_CONTACT"] if made else [] if strong else ["DEFENSIVE_CONTACT_UNVERIFIED"])

def _image_drive_candidate(game, segment, registry, segment_id):
    """Detect a real advance past a pre-existing defender in image geometry."""
    duration = segment["end_frame"] - segment["start_frame"]
    if duration < round(.65 * game.fps):
        return None
    start = segment["start_frame"]
    end = segment["end_frame"] - 1
    actor = segment["player_id"]
    support = []
    defenders = defaultdict(int)
    for view in game.views:
        before = _track_xy(game, start, actor, view)
        after = _track_xy(game, end, actor, view)
        if before is None or after is None:
            continue
        rim = np.asarray(game.views[view]["rim"][:2])
        width = game.views[view]["rim"][2]
        progress = np.linalg.norm(before - rim) - np.linalg.norm(after - rim)
        if progress < .55 * width:
            continue
        for track in game.players(start):
            if track == actor:
                continue
            defender = _track_xy(game, start, track, view)
            defender_end = _track_xy(game, end, track, view)
            if defender is None or defender_end is None:
                continue
            actor_distance = np.linalg.norm(before - rim)
            between = np.linalg.norm(defender - rim) < actor_distance
            near = np.linalg.norm(defender - before) < 1.8 * width
            crossed = np.linalg.norm(after - rim) + .3 * width < np.linalg.norm(defender_end - rim)
            if between and near and crossed:
                defenders[track] += 1
                support.append({"view": view, "progress_px": round(float(progress), 1),
                                "defender_track": track})
                break
    if not support:
        return None
    defender = max(defenders, key=defenders.get)
    # Image distance to the rim cannot establish ground-plane advantage or
    # whether the opposing player was defending. Keep this as a proposal.
    strong = False
    return _event(segment_id, "successful_drive", 0, start, end, end + 1,
                  {"driver": _role(registry, segment_id, actor, "confirmed"),
                   "defender": _role(registry, segment_id, defender, "probable")},
                  "confirmed" if strong else "uncertain", evidence=support,
                  reasons=[] if strong else ["GROUND_DEFENDER_RELATION_UNVERIFIED"])


def _drive_candidate(game, segment, registry, segment_id, teams):
    proof = drive_evidence(game, segment, teams)
    if proof is None:
        return _image_drive_candidate(game, segment, registry, segment_id)
    start, end = proof["start_frame"], proof["end_frame"]
    return _event(segment_id, "successful_drive", 0, start, end - 1, end,
                  {"driver": _role(registry, segment_id, segment["player_id"], "confirmed", start),
                   "defender": _role(registry, segment_id, proof["defender"], "confirmed", start)},
                  "confirmed" if proof["confirmed"] else "uncertain", evidence=[proof],
                  reasons=[] if proof["confirmed"] else ["TEAM_RELATION_UNKNOWN"])


def build_events(game, segment_id, registry, rim_scan=None):
    legacy = detect(game)
    events = []
    shots = []
    for index, original in enumerate(legacy["events"], 1):
        anchor = original["anchor_frame"]
        resolution = original["resolution_frame"]
        end = original["rim_window"][1] if original["rim_window"] else anchor + round(game.fps)
        if resolution is not None:
            end = max(end, resolution + 1)
        outcome = original["visual_outcome"]
        outcome_status = "confirmed" if original["confidence"] == "confirmed" else "unknown"
        existence_confirmed = bool((original["release_views"] and original["rim_window"] is not None) or
                                   (outcome == "made" and outcome_status == "confirmed"))
        roles = {"shooter": _role(registry, segment_id, original["actor_id"],
                                  "confirmed" if original["release_views"] else "probable")}
        shot = _event(
            segment_id, "shot", index, anchor, anchor, end, roles,
            "confirmed" if existence_confirmed else "uncertain",
            {"value": outcome, "status": outcome_status, "evidence_frame": resolution},
            original["evidence"],
            ([] if outcome != "unknown" else ["SHOT_OUTCOME_UNKNOWN"])
            + ([] if existence_confirmed else ["SHOT_EXISTENCE_UNVERIFIED"]),
        )
        shot['rim_window'] = original['rim_window']
        # Requested binary result policy applies only to established attempts
        # with a completed rim visit, never to motion-only or truncated proposals.
        if (existence_confirmed and outcome != 'made' and original['rim_window']
                and original['rim_window'][1] + round(.4 * game.fps) < game.end):
            crossings = [p['frame'] for p in original['evidence'] if 'frame' in p]
            result_frame = resolution if resolution is not None else round(float(np.median(crossings)))
            shot['outcome'] = {'value': 'missed', 'status': 'confirmed',
                               'evidence_frame': result_frame,
                               'method': 'completed_attempt_without_confirmed_make'}
            shot['uncertainty_reasons'] = []
        if shot['outcome']['evidence_frame'] is not None:
            shot['end_frame'] = max(anchor + 1, shot['outcome']['evidence_frame'] + 1)
        shot["shot_zone"] = original["shot_zone"]
        shot["legacy_event_id"] = original["id"]
        events.append(shot)
        shots.append(shot)

    original_controls = merge_dribble_controls(legacy["controls"], game.fps)
    game.control_segments = original_controls
    # Repeated hand-to-hand paths provide weak cooperation evidence. A single
    # controller switch is never sufficient to declare a pass or steal.
    raw_transfers = []
    for a, b in zip(original_controls, original_controls[1:]):
        if a["player_id"] == b["player_id"]:
            continue
        gap = b["start_frame"] - a["end_frame"]
        if gap < -round(.2 * game.fps) or gap > round(1.2 * game.fps):
            continue
        evidence = _transfer_evidence(game, a, b)
        if evidence is None:
            continue
        raw_transfers.append((a, b, evidence))
    team_report = infer_teams(
        [(a["player_id"], b["player_id"]) for a, b, _ in raw_transfers],
        [key.split(":", 1)[1] for key in registry.track_to_player
         if key.startswith(f"{segment_id}:")],
    )
    sides = team_report["teams"]
    transfers = []
    for a, b, evidence in raw_transfers:
        frame = b["start_frame"]
        cooperative = (sides.get(a["player_id"]) is not None
                       and sides.get(a["player_id"]) == sides.get(b["player_id"]))
        transfer = _event(segment_id, "pass" if cooperative else "ball_transfer",
                          len(transfers) + 1,
                          a["end_frame"] - 1, frame, frame + 1,
                          {"passer" if cooperative else "sender":
                           _role(registry, segment_id, a["player_id"], "confirmed"),
                           "receiver": _role(registry, segment_id, b["player_id"],
                                             "confirmed")},
                          "confirmed" if cooperative else "uncertain",
                          evidence=[evidence],
                          reasons=[] if cooperative else ["TEAM_RELATION_UNKNOWN"])
        transfers.append(transfer)
        events.append(transfer)
    # A missed attempt can produce one acquisition, including long rebounds.
    used_acquisitions = set()
    for shot in sorted(shots, key=lambda s: (s['status'] != 'confirmed', -s['anchor_frame'])):
        if shot['outcome']['value'] == 'made' or shot.get('rim_window') is None:
            continue
        resolved = shot['outcome'].get('evidence_frame')
        if resolved is None:
            crossings = [p['frame'] for p in shot['evidence'] if 'frame' in p]
            if not crossings:
                continue
            resolved = round(float(np.median(crossings)))
        next_shot = min((s['anchor_frame'] for s in shots if s is not shot
                         and s['status'] == 'confirmed' and s['anchor_frame'] > resolved), default=game.end)
        options = [c for c in original_controls if resolved < c['start_frame'] < next_shot
                   and c['start_frame'] <= resolved + round(5 * game.fps)]
        if not options:
            continue
        acquisition_proof = rebound_acquisition(game, options)
        acquisition = acquisition_proof['control']
        frame, track = acquisition['start_frame'], acquisition['player_id']
        key = (frame, track)
        if key in used_acquisitions:
            continue
        used_acquisitions.add(key)
        contacts = acquisition_proof['contacts']
        strong = (shot['status'] == 'confirmed' and shot['outcome']['value'] == 'missed'
                  and acquisition_proof['confirmed'])
        reasons = [] if strong else ['MISS_OR_STABLE_ACQUISITION_UNVERIFIED']
        rebound = _event(segment_id, 'rebound', len(events) + 1, resolved, frame,
                         min(acquisition['end_frame'], frame + round(.2 * game.fps)),
                         {'rebounder': _role(registry, segment_id, track, 'confirmed' if strong else 'probable', frame)},
                         'confirmed' if strong else 'uncertain',
                         evidence=[{'method': 'post_attempt_acquisition', 'miss_event_id': shot['event_id'],
                                    'acquisition_frame': frame, 'control_end_frame': acquisition['end_frame'],
                                    'contacts': contacts, 'control_evidence': acquisition_proof['method']}], reasons=reasons)
        rebound['related_event_ids'] = [shot['event_id']]
        shooter = shot['actor_roles'].get('shooter', {}).get('track_id')
        offense, side = sides.get(str(shooter)), sides.get(str(track))
        rebound['rebound_side'] = ('offensive' if offense and offense == side else
                                  'defensive' if offense and side else 'team_unknown')
        events.append(rebound)

    for a, b in zip(original_controls, original_controls[1:]):
        if a['player_id'] == b['player_id'] or not opposed(sides, a['player_id'], b['player_id']):
            continue
        frame = b['start_frame']
        if not 0 <= frame - a['end_frame'] <= round(1.5 * game.fps):
            continue
        # Collections belong to rebound/reset, not to a live-ball steal.
        shot_context = any(s['status'] == 'confirmed' and
                           s['anchor_frame'] <= frame <= s['end_frame'] + 2 * game.fps for s in shots)
        rebound_context = (frame, b['player_id']) in used_acquisitions
        proof = steal_evidence(game, a, b, sides)
        if shot_context or rebound_context:
            proof['reasons'].append('POST_SHOT_OR_REBOUND_COLLECTION')
            proof['confirmed'] = False
        strong = proof['confirmed']
        roles = {'collector': _role(registry, segment_id, b['player_id'], 'confirmed', frame),
                 'previous_controller': _role(registry, segment_id, a['player_id'], 'confirmed', a['end_frame'] - 1),
                 'stealer': _role(registry, segment_id, b['player_id'], 'confirmed' if strong else 'probable', frame)}
        events.append(_event(segment_id, 'steal', len(events) + 1, a['end_frame'] - 1, frame,
                             min(b['end_frame'], frame + round(.25 * game.fps)), roles,
                             'confirmed' if strong else 'uncertain', evidence=[proof], reasons=proof['reasons']))

    for segment in original_controls:
        candidate = _drive_candidate(game, segment, registry, segment_id, sides)
        if candidate:
            candidate["event_id"] = f"{segment_id}-successful_drive-{len(events) + 1:05d}"
            events.append(candidate)

    for shot in shots:
        candidate = _block_candidate(game, shot, registry, segment_id, sides)
        if candidate:
            candidate["event_id"] = f"{segment_id}-block-{len(events) + 1:05d}"
            candidate["related_event_ids"].append(shot["event_id"])
            events.append(candidate)

    # A direct, established cooperative pass to a scorer is a visual assist
    # chain only if no other controller or shot interrupts the opportunity.
    for shot in shots:
        if shot["outcome"]["value"] != "made" or shot["status"] != "confirmed":
            continue
        receiver = shot["actor_roles"].get("shooter", {}).get("track_id")
        if shot["actor_roles"].get("shooter", {}).get("role_status") != "confirmed":
            continue
        for transfer in reversed(transfers):
            if transfer["type"] != "pass" or transfer["status"] != "confirmed":
                continue
            frame = transfer["anchor_frame"]
            if not 0 <= shot["anchor_frame"] - frame <= round(4 * game.fps):
                continue
            if transfer["actor_roles"]["receiver"]["track_id"] != receiver:
                continue
            if any(other is not shot and frame < other["anchor_frame"] < shot["anchor_frame"]
                   for other in shots):
                continue
            intervening = [control for control in original_controls
                           if frame < control["start_frame"] < shot["anchor_frame"]
                           and control["player_id"] != receiver]
            if intervening:
                continue
            chain = _event(segment_id, "assist_chain", len(events) + 1,
                           transfer["start_frame"], shot["anchor_frame"],
                           shot["end_frame"],
                           {"passer": transfer["actor_roles"]["passer"],
                            "scorer": shot["actor_roles"]["shooter"]},
                           "confirmed", evidence=transfer["evidence"] + shot["evidence"])
            chain["related_event_ids"] = [transfer["event_id"], shot["event_id"]]
            events.append(chain)
            break

    for rebound in [e for e in events if e["type"] == "rebound" and e["status"] == "confirmed"]:
        actor = rebound["actor_roles"]["rebounder"]["track_id"]
        for shot in shots:
            if (shot["actor_roles"].get("shooter", {}).get("track_id") == actor
                    and 0 <= shot["anchor_frame"] - rebound["anchor_frame"] <= 2 * game.fps
                    and shot["event_id"] not in rebound["related_event_ids"]):
                shot["subtype"] = "putback"
                shot["related_event_ids"].append(rebound["event_id"])
                break
    for action in [e for e in events if e["type"] in {"steal", "successful_drive"} and e["status"] == "confirmed"]:
        role = action["actor_roles"].get("stealer" if action["type"] == "steal" else "driver", {})
        for shot in shots:
            if (shot["actor_roles"].get("shooter", {}).get("player_id") == role.get("player_id")
                    and shot["outcome"]["value"] == "made"
                    and 0 <= shot["anchor_frame"] - action["anchor_frame"] <= 4 * game.fps):
                action["related_event_ids"].append(shot["event_id"])
                shot["related_event_ids"].append(action["event_id"])
                shot.setdefault("highlight_tags", []).append(action["type"] + "_to_score")
                break

    if rim_scan:
        for visit in rim_scan["visits"]:
            frame = int(visit["peak_frame"])
            if any(abs(frame - shot["anchor_frame"]) <= round(1.5 * game.fps)
                   or shot["start_frame"] <= frame < shot["end_frame"]
                   or (shot.get("rim_window") and shot["rim_window"][0] <= frame <= shot["rim_window"][1])
                   for shot in shots):
                continue
            events.append(_event(
                segment_id, "shot", len(events) + 1,
                max(game.start, frame - round(.5 * game.fps)), frame,
                min(game.end, frame + round(.5 * game.fps)), {"shooter": _role(registry, segment_id, None)},
                "uncertain", evidence=[visit],
                reasons=["RIM_MOTION_ONLY", "BALL_OR_RELEASE_NOT_VERIFIED"],
            ))
    events.sort(key=lambda event: (event["anchor_frame"], event["event_id"]))
    timeline = build_possession(game, original_controls, shots, segment_id, sides)
    for event in events:
        state = possession_at(timeline, event["start_frame"])
        if state:
            event["possession_id"] = state["possession_id"]
    return events, legacy["audit"], timeline, team_report
