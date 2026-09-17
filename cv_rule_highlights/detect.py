"""Observed hand/ball relations and multi-view rim passage rules."""

import cv2
import numpy as np

from .data import Game


def net_motion(game, frame, track):
    """Measure localized post-passage net motion against its own baseline."""
    x1, y1, x2, y2 = game.rules.net_roi
    polygon = np.array([[0, 0], [x2 - x1 - 1, 0], [x2 - x1 - 12, y2 - y1 - 1], [8, y2 - y1 - 1]], np.int32)
    region = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillConvexPoly(region, polygon, True)
    cap = cv2.VideoCapture(game.views[game.rules.result_view]["path"])
    def image(f):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, bgr = cap.read()
        return cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) if ok else None
    def score(a, b, f):
        if a is None or b is None: return 0.0
        mask = np.ones(a.shape, bool)
        near = track[np.argmin(np.abs(track[:, 0] - f))] if len(track) else None
        if near is not None and abs(near[0] - f) <= 2:
            r = max(near[3] * 1.8, 8); xa, ya = int(near[1] - r - x1), int(near[2] - r - y1)
            xb, yb = int(near[1] + r - x1), int(near[2] + r - y1)
            mask[max(0,ya):min(mask.shape[0],yb+1), max(0,xa):min(mask.shape[1],xb+1)] = False
        # Only the white, low-saturation net fabric is evidence. This rejects
        # the orange ball and most of the dark backboard/players.
        fabric = ((a > 105) & (b > 105)) & mask & region.astype(bool)
        values = np.abs(b.astype(float) - a.astype(float))[fabric]
        return float(np.mean(values)) if values.size else 0.0
    before = [score(image(f - 1), image(f), f) for f in range(max(game.start + 1, int(frame - .5 * game.fps)), int(frame))]
    after = [score(image(f - 1), image(f), f) for f in range(int(frame) + 1, int(frame + game.rules.net_motion_seconds * game.fps))]
    cap.release()
    baseline = float(np.median(before)) if before else 0.0
    peak = float(np.percentile(after, 90)) if after else 0.0
    ratio = peak / max(baseline, 1.0)
    sustained = max((float(np.mean(after[i:i + 4])) for i in range(max(0, len(after) - 3))), default=0.0)
    return {"baseline": round(baseline, 3), "peak": round(peak, 3), "sustained": round(sustained, 3), "ratio": round(ratio, 3),
            "passed": bool(after and sustained >= game.rules.net_motion_peak and ratio >= game.rules.net_motion_ratio)}


def controls(game: Game):
    """Stable wrist contacts; dribble gaps retain identity without inventing observations."""
    contacts = []
    for frame in range(game.start, game.end):
        balls = game.balls(frame)
        ranked = []
        for player, observations in game.players(frame).items():
            distances = []
            for view, ball in balls.items():
                if view not in observations:
                    continue
                obs = observations[view]
                wrists = np.asarray(obs["keypoints_xy"], dtype=float)[[9, 10]]
                valid = np.asarray(obs["keypoints_conf"])[[9, 10]] >= game.rules.keypoint_confidence
                valid &= np.isfinite(wrists).all(axis=1)
                height = obs["bbox"][3] - obs["bbox"][1]
                if valid.any() and height > 0:
                    distances.append(float(np.linalg.norm(wrists[valid] - ball["center_xy"], axis=1).min() / height))
            if distances:
                distance = float(np.median(distances))
                if distance < game.rules.hand_distance:
                    ranked.append((distance, player))
        ranked.sort()
        # Similar distances in a contested ball are insufficient to assign an owner.
        owner = ranked[0][1] if ranked and (len(ranked) == 1 or ranked[1][0] - ranked[0][0] > 0.03) else None
        contacts.append((frame, owner))

    minimum = max(2, round(game.rules.control_seconds * game.fps))
    max_gap = round(game.rules.control_gap_seconds * game.fps)
    segments = []
    current = None
    candidate, count = None, 0
    for frame, owner in contacts:
        if current and frame - current["last_seen"] > max_gap:
            segments.append(current)
            current = None
        if current and owner == current["player_id"]:
            current["last_seen"] = frame
            candidate, count = None, 0
        elif owner is not None:
            count = count + 1 if owner == candidate else 1
            candidate = owner
            if count >= minimum:
                if current:
                    segments.append(current)
                current = {"player_id": owner, "start_frame": frame - count + 1, "last_seen": frame}
                candidate, count = None, 0
        else:
            candidate, count = None, 0
    if current:
        segments.append(current)
    for segment in segments:
        segment["end_frame"] = segment.pop("last_seen") + 1
        segment["team_id"] = game.teams.get(segment["player_id"])
    return segments


def rim_passages(samples, rim, fps, rules):
    """A descent supported by observations above, across and wholly below the rim.

    No interpolation over a long gap, no extrapolated ball, and no inference
    from a single bounding-box intersection. Returns positive AND negative evidence.
    """
    cx, cy, width, height = rim
    passages = []
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        if not (a[2] <= cy < b[2]) or b[0] - a[0] > fps * rules.observation_gap_seconds + 1e-6:
            continue
        crossing = a[0] + (b[0] - a[0]) * (cy - a[2]) / (b[2] - a[2])
        x = a[1] + (b[1] - a[1]) * (cy - a[2]) / (b[2] - a[2])
        radius = max(a[3], b[3])
        if abs(x - cx) > 2 * width:
            continue
        before = samples[:i]
        before = before[(crossing - before[:, 0] <= fps * rules.passage_seconds) & (before[:, 2] < cy - height / 2 - radius)]
        after = samples[i:]
        after = after[(after[:, 0] - crossing <= fps * rules.passage_seconds) & (after[:, 2] > cy + height / 2 + radius)]
        if not len(before) or not len(after):
            continue
        support = samples[(samples[:, 0] >= before[-1, 0]) & (samples[:, 0] <= after[0, 0])]
        if np.any(np.diff(support[:, 0]) > fps * rules.observation_gap_seconds + 1e-6):
            continue
        # Reject a downward passage whose support includes an upward reversal.
        if np.any(np.diff(support[:, 2]) < -0.05 * width):
            continue
        if abs(x - cx) + radius < width / 2:
            result = "inside"
        elif abs(x - cx) - radius > width / 2:
            result = "outside"
        else:
            result = "edge"
        passages.append({
            "frame": float(crossing), "result": result,
            "support_frames": [int(support[0, 0]), int(support[-1, 0])],
            "crossing_x": float(x),
        })
    return passages


def release_views(game, player, frame, samples):
    """Forward shot proposals from a raised hand and observed flight toward the hoop."""
    supported = []
    for view, obs in game.players(frame).get(player, {}).items():
        if view not in samples:
            continue
        points = np.asarray(obs["keypoints_xy"], dtype=float)
        conf = np.asarray(obs["keypoints_conf"])
        raised = any(
            conf[w] >= game.rules.keypoint_confidence and conf[s] >= game.rules.keypoint_confidence
            and np.isfinite(points[[w, s]]).all() and points[w, 1] < points[s, 1]
            for w, s in [(9, 5), (10, 6)]
        )
        if not raised:
            continue
        track = samples[view]
        flight = track[(track[:, 0] >= frame) & (track[:, 0] <= frame + 0.4 * game.fps)]
        if len(flight) < 3 or flight[-1, 0] - flight[0, 0] < 0.1 * game.fps:
            continue
        if np.max(np.diff(flight[:, 0])) > game.rules.observation_gap_seconds * game.fps + 1e-6:
            continue
        rim = game.views[view]["rim"]
        start, finish = flight[0, 1:3], flight[-1, 1:3]
        target = np.asarray(rim[:2]) - start
        movement = finish - start
        denominator = np.linalg.norm(target) * np.linalg.norm(movement)
        if denominator > 0 and np.dot(target, movement) / denominator > 0.5:
            if np.linalg.norm(target) - np.linalg.norm(np.asarray(rim[:2]) - finish) > 0.4 * rim[2]:
                supported.append(view)
    return supported


def shot_zone(game, player, frame):
    """Conservative ground-anchor estimate, NOT a toes-on-the-line adjudication."""
    if game.inside_arc is None or player is None or frame is None:
        return "unknown"
    positions = []
    for f in range(max(game.start, frame - round(0.25 * game.fps)), frame + 1):
        p = game.data["ground_positions_3d"].get(str(f), {}).get(player)
        if p is not None and np.isfinite(p).all():
            positions.append(p[:2])
    if not positions:
        return "unknown"
    distances = [cv2.pointPolygonTest(game.inside_arc, tuple(map(float, p)), True) for p in positions]
    if min(distances) > 0.30:
        return "inside_arc"
    if max(distances) < -0.30:
        return "outside_arc"
    return "ambiguous"


def detect(game: Game):
    segments = controls(game)
    result_view = game.rules.result_view
    if result_view not in game.views:
        raise ValueError(f"result_view {result_view!r} is not configured")
    samples = {result_view: game.ball_samples(result_view)}
    passages = []
    near_frames = set()
    track = samples[result_view]
    rim = game.views[result_view]["rim"]
    for passage in rim_passages(track, rim, game.fps, game.rules):
        passages.append({**passage, "view": result_view})
    if len(track):
        xy = (track[:, 1:3] - rim[:2]) / rim[2]
        near = (np.abs(xy[:, 0]) < 1.5) & (xy[:, 1] > -2) & (xy[:, 1] < 1.5)
        near_frames.update(track[near, 0].astype(int).tolist())

    proposals = []
    for segment in segments:
        release = segment["end_frame"] - 1
        supported = release_views(game, segment["player_id"], release, samples)
        if len(supported) >= 1:
            proposals.append({"release_frame": release, "actor_id": segment["player_id"], "release_views": supported})

    visits = []
    for frame in sorted(near_frames):
        if not visits or frame - visits[-1][1] > game.fps * game.rules.visit_gap_seconds:
            visits.append([frame, frame])
        else:
            visits[-1][1] = frame

    events = []
    used = set()
    for begin, end in visits:
        evidence = [p for p in passages if p["view"] == game.rules.result_view
                    and begin - game.fps * game.rules.passage_seconds <= p["frame"] <= end]
        part = track[(track[:, 0] >= begin - 0.5 * game.fps) & (track[:, 0] <= end)]
        if len(part) < 3:
            continue
        if not evidence:
            continue
        matches = [i for i, p in enumerate(proposals) if i not in used and begin - game.fps * game.rules.flight_seconds <= p["release_frame"] <= end]
        proposal = proposals[matches[-1]] if matches else None
        if matches:
            used.add(matches[-1])
        if proposal is None:
            # Backtracking recovers shots whose hand pose was occluded. Attribution
            # remains tentative, and no controlled contact is required for tip-ins.
            previous = [s for s in segments if begin - game.fps * game.rules.flight_seconds <= s["end_frame"] - 1 <= begin]
            last = previous[-1] if previous else None
            proposal = {
                "release_frame": last["end_frame"] - 1 if last else None,
                "actor_id": last["player_id"] if last else None,
                "release_views": [],
            }
        outcome, confidence, resolution = "unknown", "probable", None
        positive_groups = []
        for p in sorted((p for p in evidence if p["result"] == "inside"), key=lambda p: p["frame"]):
            if not positive_groups or p["frame"] - positive_groups[-1][0]["frame"] > game.fps * game.rules.fusion_seconds:
                positive_groups.append([p])
            else:
                positive_groups[-1].append(p)
        for group in positive_groups:
            t = float(np.median([p["frame"] for p in group]))
            if len(group) >= game.rules.min_make_views:
                motion = net_motion(game, t, track)
                if motion["passed"]:
                    outcome, confidence, resolution = "made", "confirmed", round(t)
                else:
                    outcome, confidence = "unknown", "unconfirmed"
                for p in evidence:
                    p["net_motion"] = motion
                break
        if outcome != "made":
            outside = [p for p in evidence if p["result"] == "outside"]
            for p in outside:
                group = [q for q in outside if abs(q["frame"] - p["frame"]) <= game.fps * game.rules.fusion_seconds]
                # One clear miss is useful, but an inside/outside disagreement
                # is an unresolved occlusion or calibration conflict.
                conflict = any(q["result"] == "inside" for q in evidence)
                if len(group) >= 2 and not conflict:
                    outcome, confidence = "missed", "confirmed"
                    resolution = round(float(np.median([q["frame"] for q in group])))
                    break
        events.append({**proposal, "rim_window": [begin, end + 1], "visual_outcome": outcome,
                       "confidence": confidence, "resolution_frame": resolution, "evidence": evidence})

    for i, proposal in enumerate(proposals):
        if i not in used:
            events.append({**proposal, "rim_window": None, "visual_outcome": "unknown",
                           "confidence": "probable", "resolution_frame": None, "evidence": []})
    for event in events:
        event["anchor_frame"] = event["release_frame"] if event["release_frame"] is not None else event["rim_window"][0]
    events.sort(key=lambda e: e["anchor_frame"])
    for i, event in enumerate(events, 1):
        actor, release = event["actor_id"], event["release_frame"]
        event.update({
            "id": f"shot_{i:05d}", "type": "shot", "team_id": game.teams.get(actor),
            "actor_confidence": "probable" if actor is not None else "unknown",
            "shot_zone": shot_zone(game, actor, release),
            "scoring_validity": "unknown", "awarded_points": None,
        })
        event["lead_in_frame"] = event["anchor_frame"]
        event["pass"] = None
        if actor is not None and game.teams.get(actor) is not None and release is not None:
            before = [s for s in segments if s["start_frame"] <= release]
            if len(before) >= 2 and before[-1]["player_id"] == actor:
                a, b = before[-2:]
                if a["player_id"] != actor and a["team_id"] == b["team_id"]:
                    if 0 <= b["start_frame"] - a["end_frame"] <= 1.5 * game.fps and release - b["start_frame"] <= 1.5 * game.fps:
                        intervening = any(a["end_frame"] <= other["anchor_frame"] < b["start_frame"] for other in events)
                        if not intervening:
                            event["pass"] = {"passer_id": a["player_id"], "receiver_id": actor, "frame": a["end_frame"] - 1, "confidence": "probable"}
                            event["lead_in_frame"] = a["end_frame"] - 1
        event["highlight_score"] = (
            75 if event["visual_outcome"] == "made" and event["shot_zone"] == "outside_arc" else
            65 if event["visual_outcome"] == "made" and event["pass"] else
            50 if event["visual_outcome"] == "made" else 10
        )

    observed = sum(bool(game.balls(f)) for f in range(game.start, game.end))
    return {
        "schema_version": "highlights-0.1", "config": str(game.config_path), "poses": str(game.poses_path),
        "fps": game.fps, "frame_range": [game.start, game.end],
        "audit": {"processed_frames": game.end - game.start, "observed_ball_frames": observed,
                  "predicted_ball_frames": sum(bool(v) for v in game.data["balls_3d_predicted"].values()),
                  "rim_views": list(game.views)},
        "controls": segments, "events": events,
    }
