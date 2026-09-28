"""Contact and ground-advantage evidence shared by non-shot event rules."""

import numpy as np


def opposed(teams, first, second):
    return teams.get(str(first)) is not None and teams.get(str(second)) is not None and teams[str(first)] != teams[str(second)]


def ground(game, frame, track):
    values = []
    for f in range(max(game.start, frame - 2), min(game.end, frame + 3)):
        quality = game.data.get('quality', {}).get(str(f), {}).get(str(track), {})
        p = game.data.get('ground_positions_3d', {}).get(str(f), {}).get(str(track))
        if (p is not None and np.isfinite(p).all() and not quality.get('predicted_track', False)
                and len(quality.get('views', [])) >= 2
                and quality.get('mean_reprojection_error_px') is not None
                and quality['mean_reprojection_error_px'] < 30):
            values.append(p[:2])
    return np.median(values, axis=0) if len(values) >= 3 else None


def contact_views(game, frame, track, radius_scale=1.8):
    evidence = []
    for view, ball in game.balls(frame).items():
        obs = game.players(frame).get(str(track), {}).get(view)
        if obs is None:
            continue
        wrists = np.asarray(obs['keypoints_xy'], float)[[9, 10]]
        valid = (np.asarray(obs['keypoints_conf'])[[9, 10]] >= game.rules.keypoint_confidence) & np.isfinite(wrists).all(axis=1)
        if not valid.any():
            continue
        box = ball['bbox']
        radius = (box[2] - box[0] + box[3] - box[1]) / 4
        distance = float(np.linalg.norm(wrists[valid] - ball['center_xy'], axis=1).min())
        if distance <= max(8, radius_scale * radius):
            evidence.append({'view': view, 'frame': frame, 'distance_px': round(distance, 2),
                             'ball_radius_px': round(radius, 2), 'method': 'observed_wrist_ball_contact'})
    return evidence


def trajectory_turn(game, frame, view):
    samples = game.ball_samples(view)
    part = samples[(samples[:, 0] >= frame - 5) & (samples[:, 0] <= frame + 5)]
    before, after = part[part[:, 0] <= frame], part[part[:, 0] >= frame]
    if len(before) < 2 or len(after) < 2 or np.max(np.diff(part[:, 0])) > 3:
        return None
    a, b = before[0], before[-1]
    c, d = after[0], after[-1]
    u, v = (b[1:3] - a[1:3]) / (b[0] - a[0]), (d[1:3] - c[1:3]) / (d[0] - c[0])
    speed_before, speed_after = np.linalg.norm(u), np.linalg.norm(v)
    if speed_before < 1 or speed_after < 1:
        return None
    cosine = float(np.dot(u, v) / (speed_before * speed_after))
    if cosine > .4:
        return None
    return {'view': view, 'frame': frame, 'cosine': round(cosine, 3),
            'support_frames': [int(a[0]), int(d[0])], 'method': 'observed_velocity_deflection'}


def drive_evidence(game, control, teams):
    if not game.court.get('usable_for_progress'):
        return None
    target = np.asarray(game.court['basket_ground_xy'])
    actor = str(control['player_id'])
    start, end = control['start_frame'], control['end_frame'] - 1
    step = max(1, round(.1 * game.fps))
    minimum = round(.7 * game.fps)
    for begin in range(start, max(start, end - minimum + 1), step):
        a = ground(game, begin, actor)
        if a is None:
            continue
        axis = target - a
        norm = np.linalg.norm(axis)
        if norm < 1:
            continue
        axis /= norm
        for defender in game.players(begin):
            if str(defender) == actor or (teams.get(actor) and teams.get(str(defender)) == teams[actor]):
                continue
            d = ground(game, begin, defender)
            if d is None:
                continue
            relative = d - a
            along = float(relative @ axis)
            lateral = float(np.linalg.norm(relative - along * axis))
            if not .2 < along < 2.5 or lateral > 1.2 or np.linalg.norm(relative) > 2.6:
                continue
            for finish in range(begin + minimum, min(end, begin + round(3 * game.fps)) + 1, step):
                b = ground(game, finish, actor)
                if b is None or (b - a) @ axis < .9 or norm - np.linalg.norm(target - b) < .7:
                    continue
                maintained = []
                for f in range(finish - round(.2 * game.fps), finish + 1, step):
                    ap, dp = ground(game, f, actor), ground(game, f, defender)
                    if ap is not None and dp is not None:
                        maintained.append(float((ap - dp) @ axis))
                if len(maintained) < 3 or min(maintained) < .25:
                    continue
                return {'defender': str(defender), 'start_frame': begin, 'end_frame': finish + 1,
                        'confirmed': opposed(teams, actor, defender), 'method': 'sustained_ground_advantage',
                        'advance_m': round(float((b - a) @ axis), 3),
                        'initial_defender_distance_m': round(float(np.linalg.norm(relative)), 3),
                        'advantage_m': round(min(maintained), 3), 'support_frames': len(maintained)}
    return None


def merge_dribble_controls(controls, fps):
    merged = []
    for item in controls:
        if (merged and merged[-1]['player_id'] == item['player_id']
                and item['start_frame'] - merged[-1]['end_frame'] <= round(.5 * fps)):
            merged[-1]['end_frame'] = item['end_frame']
        else:
            merged.append(dict(item))
    return merged


def steal_evidence(game, previous, acquired, teams):
    """A durable takeover: two-view contact with a deflection in either view."""
    reasons = []
    if not opposed(teams, previous['player_id'], acquired['player_id']):
        reasons.append('OPPOSITION_UNVERIFIED')
    minimum = round(.25 * game.fps)
    if any(c['end_frame'] - c['start_frame'] < minimum for c in (previous, acquired)):
        reasons.append('CONTROL_TOO_BRIEF')
    if not 0 <= acquired['start_frame'] - previous['end_frame'] <= round(1.2 * game.fps):
        reasons.append('CONTROL_GAP_TOO_LONG')
    contacts, turns, contest = [], [], []
    for frame in range(max(previous['start_frame'], previous['end_frame'] - 6), acquired['start_frame'] + 3):
        local = contact_views(game, frame, acquired['player_id'])
        for contact in local:
            turn = trajectory_turn(game, frame, contact['view'])
            if turn:
                # Different views may expose contact one or two frames apart.
                paired = [c for f in range(max(previous['start_frame'], frame - 2), frame + 3)
                          for c in contact_views(game, f, acquired['player_id'])]
                prior_touch = [c for f in range(max(previous['start_frame'], frame - 3), frame + 1)
                               for c in contact_views(game, f, previous['player_id'])]
                rank = (bool(prior_touch), len({c['view'] for c in paired}))
                if rank > (bool(contest), len({c['view'] for c in contacts})):
                    contacts, turns, contest = paired, [turn], prior_touch
    if len({c['view'] for c in contacts}) < 2:
        reasons.append('MULTIVIEW_TOUCH_UNVERIFIED')
    if not turns:
        reasons.append('DEFLECTION_UNVERIFIED')
    if not contest:
        reasons.append('RECEIVE_VS_STEAL_UNVERIFIED')
    return {'method': 'contact_then_durable_takeover', 'confirmed': not reasons, 'reasons': reasons,
            'previous_control': [previous['start_frame'], previous['end_frame']],
            'new_control': [acquired['start_frame'], acquired['end_frame']],
            'contacts': contacts, 'deflections': turns, 'previous_player_contacts': contest}


def rebound_acquisition(game, options):
    """Never replace the first receiver with a later, longer possession."""
    control = options[0]
    start, end = control['start_frame'], control['end_frame']
    contacts = [c for f in range(start, min(end, start + 9))
                for c in contact_views(game, f, control['player_id'])]
    stable = end - start >= round(.2 * game.fps) and bool(contacts)
    brief_multiview = (end - start >= round(.1 * game.fps)
                      and len({c['view'] for c in contacts}) >= 2
                      and len({c['frame'] for c in contacts}) >= 3)
    return {'control': control, 'contacts': contacts, 'confirmed': stable or brief_multiview,
            'method': 'stable_control' if stable else 'brief_multiview_touch' if brief_multiview
                      else 'first_acquisition_unverified'}
