"""Player visibility scoring and at most one camera cut per highlight."""

import numpy as np

from ..core.timeline import source_frame


def critical_frame(event):
    value = event['outcome'].get('evidence_frame')
    return event['anchor_frame'] if value is None else value


def clip_boundary(game, event, controls):
    key = event['start_frame']
    prior = [c for c in controls if c['start_frame'] <= key
             and min(c['end_frame'], key + 1) - c['start_frame'] >= round(.2 * game.fps)]
    control = max(prior, key=lambda c: c['end_frame']) if prior else None
    begin = control['start_frame'] if control else key
    return {'event_id': event['event_id'], 'control': control,
            'reason': 'previous_stable_control' if control else 'no_stable_control_use_event_start',
            'start': max(game.start, begin - round(.5 * game.fps)),
            'end': min(game.end, event['end_frame'] + round(1.5 * game.fps))}


def frame_quality(game, view, frame, events):
    settings = game.views[view]
    width, height = settings.get('width', 1920), settings.get('height', 1080)
    people = game.players(frame)
    actors = {str(role['track_id']) for e in events for role in e['actor_roles'].values()
              if isinstance(role, dict) and role.get('track_id') is not None}
    scores = []
    for actor in actors:
        obs = people.get(actor, {}).get(view)
        if obs is None:
            scores.append(0.0)
            continue
        box = np.asarray(obs['bbox'], float)
        if not np.isfinite(box).all():
            scores.append(0.0)
            continue
        area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
        overlap = 0.0
        for other, observations in people.items():
            if str(other) == actor or view not in observations:
                continue
            b = observations[view]['bbox']
            overlap += max(0, min(box[2], b[2]) - max(box[0], b[0])) * max(0, min(box[3], b[3]) - max(box[1], b[1])) / area
        unoccluded = 1 - min(1, overlap)
        size = min(1, (box[3] - box[1]) / (.4 * height))
        conf = np.asarray(obs.get('keypoints_conf', []), float)
        pose = float(np.mean(conf >= .35)) if len(conf) else 0.0
        stable = 0.0
        before = game.players(max(game.start, frame - 3)).get(actor, {}).get(view)
        after = game.players(min(game.end - 1, frame + 3)).get(actor, {}).get(view)
        if before and after:
            a, b = np.asarray(before['bbox'], float), np.asarray(after['bbox'], float)
            center = (box[:2] + box[2:]) / 2
            ca, cb = (a[:2] + a[2:]) / 2, (b[:2] + b[2:]) / 2
            jitter = np.linalg.norm(cb - 2 * center + ca) / max(1, box[3] - box[1])
            stable = max(0, 1 - 5 * jitter) if np.isfinite(jitter) else 0
        in_frame = float(box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height)
        scores.append(.35 * unoccluded + .25 * size + .2 * pose + .15 * stable + .05 * in_frame)
    person = float(np.mean(scores)) if scores else 0.0
    ball = float(view in game.balls(frame))
    # 80% people, 20% ball. Overlap is an occlusion proxy, not pixel segmentation.
    return .8 * person + .2 * ball


def camera_shots(game, start, end, events):
    frames = list(range(start, end, max(1, round(.2 * game.fps))))
    quality = {view: np.asarray([frame_quality(game, view, f, events) for f in frames])
               for view in game.views}
    scores = {view: float(np.mean(values)) for view, values in quality.items()}
    # Don't select a source that does not cover its entire requested interval.
    valid = {v for v, s in game.views.items() if source_frame(start, s['frame_zero']) >= 0
             and source_frame(end, s['frame_zero']) <= game.data['video_info'][v].get('total_frames', game.end)}
    scores = {v: score for v, score in scores.items() if v in valid}
    if not scores:
        raise ValueError('No camera covers the highlight interval')
    best = max(scores, key=scores.get)
    choice = [(start, end, best)]
    gain = .08
    minimum = round(2 * game.fps)
    for cut in range(start + minimum, end - minimum + 1, max(1, round(.5 * game.fps))):
        if any(abs(cut - critical_frame(e)) < .5 * game.fps for e in events):
            continue
        index = int(np.searchsorted(frames, cut))
        left = {v: float(quality[v][:index].mean()) for v in scores}
        right = {v: float(quality[v][index:].mean()) for v in scores}
        a, b = max(left, key=left.get), max(right, key=right.get)
        if a == b or left[a] - left[b] < .06 or right[b] - right[a] < .06:
            continue
        improvement = (left[a] * index + right[b] * (len(frames) - index)) / len(frames) - scores[best]
        if improvement > gain:
            gain = improvement
            choice = [(start, cut, a), (cut, end, b)]
    shots = []
    for begin, finish, view in choice:
        source = game.views[view]
        shots.append({'view': view, 'source': source['path'], 'sync_start_frame': begin,
                      'sync_end_frame': finish, 'source_start_frame': source_frame(begin, source['frame_zero']),
                      'source_end_frame': source_frame(finish, source['frame_zero'])})
    return shots, scores
