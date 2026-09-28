"""Local net deformation measured on visible fabric, excluding the moving ball."""

import cv2
import numpy as np

from ..core.timeline import source_frame


def net_deformation(frames, masks, before_count):
    """Compare visible white fabric to a pre-arrival temporal reference."""
    if before_count < 3 or len(frames) <= before_count:
        return {'passed': False, 'reason': 'NET_FRAMES_MISSING', 'series': []}
    reference = np.median(np.stack(frames[:before_count]), axis=0).astype(np.uint8)
    hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    white = (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 100)
    # Net texture is locally brighter than its background. Flat bright objects
    # or backboard paint are not evidence of moving fabric.
    local = cv2.GaussianBlur(gray, (9, 9), 0)
    white &= gray.astype(float) - local > 5
    if np.count_nonzero(white) < 12:
        return {'passed': False, 'reason': 'NET_TEXTURE_MISSING', 'series': []}
    series = []
    for image, visible in zip(frames, masks):
        current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        retained = visible & white
        coverage = np.count_nonzero(retained) / np.count_nonzero(white)
        fraction = (float(np.mean(cv2.absdiff(current, gray)[retained] > 20))
                    if coverage >= .2 else None)
        series.append({'change_fraction': fraction, 'visible_fraction': round(coverage, 3)})
    baseline = float(np.median([p['change_fraction'] for p in series[:before_count]
                                if p['change_fraction'] is not None] or [0.0]))
    after = series[before_count:]
    values = [p['change_fraction'] for p in after]
    threshold = max(.22, baseline + .16)
    longest = run = 0
    for value in values:
        run = run + 1 if value is not None and value >= threshold else 0
        longest = max(longest, run)
    return {'passed': longest >= 3, 'reason': 'NET_DEFORMATION' if longest >= 3 else 'NET_DEFORMATION_INSUFFICIENT',
            'baseline': round(baseline, 4), 'threshold': round(threshold, 4),
            'peak': round(max((v for v in values if v is not None), default=0.0), 4),
            'sustained_frames': longest, 'series': series}


def net_motion(game, frame, track):
    settings = game.views[game.rules.result_view]
    x1, y1, x2, y2 = game.rules.net_roi
    if not (0 <= x1 < x2 <= settings['width'] and 0 <= y1 < y2 <= settings['height']):
        return {'passed': False, 'reason': 'NET_ROI_INVALID', 'series': []}
    anchor = round(frame)
    start = max(game.start, anchor - round(.5 * game.fps), settings['frame_zero'])
    end = min(game.end, anchor + round(.8 * game.fps))
    cap = cv2.VideoCapture(settings['path'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame(start, settings['frame_zero']))
    images, masks = [], []
    try:
        for f in range(start, end):
            ok, image = cap.read()
            if not ok:
                break
            images.append(image[y1:y2, x1:x2])
            visible = np.ones((y2 - y1, x2 - x1), np.uint8)
            for t in (f - 1, f):
                index = int(np.searchsorted(track[:, 0], t)) if len(track) else 0
                samples = track[max(0, index - 1):min(len(track), index + 1)]
                if len(samples) == 2 and samples[-1, 0] - samples[0, 0] <= game.fps * game.rules.observation_gap_seconds:
                    # Interpolation masks an occluder only; it cannot prove a ball observation.
                    ball = [np.interp(t, samples[:, 0], samples[:, i]) for i in range(1, 4)]
                elif len(samples) and np.min(abs(samples[:, 0] - t)) <= 2:
                    ball = samples[np.argmin(abs(samples[:, 0] - t)), 1:]
                else:
                    continue
                x, y, r = ball
                cv2.circle(visible, (round(x - x1), round(y - y1)), max(8, round(1.5 * r)), 0, -1)
            for observations in game.players(f).values():
                obs = observations.get(game.rules.result_view)
                if obs is None:
                    continue
                a, b, c, d = map(round, obs['bbox'])
                if a < x2 and c > x1 and b < y2 and d > y1:
                    cv2.rectangle(visible, (a - x1, b - y1), (c - x1, d - y1), 0, -1)
            masks.append(visible.astype(bool))
    finally:
        cap.release()
    result = net_deformation(images, masks, max(0, anchor - start - 3))
    result.update({'start_frame': start, 'end_frame': start + len(images),
                   'view': game.rules.result_view, 'method': 'visible_net_texture_deformation'})
    return result
