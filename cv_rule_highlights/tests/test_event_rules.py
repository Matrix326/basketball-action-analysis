from types import SimpleNamespace

import cv2
import numpy as np

from cv_rule_highlights.src.core.data import Rules
from cv_rule_highlights.src.events.event_rules import drive_evidence, merge_dribble_controls, opposed
from cv_rule_highlights.src.perception.shot_evidence import net_deformation
from cv_rule_highlights.src.reporting.stats import compute_stats
from cv_rule_highlights.src.editing.auto_clips import eligible, combine_plans


def fabric(dx=0):
    im = np.full((60, 80, 3), 30, np.uint8)
    for x in range(15, 65, 10):
        cv2.line(im, (x + dx, 5), (x + dx - 8, 50), (220, 220, 220), 1)
        cv2.line(im, (x + dx, 5), (x + dx + 8, 50), (220, 220, 220), 1)
    return im


def test_net_deforms_and_persists():
    frames = [fabric()] * 10 + [fabric(3)] * 8
    masks = [np.ones((60, 80), bool)] * len(frames)
    assert net_deformation(frames, masks, 10)['passed']


def test_ball_crosses_static_net_without_deformation():
    frames, masks = [], []
    for f in range(18):
        im = fabric(); mask = np.ones((60, 80), np.uint8)
        if f >= 10:
            center = (40, 6 * (f - 10))
            cv2.circle(im, center, 9, (25, 80, 170), -1)
            cv2.circle(mask, center, 13, 0, -1)
        frames.append(im); masks.append(mask.astype(bool))
    result = net_deformation(frames, masks, 10)
    assert not result['passed']


def test_occluded_net_does_not_confirm():
    frames = [fabric()] * 10 + [fabric(3)] * 8
    masks = [np.ones((60, 80), bool)] * 10 + [np.zeros((60, 80), bool)] * 8
    assert not net_deformation(frames, masks, 10)['passed']


def drive_game(*, advance=True, ground_missing=False):
    positions, quality = {}, {}
    for f in range(70):
        positions[str(f)] = {'1': [0, 4 - (2 * f / 60 if advance else 0), 0], '2': [.2, 3, 0]}
        quality[str(f)] = {v: {'views': ['a', 'b'], 'mean_reprojection_error_px': 5} for v in ['1', '2']}
    game = SimpleNamespace(start=0, end=70, fps=30, court={'usable_for_progress': True, 'basket_ground_xy': [0, 0]},
                           data={'ground_positions_3d': {} if ground_missing else positions, 'quality': quality},
                           players=lambda frame: {'1': {}, '2': {}})
    return game


def test_drive_requires_existing_defender_and_sustained_advantage():
    game = drive_game(); control = {'player_id': '1', 'start_frame': 0, 'end_frame': 60}
    proof = drive_evidence(game, control, {'1': 'a', '2': 'b'})
    assert proof and proof['confirmed'] and proof['advance_m'] >= .9
    assert not drive_evidence(drive_game(advance=False), control, {'1': 'a', '2': 'b'})
    assert not drive_evidence(game, control, {'1': 'a', '2': 'a'})
    assert not drive_evidence(drive_game(ground_missing=True), control, {'1': 'a', '2': 'b'})
    assert not drive_evidence(game, control, {})['confirmed']


def test_dribble_gap_does_not_create_a_new_controller():
    spans = [{'player_id': '1', 'start_frame': 0, 'end_frame': 8},
             {'player_id': '1', 'start_frame': 16, 'end_frame': 26},
             {'player_id': '2', 'start_frame': 27, 'end_frame': 35}]
    assert len(merge_dribble_controls(spans, 30)) == 2
    assert not opposed({}, '1', '2')
    assert not opposed({'1': 'a', '2': 'a'}, '1', '2')


def event(eid, kind='shot', outcome='made', status='confirmed', actor='p1', identity='confirmed'):
    role = {'shot': 'shooter', 'successful_drive': 'driver', 'steal': 'stealer', 'block': 'blocker', 'rebound': 'rebounder', 'assist_chain': 'passer'}[kind]
    return {'event_id': eid, 'segment_id': 'segment_1', 'type': kind, 'status': status,
            'start_frame': 5, 'anchor_frame': 10, 'end_frame': 20,
            'outcome': {'value': outcome, 'status': 'confirmed', 'evidence_frame': 15},
            'actor_roles': {role: {'player_id': actor, 'track_id': '1', 'role_status': 'confirmed', 'identity_status': identity}},
            'related_event_ids': []}


def test_counts_conserve_unknown_and_unassigned_without_double_counting():
    events = [event('a'), event('b', outcome='missed'), event('c', outcome='unknown'),
              event('d', actor=None), event('e', kind='steal', status='uncertain')]
    stats = compute_stats(events, ['p1'])
    assert stats['all_players']['shot_attempts_detected'] == 4
    assert stats['players']['p1']['made'] == 1
    assert stats['unassigned']['made'] == 1
    assert stats['players']['p1']['resolved_fg_pct'] == .5
    assert stats['all_players']['steals'] == 0
    assert stats['all_players']['uncertain_counts']['steal'] == 1
    candidate = event('f'); candidate['outcome']['status'] = 'unknown'
    assert compute_stats([candidate], ['p1'])['players']['p1']['outcome_unknown'] == 1


def test_personal_clips_filter_positive_role_before_merging():
    e = event('a', kind='successful_drive')
    e['actor_roles']['defender'] = {'player_id': 'p2', 'track_id': '2', 'role_status': 'confirmed', 'identity_status': 'confirmed'}
    game = SimpleNamespace(start=0, end=100, fps=30, views={'a': {'path': 'video.mp4', 'frame_zero': 0}},
                           data={'video_info': {'a': {'total_frames': 100}}},
                           balls=lambda frame: {}, players=lambda frame: {})
    games, events = {'segment_1': game}, {'segment_1': [e]}
    assert combine_plans(games, events, player_id='p1')['eligible_events'] == 1
    assert combine_plans(games, events, player_id='p2')['clips'] == []
    assert not eligible(event('candidate', status='uncertain'))
    assert eligible(event('candidate', status='uncertain'), candidates=True)


def test_steal_rejects_catch_and_controller_flicker(monkeypatch):
    from cv_rule_highlights.src.events import event_rules as rules
    game = SimpleNamespace(fps=30)
    a = {'player_id': '1', 'start_frame': 0, 'end_frame': 20}
    b = {'player_id': '2', 'start_frame': 22, 'end_frame': 40}
    teams = {'1': 'a', '2': 'b'}
    monkeypatch.setattr(rules, 'contact_views', lambda *args: [{'view': 'a'}, {'view': 'b'}])
    monkeypatch.setattr(rules, 'trajectory_turn', lambda *args: None)
    assert not rules.steal_evidence(game, a, b, teams)['confirmed']
    monkeypatch.setattr(rules, 'trajectory_turn', lambda g, f, v: {'view': v, 'frame': f} if v == 'a' else None)
    assert rules.steal_evidence(game, a, b, teams)['confirmed']
    monkeypatch.setattr(rules, 'contact_views', lambda g, f, track: [{'view': 'a'}, {'view': 'b'}] if track == '2' else [])
    assert 'RECEIVE_VS_STEAL_UNVERIFIED' in rules.steal_evidence(game, a, b, teams)['reasons']
    assert 'CONTROL_TOO_BRIEF' in rules.steal_evidence(game, a, {**b, 'end_frame': 26}, teams)['reasons']
    assert not rules.steal_evidence(game, a, b, {'1': 'a', '2': 'a'})['confirmed']


def test_block_requires_opponent_and_rejects_scored_shot(monkeypatch):
    from cv_rule_highlights.src.events import auto_events
    game = SimpleNamespace(start=0, end=40, fps=30,
        views={v: {'rim': [250, 100, 60, 8]} for v in ['a', 'b']},
        players=lambda f: {'1': {v: {'keypoints_xy': [[50, 100]] * 17} for v in ['a', 'b']}, '2': {}},
        balls=lambda f: {v: {'center_xy': [50, 50]} for v in ['a', 'b']})
    registry = SimpleNamespace(resolve=lambda s, t, f=None: 'p' + str(t), status=lambda s, t: 'confirmed')
    monkeypatch.setattr(auto_events, 'contact_views', lambda g, f, t: [{'view': v, 'ball_radius_px': 5} for v in ['a', 'b']])
    monkeypatch.setattr(auto_events, 'trajectory_turn', lambda g, f, v: {'view': v, 'frame': f, 'cosine': -1})
    shot = event('a', outcome='unknown')
    assert auto_events._block_candidate(game, shot, registry, 'segment_1', {'1': 'a', '2': 'b'})['status'] == 'confirmed'
    assert auto_events._block_candidate(game, shot, registry, 'segment_1', {'1': 'a', '2': 'a'})['status'] == 'uncertain'
    shot['outcome']['value'] = 'made'
    assert auto_events._block_candidate(game, shot, registry, 'segment_1', {'1': 'a', '2': 'b'})['status'] == 'rejected'


def test_duplicate_atomic_event_cannot_double_count():
    import pytest
    with pytest.raises(ValueError, match='Duplicate atomic event'):
        compute_stats([event('same'), event('same')], ['p1'])


def test_frame_zero_is_included_in_clip():
    e = event('zero'); e.update(start_frame=0, anchor_frame=0, end_frame=2)
    e['outcome']['evidence_frame'] = 0
    game = SimpleNamespace(start=0, end=30, fps=30, views={'a': {'path': 'video.mp4', 'frame_zero': 0}},
        data={'video_info': {'a': {'total_frames': 30}}}, balls=lambda f: {}, players=lambda f: {})
    clip = combine_plans({'segment_1': game}, {'segment_1': [e]})['clips'][0]
    assert clip['critical_frames'] == [0] and clip['source_start_frame'] == 0


def test_event_categories_filter_before_merging_and_keep_unassigned_makes():
    from cv_rule_highlights.src.editing.event_clips import category_plans
    events = [event('made', actor=None), event('miss', outcome='missed'),
              event('unknown', outcome='unknown'), event('candidate', status='uncertain'),
              event('board', kind='rebound'), event('drive', kind='successful_drive')]
    game = SimpleNamespace(start=0, end=100, fps=30, views={'a': {'path': 'video.mp4', 'frame_zero': 0}},
        data={'video_info': {'a': {'total_frames': 100}}}, balls=lambda f: {}, players=lambda f: {})
    plans = category_plans({'segment_1': game}, {'segment_1': events})
    assert plans['made']['selected_event_ids'] == ['made']
    assert plans['rebound']['clips'][0]['event_ids'] == ['board']
    assert plans['successful_drive']['clips'][0]['event_ids'] == ['drive']
    assert plans['steal']['clips'] == [] and plans['block']['clips'] == []
    assert plans['made']['budget_seconds'] is None


def test_rebound_never_skips_first_receiver_for_later_pass(monkeypatch):
    from cv_rule_highlights.src.events import event_rules as rules
    game = SimpleNamespace(fps=30)
    brief = {'player_id': '1', 'start_frame': 100, 'end_frame': 103}
    later = {'player_id': '2', 'start_frame': 150, 'end_frame': 210}
    monkeypatch.setattr(rules, 'contact_views', lambda g, f, t: [{'view': 'a', 'frame': f}])
    proof = rules.rebound_acquisition(game, [brief, later])
    assert proof['control'] == brief and not proof['confirmed']
    monkeypatch.setattr(rules, 'contact_views', lambda g, f, t: [{'view': v, 'frame': f} for v in ['a', 'b']])
    proof = rules.rebound_acquisition(game, [brief, later])
    assert proof['control'] == brief and proof['confirmed']
    monkeypatch.setattr(rules, 'contact_views', lambda g, f, t: [])
    assert not rules.rebound_acquisition(game, [later])['confirmed']
