from types import SimpleNamespace

from cv_rule_highlights.src.editing import shot_selection as selection


def test_boundary_uses_stable_control_and_event_completion():
    game = SimpleNamespace(start=0, end=600, fps=30)
    event = {'event_id': 'e', 'start_frame': 150, 'end_frame': 220}
    stable = {'start_frame': 60, 'end_frame': 140}
    flicker = {'start_frame': 145, 'end_frame': 148}
    result = selection.clip_boundary(game, event, [stable, flicker])
    assert (result['start'], result['end']) == (45, 265)
    assert result['control'] == stable
    assert selection.clip_boundary(game, event, [])['start'] == 135


def test_camera_cut_is_contiguous_slow_and_avoids_action(monkeypatch):
    game = SimpleNamespace(start=0, end=300, fps=30,
        views={v: {'path': v + '.mp4', 'frame_zero': -10} for v in ['a', 'b']},
        data={'video_info': {v: {'total_frames': 400} for v in ['a', 'b']}})
    monkeypatch.setattr(selection, 'frame_quality', lambda g, v, f, events:
        .95 if (v == 'a') == (f < 150) else .1)
    events = [{'anchor_frame': 150, 'outcome': {}}]
    shots, _ = selection.camera_shots(game, 0, 300, events)
    assert len(shots) == 2
    assert shots[0]['view'] == 'a' and shots[1]['view'] == 'b'
    assert shots[0]['sync_end_frame'] == shots[1]['sync_start_frame']
    assert abs(shots[1]['sync_start_frame'] - 150) >= 15
    assert sum(s['source_end_frame'] - s['source_start_frame'] for s in shots) == 300
    assert all(s['sync_end_frame'] - s['sync_start_frame'] >= 60 for s in shots)
    monkeypatch.setattr(selection, 'frame_quality', lambda g, v, f, events: .8 if v == 'a' else .78)
    assert len(selection.camera_shots(game, 0, 300, events)[0]) == 1


def test_player_visibility_outweighs_ball_only_view():
    observations = {'a': {'bbox': [100, 100, 250, 500], 'keypoints_conf': [1] * 17},
                    'b': {'bbox': [100, 100, 130, 180], 'keypoints_conf': [1] * 17}}
    game = SimpleNamespace(start=0, end=100, views={'a': {}, 'b': {}},
        players=lambda f: {'1': observations, '2': {'b': observations['b']}},
        balls=lambda f: {'b': {}})
    events = [{'actor_roles': {'driver': {'track_id': '1'}}}]
    assert selection.frame_quality(game, 'a', 50, events) > selection.frame_quality(game, 'b', 50, events)
