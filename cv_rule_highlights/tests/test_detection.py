import json

import cv2
import numpy as np
import pytest
import yaml

from cv_rule_highlights.src.editing.clips import plan_clips
from cv_rule_highlights.src.core.data import Game
from cv_rule_highlights.src.perception.detect import controls, detect, rim_passages


def make_game(tmp_path, *, second_x=100, predicted=False, shooter=True, moving_net=True):
    """A controlled release, rising flight, then a rim descent in two cameras."""
    data = {
        "schema_version": "2.0-rfdetr-rtmpose",
        "video_info": {v: {"fps": 30, "width": 320, "height": 240, "path": str(tmp_path / f"{v}.mp4"), "frame_offset": 0} for v in ("a", "b")},
        "poses_2d": {}, "poses_3d": {}, "quality": {}, "ground_positions_3d": {},
        "balls_2d": {}, "balls_3d": {}, "balls_3d_predicted": {},
    }
    for frame in range(90):
        key = str(frame + 900)
        data["poses_2d"][key] = {}
        if shooter:
            joints = [[30, 180]] * 17
            joints[9] = [30, 160]
            joints[10] = [35, 160]
            obs = {"bbox": [10, 100, 60, 230], "keypoints_xy": joints, "keypoints_conf": [0.9] * 17}
            data["poses_2d"][key] = {"1": {"a": obs, "b": obs}}
            data["ground_positions_3d"][key] = {"1": [2, 2, 0]}
        if frame <= 8:
            x, y = 30, 160
        elif frame <= 20:
            x, y = 30 + (frame - 8) * 70 / 12, 160 - (frame - 8) * 120 / 12
        elif frame <= 40:
            x, y = 100, 40 + (frame - 20) * 7
        else:
            continue
        data["balls_2d"][key] = {}
        for view in ("a", "b"):
            bx = second_x if view == "b" and frame > 20 else x
            data["balls_2d"][key][view] = {"center_xy": [bx, y], "bbox": [bx - 4, y - 4, bx + 4, y + 4], "confidence": 0.9}
        data["balls_3d_predicted"][key] = predicted and frame > 20
    # An actual encoded fabric sequence exercises source-frame offsets and
    # image extraction; no mocked net verdict is used for make assertions.
    for view in ("a", "b"):
        writer = cv2.VideoWriter(str(tmp_path / f"{view}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
        for f in range(90):
            image = np.full((240, 320, 3), 30, np.uint8)
            dx = 3 if moving_net and f >= 29 else 0
            for x in range(80, 125, 8):
                cv2.line(image, (x + dx, 122), (x + dx - 8, 175), (220, 220, 220), 1)
                cv2.line(image, (x + dx, 122), (x + dx + 8, 175), (220, 220, 220), 1)
            writer.write(image)
        writer.release()
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps(data))
    config = tmp_path / "game.yaml"
    config.write_text(yaml.safe_dump({
        "poses": "poses.json", "views": {v: {"rim": [100, 100, 60, 8], "frame_zero": 900} for v in ("a", "b")},
        "teams": {1: "red"}, "rules": {"result_view": "a", "net_roi": [70, 120, 140, 180]},
    }))
    return Game.load(config)


def test_confirmed_make_has_release_and_complete_clip(tmp_path):
    game = make_game(tmp_path)
    report = detect(game)
    makes = [e for e in report["events"] if e["visual_outcome"] == "made"]
    assert len(makes) == 1
    event = makes[0]
    assert event["actor_id"] == "1"
    assert event["release_frame"] < event["resolution_frame"]
    assert abs(event["resolution_frame"] - 929) <= 1
    assert event["awarded_points"] is None
    plan = plan_clips(game, report)
    assert len(plan["clips"]) == 1
    clip = plan["clips"][0]
    assert clip["sync_start_frame"] <= event["release_frame"]
    assert clip["sync_end_frame"] > event["resolution_frame"]
    assert clip["source_start_frame"] == clip["sync_start_frame"] - 900


def test_unknown_shooter_does_not_suppress_make(tmp_path):
    game = make_game(tmp_path, shooter=False)
    event = detect(game)["events"][0]
    assert event["visual_outcome"] == "made"
    assert event["actor_id"] is None


def test_3d_prediction_does_not_erase_real_2d_evidence(tmp_path):
    game = make_game(tmp_path, predicted=True)
    report = detect(game)
    assert any(e["visual_outcome"] == "made" for e in report["events"])
    assert plan_clips(game, report)["clips"]


def test_sideways_view_conflict_blocks_a_make(tmp_path):
    game = make_game(tmp_path, second_x=150)
    report = detect(game)
    assert len(report["events"]) == 1
    assert report["events"][0]["visual_outcome"] == "unknown"
    assert {p["view"] for p in report["events"][0]["evidence"]} == {"a", "b"}
    assert not plan_clips(game, report)["clips"]


def test_long_occlusion_is_unknown_not_missed(tmp_path):
    game = make_game(tmp_path)
    for frame in range(926, 933):
        game.data["balls_2d"].pop(str(frame))
    report = detect(game)
    assert report["events"]
    assert all(e["visual_outcome"] == "unknown" for e in report["events"])


def test_two_sideways_views_cannot_confirm_a_make(tmp_path):
    game = make_game(tmp_path, second_x=150)
    for frame, observations in game.data["balls_2d"].items():
        if int(frame) > 920:
            observations["a"] = observations["b"].copy()
    report = detect(game)
    assert len(report["events"]) == 1
    assert report["events"][0]["visual_outcome"] != "made"


def test_stationary_rim_false_positive_does_not_create_a_shot(tmp_path):
    game = make_game(tmp_path, shooter=False)
    for observations in game.data["balls_2d"].values():
        for ball in observations.values():
            ball["center_xy"] = [100, 90]
            ball["bbox"] = [96, 86, 104, 94]
    assert detect(game)["events"] == []


def test_upward_crossing_and_rim_bounce_do_not_count(tmp_path):
    game = make_game(tmp_path)
    upward = np.array([[f, 100, 140 - f * 5, 4] for f in range(20)], dtype=float)
    assert rim_passages(upward, [100, 100, 60, 8], 30, game.rules) == []
    bounce = np.array([[f, 100, y, 4] for f, y in enumerate([80, 90, 98, 101, 95, 85, 70])], dtype=float)
    assert rim_passages(bounce, [100, 100, 60, 8], 30, game.rules) == []


def test_short_dribble_gap_does_not_change_owner(tmp_path):
    game = make_game(tmp_path)
    for frame in (903, 904):
        game.data["balls_2d"].pop(str(frame))
    segments = controls(game)
    assert len(segments) == 1
    assert segments[0]["player_id"] == "1"


def test_contested_contact_stays_unassigned(tmp_path):
    game = make_game(tmp_path)
    for observations in game.data["poses_2d"].values():
        observations["2"] = observations["1"]
    assert controls(game) == []


def test_view_fps_mismatch_is_rejected(tmp_path):
    game = make_game(tmp_path)
    game.data["video_info"]["b"]["fps"] = 25
    game.poses_path.write_text(json.dumps(game.data))
    with pytest.raises(ValueError, match="same positive FPS"):
        Game.load(game.config_path)


def test_budget_never_truncates_a_shot(tmp_path):
    game = make_game(tmp_path)
    report = detect(game)
    from dataclasses import replace
    game.rules = replace(game.rules, budget_seconds=0.5)
    plan = plan_clips(game, report)
    assert plan["eligible_events"] == 1
    assert not plan["clips"]


def test_projected_crossings_without_net_motion_stay_unknown(tmp_path):
    report = detect(make_game(tmp_path, moving_net=False))
    assert not any(e["visual_outcome"] == "made" for e in report["events"])


def test_ball_escapes_sideways_after_grazing_rim(tmp_path):
    game = make_game(tmp_path)
    samples = np.array([[0, 100, 75, 8], [1, 102, 95, 8], [2, 117, 102, 8], [3, 145, 119, 8]], float)
    passages = rim_passages(samples, [100, 100, 60, 8], 30, game.rules)
    assert passages and all(p["result"] != "inside" for p in passages)


def test_rim_roll_retains_arrival_until_downward_exit(tmp_path):
    game = make_game(tmp_path)
    samples = np.array([[f, 100, y, 8] for f, y in enumerate([75] + [95]*17 + [103, 115, 130])], float)
    passages = rim_passages(samples, [100, 100, 60, 8], 30, game.rules)
    assert len(passages) == 1 and passages[0]["result"] == "inside"
