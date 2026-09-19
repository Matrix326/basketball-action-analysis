import json

import pytest

from src.hoop_detection.run_hoop_detection import (
    ball_positions_from_poses,
    ball_positions_from_trajectory,
    read_hoop_file,
    write_hoop_json,
)


def test_ball_positions_from_poses_skips_predicted_and_missing(tmp_path):
    path = tmp_path / "poses_3d.json"
    path.write_text(
        json.dumps(
            {
                "balls_3d": {
                    "900": [1.0, 2.0, 3.0],
                    "901": None,
                    "902": [4.0, 5.0, 6.0],
                },
                "balls_3d_predicted": {"902": True},
            }
        ),
        encoding="utf-8",
    )
    # 901 is unobserved and 902 is a predicted hold-over, not an observation.
    assert ball_positions_from_poses(path) == [[1.0, 2.0, 3.0]]


def test_ball_positions_from_trajectory_skips_null_positions(tmp_path):
    path = tmp_path / "ball_trajectory.json"
    path.write_text(
        json.dumps(
            {
                "frames": {
                    "900": {"position": [1.0, 2.0, 3.0]},
                    "901": {"position": None},
                }
            }
        ),
        encoding="utf-8",
    )
    assert ball_positions_from_trajectory(path) == [[1.0, 2.0, 3.0]]


def test_read_hoop_file_accepts_hoop_schema(tmp_path):
    path = tmp_path / "hoop_3d.json"
    path.write_text(
        json.dumps({"hoop_center": [7.653, 1.801, 2.883]}), encoding="utf-8"
    )
    assert read_hoop_file(path) == [7.653, 1.801, 2.883]


def test_read_hoop_file_accepts_bare_point(tmp_path):
    path = tmp_path / "point.json"
    path.write_text(json.dumps([1.0, 2.0, 3.0]), encoding="utf-8")
    assert read_hoop_file(path) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "payload",
    [
        {"hoop_center": [1.0, 2.0]},
        {"hoop_center": [1.0, 2.0, 3.0, 4.0]},
        {"hoop_center": [1.0, 2.0, None]},
        {},
        [1.0, 2.0],
    ],
)
def test_read_hoop_file_rejects_bad_payload(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_hoop_file(path)


def test_write_hoop_json_records_source_and_ambiguity(tmp_path):
    path = tmp_path / "nested" / "hoop_3d.json"
    write_hoop_json(
        path,
        [7.724, -0.085, 2.893],
        start_frame=900,
        end_frame=1800,
        candidates=[{"point": [7.724, -0.085, 2.893], "views": ["view1", "view4"]}],
        source="view_support",
        ambiguous=True,
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "hoop-3d/v1"
    assert data["hoop_center"] == [7.724, -0.085, 2.893]
    assert data["height_m"] == 2.893
    assert data["source"] == "view_support"
    assert data["ambiguous"] is True
    assert data["candidates"] == [
        {"point": [7.724, -0.085, 2.893], "views": ["view1", "view4"]}
    ]
