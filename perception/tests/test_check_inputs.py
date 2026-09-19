import json

import pytest
import yaml

from src.check_inputs import check_inputs


def test_missing_config_does_not_silently_use_defaults(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config not found"):
        check_inputs(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    "options",
    [
        {"start_frame": -1, "end_frame": 10},
        {"start_frame": 10, "end_frame": 10},
        {"start_frame": 10, "limit": 0},
        {"views": ["view1"]},
        {"views": ["view1", "unknown"]},
    ],
)
def test_invalid_request_rejected_before_inference(options):
    with pytest.raises(ValueError):
        check_inputs("config/config.yaml", **options)


def _minimal_config(tmp_path, **rfdetr):
    """Config satisfying every check that precedes the RF-DETR backend one."""
    (tmp_path / "intrinsics.json").write_text(
        json.dumps({"view1": {}, "view2": {}}), encoding="utf-8"
    )
    (tmp_path / "extrinsics.json").write_text(
        json.dumps({"view1": {}, "view2": {}}), encoding="utf-8"
    )
    for name in ("bg.png", "pose.py", "pose.pth"):
        (tmp_path / name).write_text("", encoding="utf-8")
    data = {
        "project_root": str(tmp_path),
        "videos": {
            "view1": str(tmp_path / "1.mp4"),
            "view2": str(tmp_path / "2.mp4"),
        },
        "camera": {
            "intrinsics_path": str(tmp_path / "intrinsics.json"),
            "extrinsics_path": str(tmp_path / "extrinsics.json"),
            "view_to_camera": {"view1": "view1", "view2": "view2"},
        },
        "assets": {"court_background": str(tmp_path / "bg.png")},
        "pose": {
            "config": str(tmp_path / "pose.py"),
            "checkpoint": str(tmp_path / "pose.pth"),
        },
        "reid": {
            "use_appearance_embeddings": False,
            "use_deep_appearance_embeddings": False,
            "use_face_embeddings": False,
        },
        "trajectory": {"fps": 30, "start_frame": 0, "process_seconds": 1},
        "rfdetr": rfdetr,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def _request():
    return {"start_frame": 0, "end_frame": 1, "views": ["view1", "view2"]}


def test_hybrid_backend_reaches_video_checks(tmp_path):
    """`hybrid` is a shipped backend value; only unknown names are rejected."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"")
    ball_path = tmp_path / "ball.pth"
    ball_path.write_bytes(b"")
    config = _minimal_config(
        tmp_path,
        backend="hybrid",
        onnx_path=str(onnx_path),
        ball_checkpoint_path=str(ball_path),
    )
    # Reaching the video stage proves the backend/branch checks were passed.
    with pytest.raises(OSError, match="Cannot open"):
        check_inputs(config, **_request())


def test_hybrid_without_onnx_model_rejected(tmp_path):
    """hybrid runs the person detector on ONNX; the engine is not that path."""
    engine_path = tmp_path / "model.engine"
    engine_path.write_bytes(b"")
    config = _minimal_config(tmp_path, backend="hybrid", engine_path=str(engine_path))
    with pytest.raises(FileNotFoundError, match="No model file for hybrid"):
        check_inputs(config, **_request())


def test_hybrid_missing_ball_checkpoint_fails_fast(tmp_path):
    """A path that is set but absent degrades ball detection silently downstream."""
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"")
    config = _minimal_config(
        tmp_path,
        backend="hybrid",
        onnx_path=str(onnx_path),
        ball_checkpoint_path=str(tmp_path / "absent.pth"),
    )
    with pytest.raises(FileNotFoundError, match="ball_checkpoint_path"):
        check_inputs(config, **_request())


def test_unknown_backend_rejected(tmp_path):
    config = _minimal_config(tmp_path, backend="bogus", onnx_path="model.onnx")
    with pytest.raises(ValueError, match="Unsupported perception detector backend"):
        check_inputs(config, **_request())
