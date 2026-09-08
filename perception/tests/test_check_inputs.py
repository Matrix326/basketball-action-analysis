import pytest

from src.check_inputs import check_inputs


def test_missing_config_does_not_silently_use_defaults(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config not found"):
        check_inputs(tmp_path / "missing.yaml")


@pytest.mark.parametrize("options", [
    {"start_frame": -1, "end_frame": 10},
    {"start_frame": 10, "end_frame": 10},
    {"start_frame": 10, "limit": 0},
    {"views": ["view1"]},
    {"views": ["view1", "unknown"]},
])
def test_invalid_request_rejected_before_inference(options):
    with pytest.raises(ValueError):
        check_inputs("config/config.yaml", **options)
