"""Automatic-run configuration and per-segment input snapshots."""

from pathlib import Path

import yaml

from .court import court_geometry


ALLOWED_CONFIG_KEYS = {
    "schema_version", "annotation_policy", "identity_mode", "team_mode",
    "selection", "scoring_mode", "fps", "project_root", "baseline_events",
    "rules", "rims", "segments",
}


def load_run_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) - ALLOWED_CONFIG_KEYS:
        raise ValueError(f"Unapproved input keys: {sorted(set(config) - ALLOWED_CONFIG_KEYS)}")
    expected = {
        "schema_version": "automatic-highlights-config-1",
        "annotation_policy": "forbidden",
        "identity_mode": "automatic_anonymous",
        "team_mode": "automatic_or_unknown",
        "selection": "all_confirmed",
        "scoring_mode": "unspecified",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must be {value!r}")
    if config["rules"].get("budget_seconds") is not None:
        raise ValueError("No clipping budget is allowed")
    for segment_id, spec in config["segments"].items():
        allowed = {"frames", "poses", "ball_evidence", "rim_scan_cache",
                   "perception_config", "videos"}
        if set(spec) - allowed or set(spec["videos"]) != set(config["rims"]):
            raise ValueError(f"Invalid inputs or views for {segment_id}")
        if not str(segment_id).startswith("segment_") or int(spec["frames"]) <= 0:
            raise ValueError(f"Invalid segment metadata: {segment_id}")
    return config


def write_game_config(config, segment_id, spec, output):
    values = {
        "poses": spec["poses"],
        "views": {
            view: {"path": path, "rim": config["rims"][view]}
            for view, path in spec["videos"].items()
        },
        "rules": config["rules"],
        "teams": {},
    }
    if spec.get("perception_config"):
        values["court"] = court_geometry(spec["perception_config"], config["rims"])
    if spec.get("ball_evidence"):
        values["ball_evidence"] = spec["ball_evidence"]
    target = output / "inputs" / f"{segment_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False),
                      encoding="utf-8")
    return target


