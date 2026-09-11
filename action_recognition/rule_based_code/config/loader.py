"""
配置加载器
支持 YAML 配置文件、变量引用（${var}）、深度合并、运行时覆盖
"""

import os
import copy
import yaml
from typing import Any, Dict, List, Optional, Tuple


_DEFAULT_YAML_PATH = os.path.join(os.path.dirname(__file__), "default.yaml")


def _resolve_variables(cfg: Dict, root: Optional[Dict] = None) -> Dict:
    root = root or cfg
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_variables(value, root)
        elif isinstance(value, str) and "${" in value:
            resolved[key] = _interpolate(value, root)
        elif isinstance(value, list):
            resolved[key] = [
                _interpolate(v, root) if isinstance(v, str) and "${" in v else v
                for v in value
            ]
        else:
            resolved[key] = value
    return resolved


def _interpolate(s: str, root: Dict) -> Any:
    if not isinstance(s, str) or "${" not in s:
        return s

    start = s.find("${")
    end = s.find("}", start)
    if start == -1 or end == -1:
        return s

    var_name = s[start + 2:end]
    parts = var_name.split(".")
    value = root
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return s

    if start == 0 and end == len(s) - 1:
        return value

    prefix = s[:start]
    suffix = _interpolate(s[end + 1:], root)
    return str(prefix) + str(value) + str(suffix)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    def __init__(self, data: Dict):
        self._data = _resolve_variables(data)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        path = os.path.abspath(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if data.get("project_root") in (None, "", "auto"):
            data["project_root"] = os.path.dirname(os.path.dirname(path))
        return cls(data)

    @classmethod
    def from_dict(cls, data: Dict) -> "Config":
        return cls(data)

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        value = self._data
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def to_dict(self) -> Dict:
        return copy.deepcopy(self._data)

    def merge(self, override: Dict) -> "Config":
        merged = _deep_merge(self._data, override)
        return Config(merged)

    def override(self, flat_dict: Dict) -> "Config":
        data = copy.deepcopy(self._data)
        for key, value in flat_dict.items():
            parts = key.split(".")
            target = data
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value
        return Config(data)

    @property
    def player_colors_rgb(self) -> List[Tuple[int, int, int]]:
        colors = self.get("player_colors", [])
        return [tuple(c) for c in colors]

    @property
    def player_colors_bgr(self) -> List[Tuple[int, int, int]]:
        return [(b, g, r) for r, g, b in self.player_colors_rgb]

    @property
    def skeleton_connections(self) -> List[Tuple[int, int]]:
        conns = self.get("skeleton_connections", [])
        return [tuple(c) for c in conns]

    @property
    def video_paths(self) -> Dict[str, str]:
        return self.get("videos", {})

    @property
    def view_to_camera(self) -> Dict[str, str]:
        return self.get("camera.view_to_camera", {})

    def __repr__(self) -> str:
        return f"Config({self._data})"


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict] = None,
) -> Config:
    if config_path and os.path.exists(config_path):
        cfg = Config.from_yaml(config_path)
    else:
        cfg = Config.from_yaml(_DEFAULT_YAML_PATH)

    if overrides:
        cfg = cfg.merge(overrides)

    return cfg


def get_default_config() -> Config:
    return Config.from_yaml(_DEFAULT_YAML_PATH)
