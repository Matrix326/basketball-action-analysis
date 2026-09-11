"""
配置管理模块
"""

from .loader import Config, load_config, get_default_config

__all__ = [
    "Config",
    "load_config",
    "get_default_config",
]
