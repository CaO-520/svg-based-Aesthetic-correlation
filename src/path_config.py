from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYSTEM_PATH_CONFIG = PROJECT_ROOT / "configs" / "system_path.yaml"


def load_system_paths(config_path: Path = DEFAULT_SYSTEM_PATH_CONFIG) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"System path config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_system_path(
    section: str,
    key: str,
    config_path: Path = DEFAULT_SYSTEM_PATH_CONFIG,
) -> Path:
    config = load_system_paths(config_path)
    value = config.get(section, {}).get(key)
    if not value:
        raise KeyError(f"Missing {section}.{key} in {config_path}")

    return Path(os.path.expandvars(str(value))).expanduser()


def get_aesthetic_model_dir(config_path: Path = DEFAULT_SYSTEM_PATH_CONFIG) -> Path:
    return get_system_path("models", "aesthetic_predictor", config_path)
