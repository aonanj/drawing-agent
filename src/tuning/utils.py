from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(config_path: Path) -> DictConfig:
    """Load a YAML config file into an OmegaConf DictConfig."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config: DictConfig | ListConfig = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"Expected DictConfig, got {type(config).__name__}")
    return config


def flatten_config(config: DictConfig) -> Dict[str, Any]:
    """Convert DictConfig to a plain dictionary for logging libraries."""
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
