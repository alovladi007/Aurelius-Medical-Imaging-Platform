"""Configuration management for the cancer quantitative model."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for loading and merging YAML configs."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager.

        Args:
            config_path: Path to config file or directory
        """
        self.config_path = config_path
        self.config: Optional[DictConfig] = None

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> DictConfig:
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Loaded configuration as DictConfig
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.config = OmegaConf.create(config_dict)
        return self.config

    def merge(self, *configs: str) -> DictConfig:
        """Merge multiple configuration files.

        Args:
            *configs: Paths to config files to merge (later configs override earlier)

        Returns:
            Merged configuration
        """
        merged = OmegaConf.create({})

        for config_path in configs:
            cfg = self.load(config_path)
            merged = OmegaConf.merge(merged, cfg)

        self.config = merged
        return self.config

    def update(self, updates: Dict[str, Any]) -> DictConfig:
        """Update config with new values.

        Args:
            updates: Dictionary of updates to apply

        Returns:
            Updated configuration
        """
        if self.config is None:
            self.config = OmegaConf.create(updates)
        else:
            update_cfg = OmegaConf.create(updates)
            self.config = OmegaConf.merge(self.config, update_cfg)

        return self.config

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by key.

        Args:
            key: Dot-separated key (e.g., "model.backbone")
            default: Default value if key not found

        Returns:
            Config value or default
        """
        if self.config is None:
            return default

        try:
            return OmegaConf.select(self.config, key, default=default)
        except Exception:
            return default

    def save(self, output_path: str):
        """Save configuration to YAML file.

        Args:
            output_path: Path to save config
        """
        if self.config is None:
            raise ValueError("No configuration loaded")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            OmegaConf.save(self.config, f)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration as dictionary
        """
        if self.config is None:
            return {}

        return OmegaConf.to_container(self.config, resolve=True)

    def __repr__(self) -> str:
        """String representation."""
        if self.config is None:
            return "Config(empty)"
        return f"Config({OmegaConf.to_yaml(self.config)})"


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Loaded configuration
    """
    config = Config()
    return config.load(config_path)


def merge_configs(*config_paths: str) -> DictConfig:
    """Merge multiple configuration files.

    Args:
        *config_paths: Paths to config files to merge

    Returns:
        Merged configuration
    """
    config = Config()
    return config.merge(*config_paths)


def save_config(config: DictConfig, output_path: str):
    """Save configuration to YAML file.

    Args:
        config: Configuration to save
        output_path: Path to save config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        OmegaConf.save(config, f)
