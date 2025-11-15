"""Configuration management for the cancer quantitative model."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf


class Config:
    """Configuration manager for the project."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Path to configuration directory. If None, uses default.
        """
        if config_dir is None:
            # Assuming we're in src/cancer_quant_model, go up to project root
            self.config_dir = Path(__file__).parent.parent.parent / "config"
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")

    def load_yaml(self, config_file: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            config_file: Name of the config file (e.g., 'dataset.yaml')

        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def load_omega(self, config_file: str) -> DictConfig:
        """
        Load a YAML configuration file as OmegaConf.

        Args:
            config_file: Name of the config file

        Returns:
            OmegaConf configuration
        """
        config_path = self.config_dir / config_file
        return OmegaConf.load(config_path)

    def merge_configs(self, *config_files: str) -> DictConfig:
        """
        Merge multiple configuration files.

        Args:
            *config_files: Names of config files to merge

        Returns:
            Merged OmegaConf configuration
        """
        configs = [self.load_omega(cf) for cf in config_files]
        return OmegaConf.merge(*configs)

    def save_config(self, config: Dict[str, Any], output_path: Path):
        """
        Save configuration to a YAML file.

        Args:
            config: Configuration dictionary
            output_path: Path to save the config
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

    @staticmethod
    def get_default_dataset_config() -> Dict[str, Any]:
        """Get default dataset configuration."""
        return {
            "dataset": {
                "type": "folder_binary",
                "name": "histopathology_cancer",
                "paths": {
                    "raw_data_dir": "data/raw",
                    "processed_data_dir": "data/processed",
                    "splits_dir": "data/splits",
                },
                "image": {
                    "target_size": [224, 224],
                    "channels": 3,
                },
                "labels": {
                    "num_classes": 2,
                    "class_names": ["non_cancer", "cancer"],
                },
                "dataloader": {
                    "batch_size": 32,
                    "num_workers": 4,
                },
            }
        }

    @staticmethod
    def override_config(
        base_config: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Override configuration values.

        Args:
            base_config: Base configuration
            overrides: Override values (nested dict)

        Returns:
            Updated configuration
        """
        base_omega = OmegaConf.create(base_config)
        override_omega = OmegaConf.create(overrides)
        merged = OmegaConf.merge(base_omega, override_omega)
        return OmegaConf.to_container(merged, resolve=True)


def load_config(config_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convenience function to load a configuration file.

    Args:
        config_name: Name of the config file
        config_dir: Optional config directory

    Returns:
        Configuration dictionary
    """
    config_manager = Config(config_dir)
    return config_manager.load_yaml(config_name)
