"""
Configuration management for the cancer detection system.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "multimodal_cancer_detector"
    image_size: tuple = (224, 224)
    num_classes: int = 4
    num_stages: int = 5
    vision_model: str = "vit_base_patch16_224"
    efficientnet_model: str = "efficientnet_b4"
    image_embedding_dim: int = 768
    clinical_input_dim: int = 10
    clinical_hidden_dims: list = None
    clinical_embedding_dim: int = 128
    genomic_sequence_length: int = 1000
    genomic_embedding_dim: int = 256
    fusion_dim: int = 512
    dropout: float = 0.3

    def __post_init__(self):
        if self.clinical_hidden_dims is None:
            self.clinical_hidden_dims = [256, 128]


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = "./data"
    dataset_type: str = "custom"
    batch_size: int = 32
    num_workers: int = 4
    train_val_split: float = 0.8
    multimodal: bool = True
    normalize_method: str = "min_max"
    apply_clahe: bool = True
    remove_outliers: bool = True
    outlier_std: float = 3.0


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 0.000001
    detection_weight: float = 1.0
    staging_weight: float = 0.5
    risk_weight: float = 0.3
    use_focal_loss: bool = True
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    gradient_clip: float = 1.0
    label_smoothing: float = 0.1
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001
    checkpoint_dir: str = "./checkpoints"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    model_format: str = "pytorch"
    use_onnx: bool = False
    onnx_opset_version: int = 13
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    use_gpu: bool = True
    fp16: bool = False


class Config:
    """Main configuration class."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_dict = {}

        # Load default config
        default_config_path = Path(__file__).parent.parent.parent / "configs" / "default_config.yaml"
        if default_config_path.exists():
            self.config_dict = self._load_yaml(default_config_path)

        # Override with custom config if provided
        if config_path:
            custom_config = self._load_yaml(config_path)
            self.config_dict = self._deep_update(self.config_dict, custom_config)

        # Parse into dataclasses
        self.model = self._parse_model_config()
        self.data = self._parse_data_config()
        self.training = self._parse_training_config()
        self.deployment = self._parse_deployment_config()

        # Store other configs as dict
        self.evaluation = self.config_dict.get('evaluation', {})
        self.logging = self.config_dict.get('logging', {})
        self.experiment = self.config_dict.get('experiment', {})

    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _deep_update(self, base: Dict, update: Dict) -> Dict:
        """Deep update dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                base[key] = self._deep_update(base.get(key, {}), value)
            else:
                base[key] = value
        return base

    def _parse_model_config(self) -> ModelConfig:
        """Parse model configuration."""
        model_dict = self.config_dict.get('model', {})
        return ModelConfig(
            name=model_dict.get('name', 'multimodal_cancer_detector'),
            image_size=tuple(model_dict.get('image_size', [224, 224])),
            num_classes=model_dict.get('num_classes', 4),
            num_stages=model_dict.get('num_stages', 5),
            vision_model=model_dict.get('vision_model', 'vit_base_patch16_224'),
            efficientnet_model=model_dict.get('efficientnet_model', 'efficientnet_b4'),
            image_embedding_dim=model_dict.get('image_embedding_dim', 768),
            clinical_input_dim=model_dict.get('clinical_input_dim', 10),
            clinical_hidden_dims=model_dict.get('clinical_hidden_dims', [256, 128]),
            clinical_embedding_dim=model_dict.get('clinical_embedding_dim', 128),
            genomic_sequence_length=model_dict.get('genomic_sequence_length', 1000),
            genomic_embedding_dim=model_dict.get('genomic_embedding_dim', 256),
            fusion_dim=model_dict.get('fusion_dim', 512),
            dropout=model_dict.get('dropout', 0.3)
        )

    def _parse_data_config(self) -> DataConfig:
        """Parse data configuration."""
        data_dict = self.config_dict.get('data', {})
        return DataConfig(
            data_dir=data_dict.get('data_dir', './data'),
            dataset_type=data_dict.get('dataset_type', 'custom'),
            batch_size=data_dict.get('batch_size', 32),
            num_workers=data_dict.get('num_workers', 4),
            train_val_split=data_dict.get('train_val_split', 0.8),
            multimodal=data_dict.get('multimodal', True),
            normalize_method=data_dict.get('normalize_method', 'min_max'),
            apply_clahe=data_dict.get('apply_clahe', True),
            remove_outliers=data_dict.get('remove_outliers', True),
            outlier_std=data_dict.get('outlier_std', 3.0)
        )

    def _parse_training_config(self) -> TrainingConfig:
        """Parse training configuration."""
        train_dict = self.config_dict.get('training', {})
        loss_dict = train_dict.get('loss', {})
        early_stopping = train_dict.get('early_stopping', {})
        checkpoint = train_dict.get('checkpoint', {})

        return TrainingConfig(
            num_epochs=train_dict.get('num_epochs', 100),
            learning_rate=train_dict.get('learning_rate', 0.0001),
            weight_decay=train_dict.get('weight_decay', 0.01),
            optimizer=train_dict.get('optimizer', 'adamw'),
            scheduler=train_dict.get('scheduler', 'cosine'),
            warmup_epochs=train_dict.get('warmup_epochs', 5),
            min_lr=train_dict.get('min_lr', 0.000001),
            detection_weight=loss_dict.get('detection_weight', 1.0),
            staging_weight=loss_dict.get('staging_weight', 0.5),
            risk_weight=loss_dict.get('risk_weight', 0.3),
            use_focal_loss=loss_dict.get('use_focal_loss', True),
            focal_alpha=loss_dict.get('focal_alpha', 0.25),
            focal_gamma=loss_dict.get('focal_gamma', 2.0),
            gradient_clip=train_dict.get('gradient_clip', 1.0),
            label_smoothing=train_dict.get('label_smoothing', 0.1),
            early_stopping_patience=early_stopping.get('patience', 15),
            early_stopping_min_delta=early_stopping.get('min_delta', 0.001),
            checkpoint_dir=checkpoint.get('checkpoint_dir', './checkpoints')
        )

    def _parse_deployment_config(self) -> DeploymentConfig:
        """Parse deployment configuration."""
        deploy_dict = self.config_dict.get('deployment', {})
        server_dict = deploy_dict.get('server', {})

        return DeploymentConfig(
            model_format=deploy_dict.get('model_format', 'pytorch'),
            use_onnx=deploy_dict.get('use_onnx', False),
            onnx_opset_version=deploy_dict.get('onnx_opset_version', 13),
            server_host=server_dict.get('host', '0.0.0.0'),
            server_port=server_dict.get('port', 8000),
            use_gpu=deploy_dict.get('use_gpu', True),
            fp16=deploy_dict.get('fp16', False)
        )

    def save(self, output_path: str):
        """
        Save configuration to file.

        Args:
            output_path: Output file path (YAML or JSON)
        """
        output_path = Path(output_path)

        # Convert to dictionary
        config_dict = {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'training': asdict(self.training),
            'deployment': asdict(self.deployment),
            'evaluation': self.evaluation,
            'logging': self.logging,
            'experiment': self.experiment
        }

        # Save based on extension
        if output_path.suffix == '.yaml' or output_path.suffix == '.yml':
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
        elif output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {output_path.suffix}")

        logger.info(f"Configuration saved to {output_path}")

    def __str__(self) -> str:
        """String representation."""
        lines = [
            "="*80,
            "CONFIGURATION",
            "="*80,
            "",
            "MODEL:",
            f"  Name: {self.model.name}",
            f"  Image Size: {self.model.image_size}",
            f"  Num Classes: {self.model.num_classes}",
            "",
            "DATA:",
            f"  Data Dir: {self.data.data_dir}",
            f"  Batch Size: {self.data.batch_size}",
            f"  Multimodal: {self.data.multimodal}",
            "",
            "TRAINING:",
            f"  Epochs: {self.training.num_epochs}",
            f"  Learning Rate: {self.training.learning_rate}",
            f"  Optimizer: {self.training.optimizer}",
            "",
            "DEPLOYMENT:",
            f"  Model Format: {self.deployment.model_format}",
            f"  Use GPU: {self.deployment.use_gpu}",
            "",
            "="*80
        ]
        return "\n".join(lines)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use defaults.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    return Config(config_path)
