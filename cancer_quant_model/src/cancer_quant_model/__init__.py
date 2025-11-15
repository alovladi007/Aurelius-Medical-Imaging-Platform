"""Cancer Quantitative Histopathology Model.

A production-ready deep learning system for quantitative cancer research using
histopathology tissue slide images.

Features:
- Multiple architectures (ResNet, EfficientNet, ViT)
- Quantitative feature extraction (color, texture, morphology, deep features)
- Explainability via Grad-CAM
- MLflow experiment tracking
- FastAPI inference server
- Comprehensive evaluation metrics

Author: Research Team
License: MIT
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Research Team"
__license__ = "MIT"

# Import main components
from .config import Config, load_config, merge_configs
from .data import (
    FolderDataset,
    HistopathDataModule,
    HistopathDataset,
    get_inference_transforms,
    get_train_transforms,
    get_val_transforms,
)
from .models import (
    EfficientNetClassifier,
    ResNetClassifier,
    ViTClassifier,
    create_model,
)
from .training import (
    CheckpointCallback,
    EarlyStoppingCallback,
    Evaluator,
    MLflowLoggingCallback,
    Trainer,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "Config",
    "load_config",
    "merge_configs",
    # Data
    "HistopathDataset",
    "FolderDataset",
    "HistopathDataModule",
    "get_train_transforms",
    "get_val_transforms",
    "get_inference_transforms",
    # Models
    "ResNetClassifier",
    "EfficientNetClassifier",
    "ViTClassifier",
    "create_model",
    # Training
    "Trainer",
    "Evaluator",
    "EarlyStoppingCallback",
    "CheckpointCallback",
    "MLflowLoggingCallback",
]
