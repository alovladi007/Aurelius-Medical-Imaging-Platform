"""Model architectures for cancer quantitative model."""

from .efficientnet import EfficientNetClassifier, create_efficientnet_model
from .heads import AttentionPoolingHead, ClassificationHead, MultiTaskHead
from .resnet import ResNetClassifier, create_resnet_model
from .vit import ViTClassifier, create_vit_model

__all__ = [
    # Heads
    "ClassificationHead",
    "MultiTaskHead",
    "AttentionPoolingHead",
    # Models
    "ResNetClassifier",
    "EfficientNetClassifier",
    "ViTClassifier",
    # Factory functions
    "create_resnet_model",
    "create_efficientnet_model",
    "create_vit_model",
]


def create_model(config: dict, model_type: str = "resnet"):
    """Factory function to create model from config.

    Args:
        config: Model configuration
        model_type: Type of model ('resnet', 'efficientnet', 'vit')

    Returns:
        Model instance
    """
    if model_type == "resnet" or config.get("model", {}).get("name", "").startswith("resnet"):
        return create_resnet_model(config)
    elif model_type == "efficientnet" or config.get("model", {}).get("name", "").startswith(
        "efficientnet"
    ):
        return create_efficientnet_model(config)
    elif model_type == "vit" or config.get("model", {}).get("name", "").startswith("vit"):
        return create_vit_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
