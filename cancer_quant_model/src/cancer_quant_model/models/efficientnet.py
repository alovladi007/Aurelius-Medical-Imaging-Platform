"""EfficientNet-based models for cancer classification."""

from typing import List, Optional

import timm
import torch
import torch.nn as nn

from .heads import ClassificationHead


class EfficientNetClassifier(nn.Module):
    """EfficientNet-based classifier for histopathology images."""

    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
        dropout: float = 0.4,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        """Initialize EfficientNet classifier.

        Args:
            backbone: EfficientNet variant (e.g., 'efficientnet_b0', 'efficientnet_b3')
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes
            freeze_backbone: Freeze all backbone layers
            freeze_layers: Number of initial layers to freeze
            dropout: Dropout probability
            hidden_dims: Optional hidden layers before classification
            activation: Activation function
            use_batch_norm: Use batch normalization in head
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Load pretrained EfficientNet using timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="",  # Remove global pooling
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            if len(features.shape) == 4:
                # Has spatial dimensions, need pooling
                self.feature_dim = features.shape[1]
                self.needs_pooling = True
            else:
                self.feature_dim = features.shape[1]
                self.needs_pooling = False

        # Global pooling layer
        if self.needs_pooling:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = nn.Identity()

        # Freeze layers if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            # Freeze initial blocks
            blocks = list(self.backbone.children())
            for i in range(min(freeze_layers, len(blocks))):
                for param in blocks[i].parameters():
                    param.requires_grad = False

        # Classification head
        self.head = ClassificationHead(
            in_features=self.feature_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W)
            return_features: If True, return features instead of logits

        Returns:
            Output logits (B, num_classes) or features (B, feature_dim)
        """
        # Extract features
        features = self.backbone(x)

        # Global pooling if needed
        if self.needs_pooling:
            features = self.global_pool(features)
            features = features.squeeze(-1).squeeze(-1)

        if return_features:
            return features

        # Classification
        logits = self.head(features)

        return logits

    def get_feature_extractor(self) -> nn.Module:
        """Get feature extractor (backbone without head).

        Returns:
            Feature extractor module
        """
        return self.backbone

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_efficientnet_model(config: dict) -> EfficientNetClassifier:
    """Create EfficientNet model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        EfficientNet classifier model
    """
    model_cfg = config.get("model", {})
    head_cfg = model_cfg.get("head", {})

    return EfficientNetClassifier(
        backbone=model_cfg.get("backbone", "efficientnet_b3"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 2),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        freeze_layers=model_cfg.get("freeze_layers", 0),
        dropout=head_cfg.get("dropout", 0.4),
        hidden_dims=head_cfg.get("hidden_dims", None),
        activation=head_cfg.get("activation", "relu"),
        use_batch_norm=head_cfg.get("use_batch_norm", True),
    )
