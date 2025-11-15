"""ResNet-based models for cancer classification."""

from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models

from .heads import ClassificationHead


class ResNetClassifier(nn.Module):
    """ResNet-based classifier for histopathology images."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
        dropout: float = 0.3,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        """Initialize ResNet classifier.

        Args:
            backbone: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes
            freeze_backbone: Freeze all backbone layers
            freeze_layers: Number of initial layers to freeze (0 = none)
            dropout: Dropout probability
            hidden_dims: Optional hidden layers before classification
            activation: Activation function
            use_batch_norm: Use batch normalization in head
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Load pretrained ResNet
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "resnet101":
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == "resnet152":
            self.backbone = models.resnet152(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown ResNet variant: {backbone}")

        # Remove original FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Freeze layers if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            # Freeze initial layers
            layers = list(self.backbone.children())
            for i in range(min(freeze_layers, len(layers))):
                for param in layers[i].parameters():
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
        features = features.squeeze(-1).squeeze(-1)  # (B, feature_dim)

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


def create_resnet_model(config: dict) -> ResNetClassifier:
    """Create ResNet model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        ResNet classifier model
    """
    model_cfg = config.get("model", {})
    head_cfg = model_cfg.get("head", {})

    return ResNetClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 2),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        freeze_layers=model_cfg.get("freeze_layers", 0),
        dropout=head_cfg.get("dropout", 0.3),
        hidden_dims=head_cfg.get("hidden_dims", None),
        activation=head_cfg.get("activation", "relu"),
        use_batch_norm=head_cfg.get("use_batch_norm", True),
    )
