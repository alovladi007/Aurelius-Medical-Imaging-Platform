"""Vision Transformer (ViT) models for cancer classification."""

from typing import List, Optional

import timm
import torch
import torch.nn as nn

from .heads import ClassificationHead


class ViTClassifier(nn.Module):
    """Vision Transformer classifier for histopathology images."""

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 2,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
        dropout: float = 0.2,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "gelu",
        use_batch_norm: bool = False,
    ):
        """Initialize ViT classifier.

        Args:
            backbone: ViT variant (e.g., 'vit_tiny_patch16_224', 'vit_base_patch16_224')
            pretrained: Use ImageNet pretrained weights
            num_classes: Number of output classes
            freeze_backbone: Freeze all backbone layers
            freeze_layers: Number of initial transformer blocks to freeze
            dropout: Dropout probability
            hidden_dims: Optional hidden layers before classification
            activation: Activation function (typically 'gelu' for ViT)
            use_batch_norm: Use batch normalization in head (typically False for ViT)
        """
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        # Load pretrained ViT using timm
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Freeze layers if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            # Freeze initial transformer blocks
            if hasattr(self.backbone, "blocks"):
                for i in range(min(freeze_layers, len(self.backbone.blocks))):
                    for param in self.backbone.blocks[i].parameters():
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
        features = self.backbone(x)  # (B, feature_dim)

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

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention maps from the last transformer block.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Attention maps
        """
        # This is a simplified version
        # For full implementation, you'd need to register hooks on attention layers
        return None

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def create_vit_model(config: dict) -> ViTClassifier:
    """Create ViT model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        ViT classifier model
    """
    model_cfg = config.get("model", {})
    head_cfg = model_cfg.get("head", {})

    return ViTClassifier(
        backbone=model_cfg.get("backbone", "vit_base_patch16_224"),
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 2),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        freeze_layers=model_cfg.get("freeze_layers", 0),
        dropout=head_cfg.get("dropout", 0.2),
        hidden_dims=head_cfg.get("hidden_dims", None),
        activation=head_cfg.get("activation", "gelu"),
        use_batch_norm=head_cfg.get("use_batch_norm", False),
    )
