"""Vision Transformer (ViT) model implementation."""

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn

from cancer_quant_model.models.heads import ClassificationHead


class ViTModel(nn.Module):
    """Vision Transformer model for histopathology classification."""

    def __init__(
        self,
        variant: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        use_custom_head: bool = True,
        hidden_dims: Optional[List[int]] = None,
        extract_features: bool = False,
    ):
        """
        Initialize ViT model.

        Args:
            variant: ViT variant (vit_tiny, vit_small, vit_base, vit_large)
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone weights
            freeze_layers: Number of transformer blocks to freeze
            dropout: Dropout rate
            drop_path_rate: Stochastic depth rate
            use_custom_head: Use custom classification head
            hidden_dims: Hidden dimensions for custom head
            extract_features: Return features in addition to logits
        """
        super().__init__()

        self.variant = variant
        self.num_classes = num_classes
        self.extract_features = extract_features

        # Load pretrained model
        self.backbone = timm.create_model(
            variant,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_path_rate=drop_path_rate,
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Classification head
        if use_custom_head:
            self.head = ClassificationHead(
                in_features=self.feature_dim,
                num_classes=num_classes,
                hidden_dims=hidden_dims or [384],
                dropout=dropout,
                activation="gelu",
                batch_norm=False,
                use_layer_norm=True,
            )
        else:
            layers = []
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(self.feature_dim, num_classes))
            self.head = nn.Sequential(*layers)

    def _freeze_layers(self, num_blocks: int):
        """Freeze transformer blocks."""
        if hasattr(self.backbone, "blocks"):
            for i in range(min(num_blocks, len(self.backbone.blocks))):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        """Forward pass."""
        # Backbone (returns CLS token or pooled features)
        features = self.backbone(x)  # (B, feature_dim)

        # Classification
        logits = self.head(features)

        if self.extract_features:
            return {"logits": logits, "features": features}
        else:
            return logits

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim


def build_vit_model(config: Dict) -> ViTModel:
    """Build ViT model from config."""
    model_config = config.get("model", {})

    return ViTModel(
        variant=model_config.get("variant", "vit_base_patch16_224"),
        num_classes=model_config.get("head", {}).get("num_classes", 2),
        pretrained=model_config.get("pretrained", True),
        freeze_backbone=model_config.get("freeze_backbone", False),
        freeze_layers=model_config.get("freeze_layers", 0),
        dropout=model_config.get("head", {}).get("dropout", 0.1),
        drop_path_rate=model_config.get("transformer", {}).get("drop_path_rate", 0.1),
        use_custom_head=model_config.get("head", {}).get("use_custom_head", True),
        hidden_dims=model_config.get("head", {}).get("hidden_dims", [384]),
        extract_features=model_config.get("extract_features", False),
    )
