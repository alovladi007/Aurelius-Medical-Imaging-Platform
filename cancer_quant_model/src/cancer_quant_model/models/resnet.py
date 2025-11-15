"""ResNet model implementation."""

from typing import Dict, List, Optional

import timm
import torch
import torch.nn as nn

from cancer_quant_model.models.heads import ClassificationHead, GeM


class ResNetModel(nn.Module):
    """ResNet model for histopathology classification."""

    def __init__(
        self,
        variant: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        freeze_layers: int = 0,
        dropout: float = 0.5,
        global_pool: str = "avg",
        use_custom_head: bool = False,
        hidden_dims: Optional[List[int]] = None,
        extract_features: bool = False,
    ):
        """
        Initialize ResNet model.

        Args:
            variant: ResNet variant (resnet18, resnet34, resnet50, resnet101, resnet152)
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone weights
            freeze_layers: Number of initial layers to freeze
            dropout: Dropout rate
            global_pool: Global pooling type ('avg', 'max', 'gem')
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
            global_pool="",  # Remove global pooling
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)
            self.feature_dim = dummy_output.shape[1]

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Global pooling
        if global_pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        elif global_pool == "max":
            self.global_pool = nn.AdaptiveMaxPool2d(1)
        elif global_pool == "gem":
            self.global_pool = GeM()
        else:
            raise ValueError(f"Unknown global_pool: {global_pool}")

        # Classification head
        if use_custom_head:
            self.head = ClassificationHead(
                in_features=self.feature_dim,
                num_classes=num_classes,
                hidden_dims=hidden_dims,
                dropout=dropout,
            )
        else:
            layers = []
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(self.feature_dim, num_classes))
            self.head = nn.Sequential(*layers)

    def _freeze_layers(self, num_layers: int):
        """Freeze initial layers."""
        # For ResNet, freeze initial blocks
        modules_to_freeze = []

        if hasattr(self.backbone, "conv1"):
            modules_to_freeze.append(self.backbone.conv1)
        if hasattr(self.backbone, "bn1"):
            modules_to_freeze.append(self.backbone.bn1)

        # Freeze layer blocks
        for i in range(1, min(num_layers + 1, 5)):
            layer_name = f"layer{i}"
            if hasattr(self.backbone, layer_name):
                modules_to_freeze.append(getattr(self.backbone, layer_name))

        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Logits (B, num_classes) or dict with logits and features
        """
        # Backbone
        features = self.backbone(x)  # (B, C, H, W)

        # Global pooling
        pooled = self.global_pool(features)  # (B, C, 1, 1)
        pooled = pooled.flatten(1)  # (B, C)

        # Classification
        logits = self.head(pooled)

        if self.extract_features:
            return {"logits": logits, "features": pooled}
        else:
            return logits

    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.feature_dim


def build_resnet_model(config: Dict) -> ResNetModel:
    """
    Build ResNet model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        ResNet model
    """
    model_config = config.get("model", {})

    return ResNetModel(
        variant=model_config.get("variant", "resnet50"),
        num_classes=model_config.get("head", {}).get("num_classes", 2),
        pretrained=model_config.get("pretrained", True),
        freeze_backbone=model_config.get("freeze_backbone", False),
        freeze_layers=model_config.get("freeze_layers", 0),
        dropout=model_config.get("head", {}).get("dropout", 0.5),
        global_pool=model_config.get("global_pool", "avg"),
        use_custom_head=model_config.get("head", {}).get("use_custom_head", False),
        hidden_dims=model_config.get("head", {}).get("hidden_dims", None),
        extract_features=model_config.get("extract_features", False),
    )
