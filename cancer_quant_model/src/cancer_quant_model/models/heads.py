"""Classification heads for models."""

from typing import List, Optional

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Custom classification head."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.5,
        activation: str = "relu",
        batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        """
        Initialize classification head.

        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            hidden_dims: Hidden layer dimensions (None for single linear layer)
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'silu')
            batch_norm: Use batch normalization
            use_layer_norm: Use layer normalization instead of batch norm
        """
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        # Build layers
        layers = []

        if hidden_dims is None:
            # Simple linear head
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_features, num_classes))
        else:
            # Multi-layer head
            prev_dim = in_features

            for hidden_dim in hidden_dims:
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

                layers.append(nn.Linear(prev_dim, hidden_dim))

                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                elif batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))

                # Activation
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "silu":
                    layers.append(nn.SiLU(inplace=True))

                prev_dim = hidden_dim

            # Final layer
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(prev_dim, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.head(x)


class AttentionPool2d(nn.Module):
    """Attention-based global pooling."""

    def __init__(self, in_features: int, num_heads: int = 8):
        """
        Initialize attention pooling.

        Args:
            in_features: Number of input features
            num_heads: Number of attention heads
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            batch_first=True,
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Pooled features (B, C)
        """
        b, c, h, w = x.shape

        # Reshape to sequence
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # (B, 1, C)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+H*W, C)

        # Apply attention
        attn_output, _ = self.attention(cls_tokens, x, x)  # (B, 1, C)

        # Return CLS token output
        return attn_output.squeeze(1)  # (B, C)


class GeM(nn.Module):
    """Generalized Mean Pooling."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        """
        Initialize GeM pooling.

        Args:
            p: Power parameter
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.gem(x, p=self.p, eps=self.eps)

    @staticmethod
    def gem(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
        """Generalized mean pooling."""
        return nn.functional.avg_pool2d(
            x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))
        ).pow(1.0 / p)
