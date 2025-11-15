"""Classification heads for cancer quantitative models."""

from typing import List, Optional

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Flexible classification head with optional hidden layers."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
    ):
        """Initialize classification head.

        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            hidden_dims: Optional list of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes

        # Build layers
        layers = []

        if hidden_dims:
            # Add hidden layers
            prev_dim = in_features
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))

                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))

                # Activation
                if activation == "relu":
                    layers.append(nn.ReLU(inplace=True))
                elif activation == "gelu":
                    layers.append(nn.GELU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU(0.2, inplace=True))

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

                prev_dim = hidden_dim

            # Final classification layer
            layers.append(nn.Linear(prev_dim, num_classes))

        else:
            # Simple head: just dropout and linear
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(in_features, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, in_features)

        Returns:
            Output logits (B, num_classes)
        """
        return self.head(x)


class MultiTaskHead(nn.Module):
    """Multi-task head for joint classification and regression."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_regression_targets: int = 0,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ):
        """Initialize multi-task head.

        Args:
            in_features: Number of input features
            num_classes: Number of classification classes
            num_regression_targets: Number of regression targets
            hidden_dims: Optional shared hidden layers
            dropout: Dropout probability
        """
        super().__init__()

        # Shared layers
        if hidden_dims:
            shared_layers = []
            prev_dim = in_features

            for hidden_dim in hidden_dims:
                shared_layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = hidden_dim

            self.shared = nn.Sequential(*shared_layers)
            final_dim = prev_dim
        else:
            self.shared = nn.Identity()
            final_dim = in_features

        # Task-specific heads
        self.classification_head = nn.Linear(final_dim, num_classes)

        if num_regression_targets > 0:
            self.regression_head = nn.Linear(final_dim, num_regression_targets)
        else:
            self.regression_head = None

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Input features

        Returns:
            Dictionary with 'classification' and optionally 'regression' outputs
        """
        shared_features = self.shared(x)

        outputs = {
            "classification": self.classification_head(shared_features),
        }

        if self.regression_head is not None:
            outputs["regression"] = self.regression_head(shared_features)

        return outputs


class AttentionPoolingHead(nn.Module):
    """Classification head with attention-based pooling for tile aggregation."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        """Initialize attention pooling head.

        Args:
            in_features: Number of input features per tile
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for attention
            dropout: Dropout probability
        """
        super().__init__()

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention pooling.

        Args:
            x: Input features (B, N, in_features) where N is number of tiles

        Returns:
            Output logits (B, num_classes)
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # (B, N, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted pooling
        weighted_features = torch.sum(x * attention_weights, dim=1)  # (B, in_features)

        # Classification
        return self.classifier(weighted_features)
