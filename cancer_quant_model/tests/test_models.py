"""Tests for model architectures."""

import pytest
import torch

from cancer_quant_model.models.resnet import ResNetModel
from cancer_quant_model.models.efficientnet import EfficientNetModel
from cancer_quant_model.models.vit import ViTModel


class TestModels:
    """Tests for model architectures."""

    def test_resnet_forward(self):
        """Test ResNet forward pass."""
        model = ResNetModel(
            variant="resnet18", num_classes=2, pretrained=False, extract_features=False
        )

        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 2)

    def test_resnet_with_features(self):
        """Test ResNet with feature extraction."""
        model = ResNetModel(
            variant="resnet18", num_classes=2, pretrained=False, extract_features=True
        )

        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert isinstance(output, dict)
        assert "logits" in output
        assert "features" in output
        assert output["logits"].shape == (2, 2)

    def test_efficientnet_forward(self):
        """Test EfficientNet forward pass."""
        model = EfficientNetModel(
            variant="efficientnet_b0", num_classes=2, pretrained=False
        )

        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 2)

    @pytest.mark.slow
    def test_vit_forward(self):
        """Test ViT forward pass."""
        model = ViTModel(
            variant="vit_tiny_patch16_224", num_classes=2, pretrained=False
        )

        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 2)
