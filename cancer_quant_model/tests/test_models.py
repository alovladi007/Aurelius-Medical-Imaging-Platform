"""Tests for model architectures."""
import pytest
import torch

from cancer_quant_model.models import ResNetClassifier, EfficientNetClassifier, ViTClassifier


def test_resnet_forward():
    """Test ResNet forward pass."""
    model = ResNetClassifier(
        backbone="resnet18",
        pretrained=False,
        num_classes=2,
    )
    
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 2)


def test_efficientnet_forward():
    """Test EfficientNet forward pass."""
    model = EfficientNetClassifier(
        backbone="efficientnet_b0",
        pretrained=False,
        num_classes=2,
    )
    
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 2)


def test_vit_forward():
    """Test ViT forward pass."""
    model = ViTClassifier(
        backbone="vit_tiny_patch16_224",
        pretrained=False,
        num_classes=2,
    )
    
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    
    assert output.shape == (2, 2)


def test_feature_extraction():
    """Test feature extraction."""
    model = ResNetClassifier(
        backbone="resnet18",
        pretrained=False,
        num_classes=2,
    )
    
    x = torch.randn(2, 3, 224, 224)
    features = model(x, return_features=True)
    
    assert features.shape == (2, 512)  # resnet18 feature dim


if __name__ == "__main__":
    pytest.main([__file__])
