"""Tests for feature extraction."""
import numpy as np
import pytest

from cancer_quant_model.utils.feature_utils import (
    extract_color_features,
    extract_texture_features,
    extract_classic_features,
)


def test_color_features():
    """Test color feature extraction."""
    # Create synthetic image
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    features = extract_color_features(image)
    
    # Check that we have expected features
    assert "R_mean" in features
    assert "G_std" in features
    assert "B_skew" in features
    assert "H_mean" in features
    
    # Check all values are numeric
    for v in features.values():
        assert isinstance(v, (int, float))


def test_texture_features():
    """Test texture feature extraction."""
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    features = extract_texture_features(image)
    
    # Check for some expected features
    assert any("haralick" in k for k in features.keys())
    assert "lbp_mean" in features or "lbp_entropy" in features or len(features) > 0
    
    # Check all values are numeric
    for v in features.values():
        assert isinstance(v, (int, float))


def test_classic_features():
    """Test combined classic feature extraction."""
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    features = extract_classic_features(image)
    
    # Should have many features
    assert len(features) > 20
    
    # Check all values are numeric and no NaNs
    for k, v in features.items():
        assert isinstance(v, (int, float))
        assert not np.isnan(v), f"NaN found in feature: {k}"


if __name__ == "__main__":
    pytest.main([__file__])
