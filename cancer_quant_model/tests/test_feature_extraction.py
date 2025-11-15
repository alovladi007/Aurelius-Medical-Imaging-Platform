"""Tests for feature extraction."""

import numpy as np
import pytest
from PIL import Image

from cancer_quant_model.utils.feature_utils import QuantitativeFeatureExtractor


class TestFeatureExtraction:
    """Tests for quantitative feature extraction."""

    @pytest.fixture
    def synthetic_image(self):
        """Create a synthetic image for testing."""
        # Create a simple synthetic histopathology-like image
        img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)

        # Add some "nuclei-like" dark spots
        for _ in range(20):
            x = np.random.randint(10, 246)
            y = np.random.randint(10, 246)
            img[y - 5 : y + 5, x - 5 : x + 5] = np.random.randint(20, 80, (10, 10, 3))

        return img

    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = QuantitativeFeatureExtractor()
        assert extractor is not None

    def test_color_features(self, synthetic_image):
        """Test color feature extraction."""
        extractor = QuantitativeFeatureExtractor()
        features = extractor.extract_color_features(synthetic_image)

        # Check that features are returned
        assert len(features) > 0

        # Check for expected feature names
        assert "color_r_mean" in features
        assert "color_g_mean" in features
        assert "color_b_mean" in features
        assert "color_r_std" in features

        # Check that values are numeric and not NaN
        for key, value in features.items():
            assert isinstance(value, (int, float, np.integer, np.floating))
            assert not np.isnan(value)

    def test_texture_features(self, synthetic_image):
        """Test texture feature extraction."""
        extractor = QuantitativeFeatureExtractor()
        features = extractor.extract_texture_features(synthetic_image)

        # Check that features are returned
        assert len(features) > 0

        # Check for GLCM features
        assert "glcm_contrast_mean" in features
        assert "glcm_homogeneity_mean" in features
        assert "glcm_energy_mean" in features

        # Check for LBP features
        assert "lbp_mean" in features

        # Check that values are numeric and not NaN
        for key, value in features.items():
            assert isinstance(value, (int, float, np.integer, np.floating))
            assert not np.isnan(value)

    def test_morphological_features(self, synthetic_image):
        """Test morphological feature extraction."""
        extractor = QuantitativeFeatureExtractor()
        features = extractor.extract_morphological_features(synthetic_image)

        # Check that features are returned
        assert len(features) > 0

        # Check for expected features
        assert "morph_num_objects" in features
        assert "morph_edge_density" in features

        # Check that values are numeric and not NaN
        for key, value in features.items():
            assert isinstance(value, (int, float, np.integer, np.floating))
            assert not np.isnan(value)

    def test_frequency_features(self, synthetic_image):
        """Test frequency domain feature extraction."""
        extractor = QuantitativeFeatureExtractor()
        features = extractor.extract_frequency_features(synthetic_image)

        # Check that features are returned
        assert len(features) > 0

        # Check for expected features
        assert "freq_mean" in features
        assert "freq_power_low" in features
        assert "freq_power_high" in features

        # Check that values are numeric and not NaN
        for key, value in features.items():
            assert isinstance(value, (int, float, np.integer, np.floating))
            assert not np.isnan(value)

    def test_all_features(self, synthetic_image):
        """Test extraction of all features."""
        extractor = QuantitativeFeatureExtractor()
        features = extractor.extract_all_features(synthetic_image)

        # Should have many features from all categories
        assert len(features) > 50

        # Check that all values are valid
        for key, value in features.items():
            assert isinstance(value, (int, float, np.integer, np.floating))
            assert not np.isnan(value), f"Feature {key} is NaN"
            assert not np.isinf(value), f"Feature {key} is infinite"

    def test_feature_reproducibility(self, synthetic_image):
        """Test that features are reproducible."""
        extractor = QuantitativeFeatureExtractor()

        features1 = extractor.extract_all_features(synthetic_image)
        features2 = extractor.extract_all_features(synthetic_image)

        # Features should be identical
        assert len(features1) == len(features2)
        for key in features1.keys():
            assert key in features2
            np.testing.assert_almost_equal(features1[key], features2[key])
