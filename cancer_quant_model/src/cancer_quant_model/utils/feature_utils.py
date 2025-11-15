"""Feature extraction utilities for quantitative analysis."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage import feature, measure
from skimage.feature import graycomatrix, graycoprops


class QuantitativeFeatureExtractor:
    """Extract quantitative features from histopathology images."""

    def __init__(self):
        """Initialize feature extractor."""
        pass

    def extract_all_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract all quantitative features from an image.

        Args:
            image: Input image (H, W, C) in RGB format, uint8 [0, 255]

        Returns:
            Dictionary of features
        """
        features = {}

        # Color features
        features.update(self.extract_color_features(image))

        # Texture features
        features.update(self.extract_texture_features(image))

        # Morphological features
        features.update(self.extract_morphological_features(image))

        # Frequency features
        features.update(self.extract_frequency_features(image))

        return features

    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract color statistics features.

        Args:
            image: Input image (H, W, C) in RGB, uint8

        Returns:
            Dictionary of color features
        """
        features = {}

        # Per-channel statistics
        for i, channel in enumerate(["r", "g", "b"]):
            channel_data = image[:, :, i].flatten()

            features[f"color_{channel}_mean"] = np.mean(channel_data)
            features[f"color_{channel}_std"] = np.std(channel_data)
            features[f"color_{channel}_median"] = np.median(channel_data)
            features[f"color_{channel}_min"] = np.min(channel_data)
            features[f"color_{channel}_max"] = np.max(channel_data)
            features[f"color_{channel}_q25"] = np.percentile(channel_data, 25)
            features[f"color_{channel}_q75"] = np.percentile(channel_data, 75)

        # HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(["h", "s", "v"]):
            channel_data = hsv[:, :, i].flatten()
            features[f"color_{channel}_mean"] = np.mean(channel_data)
            features[f"color_{channel}_std"] = np.std(channel_data)

        # LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        for i, channel in enumerate(["l", "a_star", "b_star"]):
            channel_data = lab[:, :, i].flatten()
            features[f"color_{channel}_mean"] = np.mean(channel_data)
            features[f"color_{channel}_std"] = np.std(channel_data)

        return features

    def extract_texture_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract texture features using GLCM.

        Args:
            image: Input image (H, W, C), uint8

        Returns:
            Dictionary of texture features
        """
        features = {}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # GLCM parameters
        distances = [1, 3, 5]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

        # Compute GLCM
        glcm = graycomatrix(
            gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True
        )

        # Extract properties
        for prop in properties:
            prop_values = graycoprops(glcm, prop)

            features[f"glcm_{prop}_mean"] = np.mean(prop_values)
            features[f"glcm_{prop}_std"] = np.std(prop_values)
            features[f"glcm_{prop}_min"] = np.min(prop_values)
            features[f"glcm_{prop}_max"] = np.max(prop_values)

        # Local Binary Patterns (LBP)
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")

        features["lbp_mean"] = np.mean(lbp)
        features["lbp_std"] = np.std(lbp)
        features["lbp_median"] = np.median(lbp)

        # Histogram of LBP
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        lbp_hist = lbp_hist.astype(float) / np.sum(lbp_hist)

        for i in range(min(10, len(lbp_hist))):  # First 10 bins
            features[f"lbp_hist_bin_{i}"] = lbp_hist[i]

        return features

    def extract_morphological_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract morphological features.

        Args:
            image: Input image (H, W, C), uint8

        Returns:
            Dictionary of morphological features
        """
        features = {}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Threshold to create binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (assume tissue is darker)
        if np.mean(binary) > 127:
            binary = 255 - binary

        # Connected components
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if len(regions) > 0:
            # Area statistics
            areas = [r.area for r in regions]
            features["morph_num_objects"] = len(regions)
            features["morph_area_mean"] = np.mean(areas)
            features["morph_area_std"] = np.std(areas)
            features["morph_area_total"] = np.sum(areas)

            # Perimeter statistics
            perimeters = [r.perimeter for r in regions]
            features["morph_perimeter_mean"] = np.mean(perimeters)
            features["morph_perimeter_std"] = np.std(perimeters)

            # Eccentricity
            eccentricities = [r.eccentricity for r in regions]
            features["morph_eccentricity_mean"] = np.mean(eccentricities)
            features["morph_eccentricity_std"] = np.std(eccentricities)

            # Solidity
            solidities = [r.solidity for r in regions]
            features["morph_solidity_mean"] = np.mean(solidities)
            features["morph_solidity_std"] = np.std(solidities)

            # Circularity (4π × area / perimeter²)
            circularities = [
                (4 * np.pi * r.area) / (r.perimeter**2) if r.perimeter > 0 else 0
                for r in regions
            ]
            features["morph_circularity_mean"] = np.mean(circularities)
            features["morph_circularity_std"] = np.std(circularities)
        else:
            # No objects found
            features["morph_num_objects"] = 0
            for key in [
                "area_mean",
                "area_std",
                "area_total",
                "perimeter_mean",
                "perimeter_std",
                "eccentricity_mean",
                "eccentricity_std",
                "solidity_mean",
                "solidity_std",
                "circularity_mean",
                "circularity_std",
            ]:
                features[f"morph_{key}"] = 0.0

        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features["morph_edge_density"] = np.sum(edges > 0) / edges.size

        return features

    def extract_frequency_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT.

        Args:
            image: Input image (H, W, C), uint8

        Returns:
            Dictionary of frequency features
        """
        features = {}

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)

        # Compute FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)

        # Log transform for better visualization/analysis
        magnitude_spectrum_log = np.log1p(magnitude_spectrum)

        features["freq_mean"] = np.mean(magnitude_spectrum_log)
        features["freq_std"] = np.std(magnitude_spectrum_log)
        features["freq_max"] = np.max(magnitude_spectrum_log)

        # Power in different frequency bands
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2

        # Low frequency (center region)
        radius_low = min(h, w) // 8
        y, x = np.ogrid[:h, :w]
        mask_low = ((x - center_w) ** 2 + (y - center_h) ** 2) <= radius_low**2
        features["freq_power_low"] = np.sum(magnitude_spectrum[mask_low])

        # High frequency (outer region)
        radius_high = min(h, w) // 4
        mask_high = ((x - center_w) ** 2 + (y - center_h) ** 2) > radius_high**2
        features["freq_power_high"] = np.sum(magnitude_spectrum[mask_high])

        # Mid frequency
        mask_mid = ~mask_low & ~mask_high
        features["freq_power_mid"] = np.sum(magnitude_spectrum[mask_mid])

        # Ratio of high to low frequency
        if features["freq_power_low"] > 0:
            features["freq_high_low_ratio"] = (
                features["freq_power_high"] / features["freq_power_low"]
            )
        else:
            features["freq_high_low_ratio"] = 0.0

        return features


def extract_patch_features(
    image: np.ndarray, patch_size: int = 224, stride: Optional[int] = None
) -> Tuple[List[Dict[str, float]], List[Tuple[int, int]]]:
    """
    Extract features from patches of a large image.

    Args:
        image: Large input image (H, W, C)
        patch_size: Size of patches
        stride: Stride for patch extraction (default: patch_size // 2)

    Returns:
        Tuple of (list of feature dicts, list of patch coordinates)
    """
    if stride is None:
        stride = patch_size // 2

    extractor = QuantitativeFeatureExtractor()

    h, w = image.shape[:2]
    features_list = []
    coords_list = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y : y + patch_size, x : x + patch_size]

            # Extract features from patch
            features = extractor.extract_all_features(patch)
            features_list.append(features)
            coords_list.append((y, x))

    return features_list, coords_list
