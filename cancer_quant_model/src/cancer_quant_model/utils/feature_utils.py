"""Feature extraction utilities for quantitative analysis."""

from typing import Dict, Optional, Tuple

import cv2
import mahotas
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from skimage import color, feature, measure, morphology
from skimage.filters import threshold_otsu


def extract_color_features(image: np.ndarray) -> Dict[str, float]:
    """Extract color statistical features from image.

    Args:
        image: RGB image (H, W, 3) with values in [0, 255]

    Returns:
        Dictionary of color features
    """
    features = {}

    # Ensure image is in correct range
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # RGB statistics
    for i, channel in enumerate(["R", "G", "B"]):
        channel_data = image[:, :, i].flatten()

        features[f"{channel}_mean"] = float(np.mean(channel_data))
        features[f"{channel}_std"] = float(np.std(channel_data))
        features[f"{channel}_skew"] = float(stats.skew(channel_data))
        features[f"{channel}_kurtosis"] = float(stats.kurtosis(channel_data))
        features[f"{channel}_median"] = float(np.median(channel_data))
        features[f"{channel}_min"] = float(np.min(channel_data))
        features[f"{channel}_max"] = float(np.max(channel_data))
        features[f"{channel}_range"] = float(np.ptp(channel_data))

    # HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    for i, channel in enumerate(["H", "S", "V"]):
        channel_data = hsv[:, :, i].flatten()

        features[f"{channel}_mean"] = float(np.mean(channel_data))
        features[f"{channel}_std"] = float(np.std(channel_data))

    # LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    for i, channel in enumerate(["L", "A", "B_lab"]):
        channel_data = lab[:, :, i].flatten()

        features[f"{channel}_mean"] = float(np.mean(channel_data))
        features[f"{channel}_std"] = float(np.std(channel_data))

    return features


def extract_texture_features(image: np.ndarray) -> Dict[str, float]:
    """Extract texture features from image.

    Args:
        image: RGB or grayscale image

    Returns:
        Dictionary of texture features
    """
    features = {}

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        if image.max() <= 1.0:
            gray = color.rgb2gray(image)
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    # Haralick texture features (GLCM-based)
    try:
        haralick_features = mahotas.features.haralick(gray, ignore_zeros=True, return_mean=True)

        haralick_names = [
            "angular_second_moment",
            "contrast",
            "correlation",
            "sum_of_squares_variance",
            "inverse_difference_moment",
            "sum_average",
            "sum_variance",
            "sum_entropy",
            "entropy",
            "difference_variance",
            "difference_entropy",
            "info_measure_corr_1",
            "info_measure_corr_2",
        ]

        for name, value in zip(haralick_names, haralick_features):
            features[f"haralick_{name}"] = float(value)

    except Exception as e:
        # If Haralick fails, fill with zeros
        for i in range(13):
            features[f"haralick_feature_{i}"] = 0.0

    # Local Binary Patterns (LBP)
    try:
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method="uniform")
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2), density=True
        )

        # Statistical features of LBP
        features["lbp_mean"] = float(np.mean(lbp))
        features["lbp_std"] = float(np.std(lbp))
        features["lbp_entropy"] = float(stats.entropy(lbp_hist + 1e-10))

    except Exception:
        features["lbp_mean"] = 0.0
        features["lbp_std"] = 0.0
        features["lbp_entropy"] = 0.0

    # GLCM properties (alternative texture features)
    try:
        # Compute GLCM
        glcm = feature.graycomatrix(
            gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256
        )

        # Extract properties
        properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
        for prop in properties:
            prop_values = feature.graycoprops(glcm, prop).flatten()
            features[f"glcm_{prop}_mean"] = float(np.mean(prop_values))
            features[f"glcm_{prop}_std"] = float(np.std(prop_values))

    except Exception:
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
            features[f"glcm_{prop}_mean"] = 0.0
            features[f"glcm_{prop}_std"] = 0.0

    return features


def extract_morphological_features(image: np.ndarray) -> Dict[str, float]:
    """Extract morphological features (nuclei detection and shape analysis).

    Args:
        image: RGB image

    Returns:
        Dictionary of morphological features
    """
    features = {}

    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            if image.max() <= 1.0:
                gray = color.rgb2gray(image)
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Simple thresholding for nuclei detection
        try:
            threshold = threshold_otsu(gray)
            binary = gray < threshold  # Dark regions (nuclei)
        except Exception:
            binary = gray < np.mean(gray)

        # Clean up binary image
        binary = morphology.remove_small_objects(binary, min_size=20)
        binary = morphology.remove_small_holes(binary, area_threshold=20)

        # Label regions
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if len(regions) > 0:
            # Count features
            features["nuclei_count"] = len(regions)

            # Shape features
            areas = [r.area for r in regions]
            perimeters = [r.perimeter for r in regions]
            eccentricities = [r.eccentricity for r in regions]
            solidities = [r.solidity for r in regions]

            features["nuclei_area_mean"] = float(np.mean(areas))
            features["nuclei_area_std"] = float(np.std(areas))
            features["nuclei_area_median"] = float(np.median(areas))
            features["nuclei_area_min"] = float(np.min(areas))
            features["nuclei_area_max"] = float(np.max(areas))

            features["nuclei_perimeter_mean"] = float(np.mean(perimeters))
            features["nuclei_perimeter_std"] = float(np.std(perimeters))

            features["nuclei_eccentricity_mean"] = float(np.mean(eccentricities))
            features["nuclei_eccentricity_std"] = float(np.std(eccentricities))

            features["nuclei_solidity_mean"] = float(np.mean(solidities))
            features["nuclei_solidity_std"] = float(np.std(solidities))

            # Density
            total_area = gray.shape[0] * gray.shape[1]
            features["nuclei_density"] = float(len(regions) / total_area * 10000)  # per 100x100

        else:
            # No regions detected
            features["nuclei_count"] = 0
            features["nuclei_area_mean"] = 0.0
            features["nuclei_area_std"] = 0.0
            features["nuclei_area_median"] = 0.0
            features["nuclei_area_min"] = 0.0
            features["nuclei_area_max"] = 0.0
            features["nuclei_perimeter_mean"] = 0.0
            features["nuclei_perimeter_std"] = 0.0
            features["nuclei_eccentricity_mean"] = 0.0
            features["nuclei_eccentricity_std"] = 0.0
            features["nuclei_solidity_mean"] = 0.0
            features["nuclei_solidity_std"] = 0.0
            features["nuclei_density"] = 0.0

    except Exception as e:
        # If morphological analysis fails, fill with zeros
        features["nuclei_count"] = 0
        for key in [
            "nuclei_area_mean",
            "nuclei_area_std",
            "nuclei_area_median",
            "nuclei_area_min",
            "nuclei_area_max",
            "nuclei_perimeter_mean",
            "nuclei_perimeter_std",
            "nuclei_eccentricity_mean",
            "nuclei_eccentricity_std",
            "nuclei_solidity_mean",
            "nuclei_solidity_std",
            "nuclei_density",
        ]:
            features[key] = 0.0

    return features


def extract_classic_features(image: np.ndarray) -> Dict[str, float]:
    """Extract all classic (non-deep learning) features.

    Args:
        image: RGB image

    Returns:
        Dictionary of all classic features
    """
    features = {}

    # Extract all feature types
    features.update(extract_color_features(image))
    features.update(extract_texture_features(image))
    features.update(extract_morphological_features(image))

    return features


def extract_deep_features(
    model: nn.Module,
    image: torch.Tensor,
    layer_name: Optional[str] = None,
    device: str = "cpu",
) -> np.ndarray:
    """Extract deep features from a model's intermediate layer.

    Args:
        model: PyTorch model
        image: Input image tensor (C, H, W) or (B, C, H, W)
        layer_name: Name of layer to extract features from (if None, use penultimate layer)
        device: Device to run inference on

    Returns:
        Feature vector as numpy array
    """
    model = model.to(device)
    model.eval()

    # Add batch dimension if needed
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    features = []

    def hook_fn(module, input, output):
        """Hook function to capture intermediate features."""
        features.append(output)

    # Register hook
    if layer_name:
        # Find layer by name
        for name, module in model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)
                break
    else:
        # Use penultimate layer (before final classification)
        # For most models, this is typically avgpool or the last feature layer
        modules = list(model.children())
        if len(modules) > 1:
            handle = modules[-2].register_forward_hook(hook_fn)
        else:
            # Fallback: hook the entire model
            handle = model.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model(image)

    # Remove hook
    handle.remove()

    # Extract features
    if len(features) > 0:
        feat = features[0]

        # Global average pooling if spatial dimensions exist
        if len(feat.shape) == 4:  # (B, C, H, W)
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.squeeze(-1).squeeze(-1)

        # Convert to numpy
        feat = feat.cpu().numpy()

        # Remove batch dimension if single image
        if feat.shape[0] == 1:
            feat = feat.squeeze(0)

        return feat
    else:
        # Fallback: return empty array
        return np.array([])


def extract_all_features(
    image: np.ndarray,
    model: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Extract both classic and deep features.

    Args:
        image: RGB image (numpy array or PIL Image)
        model: Optional PyTorch model for deep feature extraction
        device: Device for deep feature extraction

    Returns:
        Dictionary of all features
    """
    features = {}

    # Classic features
    features.update(extract_classic_features(image))

    # Deep features (if model provided)
    if model is not None:
        # Convert image to tensor
        if isinstance(image, np.ndarray):
            # Normalize to [0, 1]
            if image.max() > 1.0:
                image_tensor = torch.from_numpy(image / 255.0).float()
            else:
                image_tensor = torch.from_numpy(image).float()

            # Convert to (C, H, W)
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)
            elif image_tensor.shape[-1] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)

        deep_feat = extract_deep_features(model, image_tensor, device=device)

        # Add deep features with indexed names
        for i, val in enumerate(deep_feat):
            features[f"deep_feature_{i}"] = float(val)

    return features
