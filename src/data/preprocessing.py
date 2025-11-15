"""
Preprocessing utilities for medical images and multimodal data.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Dict, List
import cv2
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class MedicalImagePreprocessor:
    """Preprocessor for medical imaging data."""

    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize_method: str = 'min_max',
                 apply_clahe: bool = False,
                 remove_outliers: bool = True,
                 outlier_std: float = 3.0):
        """
        Initialize medical image preprocessor.

        Args:
            target_size: Target image size (height, width)
            normalize_method: Normalization method ('min_max', 'z_score', 'none')
            apply_clahe: Whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            remove_outliers: Whether to clip outlier values
            outlier_std: Number of standard deviations for outlier removal
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.apply_clahe = apply_clahe
        self.remove_outliers = remove_outliers
        self.outlier_std = outlier_std

        if apply_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess medical image.

        Args:
            image: Input image array

        Returns:
            Preprocessed image array
        """
        # Remove outliers
        if self.remove_outliers:
            image = self._remove_outliers(image)

        # Apply CLAHE if enabled
        if self.apply_clahe:
            image = self._apply_clahe(image)

        # Resize
        image = self._resize(image)

        # Normalize
        image = self._normalize(image)

        return image

    def _remove_outliers(self, image: np.ndarray) -> np.ndarray:
        """Remove outlier pixel values."""
        mean = image.mean()
        std = image.std()
        lower = mean - self.outlier_std * std
        upper = mean + self.outlier_std * std
        return np.clip(image, lower, upper)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        # Convert to uint8 for CLAHE
        if image.max() <= 1.0:
            image_uint8 = (image * 255).astype(np.uint8)
        else:
            image_uint8 = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            clahe_image = self.clahe.apply(image_uint8)
        else:
            # Apply to each channel
            clahe_channels = []
            for i in range(image.shape[2]):
                clahe_channel = self.clahe.apply(image_uint8[:, :, i])
                clahe_channels.append(clahe_channel)
            clahe_image = np.stack(clahe_channels, axis=2)

        # Convert back to float
        return clahe_image.astype(np.float32) / 255.0

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size."""
        if image.shape[:2] != self.target_size:
            if image.ndim == 2:
                image = cv2.resize(image, self.target_size[::-1],
                                 interpolation=cv2.INTER_LINEAR)
            else:
                image = cv2.resize(image, self.target_size[::-1],
                                 interpolation=cv2.INTER_LINEAR)
        return image

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values."""
        if self.normalize_method == 'min_max':
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
        elif self.normalize_method == 'z_score':
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std
        elif self.normalize_method == 'none':
            pass
        else:
            raise ValueError(f"Unknown normalization method: {self.normalize_method}")

        return image


class ClinicalDataPreprocessor:
    """Preprocessor for clinical/tabular data."""

    def __init__(self,
                 categorical_features: Optional[List[str]] = None,
                 numerical_features: Optional[List[str]] = None,
                 normalize_numerical: bool = True,
                 handle_missing: str = 'mean'):
        """
        Initialize clinical data preprocessor.

        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            normalize_numerical: Whether to normalize numerical features
            handle_missing: How to handle missing values ('mean', 'median', 'zero', 'drop')
        """
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.normalize_numerical = normalize_numerical
        self.handle_missing = handle_missing

        # Statistics for normalization
        self.feature_means = {}
        self.feature_stds = {}
        self.feature_mins = {}
        self.feature_maxs = {}
        self.fitted = False

    def fit(self, data: Dict[str, np.ndarray]):
        """
        Fit preprocessor to data (compute statistics).

        Args:
            data: Dictionary mapping feature names to arrays
        """
        for feature in self.numerical_features:
            if feature in data:
                values = data[feature]
                valid_values = values[~np.isnan(values)]

                if len(valid_values) > 0:
                    self.feature_means[feature] = valid_values.mean()
                    self.feature_stds[feature] = valid_values.std()
                    self.feature_mins[feature] = valid_values.min()
                    self.feature_maxs[feature] = valid_values.max()

        self.fitted = True

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Transform clinical data.

        Args:
            data: Dictionary mapping feature names to arrays

        Returns:
            Preprocessed data dictionary
        """
        processed_data = {}

        # Process numerical features
        for feature in self.numerical_features:
            if feature in data:
                values = data[feature].copy()

                # Handle missing values
                values = self._handle_missing_values(values, feature)

                # Normalize if enabled
                if self.normalize_numerical and self.fitted:
                    if feature in self.feature_stds and self.feature_stds[feature] > 0:
                        values = (values - self.feature_means[feature]) / self.feature_stds[feature]

                processed_data[feature] = values

        # Process categorical features (one-hot encoding would go here)
        for feature in self.categorical_features:
            if feature in data:
                processed_data[feature] = data[feature]

        return processed_data

    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def _handle_missing_values(self, values: np.ndarray, feature: str) -> np.ndarray:
        """Handle missing values in feature."""
        if not np.any(np.isnan(values)):
            return values

        if self.handle_missing == 'mean' and feature in self.feature_means:
            values[np.isnan(values)] = self.feature_means[feature]
        elif self.handle_missing == 'median':
            median = np.nanmedian(values)
            values[np.isnan(values)] = median
        elif self.handle_missing == 'zero':
            values[np.isnan(values)] = 0
        elif self.handle_missing == 'drop':
            # Return only non-nan values (caller must handle shape change)
            values = values[~np.isnan(values)]

        return values


class GenomicDataPreprocessor:
    """Preprocessor for genomic sequence data."""

    def __init__(self,
                 sequence_length: int = 1000,
                 encoding: str = 'one_hot'):
        """
        Initialize genomic data preprocessor.

        Args:
            sequence_length: Target sequence length
            encoding: Encoding method ('one_hot', 'ordinal')
        """
        self.sequence_length = sequence_length
        self.encoding = encoding

        # Nucleotide mapping
        self.nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.num_nucleotides = len(self.nucleotide_map)

    def __call__(self, sequence: str) -> np.ndarray:
        """
        Preprocess genomic sequence.

        Args:
            sequence: DNA/RNA sequence string

        Returns:
            Encoded sequence array
        """
        # Convert to uppercase
        sequence = sequence.upper()

        # Pad or truncate to target length
        if len(sequence) < self.sequence_length:
            sequence = sequence + 'N' * (self.sequence_length - len(sequence))
        else:
            sequence = sequence[:self.sequence_length]

        # Encode
        if self.encoding == 'one_hot':
            return self._one_hot_encode(sequence)
        elif self.encoding == 'ordinal':
            return self._ordinal_encode(sequence)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode DNA sequence."""
        encoded = np.zeros((self.sequence_length, self.num_nucleotides), dtype=np.float32)

        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.nucleotide_map:
                encoded[i, self.nucleotide_map[nucleotide]] = 1.0

        return encoded

    def _ordinal_encode(self, sequence: str) -> np.ndarray:
        """Ordinal encode DNA sequence."""
        encoded = np.zeros(self.sequence_length, dtype=np.int64)

        for i, nucleotide in enumerate(sequence):
            if nucleotide in self.nucleotide_map:
                encoded[i] = self.nucleotide_map[nucleotide]

        return encoded


def preprocess_batch(images: List[np.ndarray],
                    preprocessor: MedicalImagePreprocessor) -> torch.Tensor:
    """
    Preprocess a batch of images.

    Args:
        images: List of image arrays
        preprocessor: Preprocessor instance

    Returns:
        Batched tensor of preprocessed images
    """
    processed_images = []

    for image in images:
        processed = preprocessor(image)
        # Add channel dimension if grayscale
        if processed.ndim == 2:
            processed = processed[np.newaxis, ...]
        processed_images.append(processed)

    # Stack into batch
    batch = np.stack(processed_images, axis=0)
    return torch.from_numpy(batch).float()
