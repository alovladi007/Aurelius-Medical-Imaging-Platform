"""
Data augmentation for medical images with domain-specific transformations.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict, Callable
import random
import cv2
from scipy import ndimage


class MedicalImageAugmentation:
    """
    Medical image augmentation with transformations appropriate for clinical data.
    Carefully designed to preserve diagnostic information.
    """

    def __init__(self,
                 rotation_range: float = 15.0,
                 width_shift_range: float = 0.1,
                 height_shift_range: float = 0.1,
                 zoom_range: Tuple[float, float] = (0.9, 1.1),
                 horizontal_flip: bool = True,
                 vertical_flip: bool = False,
                 brightness_range: Optional[Tuple[float, float]] = (0.8, 1.2),
                 contrast_range: Optional[Tuple[float, float]] = (0.8, 1.2),
                 noise_std: float = 0.01,
                 elastic_alpha: float = 0,
                 elastic_sigma: float = 0,
                 augmentation_probability: float = 0.5):
        """
        Initialize medical image augmentation.

        Args:
            rotation_range: Degrees of rotation
            width_shift_range: Fraction of total width for horizontal shifts
            height_shift_range: Fraction of total height for vertical shifts
            zoom_range: Range for random zoom (min, max)
            horizontal_flip: Whether to apply horizontal flips
            vertical_flip: Whether to apply vertical flips
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            noise_std: Standard deviation for Gaussian noise
            elastic_alpha: Alpha parameter for elastic deformation
            elastic_sigma: Sigma parameter for elastic deformation
            augmentation_probability: Probability of applying each augmentation
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.augmentation_probability = augmentation_probability

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to image.

        Args:
            image: Input image array

        Returns:
            Augmented image array
        """
        # Random rotation
        if random.random() < self.augmentation_probability:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = self._rotate(image, angle)

        # Random shifts
        if random.random() < self.augmentation_probability:
            image = self._shift(image)

        # Random zoom
        if random.random() < self.augmentation_probability:
            zoom_factor = random.uniform(*self.zoom_range)
            image = self._zoom(image, zoom_factor)

        # Random flips
        if self.horizontal_flip and random.random() < self.augmentation_probability:
            image = np.fliplr(image)

        if self.vertical_flip and random.random() < self.augmentation_probability:
            image = np.flipud(image)

        # Brightness adjustment
        if self.brightness_range and random.random() < self.augmentation_probability:
            brightness_factor = random.uniform(*self.brightness_range)
            image = self._adjust_brightness(image, brightness_factor)

        # Contrast adjustment
        if self.contrast_range and random.random() < self.augmentation_probability:
            contrast_factor = random.uniform(*self.contrast_range)
            image = self._adjust_contrast(image, contrast_factor)

        # Add Gaussian noise
        if self.noise_std > 0 and random.random() < self.augmentation_probability:
            image = self._add_gaussian_noise(image, self.noise_std)

        # Elastic deformation
        if self.elastic_alpha > 0 and random.random() < self.augmentation_probability:
            image = self._elastic_transform(image, self.elastic_alpha, self.elastic_sigma)

        return image

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        if image.ndim == 2:
            return ndimage.rotate(image, angle, reshape=False, order=1, mode='nearest')
        elif image.ndim == 3:
            # For multi-channel images, rotate each channel
            rotated_channels = []
            for i in range(image.shape[2]):
                rotated_channel = ndimage.rotate(
                    image[:, :, i], angle, reshape=False, order=1, mode='nearest'
                )
                rotated_channels.append(rotated_channel)
            return np.stack(rotated_channels, axis=2)
        return image

    def _shift(self, image: np.ndarray) -> np.ndarray:
        """Apply random spatial shifts."""
        h, w = image.shape[:2]
        tx = random.uniform(-self.width_shift_range, self.width_shift_range) * w
        ty = random.uniform(-self.height_shift_range, self.height_shift_range) * h

        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        if image.ndim == 2:
            shifted = cv2.warpAffine(image, translation_matrix, (w, h),
                                    borderMode=cv2.BORDER_REFLECT)
        else:
            shifted = cv2.warpAffine(image, translation_matrix, (w, h),
                                    borderMode=cv2.BORDER_REFLECT)

        return shifted

    def _zoom(self, image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Apply random zoom."""
        h, w = image.shape[:2]
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))

        # Resize
        if image.ndim == 2:
            zoomed = cv2.resize(image, (zw, zh), interpolation=cv2.INTER_LINEAR)
        else:
            zoomed = cv2.resize(image, (zw, zh), interpolation=cv2.INTER_LINEAR)

        # Crop or pad to original size
        if zoom_factor > 1:
            # Crop center
            y1 = (zh - h) // 2
            x1 = (zw - w) // 2
            zoomed = zoomed[y1:y1+h, x1:x1+w]
        else:
            # Pad
            pad_h = (h - zh) // 2
            pad_w = (w - zw) // 2
            if image.ndim == 2:
                zoomed = np.pad(zoomed, ((pad_h, h-zh-pad_h), (pad_w, w-zw-pad_w)),
                              mode='reflect')
            else:
                zoomed = np.pad(zoomed,
                              ((pad_h, h-zh-pad_h), (pad_w, w-zw-pad_w), (0, 0)),
                              mode='reflect')

        return zoomed

    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        adjusted = image * factor
        return np.clip(adjusted, 0, 1) if image.max() <= 1 else adjusted

    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast."""
        mean = image.mean()
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 1) if image.max() <= 1 else adjusted

    def _add_gaussian_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to image."""
        noise = np.random.normal(0, std, image.shape)
        noisy = image + noise
        return np.clip(noisy, 0, 1) if image.max() <= 1 else noisy

    def _elastic_transform(self, image: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
        """
        Apply elastic deformation to image.
        Based on: Simard et al., "Best Practices for Convolutional Neural Networks"
        """
        random_state = np.random.RandomState(None)

        shape = image.shape[:2]
        dx = ndimage.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha
        dy = ndimage.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0
        ) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = (np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)))

        if image.ndim == 2:
            distorted = ndimage.map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape)
        else:
            distorted_channels = []
            for i in range(image.shape[2]):
                distorted_channel = ndimage.map_coordinates(
                    image[:, :, i], indices, order=1, mode='reflect'
                ).reshape(shape)
                distorted_channels.append(distorted_channel)
            distorted = np.stack(distorted_channels, axis=2)

        return distorted


class Compose:
    """Compose multiple transformations."""

    def __init__(self, transforms: List[Callable]):
        """
        Initialize composition.

        Args:
            transforms: List of transformation functions
        """
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply all transformations sequentially."""
        for transform in self.transforms:
            image = transform(image)
        return image


class RandomApply:
    """Apply transformation with given probability."""

    def __init__(self, transform: Callable, probability: float = 0.5):
        """
        Initialize random application.

        Args:
            transform: Transformation function
            probability: Probability of applying transformation
        """
        self.transform = transform
        self.probability = probability

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply transformation with probability."""
        if random.random() < self.probability:
            return self.transform(image)
        return image


class Normalize:
    """Normalize image to zero mean and unit variance."""

    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        """
        Initialize normalization.

        Args:
            mean: Mean value (computed from data if None)
            std: Standard deviation (computed from data if None)
        """
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Normalize image."""
        mean = self.mean if self.mean is not None else image.mean()
        std = self.std if self.std is not None else image.std()

        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image - mean

        return normalized


class ToTensor:
    """Convert numpy array to PyTorch tensor."""

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Convert to tensor."""
        # Add channel dimension if grayscale
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        # Move channel axis to first position if needed
        elif image.ndim == 3:
            image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image.copy()).float()


def get_training_augmentation() -> Compose:
    """Get standard training augmentation pipeline."""
    return Compose([
        MedicalImageAugmentation(
            rotation_range=15.0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=(0.9, 1.1),
            horizontal_flip=True,
            brightness_range=(0.85, 1.15),
            contrast_range=(0.85, 1.15),
            noise_std=0.01,
            augmentation_probability=0.5
        ),
        Normalize(),
        ToTensor()
    ])


def get_validation_augmentation() -> Compose:
    """Get validation augmentation pipeline (minimal transformations)."""
    return Compose([
        Normalize(),
        ToTensor()
    ])
