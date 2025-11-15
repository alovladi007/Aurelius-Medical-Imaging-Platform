"""Data augmentation and transformation utilities."""

from typing import Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augmentation_strength: str = "medium",
) -> A.Compose:
    """Get training transformations with augmentation.

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std
        augmentation_strength: Strength of augmentation ('light', 'medium', 'strong')

    Returns:
        Albumentations composition
    """
    # Base transforms
    transforms = [
        A.Resize(img_size, img_size),
    ]

    # Augmentation based on strength
    if augmentation_strength == "light":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
            ]
        )

    elif augmentation_strength == "medium":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=90, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5, border_mode=0
                ),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10, 50), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    ],
                    p=0.2,
                ),
            ]
        )

    elif augmentation_strength == "strong":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Rotate(limit=180, p=0.7),
                A.ShiftScaleRotate(
                    shift_limit=0.2, scale_limit=0.2, rotate_limit=0, p=0.7, border_mode=0
                ),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.7),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10, 100), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                        A.MotionBlur(blur_limit=7, p=1.0),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                        A.GridDistortion(p=1.0),
                        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1.0),
                    ],
                    p=0.2,
                ),
                A.CLAHE(clip_limit=4.0, p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
                ),
            ]
        )

    # Normalization and tensor conversion
    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def get_val_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get validation/test transformations (no augmentation).

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Albumentations composition
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get inference transformations (same as validation).

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Albumentations composition
    """
    return get_val_transforms(img_size=img_size, mean=mean, std=std)


def get_stain_augmentation_transforms(
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Get transforms with stain augmentation for histopathology.

    Args:
        img_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Albumentations composition
    """
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=90, p=0.5),
            # Stain-like color augmentation
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.7
            ),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.08, p=0.7),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            # Subtle blur and noise
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(var_limit=(5, 30), p=1.0),
                ],
                p=0.2,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Denormalize image tensor.

    Args:
        tensor: Normalized image tensor (C, H, W)
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    return tensor * std + mean


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert image tensor to numpy array for visualization.

    Args:
        tensor: Image tensor (C, H, W) in [0, 1] range

    Returns:
        Numpy array (H, W, C) in [0, 255] range
    """
    # Move to CPU and convert to numpy
    array = tensor.cpu().numpy()

    # Transpose from (C, H, W) to (H, W, C)
    if len(array.shape) == 3:
        array = np.transpose(array, (1, 2, 0))

    # Clip to [0, 1] and scale to [0, 255]
    array = np.clip(array, 0, 1)
    array = (array * 255).astype(np.uint8)

    return array
