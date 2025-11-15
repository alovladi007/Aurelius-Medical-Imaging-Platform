"""Data transformations for histopathology images."""

from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[Dict] = None,
) -> A.Compose:
    """
    Get training transforms with augmentation.

    Args:
        image_size: Target image size (H, W)
        augmentation_config: Augmentation configuration

    Returns:
        Albumentations composition
    """
    if augmentation_config is None:
        augmentation_config = {
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotation_degrees": 90,
            "brightness": 0.2,
            "contrast": 0.2,
            "saturation": 0.2,
            "hue": 0.1,
            "gaussian_blur": 0.1,
            "gaussian_noise": 0.05,
        }

    transforms_list = []

    # Resize
    transforms_list.append(A.Resize(height=image_size[0], width=image_size[1]))

    # Geometric augmentations
    if augmentation_config.get("horizontal_flip", False):
        transforms_list.append(A.HorizontalFlip(p=0.5))

    if augmentation_config.get("vertical_flip", False):
        transforms_list.append(A.VerticalFlip(p=0.5))

    if augmentation_config.get("rotation_degrees", 0) > 0:
        # For histopathology, only multiples of 90 degrees make sense
        transforms_list.append(A.RandomRotate90(p=0.5))

    # Color augmentations
    if any(
        [
            augmentation_config.get("brightness", 0) > 0,
            augmentation_config.get("contrast", 0) > 0,
            augmentation_config.get("saturation", 0) > 0,
            augmentation_config.get("hue", 0) > 0,
        ]
    ):
        transforms_list.append(
            A.ColorJitter(
                brightness=augmentation_config.get("brightness", 0.2),
                contrast=augmentation_config.get("contrast", 0.2),
                saturation=augmentation_config.get("saturation", 0.2),
                hue=augmentation_config.get("hue", 0.1),
                p=0.5,
            )
        )

    # Advanced augmentations
    if augmentation_config.get("gaussian_blur", 0) > 0:
        transforms_list.append(
            A.GaussianBlur(blur_limit=(3, 7), p=augmentation_config["gaussian_blur"])
        )

    if augmentation_config.get("gaussian_noise", 0) > 0:
        noise_std = augmentation_config["gaussian_noise"] * 255  # Convert to [0, 255] scale
        transforms_list.append(A.GaussNoise(var_limit=(10.0, noise_std), p=0.3))

    # Elastic transform (useful for histopathology)
    if augmentation_config.get("elastic_transform", False):
        transforms_list.append(
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3
            )
        )

    # Grid distortion
    if augmentation_config.get("grid_distortion", False):
        transforms_list.append(A.GridDistortion(p=0.3))

    # Normalization (ImageNet stats)
    normalize_mean = augmentation_config.get("normalize_mean", [0.485, 0.456, 0.406])
    normalize_std = augmentation_config.get("normalize_std", [0.229, 0.224, 0.225])

    transforms_list.append(A.Normalize(mean=normalize_mean, std=normalize_std))

    # Convert to tensor
    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize_mean: List[float] = [0.485, 0.456, 0.406],
    normalize_std: List[float] = [0.229, 0.224, 0.225],
) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).

    Args:
        image_size: Target image size (H, W)
        normalize_mean: Normalization mean
        normalize_std: Normalization std

    Returns:
        Albumentations composition
    """
    return A.Compose(
        [
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=normalize_mean, std=normalize_std),
            ToTensorV2(),
        ]
    )


def get_tta_transforms(
    image_size: Tuple[int, int] = (224, 224),
    normalize_mean: List[float] = [0.485, 0.456, 0.406],
    normalize_std: List[float] = [0.229, 0.224, 0.225],
    n_transforms: int = 5,
) -> List[A.Compose]:
    """
    Get test-time augmentation transforms.

    Args:
        image_size: Target image size
        normalize_mean: Normalization mean
        normalize_std: Normalization std
        n_transforms: Number of TTA transforms

    Returns:
        List of transform compositions
    """
    tta_transforms = []

    base_transforms = [
        A.Resize(height=image_size[0], width=image_size[1]),
    ]

    augmentations = [
        [],  # Original
        [A.HorizontalFlip(p=1.0)],
        [A.VerticalFlip(p=1.0)],
        [A.RandomRotate90(p=1.0)],
        [A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)],
    ]

    for aug in augmentations[:n_transforms]:
        transforms_list = base_transforms + aug
        transforms_list.extend(
            [
                A.Normalize(mean=normalize_mean, std=normalize_std),
                ToTensorV2(),
            ]
        )
        tta_transforms.append(A.Compose(transforms_list))

    return tta_transforms


def denormalize(
    tensor: torch.Tensor,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
) -> torch.Tensor:
    """
    Denormalize a tensor.

    Args:
        tensor: Input tensor (C, H, W) or (B, C, H, W)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    if tensor.dim() == 4:  # Batch
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)

    return tensor * std + mean


class MixupCutmix:
    """Mixup and Cutmix augmentation."""

    def __init__(
        self,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        prob: float = 0.5,
        num_classes: int = 2,
    ):
        """
        Initialize Mixup/Cutmix.

        Args:
            mixup_alpha: Mixup alpha parameter (0 to disable)
            cutmix_alpha: Cutmix alpha parameter (0 to disable)
            prob: Probability of applying mixup/cutmix
            num_classes: Number of classes
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes

    def __call__(
        self, batch: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup or cutmix.

        Args:
            batch: Input batch (B, C, H, W)
            labels: Labels (B,)

        Returns:
            Tuple of (mixed batch, mixed labels)
        """
        if np.random.rand() > self.prob:
            return batch, labels

        # Convert labels to one-hot
        if labels.dim() == 1:
            labels_onehot = torch.nn.functional.one_hot(labels, self.num_classes).float()
        else:
            labels_onehot = labels

        # Choose mixup or cutmix
        use_cutmix = (
            self.cutmix_alpha > 0
            and np.random.rand() < 0.5
            or self.mixup_alpha == 0
        )

        if use_cutmix:
            batch, labels_onehot = self._cutmix(batch, labels_onehot)
        else:
            batch, labels_onehot = self._mixup(batch, labels_onehot)

        return batch, labels_onehot

    def _mixup(
        self, batch: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply mixup."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        batch_size = batch.shape[0]
        index = torch.randperm(batch_size).to(batch.device)

        mixed_batch = lam * batch + (1 - lam) * batch[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_batch, mixed_labels

    def _cutmix(
        self, batch: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cutmix."""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        batch_size = batch.shape[0]
        index = torch.randperm(batch_size).to(batch.device)

        # Generate random box
        _, _, h, w = batch.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        # Apply cutmix
        mixed_batch = batch.clone()
        mixed_batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_batch, mixed_labels
