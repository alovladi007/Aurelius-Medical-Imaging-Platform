"""DataModule for managing train/val/test dataloaders."""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from cancer_quant_model.data.dataset import HistopathDataset
from cancer_quant_model.data.transforms import get_train_transforms, get_val_transforms
from cancer_quant_model.utils.logging_utils import get_logger
from cancer_quant_model.utils.seed_utils import worker_init_fn, get_generator

logger = get_logger(__name__)


class HistopathDataModule:
    """DataModule for histopathology data."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        image_column: str = "image_path",
        label_column: str = "label",
        metadata_columns: Optional[list] = None,
        image_size: tuple = (224, 224),
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        augmentation_config: Optional[Dict] = None,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        multi_label: bool = False,
        seed: int = 42,
    ):
        """
        Initialize DataModule.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (optional)
            test_df: Test DataFrame (optional)
            image_column: Column name for image paths
            label_column: Column name for labels
            metadata_columns: Additional metadata columns
            image_size: Target image size (H, W)
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch
            augmentation_config: Augmentation configuration
            normalize_mean: Normalization mean
            normalize_std: Normalization std
            multi_label: Multi-label classification
            seed: Random seed
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.image_column = image_column
        self.label_column = label_column
        self.metadata_columns = metadata_columns

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.prefetch_factor = prefetch_factor

        self.augmentation_config = augmentation_config
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.multi_label = multi_label
        self.seed = seed

        # Create datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._setup()

    def _setup(self):
        """Setup datasets."""
        # Get transforms
        train_transform = get_train_transforms(
            image_size=self.image_size,
            augmentation_config=self.augmentation_config,
        )

        val_transform = get_val_transforms(
            image_size=self.image_size,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
        )

        # Create datasets
        self.train_dataset = HistopathDataset(
            data_df=self.train_df,
            image_column=self.image_column,
            label_column=self.label_column,
            metadata_columns=self.metadata_columns,
            transform=train_transform,
            multi_label=self.multi_label,
        )

        if self.val_df is not None:
            self.val_dataset = HistopathDataset(
                data_df=self.val_df,
                image_column=self.image_column,
                label_column=self.label_column,
                metadata_columns=self.metadata_columns,
                transform=val_transform,
                multi_label=self.multi_label,
            )

        if self.test_df is not None:
            self.test_dataset = HistopathDataset(
                data_df=self.test_df,
                image_column=self.image_column,
                label_column=self.label_column,
                metadata_columns=self.metadata_columns,
                transform=val_transform,
                multi_label=self.multi_label,
            )

        logger.info("DataModule setup complete")

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            worker_init_fn=lambda worker_id: worker_init_fn(worker_id, self.seed),
            generator=get_generator(self.seed),
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """
        Compute class weights for imbalanced datasets.

        Returns:
            Class weights tensor
        """
        if self.multi_label:
            logger.warning("Class weights not implemented for multi-label")
            return None

        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np

        labels = self.train_dataset.get_labels()
        classes = np.unique(labels)

        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        logger.info(f"Computed class weights: {weights_tensor}")

        return weights_tensor

    def get_num_classes(self) -> int:
        """Get number of classes."""
        labels = self.train_dataset.get_labels()
        if self.multi_label:
            # Assuming labels are binary vectors
            return labels.shape[1] if len(labels.shape) > 1 else 2
        else:
            return len(set(labels))

    def summary(self) -> Dict:
        """Get summary of the data module."""
        summary = {
            "num_train": len(self.train_dataset) if self.train_dataset else 0,
            "num_val": len(self.val_dataset) if self.val_dataset else 0,
            "num_test": len(self.test_dataset) if self.test_dataset else 0,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "image_size": self.image_size,
            "num_classes": self.get_num_classes(),
        }

        if self.train_dataset:
            summary["train_class_dist"] = self.train_dataset.get_class_distribution()

        if self.val_dataset:
            summary["val_class_dist"] = self.val_dataset.get_class_distribution()

        return summary
