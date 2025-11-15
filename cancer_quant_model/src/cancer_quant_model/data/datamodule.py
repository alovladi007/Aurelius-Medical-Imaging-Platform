"""DataModule for organizing train/val/test dataloaders."""

from pathlib import Path
from typing import Optional

import pandas as pd
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .dataset import FolderDataset, HistopathDataset
from .transforms import get_train_transforms, get_val_transforms


class HistopathDataModule:
    """Data module for histopathology datasets."""

    def __init__(
        self,
        config: DictConfig,
        train_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """Initialize data module.

        Args:
            config: Dataset configuration
            train_csv: Path to train CSV
            val_csv: Path to val CSV
            test_csv: Path to test CSV
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Pin memory for faster GPU transfer
        """
        self.config = config
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Datasets (to be initialized)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Class information
        self.num_classes = config.get("classes", {}).get("num_classes", 2)
        self.class_names = config.get("classes", {}).get("class_names", None)

        # Image settings
        img_settings = config.get("image_settings", {})
        self.img_size = img_settings.get("target_size", 224)

        # Transform settings
        preprocessing = config.get("preprocessing", {})
        self.mean = tuple(preprocessing.get("mean", [0.485, 0.456, 0.406]))
        self.std = tuple(preprocessing.get("std", [0.229, 0.224, 0.225]))

        # Augmentation settings
        aug_config = config.get("augmentation", {})
        self.aug_strength = aug_config.get("train", {}).get("strength", "medium")

    def setup(self):
        """Set up datasets."""
        # Get transforms
        train_transform = get_train_transforms(
            img_size=self.img_size,
            mean=self.mean,
            std=self.std,
            augmentation_strength=self.aug_strength,
        )

        val_transform = get_val_transforms(
            img_size=self.img_size,
            mean=self.mean,
            std=self.std,
        )

        # Create datasets based on configuration
        dataset_type = self.config.get("dataset_type", "folder_binary")

        if dataset_type == "csv_labels":
            # CSV-based dataset
            data_root = self.config.get("data_root", None)

            if self.train_csv:
                self.train_dataset = HistopathDataset(
                    data_root=data_root,
                    csv_path=self.train_csv,
                    transform=train_transform,
                    class_names=self.class_names,
                )

            if self.val_csv:
                self.val_dataset = HistopathDataset(
                    data_root=data_root,
                    csv_path=self.val_csv,
                    transform=val_transform,
                    class_names=self.class_names,
                )

            if self.test_csv:
                self.test_dataset = HistopathDataset(
                    data_root=data_root,
                    csv_path=self.test_csv,
                    transform=val_transform,
                    class_names=self.class_names,
                )

        elif dataset_type == "folder_binary":
            # Folder-based dataset
            folder_struct = self.config.get("folder_structure", {})

            train_dir = folder_struct.get("train_dir", "data/raw/train")
            val_dir = folder_struct.get("val_dir", None)
            test_dir = folder_struct.get("test_dir", None)

            # Override with CSV paths if provided
            if self.train_csv:
                self.train_dataset = HistopathDataset(
                    csv_path=self.train_csv,
                    transform=train_transform,
                    class_names=self.class_names,
                )
            elif Path(train_dir).exists():
                self.train_dataset = FolderDataset(
                    root_dir=train_dir,
                    transform=train_transform,
                )
                # Get class names from folder dataset
                if self.class_names is None:
                    self.class_names = self.train_dataset.class_names

            if self.val_csv:
                self.val_dataset = HistopathDataset(
                    csv_path=self.val_csv,
                    transform=val_transform,
                    class_names=self.class_names,
                )
            elif val_dir and Path(val_dir).exists():
                self.val_dataset = FolderDataset(
                    root_dir=val_dir,
                    transform=val_transform,
                )

            if self.test_csv:
                self.test_dataset = HistopathDataset(
                    csv_path=self.test_csv,
                    transform=val_transform,
                    class_names=self.class_names,
                )
            elif test_dir and Path(test_dir).exists():
                self.test_dataset = FolderDataset(
                    root_dir=test_dir,
                    transform=val_transform,
                )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader.

        Returns:
            Training DataLoader
        """
        if self.train_dataset is None:
            raise ValueError("Train dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,  # Drop last incomplete batch for training
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader.

        Returns:
            Validation DataLoader or None
        """
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Get test dataloader.

        Returns:
            Test DataLoader or None
        """
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def get_class_weights(self):
        """Get class weights from training dataset.

        Returns:
            Class weights tensor or None
        """
        if self.train_dataset is None:
            return None

        return self.train_dataset.get_class_weights()

    def get_dataset_info(self) -> dict:
        """Get information about datasets.

        Returns:
            Dictionary with dataset statistics
        """
        info = {
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "img_size": self.img_size,
        }

        if self.train_dataset:
            info["train_size"] = len(self.train_dataset)
            if hasattr(self.train_dataset, "get_class_distribution"):
                info["train_distribution"] = self.train_dataset.get_class_distribution()

        if self.val_dataset:
            info["val_size"] = len(self.val_dataset)
            if hasattr(self.val_dataset, "get_class_distribution"):
                info["val_distribution"] = self.val_dataset.get_class_distribution()

        if self.test_dataset:
            info["test_size"] = len(self.test_dataset)
            if hasattr(self.test_dataset, "get_class_distribution"):
                info["test_distribution"] = self.test_dataset.get_class_distribution()

        return info
