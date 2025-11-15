"""Dataset classes for histopathology images."""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from cancer_quant_model.utils.logging_utils import get_logger

logger = get_logger(__name__)


class HistopathDataset(Dataset):
    """Histopathology image dataset."""

    def __init__(
        self,
        data_df: pd.DataFrame,
        image_column: str = "image_path",
        label_column: str = "label",
        metadata_columns: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        multi_label: bool = False,
    ):
        """
        Initialize histopathology dataset.

        Args:
            data_df: DataFrame with image paths and labels
            image_column: Column name for image paths
            label_column: Column name for labels
            metadata_columns: Additional metadata columns to include
            transform: Albumentations transform
            multi_label: Whether this is multi-label classification
        """
        self.data_df = data_df.reset_index(drop=True)
        self.image_column = image_column
        self.label_column = label_column
        self.metadata_columns = metadata_columns or []
        self.transform = transform
        self.multi_label = multi_label

        # Validate data
        self._validate_data()

        logger.info(f"Initialized dataset with {len(self)} samples")

    def _validate_data(self):
        """Validate dataset."""
        # Check required columns exist
        if self.image_column not in self.data_df.columns:
            raise ValueError(f"Image column '{self.image_column}' not found in DataFrame")

        if self.label_column not in self.data_df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in DataFrame")

        # Check image files exist
        missing_files = []
        for idx, row in self.data_df.iterrows():
            image_path = row[self.image_column]
            if not os.path.exists(image_path):
                missing_files.append(image_path)

        if missing_files:
            logger.warning(
                f"Found {len(missing_files)} missing image files. "
                f"First 5: {missing_files[:5]}"
            )

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.data_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image tensor, label tensor, metadata dict)
        """
        row = self.data_df.iloc[idx]

        # Load image
        image_path = row[self.image_column]
        image = self._load_image(image_path)

        # Get label
        label = row[self.label_column]

        # Handle multi-label
        if self.multi_label:
            if isinstance(label, str):
                label = [int(x) for x in label.split(",")]
            label = torch.tensor(label, dtype=torch.float32)
        else:
            label = torch.tensor(int(label), dtype=torch.long)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0

        # Collect metadata
        metadata = {"image_path": image_path, "idx": idx}
        for col in self.metadata_columns:
            if col in row:
                metadata[col] = row[col]

        return image, label, metadata

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from file.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (H, W, C) in RGB, uint8
        """
        try:
            image = Image.open(image_path)

            # Convert to RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            image = np.array(image)

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return np.zeros((224, 224, 3), dtype=np.uint8)

    def get_labels(self) -> np.ndarray:
        """
        Get all labels.

        Returns:
            Array of labels
        """
        labels = self.data_df[self.label_column].values
        if not self.multi_label:
            labels = labels.astype(int)
        return labels

    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get class distribution.

        Returns:
            Dictionary mapping class -> count
        """
        if self.multi_label:
            logger.warning("Class distribution not implemented for multi-label")
            return {}

        labels = self.get_labels()
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class FolderBinaryDataset(Dataset):
    """
    Dataset for binary classification from folder structure.
    Expected structure: root/{0,1}/*.png
    """

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[Callable] = None,
        extensions: List[str] = [".png", ".jpg", ".jpeg"],
    ):
        """
        Initialize folder binary dataset.

        Args:
            root_dir: Root directory with 0/ and 1/ subfolders
            transform: Albumentations transform
            extensions: Valid image extensions
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions

        # Collect image paths and labels
        self.samples = self._collect_samples()

        logger.info(f"Initialized folder dataset with {len(self)} samples")

    def _collect_samples(self) -> List[Tuple[Path, int]]:
        """
        Collect image samples.

        Returns:
            List of (image_path, label) tuples
        """
        samples = []

        for label in [0, 1]:
            label_dir = self.root_dir / str(label)

            if not label_dir.exists():
                logger.warning(f"Label directory not found: {label_dir}")
                continue

            for ext in self.extensions:
                for image_path in label_dir.glob(f"*{ext}"):
                    samples.append((image_path, label))

        return samples

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Get a sample."""
        image_path, label = self.samples[idx]

        # Load image
        image = np.array(Image.open(image_path).convert("RGB"))

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        else:
            image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0

        label = torch.tensor(label, dtype=torch.long)

        metadata = {"image_path": str(image_path), "idx": idx}

        return image, label, metadata

    def get_labels(self) -> np.ndarray:
        """Get all labels."""
        return np.array([label for _, label in self.samples])


def create_split_dataframes(
    data_dir: Path,
    dataset_type: str = "folder_binary",
    csv_path: Optional[Path] = None,
    csv_image_col: str = "image_id",
    csv_label_col: str = "label",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test split DataFrames.

    Args:
        data_dir: Data directory
        dataset_type: "folder_binary" or "csv_labels"
        csv_path: Path to CSV file (for csv_labels mode)
        csv_image_col: Image column name in CSV
        csv_label_col: Label column name in CSV
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        stratify: Use stratified split
        seed: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split

    if dataset_type == "folder_binary":
        # Collect from folder structure
        samples = []

        for label in [0, 1]:
            label_dir = data_dir / str(label)
            if label_dir.exists():
                for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    for image_path in label_dir.glob(f"*{ext}"):
                        samples.append({"image_path": str(image_path), "label": label})

        df = pd.DataFrame(samples)

    elif dataset_type == "csv_labels":
        # Load from CSV
        df = pd.read_csv(csv_path)

        # Create full image paths
        df["image_path"] = df[csv_image_col].apply(lambda x: str(data_dir / x))
        df = df.rename(columns={csv_label_col: "label"})

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Split data
    labels = df["label"].values if stratify else None

    # First split: train and (val+test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )

    # Second split: val and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    temp_labels = temp_df["label"].values if stratify else None

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=seed,
        stratify=temp_labels,
    )

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df
