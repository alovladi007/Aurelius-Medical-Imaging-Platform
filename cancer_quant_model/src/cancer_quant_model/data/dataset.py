"""Dataset classes for cancer histopathology images."""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HistopathDataset(Dataset):
    """Dataset for histopathology images with various loading modes."""

    def __init__(
        self,
        data_root: Optional[str] = None,
        csv_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        metadata_cols: Optional[List[str]] = None,
    ):
        """Initialize histopathology dataset.

        Args:
            data_root: Root directory for images
            csv_path: Path to CSV with image_path and label columns
            image_paths: List of image paths (alternative to CSV)
            labels: List of labels (alternative to CSV)
            transform: Transform to apply to images
            class_names: Names of classes
            metadata_cols: Additional metadata columns to load from CSV
        """
        self.data_root = Path(data_root) if data_root else None
        self.transform = transform
        self.class_names = class_names
        self.metadata_cols = metadata_cols or []

        # Load data
        if csv_path:
            self._load_from_csv(csv_path)
        elif image_paths and labels:
            self.image_paths = image_paths
            self.labels = labels
            self.metadata = [{}] * len(image_paths)
        else:
            raise ValueError("Must provide either csv_path or (image_paths and labels)")

    def _load_from_csv(self, csv_path: str):
        """Load data from CSV file.

        Args:
            csv_path: Path to CSV file
        """
        df = pd.read_csv(csv_path)

        # Check required columns
        if "image_path" not in df.columns:
            raise ValueError("CSV must contain 'image_path' column")
        if "label" not in df.columns:
            raise ValueError("CSV must contain 'label' column")

        self.image_paths = df["image_path"].tolist()
        self.labels = df["label"].tolist()

        # Load metadata
        self.metadata = []
        for _, row in df.iterrows():
            meta = {}
            for col in self.metadata_cols:
                if col in df.columns:
                    meta[col] = row[col]
            self.metadata.append(meta)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (image_tensor, label, metadata_dict)
        """
        # Load image
        image_path = self.image_paths[idx]

        # Resolve full path if data_root provided
        if self.data_root:
            full_path = self.data_root / image_path
        else:
            full_path = Path(image_path)

        # Load image
        try:
            image = cv2.imread(str(full_path))
            if image is None:
                # Try PIL as fallback
                image = np.array(Image.open(full_path).convert("RGB"))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {full_path}: {e}")

        # Get label
        label = self.labels[idx]

        # Get metadata
        metadata = self.metadata[idx].copy()
        metadata["image_path"] = str(image_path)
        metadata["idx"] = idx

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label, metadata

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets.

        Returns:
            Tensor of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(self.labels)
        weights = compute_class_weight("balanced", classes=classes, y=self.labels)

        return torch.tensor(weights, dtype=torch.float32)

    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution.

        Returns:
            Dictionary of {class_label: count}
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class FolderDataset(Dataset):
    """Dataset that loads from folder structure (e.g., train/0/, train/1/)."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ):
        """Initialize folder-based dataset.

        Args:
            root_dir: Root directory with class subdirectories
            transform: Transform to apply to images
            extensions: Allowed file extensions
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.extensions = extensions

        # Scan directory
        self.image_paths = []
        self.labels = []
        self.class_names = []

        # Get class directories
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        for class_idx, class_dir in enumerate(class_dirs):
            self.class_names.append(class_dir.name)

            # Get all images in this class
            for ext in extensions:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_idx)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {root_dir}")

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (image_tensor, label, metadata_dict)
        """
        # Load image
        image_path = self.image_paths[idx]

        try:
            image = cv2.imread(image_path)
            if image is None:
                image = np.array(Image.open(image_path).convert("RGB"))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")

        # Get label
        label = self.labels[idx]

        # Metadata
        metadata = {
            "image_path": image_path,
            "class_name": self.class_names[label],
            "idx": idx,
        }

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label, metadata

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights."""
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(self.labels)
        weights = compute_class_weight("balanced", classes=classes, y=self.labels)

        return torch.tensor(weights, dtype=torch.float32)


class TileDataset(Dataset):
    """Dataset for handling tiled/patched images with aggregation."""

    def __init__(
        self,
        tiles_df: pd.DataFrame,
        transform: Optional[Callable] = None,
        aggregate_by: str = "patient_id",
    ):
        """Initialize tile dataset.

        Args:
            tiles_df: DataFrame with columns: tile_path, label, patient_id (or other grouping)
            transform: Transform to apply
            aggregate_by: Column to aggregate tiles by
        """
        self.tiles_df = tiles_df
        self.transform = transform
        self.aggregate_by = aggregate_by

        # Group tiles
        if aggregate_by and aggregate_by in tiles_df.columns:
            self.groups = tiles_df.groupby(aggregate_by).groups
            self.group_ids = list(self.groups.keys())
        else:
            # No aggregation, treat each tile independently
            self.groups = None
            self.group_ids = None

    def __len__(self) -> int:
        """Get dataset length."""
        if self.groups:
            return len(self.group_ids)
        else:
            return len(self.tiles_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Get item by index.

        Args:
            idx: Index

        Returns:
            If grouped: returns all tiles for the group
            If not grouped: returns single tile
        """
        if self.groups:
            # Get all tiles for this group
            group_id = self.group_ids[idx]
            tile_indices = self.groups[group_id]

            tiles = []
            for tile_idx in tile_indices:
                row = self.tiles_df.iloc[tile_idx]
                tile_path = row["tile_path"]

                # Load tile
                image = cv2.imread(tile_path)
                if image is None:
                    image = np.array(Image.open(tile_path).convert("RGB"))
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if self.transform:
                    transformed = self.transform(image=image)
                    image = transformed["image"]

                tiles.append(image)

            # Stack tiles
            tiles_tensor = torch.stack(tiles)

            # Get label (assume same for all tiles in group)
            label = int(self.tiles_df.iloc[tile_indices[0]]["label"])

            metadata = {
                "group_id": group_id,
                "num_tiles": len(tiles),
            }

            return tiles_tensor, label, metadata

        else:
            # Single tile
            row = self.tiles_df.iloc[idx]
            tile_path = row["tile_path"]

            image = cv2.imread(tile_path)
            if image is None:
                image = np.array(Image.open(tile_path).convert("RGB"))
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            label = int(row["label"])

            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]

            metadata = {"tile_path": tile_path, "idx": idx}

            return image, label, metadata
