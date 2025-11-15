"""Tests for dataset module."""

import numpy as np
import pandas as pd
import pytest
import torch

from cancer_quant_model.data.dataset import HistopathDataset
from cancer_quant_model.data.transforms import get_val_transforms


class TestHistopathDataset:
    """Tests for HistopathDataset."""

    def test_dataset_creation(self, tmp_path):
        """Test dataset creation."""
        # Create dummy data
        df = pd.DataFrame(
            {
                "image_path": [str(tmp_path / "img1.png"), str(tmp_path / "img2.png")],
                "label": [0, 1],
            }
        )

        # Create dummy images
        for img_path in df["image_path"]:
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            from PIL import Image

            Image.fromarray(img).save(img_path)

        # Create dataset
        transform = get_val_transforms()
        dataset = HistopathDataset(data_df=df, transform=transform)

        assert len(dataset) == 2

    def test_dataset_getitem(self, tmp_path):
        """Test dataset __getitem__."""
        # Create dummy data
        img_path = tmp_path / "test.png"
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        from PIL import Image

        Image.fromarray(img).save(img_path)

        df = pd.DataFrame({"image_path": [str(img_path)], "label": [1]})

        transform = get_val_transforms()
        dataset = HistopathDataset(data_df=df, transform=transform)

        image, label, metadata = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert label.item() == 1
        assert "image_path" in metadata
