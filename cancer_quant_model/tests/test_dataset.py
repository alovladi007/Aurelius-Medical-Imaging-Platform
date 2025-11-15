"""Tests for dataset classes."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cancer_quant_model.data import FolderDataset, HistopathDataset
from cancer_quant_model.data.transforms import get_val_transforms


def create_synthetic_images(output_dir, num_images=5):
    """Create synthetic images for testing."""
    output_dir = Path(output_dir)
    for i in range(num_images):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img).save(output_dir / f"image_{i}.png")


def test_folder_dataset():
    """Test FolderDataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create class directories
        (tmpdir / "class_0").mkdir()
        (tmpdir / "class_1").mkdir()
        
        # Create images
        create_synthetic_images(tmpdir / "class_0", num_images=5)
        create_synthetic_images(tmpdir / "class_1", num_images=5)
        
        # Create dataset
        transform = get_val_transforms(img_size=224)
        dataset = FolderDataset(str(tmpdir), transform=transform)
        
        assert len(dataset) == 10
        assert len(dataset.class_names) == 2
        
        # Test __getitem__
        image, label, metadata = dataset[0]
        assert image.shape == (3, 224, 224)
        assert label in [0, 1]
        assert "image_path" in metadata


def test_histopath_dataset():
    """Test HistopathDataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create images
        create_synthetic_images(tmpdir, num_images=5)
        
        # Create image paths and labels
        image_paths = [str(tmpdir / f"image_{i}.png") for i in range(5)]
        labels = [0, 1, 0, 1, 0]
        
        # Create dataset
        transform = get_val_transforms(img_size=224)
        dataset = HistopathDataset(
            image_paths=image_paths,
            labels=labels,
            transform=transform,
        )
        
        assert len(dataset) == 5
        
        # Test __getitem__
        image, label, metadata = dataset[0]
        assert image.shape == (3, 224, 224)
        assert label == 0


if __name__ == "__main__":
    pytest.main([__file__])
