"""Tests for training loop."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image

from cancer_quant_model.data.datamodule import HistopathDataModule
from cancer_quant_model.models.resnet import ResNetModel


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Create a synthetic dataset for testing."""
    # Create directories
    train_dir = tmp_path / "train"
    train_dir.mkdir()

    (train_dir / "0").mkdir()
    (train_dir / "1").mkdir()

    # Create synthetic images
    images_per_class = 10

    for class_id in [0, 1]:
        class_dir = train_dir / str(class_id)
        for i in range(images_per_class):
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(class_dir / f"image_{i}.png")

    # Create CSV splits
    image_paths = []
    labels = []

    for class_id in [0, 1]:
        class_dir = train_dir / str(class_id)
        for img_file in class_dir.glob("*.png"):
            image_paths.append(str(img_file))
            labels.append(class_id)

    df = pd.DataFrame({"image_path": image_paths, "label": labels})

    # Split into train/val
    train_df = df.iloc[:12]
    val_df = df.iloc[12:]

    return train_df, val_df


class TestTrainingLoop:
    """Tests for training loop."""

    def test_model_forward_backward(self):
        """Test forward and backward pass."""
        model = ResNetModel(
            variant="resnet18",
            num_classes=2,
            pretrained=False,
        )

        # Forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        assert output.shape == (2, 2)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "Model should have gradients after backward"

    def test_datamodule_creation(self, synthetic_dataset):
        """Test datamodule creation."""
        train_df, val_df = synthetic_dataset

        datamodule = HistopathDataModule(
            train_df=train_df,
            val_df=val_df,
            image_size=(64, 64),
            batch_size=4,
            num_workers=0,
        )

        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        assert len(train_loader) > 0
        assert len(val_loader) > 0

        # Test batch
        batch = next(iter(train_loader))
        images, labels, metadata = batch

        assert images.shape == (4, 3, 64, 64)
        assert labels.shape == (4,)

    @pytest.mark.slow
    def test_short_training_run(self, synthetic_dataset, tmp_path):
        """Test a very short training run."""
        train_df, val_df = synthetic_dataset

        # Create datamodule
        datamodule = HistopathDataModule(
            train_df=train_df,
            val_df=val_df,
            image_size=(64, 64),
            batch_size=4,
            num_workers=0,
        )

        # Create model
        model = ResNetModel(
            variant="resnet18",
            num_classes=2,
            pretrained=False,
        )

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        device = "cpu"
        model = model.to(device)

        # Train for 2 batches
        model.train()
        train_loader = datamodule.train_dataloader()

        initial_loss = None
        final_loss = None

        for batch_idx, (images, labels, metadata) in enumerate(train_loader):
            if batch_idx >= 2:
                break

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Check that training ran
        assert initial_loss is not None
        assert final_loss is not None
        # Loss should change (not necessarily decrease in 2 steps)
        assert initial_loss != final_loss
