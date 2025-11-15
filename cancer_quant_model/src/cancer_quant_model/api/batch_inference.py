"""Batch inference for processing multiple images."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data.dataset import HistopathDataset
from ..data.transforms import get_inference_transforms
from ..utils.logging_utils import get_logger

logger = get_logger()


class BatchInference:
    """Batch inference handler."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        batch_size: int = 32,
        num_workers: int = 4,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize batch inference.

        Args:
            model: Trained model
            device: Device to run inference on
            batch_size: Batch size
            num_workers: Number of data loading workers
            class_names: Names of classes
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = class_names

    def predict_from_paths(
        self,
        image_paths: List[str],
        img_size: int = 224,
    ) -> pd.DataFrame:
        """Run batch inference on list of image paths.

        Args:
            image_paths: List of image paths
            img_size: Image size for transformation

        Returns:
            DataFrame with predictions
        """
        logger.info(f"Running inference on {len(image_paths)} images")

        # Create dataset
        transforms = get_inference_transforms(img_size=img_size)

        # Create dummy labels (not used for inference)
        dummy_labels = [0] * len(image_paths)

        dataset = HistopathDataset(
            image_paths=image_paths,
            labels=dummy_labels,
            transform=transforms,
        )

        # Create dataloader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )

        # Run inference
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, _, metadata in tqdm(dataloader, desc="Inference"):
                images = images.to(self.device)

                # Forward pass
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Create results dataframe
        all_probs = np.array(all_probs)

        results = {
            "image_path": image_paths,
            "predicted_class": all_preds,
        }

        # Add probabilities
        num_classes = all_probs.shape[1]
        for i in range(num_classes):
            class_name = self.class_names[i] if self.class_names else f"class_{i}"
            results[f"prob_{class_name}"] = all_probs[:, i]

        # Add predicted class name
        if self.class_names:
            results["predicted_class_name"] = [
                self.class_names[pred] for pred in all_preds
            ]

        # Add confidence
        results["confidence"] = [all_probs[i, pred] for i, pred in enumerate(all_preds)]

        df = pd.DataFrame(results)

        logger.info("Inference completed")

        return df

    def predict_from_csv(
        self,
        csv_path: str,
        image_path_column: str = "image_path",
        img_size: int = 224,
    ) -> pd.DataFrame:
        """Run batch inference from CSV file.

        Args:
            csv_path: Path to CSV with image paths
            image_path_column: Column name containing image paths
            img_size: Image size

        Returns:
            DataFrame with predictions
        """
        # Read CSV
        df = pd.read_csv(csv_path)

        if image_path_column not in df.columns:
            raise ValueError(f"Column {image_path_column} not found in CSV")

        image_paths = df[image_path_column].tolist()

        # Run inference
        predictions_df = self.predict_from_paths(image_paths, img_size=img_size)

        # Merge with original dataframe
        result_df = pd.merge(
            df,
            predictions_df,
            left_on=image_path_column,
            right_on="image_path",
            how="left",
        )

        return result_df


def run_batch_inference(
    model_checkpoint: str,
    image_paths: List[str],
    output_path: str,
    config: Optional[dict] = None,
    batch_size: int = 32,
    device: str = "cuda",
):
    """Run batch inference and save results.

    Args:
        model_checkpoint: Path to model checkpoint
        image_paths: List of image paths
        output_path: Path to save predictions
        config: Model configuration
        batch_size: Batch size
        device: Device to use
    """
    from ..models import create_model

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint, map_location=device)

    # Get config
    if config is None:
        config = checkpoint.get("config", {})

    # Create model
    model_type = config.get("model", {}).get("backbone", "resnet50")

    if "resnet" in model_type:
        model = create_model(config, model_type="resnet")
    elif "efficientnet" in model_type:
        model = create_model(config, model_type="efficientnet")
    else:
        model = create_model(config, model_type="resnet")

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get class names
    class_names = config.get("classes", {}).get("class_names", None)

    # Run inference
    batch_inference = BatchInference(
        model=model,
        device=device,
        batch_size=batch_size,
        class_names=class_names,
    )

    predictions_df = batch_inference.predict_from_paths(image_paths)

    # Save results
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
