"""Evaluation loop for trained models."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from cancer_quant_model.utils.logging_utils import get_logger
from cancer_quant_model.utils.metrics_utils import (
    compute_classification_metrics,
    compute_confusion_matrix,
    get_classification_report,
)

logger = get_logger(__name__)


class Evaluator:
    """Evaluator for cancer histopathology models."""

    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: str = "cuda",
        save_predictions: bool = True,
        output_dir: Path = Path("experiments/predictions"),
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device for evaluation
            save_predictions: Save predictions to disk
            output_dir: Output directory for predictions
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.save_predictions = save_predictions
        self.output_dir = Path(output_dir)

        if save_predictions:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(
        self, class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model on test set.

        Args:
            class_names: Names of classes

        Returns:
            Dictionary with evaluation results
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        all_image_paths = []
        all_features = []

        logger.info("Running evaluation...")

        for images, labels, metadata in tqdm(self.test_loader, desc="Evaluating"):
            images = images.to(self.device)

            # Forward pass
            outputs = self.model(images)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
                if "features" in outputs:
                    all_features.extend(outputs["features"].cpu().numpy())
            else:
                logits = outputs

            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            # Collect metadata
            if "image_path" in metadata:
                all_image_paths.extend(metadata["image_path"])

        # Convert to arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        logger.info("Computing metrics...")
        metrics = compute_classification_metrics(all_labels, all_preds, all_probs)

        # Confusion matrix
        cm = compute_confusion_matrix(all_labels, all_preds)
        cm_normalized = compute_confusion_matrix(all_labels, all_preds, normalize="true")

        # Classification report
        report = get_classification_report(all_labels, all_preds, target_names=class_names)

        # Prepare results
        results = {
            "metrics": metrics,
            "confusion_matrix": cm,
            "confusion_matrix_normalized": cm_normalized,
            "classification_report": report,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }

        if all_features:
            results["features"] = np.array(all_features)

        if all_image_paths:
            results["image_paths"] = all_image_paths

        # Log metrics
        logger.info("\n=== Evaluation Results ===")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        logger.info(f"\n{report}")

        # Save predictions
        if self.save_predictions:
            self._save_predictions(results, class_names)

        return results

    def _save_predictions(self, results: Dict, class_names: Optional[List[str]] = None):
        """Save predictions to disk."""
        # Create predictions DataFrame
        pred_data = {
            "label": results["labels"],
            "prediction": results["predictions"],
        }

        # Add probabilities
        num_classes = results["probabilities"].shape[1]
        for i in range(num_classes):
            class_name = class_names[i] if class_names else f"class_{i}"
            pred_data[f"prob_{class_name}"] = results["probabilities"][:, i]

        # Add image paths if available
        if "image_paths" in results:
            pred_data["image_path"] = results["image_paths"]

        df = pd.DataFrame(pred_data)

        # Save to CSV
        csv_path = self.output_dir / "predictions.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")

        # Save features if available
        if "features" in results:
            features_path = self.output_dir / "features.npy"
            np.save(features_path, results["features"])
            logger.info(f"Saved features to {features_path}")

        # Save confusion matrix
        cm_path = self.output_dir / "confusion_matrix.npy"
        np.save(cm_path, results["confusion_matrix"])
        logger.info(f"Saved confusion matrix to {cm_path}")

    def analyze_errors(
        self, results: Dict, class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Analyze prediction errors.

        Args:
            results: Evaluation results
            class_names: Names of classes

        Returns:
            Dictionary with error analysis
        """
        preds = results["predictions"]
        labels = results["labels"]
        probs = results["probabilities"]

        # Find errors
        errors = preds != labels
        error_indices = np.where(errors)[0]

        logger.info(f"Found {len(error_indices)} errors ({100*len(error_indices)/len(labels):.2f}%)")

        # High confidence errors
        max_probs = np.max(probs, axis=1)
        high_conf_errors = errors & (max_probs > 0.9)
        high_conf_error_indices = np.where(high_conf_errors)[0]

        logger.info(f"High confidence errors: {len(high_conf_error_indices)}")

        # Low confidence correct
        low_conf_correct = (~errors) & (max_probs < 0.6)
        low_conf_correct_indices = np.where(low_conf_correct)[0]

        logger.info(f"Low confidence correct: {len(low_conf_correct_indices)}")

        error_analysis = {
            "num_errors": len(error_indices),
            "error_rate": len(error_indices) / len(labels),
            "error_indices": error_indices,
            "high_conf_error_indices": high_conf_error_indices,
            "low_conf_correct_indices": low_conf_correct_indices,
        }

        return error_analysis
