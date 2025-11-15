"""Evaluation loop for cancer classification models."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging_utils import get_logger
from ..utils.metrics_utils import compute_all_metrics, find_optimal_threshold
from ..utils.viz_utils import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
)

logger = get_logger()


class Evaluator:
    """Evaluator class for model evaluation."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        num_classes: int = 2,
        class_names: Optional[List[str]] = None,
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to run evaluation on
            num_classes: Number of classes
            class_names: Names of classes
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]

    @torch.no_grad()
    def evaluate(
        self,
        save_dir: Optional[str] = None,
        save_predictions: bool = True,
        create_visualizations: bool = True,
    ) -> Dict:
        """Run evaluation.

        Args:
            save_dir: Directory to save results
            save_predictions: Whether to save predictions
            create_visualizations: Whether to create visualizations

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")

        # Collect predictions
        all_preds = []
        all_labels = []
        all_probs = []
        all_image_paths = []

        for images, labels, metadata in tqdm(self.test_loader, desc="Evaluating"):
            images = images.to(self.device)

            # Forward pass
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

            # Collect image paths
            for meta in metadata:
                all_image_paths.append(meta.get("image_path", ""))

        # Convert to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        # Compute all metrics
        metrics = compute_all_metrics(
            all_labels,
            all_preds,
            all_probs,
            num_classes=self.num_classes,
            class_names=self.class_names,
            compute_per_class=True,
        )

        # Find optimal threshold
        optimal_threshold, optimal_f1 = find_optimal_threshold(
            all_labels, all_probs, metric="f1"
        )

        metrics["optimal_threshold"] = optimal_threshold
        metrics["optimal_f1"] = optimal_f1

        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"AUC-ROC: {metrics.get('auc_roc', 0.0):.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")

        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics
            with open(save_dir / "metrics.json", "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                json_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, np.ndarray):
                        json_metrics[k] = v.tolist()
                    elif isinstance(v, (np.int64, np.float64)):
                        json_metrics[k] = float(v)
                    else:
                        json_metrics[k] = v

                json.dump(json_metrics, f, indent=2)

            logger.info(f"Saved metrics to {save_dir / 'metrics.json'}")

            # Save predictions
            if save_predictions:
                predictions_df = pd.DataFrame(
                    {
                        "image_path": all_image_paths,
                        "true_label": all_labels,
                        "predicted_label": all_preds,
                        **{
                            f"prob_class_{i}": all_probs[:, i]
                            for i in range(self.num_classes)
                        },
                    }
                )

                predictions_df.to_csv(save_dir / "predictions.csv", index=False)
                logger.info(f"Saved predictions to {save_dir / 'predictions.csv'}")

            # Create visualizations
            if create_visualizations:
                viz_dir = save_dir / "visualizations"
                viz_dir.mkdir(exist_ok=True)

                # Confusion matrix
                plot_confusion_matrix(
                    all_labels,
                    all_preds,
                    class_names=self.class_names,
                    save_path=viz_dir / "confusion_matrix.png",
                )

                # ROC curve
                plot_roc_curves(
                    all_labels,
                    all_probs,
                    class_names=self.class_names,
                    save_path=viz_dir / "roc_curve.png",
                )

                # PR curve
                plot_precision_recall_curves(
                    all_labels,
                    all_probs,
                    class_names=self.class_names,
                    save_path=viz_dir / "pr_curve.png",
                )

                # Calibration curve
                plot_calibration_curve(
                    all_labels,
                    all_probs,
                    save_path=viz_dir / "calibration_curve.png",
                )

                logger.info(f"Saved visualizations to {viz_dir}")

        return metrics
