"""Metrics utilities for model evaluation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 2,
    average: str = "binary",
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Predicted probabilities (N, num_classes) or (N,) for binary
        num_classes: Number of classes
        average: Averaging method for multi-class ('micro', 'macro', 'weighted', 'binary')
        prefix: Prefix for metric names

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Accuracy
    metrics[f"{prefix}accuracy"] = accuracy_score(y_true, y_pred)
    metrics[f"{prefix}balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    # Precision, Recall, F1
    metrics[f"{prefix}precision"] = precision_score(
        y_true, y_pred, average=average, zero_division=0
    )
    metrics[f"{prefix}recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics[f"{prefix}f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Specificity (for binary classification)
    if num_classes == 2 and average == "binary":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics[f"{prefix}specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC AUC and PR AUC (if probabilities provided)
    if y_prob is not None:
        try:
            if num_classes == 2:
                # For binary, use probabilities of positive class
                if y_prob.ndim == 2:
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob

                metrics[f"{prefix}auroc"] = roc_auc_score(y_true, y_prob_pos)
                metrics[f"{prefix}auprc"] = average_precision_score(y_true, y_prob_pos)
            else:
                # For multi-class
                multi_avg = "macro" if average == "binary" else average
                metrics[f"{prefix}auroc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average=multi_avg
                )
        except ValueError as e:
            # Handle edge cases (e.g., only one class present)
            pass

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', None)

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def compute_roc_auc(
    y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 2
) -> Dict[str, float]:
    """
    Compute ROC AUC scores.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        num_classes: Number of classes

    Returns:
        Dictionary with ROC AUC scores
    """
    results = {}

    if num_classes == 2:
        # Binary classification
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]
        results["auroc"] = roc_auc_score(y_true, y_prob)
    else:
        # Multi-class
        results["auroc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        results["auroc_weighted"] = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="weighted"
        )

        # Per-class AUC
        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            results[f"auroc_class_{i}"] = roc_auc_score(y_true_binary, y_prob[:, i])

    return results


def get_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, target_names: Optional[List[str]] = None
) -> str:
    """
    Get classification report as string.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names of target classes

    Returns:
        Classification report string
    """
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


class MetricsTracker:
    """Track metrics across epochs/batches."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = {}
        self.counts = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        Update metrics with new values.

        Args:
            metrics: Dictionary of metric values
            n: Number of samples (for averaging)
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0

            self.metrics[key] += value * n
            self.counts[key] += n

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics.

        Returns:
            Dictionary of averaged metrics
        """
        return {key: self.metrics[key] / self.counts[key] for key in self.metrics}

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


class ConfusionMatrixTracker:
    """Track confusion matrix across batches."""

    def __init__(self, num_classes: int):
        """
        Initialize confusion matrix tracker.

        Args:
            num_classes: Number of classes
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Update confusion matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
        self.confusion_matrix += cm

    def compute(self, normalize: Optional[str] = None) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            normalize: Normalization mode

        Returns:
            Confusion matrix
        """
        cm = self.confusion_matrix.copy()

        if normalize == "true":
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            cm = cm.astype(float) / cm.sum(axis=0, keepdims=True)
        elif normalize == "all":
            cm = cm.astype(float) / cm.sum()

        return cm

    def reset(self):
        """Reset confusion matrix."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
