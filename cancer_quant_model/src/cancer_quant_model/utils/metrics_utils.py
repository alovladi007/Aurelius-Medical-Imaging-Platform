"""Metrics computation utilities for model evaluation."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
) -> Dict[str, float]:
    """Compute basic classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy ('binary', 'macro', 'micro', 'weighted')

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }

    return metrics


def compute_auc_scores(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Compute AUC-ROC and AUC-PR scores.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities (shape: [n_samples, n_classes] or [n_samples])
        num_classes: Number of classes

    Returns:
        Dictionary of AUC scores
    """
    metrics = {}

    try:
        if num_classes == 2:
            # Binary classification
            if len(y_proba.shape) == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba

            # ROC-AUC
            metrics["auc_roc"] = roc_auc_score(y_true, y_proba_pos)

            # PR-AUC
            precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
            metrics["auc_pr"] = auc(recall, precision)

        else:
            # Multi-class classification
            metrics["auc_roc_macro"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
            metrics["auc_roc_weighted"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted"
            )

    except Exception as e:
        print(f"Warning: Could not compute AUC scores: {e}")
        metrics["auc_roc"] = 0.0
        metrics["auc_pr"] = 0.0

    return metrics


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute Brier score for probability calibration.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class

    Returns:
        Brier score
    """
    try:
        if len(y_proba.shape) == 2:
            y_proba = y_proba[:, 1]
        return brier_score_loss(y_true, y_proba)
    except Exception as e:
        print(f"Warning: Could not compute Brier score: {e}")
        return 0.0


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        class_names: Names of classes

    Returns:
        Dictionary of per-class metrics
    """
    num_classes = y_proba.shape[1] if len(y_proba.shape) == 2 else 2

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    per_class = {}

    # Precision, recall, F1 per class
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
        }

        # ROC-AUC per class (OvR)
        if num_classes > 2:
            try:
                y_true_binary = (y_true == i).astype(int)
                y_proba_class = y_proba[:, i]
                per_class[class_name]["auc_roc"] = roc_auc_score(y_true_binary, y_proba_class)
            except Exception:
                per_class[class_name]["auc_roc"] = 0.0

    return per_class


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins

    Returns:
        Expected Calibration Error
    """
    if len(y_proba.shape) == 2:
        y_proba = y_proba[:, 1]

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    num_classes: int = 2,
    class_names: Optional[List[str]] = None,
    compute_per_class: bool = True,
) -> Dict[str, any]:
    """Compute all evaluation metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        num_classes: Number of classes
        class_names: Names of classes
        compute_per_class: Whether to compute per-class metrics

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # Basic classification metrics
    average = "binary" if num_classes == 2 else "macro"
    metrics.update(compute_classification_metrics(y_true, y_pred, average=average))

    # AUC scores
    metrics.update(compute_auc_scores(y_true, y_proba, num_classes=num_classes))

    # Brier score
    metrics["brier_score"] = compute_brier_score(y_true, y_proba)

    # Expected Calibration Error
    metrics["ece"] = compute_expected_calibration_error(y_true, y_proba)

    # Confusion matrix
    metrics["confusion_matrix"] = compute_confusion_matrix(y_true, y_pred).tolist()

    # Per-class metrics
    if compute_per_class and num_classes > 1:
        metrics["per_class"] = compute_per_class_metrics(
            y_true, y_pred, y_proba, class_names=class_names
        )

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find optimal classification threshold.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')

    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    if len(y_proba.shape) == 2:
        y_proba = y_proba[:, 1]

    # Try different thresholds
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(y_true, y_pred_thresh)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return float(best_threshold), float(best_score)
