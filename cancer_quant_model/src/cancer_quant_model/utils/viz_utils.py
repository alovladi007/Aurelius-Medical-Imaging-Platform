"""Visualization utilities for the cancer quantitative model."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Names of classes
        normalize: Normalize the confusion matrix
        title: Plot title
        cmap: Colormap
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=cmap, values_format=".2f" if normalize else "d")

    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)

    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_title(title, fontsize=14)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_training_history(
    history: dict,
    metrics: List[str] = ["loss", "accuracy"],
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot training history.

    Args:
        history: Dictionary with training history
            Expected keys: 'train_{metric}', 'val_{metric}'
        metrics: List of metrics to plot
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"

        if train_key in history:
            ax.plot(history[train_key], label=f"Train {metric}", marker="o")
        if val_key in history:
            ax.plot(history[val_key], label=f"Val {metric}", marker="s")

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} over Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_image_grid(
    images: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    predictions: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    nrow: int = 8,
    title: str = "Image Grid",
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot a grid of images.

    Args:
        images: Tensor of images (B, C, H, W)
        labels: True labels
        predictions: Predicted labels
        class_names: Names of classes
        nrow: Number of images per row
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    batch_size = images.shape[0]
    ncol = min(nrow, batch_size)
    nrow = (batch_size + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axes = axes.flatten() if batch_size > 1 else [axes]

    for idx in range(batch_size):
        ax = axes[idx]

        # Convert to numpy and handle normalization
        img = images[idx].cpu().numpy()
        if img.shape[0] == 3:  # RGB
            img = np.transpose(img, (1, 2, 0))
            # Denormalize if needed (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        else:  # Grayscale
            img = img.squeeze()

        ax.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        ax.axis("off")

        # Add title with labels
        if labels is not None or predictions is not None:
            title_parts = []
            if labels is not None:
                label = labels[idx].item() if torch.is_tensor(labels[idx]) else labels[idx]
                label_name = class_names[label] if class_names else str(label)
                title_parts.append(f"GT: {label_name}")
            if predictions is not None:
                pred = (
                    predictions[idx].item()
                    if torch.is_tensor(predictions[idx])
                    else predictions[idx]
                )
                pred_name = class_names[pred] if class_names else str(pred)
                title_parts.append(f"Pred: {pred_name}")

            ax.set_title(" | ".join(title_parts), fontsize=8)

    # Hide empty subplots
    for idx in range(batch_size, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_feature_distributions(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    max_features: int = 10,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot distributions of features across classes.

    Args:
        features: Feature matrix (N, F)
        labels: Labels (N,)
        feature_names: Names of features
        class_names: Names of classes
        max_features: Maximum number of features to plot
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    n_features = min(features.shape[1], max_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]

    for idx in range(n_features):
        ax = axes[idx]

        for label in np.unique(labels):
            mask = labels == label
            label_name = class_names[label] if class_names else f"Class {label}"
            ax.hist(features[mask, idx], alpha=0.6, label=label_name, bins=30)

        feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {feature_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
        title: Plot title
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(8, 8))

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Plot Grad-CAM overlay on image.

    Args:
        image: Original image (H, W, C) in [0, 1]
        heatmap: Grad-CAM heatmap (H, W) in [0, 1]
        alpha: Overlay transparency
        colormap: Colormap for heatmap
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap, cmap=colormap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap=colormap, alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
