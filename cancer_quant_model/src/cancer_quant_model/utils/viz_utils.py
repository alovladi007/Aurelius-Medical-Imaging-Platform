"""Visualization utilities for model evaluation and analysis."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
):
    """Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        normalize=normalize,
        cmap="Blues",
        ax=ax,
    )

    if normalize:
        ax.set_title(f"Confusion Matrix (normalized by {normalize})")
    else:
        ax.set_title("Confusion Matrix")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """Plot ROC curves.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_names: Names of classes
        save_path: Path to save figure
        figsize: Figure size
    """
    num_classes = y_proba.shape[1] if len(y_proba.shape) == 2 else 2

    fig, ax = plt.subplots(figsize=figsize)

    if num_classes == 2:
        # Binary classification
        if len(y_proba.shape) == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")

    else:
        # Multi-class: plot OvR curves
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]

        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)

            ax.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random classifier")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """Plot Precision-Recall curves.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_names: Names of classes
        save_path: Path to save figure
        figsize: Figure size
    """
    num_classes = y_proba.shape[1] if len(y_proba.shape) == 2 else 2

    fig, ax = plt.subplots(figsize=figsize)

    if num_classes == 2:
        # Binary classification
        if len(y_proba.shape) == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba

        precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
        pr_auc = auc(recall, precision)

        ax.plot(recall, precision, lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")

    else:
        # Multi-class: plot OvR curves
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]

        for i in range(num_classes):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, i])
            pr_auc = auc(recall, precision)

            ax.plot(recall, precision, lw=2, label=f"{class_names[i]} (AUC = {pr_auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
):
    """Plot calibration curve.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        n_bins: Number of bins
        save_path: Path to save figure
        figsize: Figure size
    """
    if len(y_proba.shape) == 2:
        y_proba = y_proba[:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")

    # Plot calibration curve
    ax.plot(prob_pred, prob_true, marker="o", lw=2, label="Model")
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Perfect calibration")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """Plot distribution of prediction probabilities by true class.

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_names: Names of classes
        save_path: Path to save figure
        figsize: Figure size
    """
    if len(y_proba.shape) == 2:
        y_proba = y_proba[:, 1]

    if class_names is None:
        class_names = ["Class 0", "Class 1"]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot distributions
    for label in np.unique(y_true):
        mask = y_true == label
        ax.hist(
            y_proba[mask],
            bins=50,
            alpha=0.6,
            label=class_names[label],
            density=True,
        )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution by True Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def save_prediction_samples(
    images: List[np.ndarray],
    y_true: List[int],
    y_pred: List[int],
    y_proba: List[float],
    save_dir: str,
    class_names: Optional[List[str]] = None,
    max_samples: int = 50,
):
    """Save sample predictions with images.

    Args:
        images: List of images
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        save_dir: Directory to save samples
        class_names: Names of classes
        max_samples: Maximum number of samples to save
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        num_classes = max(max(y_true), max(y_pred)) + 1
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Create subplots
    n_samples = min(len(images), max_samples)
    n_cols = 5
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    axes = axes.flatten() if n_samples > 1 else [axes]

    for idx in range(n_samples):
        ax = axes[idx]
        img = images[idx]

        # Normalize image for display
        if img.max() > 1.0:
            img = img / 255.0

        ax.imshow(img)

        # Create title with prediction info
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        prob = y_proba[idx]

        correct = "✓" if y_true[idx] == y_pred[idx] else "✗"
        color = "green" if y_true[idx] == y_pred[idx] else "red"

        title = f"{correct} True: {true_label}\nPred: {pred_label} ({prob:.2f})"
        ax.set_title(title, fontsize=8, color=color)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "prediction_samples.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_metrics: Optional[dict] = None,
    val_metrics: Optional[dict] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5),
):
    """Plot training curves (loss and metrics).

    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_metrics: Dictionary of training metrics per epoch
        val_metrics: Dictionary of validation metrics per epoch
        save_path: Path to save figure
        figsize: Figure size
    """
    n_plots = 1 + (1 if train_metrics else 0)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Plot losses
    axes[0].plot(epochs, train_losses, "o-", label="Train Loss")
    if val_losses:
        axes[0].plot(epochs, val_losses, "s-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot metrics
    if train_metrics and n_plots > 1:
        for metric_name in train_metrics.keys():
            axes[1].plot(epochs, train_metrics[metric_name], "o-", label=f"Train {metric_name}")
            if val_metrics and metric_name in val_metrics:
                axes[1].plot(epochs, val_metrics[metric_name], "s-", label=f"Val {metric_name}")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Metric Value")
        axes[1].set_title("Training and Validation Metrics")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
