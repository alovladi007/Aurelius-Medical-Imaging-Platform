"""
Visualization utilities for training and evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

sns.set_style('whitegrid')


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Plot training history including loss and metrics.

    Args:
        history: Dictionary with training history
        save_path: Path to save figure
        figsize: Figure size
    """
    metrics = [k for k in history.keys() if not k.startswith('val_')]
    n_metrics = len(metrics)

    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot training metric
        ax.plot(history[metric], label=f'Train {metric}', linewidth=2)

        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Val {metric}', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")

    plt.close()


def plot_predictions(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: List[str],
    probabilities: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
    max_images: int = 16
):
    """
    Plot predictions vs ground truth.

    Args:
        images: Batch of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        probabilities: Class probabilities
        save_path: Path to save figure
        max_images: Maximum number of images to plot
    """
    n_images = min(len(images), max_images)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_images > 1 else [axes]

    for idx in range(n_images):
        ax = axes[idx]

        # Get image
        img = images[idx].cpu().numpy()
        if img.shape[0] == 1:  # Grayscale
            img = img[0]
            cmap = 'gray'
        else:  # RGB
            img = np.transpose(img, (1, 2, 0))
            cmap = None

        # Normalize for display
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Plot image
        ax.imshow(img, cmap=cmap)

        # Create title
        true_label = class_names[true_labels[idx]]
        pred_label = class_names[pred_labels[idx]]

        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'

        title = f'True: {true_label}\nPred: {pred_label}'
        if probabilities is not None:
            conf = probabilities[idx, pred_labels[idx]].item()
            title += f'\nConf: {conf:.2f}'

        ax.set_title(title, color=color, fontweight='bold')
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Predictions saved to {save_path}")

    plt.close()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None
):
    """
    Plot class distribution.

    Args:
        labels: Array of labels
        class_names: List of class names
        save_path: Path to save figure
    """
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')

    plt.xlabel('Cancer Type')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")

    plt.close()


def plot_attention_maps(
    attention_weights: torch.Tensor,
    images: torch.Tensor,
    save_path: Optional[str] = None,
    n_images: int = 4
):
    """
    Plot attention maps overlaid on images.

    Args:
        attention_weights: Attention weight tensors
        images: Input images
        save_path: Path to save figure
        n_images: Number of images to plot
    """
    n_images = min(len(images), n_images)

    fig, axes = plt.subplots(n_images, 2, figsize=(10, 5 * n_images))
    if n_images == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_images):
        # Original image
        img = images[idx].cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]
            cmap = 'gray'
        else:
            img = np.transpose(img, (1, 2, 0))
            cmap = None

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[idx, 0].imshow(img, cmap=cmap)
        axes[idx, 0].set_title('Original Image')
        axes[idx, 0].axis('off')

        # Attention map
        attn = attention_weights[idx].cpu().numpy()
        if attn.ndim > 2:
            attn = attn.mean(axis=0)  # Average over attention heads

        # Resize attention to image size
        from scipy.ndimage import zoom
        zoom_factor = img.shape[0] / attn.shape[0]
        attn_resized = zoom(attn, zoom_factor, order=1)

        # Overlay attention
        axes[idx, 1].imshow(img, cmap=cmap)
        axes[idx, 1].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[idx, 1].set_title('Attention Map')
        axes[idx, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention maps saved to {save_path}")

    plt.close()


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Plot comparison of metrics across different models/experiments.

    Args:
        metrics_dict: Dictionary mapping experiment names to metrics
        save_path: Path to save figure
    """
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T

    # Select key metrics
    key_metrics = ['accuracy', 'f1_macro', 'roc_auc_ovr', 'precision_macro', 'recall_macro']
    key_metrics = [m for m in key_metrics if m in df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    df[key_metrics].plot(kind='bar', ax=ax)

    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison Across Experiments')
    ax.legend(title='Metric')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")

    plt.close()


def create_evaluation_report(
    metrics: Dict[str, float],
    confusion_matrix_path: Optional[str] = None,
    roc_curves_path: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """
    Create comprehensive evaluation report with visualizations.

    Args:
        metrics: Dictionary of computed metrics
        confusion_matrix_path: Path to confusion matrix image
        roc_curves_path: Path to ROC curves image
        output_dir: Directory to save report
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics to JSON
        import json
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {metrics_file}")

        # Create HTML report
        html_report = output_dir / "evaluation_report.html"
        with open(html_report, 'w') as f:
            f.write("<html><head><title>Evaluation Report</title></head><body>")
            f.write("<h1>Cancer Detection Evaluation Report</h1>")

            f.write("<h2>Metrics</h2>")
            f.write("<table border='1'>")
            f.write("<tr><th>Metric</th><th>Value</th></tr>")
            for key, value in sorted(metrics.items()):
                f.write(f"<tr><td>{key}</td><td>{value:.4f}</td></tr>")
            f.write("</table>")

            if confusion_matrix_path:
                f.write("<h2>Confusion Matrix</h2>")
                f.write(f"<img src='{Path(confusion_matrix_path).name}' width='800'>")

            if roc_curves_path:
                f.write("<h2>ROC Curves</h2>")
                f.write(f"<img src='{Path(roc_curves_path).name}' width='800'>")

            f.write("</body></html>")

        print(f"HTML report saved to {html_report}")
