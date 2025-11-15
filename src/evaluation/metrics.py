"""
Comprehensive evaluation metrics for cancer detection models.
Includes classification, regression, and medical-specific metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class CancerDetectionMetrics:
    """Comprehensive metrics calculator for cancer detection models."""

    def __init__(self, num_classes: int = 4, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of cancer types
            class_names: Names of cancer types
        """
        self.num_classes = num_classes
        self.class_names = class_names or [
            "Lung Cancer", "Breast Cancer", "Prostate Cancer", "Colorectal Cancer"
        ]
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.all_predictions = []
        self.all_labels = []
        self.all_probabilities = []
        self.all_stages = []
        self.all_stage_preds = []
        self.all_risks = []
        self.all_risk_preds = []

    def update(self, predictions: torch.Tensor, labels: torch.Tensor,
               probabilities: Optional[torch.Tensor] = None,
               stages: Optional[torch.Tensor] = None,
               stage_preds: Optional[torch.Tensor] = None,
               risks: Optional[torch.Tensor] = None,
               risk_preds: Optional[torch.Tensor] = None):
        """
        Update metrics with batch results.

        Args:
            predictions: Predicted class labels
            labels: True class labels
            probabilities: Class probabilities
            stages: True cancer stages
            stage_preds: Predicted cancer stages
            risks: True risk scores
            risk_preds: Predicted risk scores
        """
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())

        if probabilities is not None:
            self.all_probabilities.extend(probabilities.cpu().numpy())

        if stages is not None and stage_preds is not None:
            self.all_stages.extend(stages.cpu().numpy())
            self.all_stage_preds.extend(stage_preds.cpu().numpy())

        if risks is not None and risk_preds is not None:
            self.all_risks.extend(risks.cpu().numpy())
            self.all_risk_preds.extend(risk_preds.cpu().numpy())

    def compute_classification_metrics(self) -> Dict[str, float]:
        """Compute classification metrics for cancer detection."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }

        # Per-class metrics
        precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
        recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1s = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, class_name in enumerate(self.class_names[:len(precisions)]):
            metrics[f'precision_{class_name}'] = precisions[i]
            metrics[f'recall_{class_name}'] = recalls[i]
            metrics[f'f1_{class_name}'] = f1s[i]

        # ROC-AUC if probabilities available
        if len(self.all_probabilities) > 0:
            y_probs = np.array(self.all_probabilities)
            try:
                # Multi-class ROC-AUC
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='macro'
                )
                metrics['roc_auc_ovo'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovo', average='macro'
                )

                # Per-class ROC-AUC
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                for i, class_name in enumerate(self.class_names):
                    if i < y_probs.shape[1]:
                        try:
                            metrics[f'roc_auc_{class_name}'] = roc_auc_score(
                                y_true_bin[:, i], y_probs[:, i]
                            )
                        except ValueError:
                            metrics[f'roc_auc_{class_name}'] = 0.0
            except ValueError as e:
                print(f"Warning: Could not compute ROC-AUC: {e}")

        return metrics

    def compute_staging_metrics(self) -> Dict[str, float]:
        """Compute metrics for cancer staging."""
        if not self.all_stages or not self.all_stage_preds:
            return {}

        y_true = np.array(self.all_stages)
        y_pred = np.array(self.all_stage_preds)

        metrics = {
            'staging_accuracy': accuracy_score(y_true, y_pred),
            'staging_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'staging_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'staging_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        }

        return metrics

    def compute_risk_metrics(self) -> Dict[str, float]:
        """Compute metrics for risk assessment."""
        if not self.all_risks or not self.all_risk_preds:
            return {}

        y_true = np.array(self.all_risks)
        y_pred = np.array(self.all_risk_preds)

        metrics = {
            'risk_mse': mean_squared_error(y_true, y_pred),
            'risk_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'risk_mae': mean_absolute_error(y_true, y_pred),
            'risk_r2': r2_score(y_true, y_pred),
        }

        return metrics

    def compute_medical_metrics(self) -> Dict[str, float]:
        """Compute medical-specific metrics."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Sensitivity (True Positive Rate) and Specificity per class
        metrics = {}
        for i, class_name in enumerate(self.class_names):
            if i < len(cm):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - tp - fn - fp

                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

                metrics[f'sensitivity_{class_name}'] = sensitivity
                metrics[f'specificity_{class_name}'] = specificity
                metrics[f'ppv_{class_name}'] = ppv
                metrics[f'npv_{class_name}'] = npv

        return metrics

    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all available metrics."""
        metrics = {}
        metrics.update(self.compute_classification_metrics())
        metrics.update(self.compute_staging_metrics())
        metrics.update(self.compute_risk_metrics())
        metrics.update(self.compute_medical_metrics())
        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot and optionally save confusion matrix."""
        cm = self.get_confusion_matrix()

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names[:len(cm)],
                   yticklabels=self.class_names[:len(cm)])
        plt.title('Confusion Matrix - Cancer Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_curves(self, save_path: Optional[str] = None):
        """Plot ROC curves for each cancer type."""
        if not self.all_probabilities:
            print("No probabilities available for ROC curves")
            return

        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        y_true = np.array(self.all_labels)
        y_probs = np.array(self.all_probabilities)
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))

        plt.figure(figsize=(12, 8))

        for i, class_name in enumerate(self.class_names):
            if i < y_probs.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Cancer Detection')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_classification_report(self) -> str:
        """Generate detailed classification report."""
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )

    def print_summary(self):
        """Print a comprehensive metrics summary."""
        metrics = self.compute_all_metrics()

        print("\n" + "="*80)
        print("CANCER DETECTION EVALUATION SUMMARY")
        print("="*80)

        print("\n📊 CLASSIFICATION METRICS:")
        print(f"  Accuracy:           {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision (macro):  {metrics.get('precision_macro', 0):.4f}")
        print(f"  Recall (macro):     {metrics.get('recall_macro', 0):.4f}")
        print(f"  F1-Score (macro):   {metrics.get('f1_macro', 0):.4f}")
        if 'roc_auc_ovr' in metrics:
            print(f"  ROC-AUC (OvR):      {metrics.get('roc_auc_ovr', 0):.4f}")

        print("\n🎯 PER-CLASS PERFORMANCE:")
        for class_name in self.class_names:
            if f'f1_{class_name}' in metrics:
                print(f"  {class_name}:")
                print(f"    Precision: {metrics.get(f'precision_{class_name}', 0):.4f}")
                print(f"    Recall:    {metrics.get(f'recall_{class_name}', 0):.4f}")
                print(f"    F1-Score:  {metrics.get(f'f1_{class_name}', 0):.4f}")

        if 'staging_accuracy' in metrics:
            print("\n📈 STAGING METRICS:")
            print(f"  Accuracy:   {metrics.get('staging_accuracy', 0):.4f}")
            print(f"  F1-Score:   {metrics.get('staging_f1', 0):.4f}")

        if 'risk_r2' in metrics:
            print("\n⚠️  RISK ASSESSMENT METRICS:")
            print(f"  R² Score:   {metrics.get('risk_r2', 0):.4f}")
            print(f"  RMSE:       {metrics.get('risk_rmse', 0):.4f}")
            print(f"  MAE:        {metrics.get('risk_mae', 0):.4f}")

        print("\n" + "="*80)

        print("\n📋 DETAILED CLASSIFICATION REPORT:")
        print(self.generate_classification_report())


def calculate_inference_metrics(predictions: Dict, ground_truth: Dict) -> Dict[str, float]:
    """
    Calculate metrics for inference results.

    Args:
        predictions: Dictionary with prediction results
        ground_truth: Dictionary with ground truth labels

    Returns:
        Dictionary of computed metrics
    """
    metrics = {}

    if 'cancer_type' in predictions and 'cancer_type' in ground_truth:
        metrics['accuracy'] = accuracy_score(
            ground_truth['cancer_type'],
            predictions['cancer_type']
        )

    if 'confidence' in predictions:
        metrics['mean_confidence'] = np.mean(predictions['confidence'])
        metrics['std_confidence'] = np.std(predictions['confidence'])

    return metrics
