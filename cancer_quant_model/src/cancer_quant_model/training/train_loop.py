"""Training loop with MLflow tracking."""

import time
from pathlib import Path
from typing import Dict, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from cancer_quant_model.utils.logging_utils import get_logger
from cancer_quant_model.utils.metrics_utils import MetricsTracker, compute_classification_metrics

logger = get_logger(__name__)


class Trainer:
    """Trainer for cancer histopathology models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        mixed_precision: bool = True,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1,
        max_epochs: int = 50,
        checkpoint_dir: Path = Path("experiments/checkpoints"),
        mlflow_tracking_uri: str = "experiments/mlruns",
        experiment_name: str = "cancer_quant_model",
        run_name: Optional[str] = None,
        log_every_n_steps: int = 10,
        save_top_k: int = 3,
        monitor: str = "val_auroc",
        mode: str = "max",
        early_stopping_patience: int = 10,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device for training
            mixed_precision: Use mixed precision training
            gradient_clip_val: Gradient clipping value
            accumulate_grad_batches: Gradient accumulation steps
            max_epochs: Maximum number of epochs
            checkpoint_dir: Directory for saving checkpoints
            mlflow_tracking_uri: MLflow tracking URI
            experiment_name: MLflow experiment name
            run_name: MLflow run name
            log_every_n_steps: Log metrics every N steps
            save_top_k: Save top k checkpoints
            monitor: Metric to monitor for checkpointing
            mode: 'min' or 'max' for monitored metric
            early_stopping_patience: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches

        self.max_epochs = max_epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_every_n_steps = log_every_n_steps
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode
        self.early_stopping_patience = early_stopping_patience

        # MLflow setup
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        if run_name is None:
            run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_name = run_name

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("-inf") if mode == "max" else float("inf")
        self.epochs_without_improvement = 0
        self.checkpoint_scores = []

        logger.info(f"Trainer initialized. Device: {device}, Mixed precision: {mixed_precision}")

    def train(self):
        """Run training loop."""
        with mlflow.start_run(run_name=self.run_name):
            # Log hyperparameters
            self._log_hyperparameters()

            logger.info(f"Starting training for {self.max_epochs} epochs")

            for epoch in range(self.max_epochs):
                self.current_epoch = epoch

                # Training epoch
                train_metrics = self.train_epoch()
                logger.info(
                    f"Epoch {epoch+1}/{self.max_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics.get('accuracy', 0):.4f}"
                )

                # Validation epoch
                if self.val_loader is not None:
                    val_metrics = self.validate_epoch()
                    logger.info(
                        f"Epoch {epoch+1}/{self.max_epochs} - "
                        f"Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics.get('accuracy', 0):.4f}, "
                        f"Val AUROC: {val_metrics.get('auroc', 0):.4f}"
                    )

                    # Log to MLflow
                    mlflow.log_metrics(
                        {f"train_{k}": v for k, v in train_metrics.items()}, step=epoch
                    )
                    mlflow.log_metrics(
                        {f"val_{k}": v for k, v in val_metrics.items()}, step=epoch
                    )

                    # Checkpointing
                    current_metric = val_metrics.get(self.monitor.replace("val_", ""), 0)
                    self._save_checkpoint(epoch, current_metric)

                    # Early stopping
                    if self._check_early_stopping(current_metric):
                        logger.info(f"Early stopping triggered after epoch {epoch+1}")
                        break
                else:
                    mlflow.log_metrics(
                        {f"train_{k}": v for k, v in train_metrics.items()}, step=epoch
                    )

                # LR scheduling
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_metrics.get("loss", 0))
                    else:
                        self.scheduler.step()

                    # Log learning rate
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    mlflow.log_metric("learning_rate", current_lr, step=epoch)

            logger.info("Training completed")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics_tracker = MetricsTracker()

        all_preds = []
        all_labels = []
        all_probs = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch_idx, (images, labels, metadata) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    if isinstance(outputs, dict):
                        logits = outputs["logits"]
                    else:
                        logits = outputs
                    loss = self.criterion(logits, labels)
            else:
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs
                loss = self.criterion(logits, labels)

            # Backward pass
            loss = loss / self.accumulate_grad_batches

            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Collect metrics
            metrics_tracker.update({"loss": loss.item() * self.accumulate_grad_batches}, n=1)

            # Collect predictions
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * self.accumulate_grad_batches})

        # Compute epoch metrics
        epoch_metrics = metrics_tracker.compute()

        # Compute classification metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        classification_metrics = compute_classification_metrics(
            all_labels, all_preds, all_probs
        )
        epoch_metrics.update(classification_metrics)

        return epoch_metrics

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        metrics_tracker = MetricsTracker()

        all_preds = []
        all_labels = []
        all_probs = []

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")

        for images, labels, metadata in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            loss = self.criterion(logits, labels)

            # Collect metrics
            metrics_tracker.update({"loss": loss.item()}, n=images.size(0))

            # Collect predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        # Compute epoch metrics
        epoch_metrics = metrics_tracker.compute()

        # Compute classification metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        classification_metrics = compute_classification_metrics(
            all_labels, all_preds, all_probs
        )
        epoch_metrics.update(classification_metrics)

        return epoch_metrics

    def _save_checkpoint(self, epoch: int, metric_value: float):
        """Save model checkpoint."""
        is_best = False

        if self.mode == "max":
            is_best = metric_value > self.best_metric
        else:
            is_best = metric_value < self.best_metric

        if is_best:
            self.best_metric = metric_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "metric_value": metric_value,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            mlflow.log_artifact(str(best_path))
            logger.info(f"Saved best checkpoint: {best_path}")

        # Save top-k checkpoints
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}_metric_{metric_value:.4f}.pt"
        torch.save(checkpoint, checkpoint_path)

        self.checkpoint_scores.append((metric_value, checkpoint_path))
        self.checkpoint_scores.sort(reverse=(self.mode == "max"))

        # Remove checkpoints beyond top-k
        if len(self.checkpoint_scores) > self.save_top_k:
            for _, old_path in self.checkpoint_scores[self.save_top_k :]:
                if old_path.exists():
                    old_path.unlink()
            self.checkpoint_scores = self.checkpoint_scores[: self.save_top_k]

    def _check_early_stopping(self, current_metric: float) -> bool:
        """Check if early stopping should be triggered."""
        if self.epochs_without_improvement >= self.early_stopping_patience:
            return True
        return False

    def _log_hyperparameters(self):
        """Log hyperparameters to MLflow."""
        params = {
            "max_epochs": self.max_epochs,
            "batch_size": self.train_loader.batch_size,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "optimizer": self.optimizer.__class__.__name__,
            "mixed_precision": self.mixed_precision,
            "gradient_clip_val": self.gradient_clip_val,
            "accumulate_grad_batches": self.accumulate_grad_batches,
        }
        mlflow.log_params(params)
