"""Callbacks for training loop."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch

from cancer_quant_model.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Callback:
    """Base callback class."""

    def on_train_start(self, trainer):
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer, epoch):
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch, metrics):
        """Called at the end of each epoch."""
        pass

    def on_batch_start(self, trainer, batch_idx):
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, trainer, batch_idx, loss):
        """Called at the end of each batch."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        mode: str = "min",
        min_delta: float = 0.0,
        verbose: bool = True,
    ):
        """
        Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which training will be stopped
            mode: 'min' or 'max'
            min_delta: Minimum change to qualify as an improvement
            verbose: Print messages
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.epochs_without_improvement = 0
        self.should_stop = False

    def on_epoch_end(self, trainer, epoch, metrics):
        """Check if training should stop."""
        current_value = metrics.get(self.monitor)

        if current_value is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return

        improved = False
        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.epochs_without_improvement = 0
            if self.verbose:
                logger.info(
                    f"Metric {self.monitor} improved to {current_value:.4f}"
                )
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                logger.info(
                    f"No improvement in {self.monitor} for {self.epochs_without_improvement} epochs"
                )

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            if self.verbose:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs"
                )


class CheckpointCallback(Callback):
    """Model checkpointing callback."""

    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_top_k: Save top k checkpoints
            save_last: Save last checkpoint
            verbose: Print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.checkpoints = []  # List of (metric_value, path)

    def on_epoch_end(self, trainer, epoch, metrics):
        """Save checkpoint if metric improved."""
        current_value = metrics.get(self.monitor)

        if current_value is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return

        # Save last checkpoint
        if self.save_last:
            last_path = self.checkpoint_dir / "last.pt"
            self._save_checkpoint(trainer, last_path, epoch, current_value)

        # Check if this is a top-k checkpoint
        is_best = False
        if self.mode == "min":
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value

        if is_best:
            self.best_value = current_value
            best_path = self.checkpoint_dir / "best.pt"
            self._save_checkpoint(trainer, best_path, epoch, current_value)

            if self.verbose:
                logger.info(
                    f"Saved best checkpoint with {self.monitor}={current_value:.4f}"
                )

        # Save top-k checkpoints
        checkpoint_path = (
            self.checkpoint_dir / f"epoch_{epoch:03d}_metric_{current_value:.4f}.pt"
        )
        self._save_checkpoint(trainer, checkpoint_path, epoch, current_value)

        self.checkpoints.append((current_value, checkpoint_path))

        # Sort and keep only top-k
        self.checkpoints.sort(reverse=(self.mode == "max"))
        if len(self.checkpoints) > self.save_top_k:
            for _, old_path in self.checkpoints[self.save_top_k :]:
                if old_path.exists() and old_path.name not in ["best.pt", "last.pt"]:
                    old_path.unlink()
            self.checkpoints = self.checkpoints[: self.save_top_k]

    def _save_checkpoint(
        self, trainer, path: Path, epoch: int, metric_value: float
    ):
        """Save a checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "metric_value": metric_value,
            "best_value": self.best_value,
        }

        if trainer.scheduler is not None:
            checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        torch.save(checkpoint, path)


class MLflowLoggingCallback(Callback):
    """MLflow logging callback."""

    def __init__(self, log_every_n_steps: int = 10):
        """
        Initialize MLflow logging callback.

        Args:
            log_every_n_steps: Log metrics every N steps
        """
        self.log_every_n_steps = log_every_n_steps
        self.step = 0

    def on_train_start(self, trainer):
        """Log hyperparameters."""
        params = {
            "model_name": trainer.model.__class__.__name__,
            "optimizer": trainer.optimizer.__class__.__name__,
            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
            "batch_size": trainer.train_loader.batch_size,
            "max_epochs": trainer.max_epochs,
        }

        mlflow.log_params(params)

    def on_epoch_end(self, trainer, epoch, metrics):
        """Log epoch metrics."""
        # Log all metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=epoch)

        # Log learning rate
        if trainer.scheduler is not None:
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

    def on_batch_end(self, trainer, batch_idx, loss):
        """Log batch loss."""
        self.step += 1
        if self.step % self.log_every_n_steps == 0:
            mlflow.log_metric("batch_loss", loss, step=self.step)


class MetricHistoryCallback(Callback):
    """Track metric history."""

    def __init__(self):
        """Initialize metric history callback."""
        self.history = {}

    def on_epoch_end(self, trainer, epoch, metrics):
        """Record metrics."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def save_history(self, path: Path):
        """Save history to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(f"Saved metric history to {path}")


class LearningRateSchedulerCallback(Callback):
    """Learning rate scheduler callback."""

    def __init__(self, scheduler, monitor: Optional[str] = None):
        """
        Initialize LR scheduler callback.

        Args:
            scheduler: PyTorch LR scheduler
            monitor: Metric to monitor (for ReduceLROnPlateau)
        """
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, trainer, epoch, metrics):
        """Step the scheduler."""
        if isinstance(
            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            if self.monitor is None:
                logger.warning(
                    "ReduceLROnPlateau requires a monitor metric"
                )
                return

            metric_value = metrics.get(self.monitor)
            if metric_value is not None:
                self.scheduler.step(metric_value)
        else:
            self.scheduler.step()


class GradientNormCallback(Callback):
    """Track gradient norms."""

    def __init__(self, log_every_n_steps: int = 100):
        """
        Initialize gradient norm callback.

        Args:
            log_every_n_steps: Log every N steps
        """
        self.log_every_n_steps = log_every_n_steps
        self.step = 0

    def on_batch_end(self, trainer, batch_idx, loss):
        """Log gradient norm."""
        self.step += 1

        if self.step % self.log_every_n_steps == 0:
            total_norm = 0.0
            for p in trainer.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            mlflow.log_metric("gradient_norm", total_norm, step=self.step)
