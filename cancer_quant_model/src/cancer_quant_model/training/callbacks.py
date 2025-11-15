"""Training callbacks for model training."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import numpy as np
import torch


class Callback:
    """Base callback class."""

    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer, epoch):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch, logs):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer, batch_idx):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, trainer, batch_idx, logs):
        """Called at the end of each batch."""
        pass


class EarlyStoppingCallback(Callback):
    """Early stopping callback to stop training when metric stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.001,
        verbose: bool = True,
    ):
        """Initialize early stopping callback.

        Args:
            monitor: Metric to monitor
            mode: 'min' or 'max' - whether lower or higher is better
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        super().__init__()

        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def on_epoch_end(self, trainer, epoch, logs):
        """Check if should stop training."""
        if self.monitor not in logs:
            return

        current_score = logs[self.monitor]

        if self.best_score is None:
            self.best_score = current_score
            return

        # Check improvement
        if self.mode == "min":
            improved = (self.best_score - current_score) > self.min_delta
        else:
            improved = (current_score - self.best_score) > self.min_delta

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"\nEarly stopping triggered after {epoch + 1} epochs. "
                        f"No improvement in {self.monitor} for {self.patience} epochs."
                    )


class CheckpointCallback(Callback):
    """Checkpoint saving callback."""

    def __init__(
        self,
        dirpath: str = "experiments/logs",
        filename: str = "checkpoint-{epoch:02d}-{val_loss:.3f}",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
        verbose: bool = True,
    ):
        """Initialize checkpoint callback.

        Args:
            dirpath: Directory to save checkpoints
            filename: Filename pattern
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_top_k: Save top K checkpoints
            save_last: Save last checkpoint
            verbose: Print messages
        """
        super().__init__()

        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)

        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.verbose = verbose

        self.best_scores = []
        self.best_paths = []

    def on_epoch_end(self, trainer, epoch, logs):
        """Save checkpoint if criteria met."""
        if self.monitor not in logs:
            return

        current_score = logs[self.monitor]

        # Format filename
        filename = self.filename.format(epoch=epoch, **logs)
        filepath = self.dirpath / (filename + ".pt")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            "logs": logs,
            "config": trainer.config,
        }

        # Check if this is a top-K checkpoint
        should_save = False

        if len(self.best_scores) < self.save_top_k:
            should_save = True
        else:
            if self.mode == "min":
                if current_score < max(self.best_scores):
                    should_save = True
            else:
                if current_score > min(self.best_scores):
                    should_save = True

        if should_save:
            torch.save(checkpoint, filepath)

            if self.verbose:
                print(f"\nSaved checkpoint: {filepath}")

            # Track best checkpoints
            self.best_scores.append(current_score)
            self.best_paths.append(filepath)

            # Remove worst checkpoint if exceed save_top_k
            if len(self.best_scores) > self.save_top_k:
                if self.mode == "min":
                    worst_idx = np.argmax(self.best_scores)
                else:
                    worst_idx = np.argmin(self.best_scores)

                # Remove worst checkpoint file
                worst_path = self.best_paths.pop(worst_idx)
                if worst_path.exists():
                    worst_path.unlink()

                self.best_scores.pop(worst_idx)

        # Save last checkpoint
        if self.save_last:
            last_path = self.dirpath / "last.pt"
            torch.save(checkpoint, last_path)


class MLflowLoggingCallback(Callback):
    """MLflow experiment tracking callback."""

    def __init__(
        self,
        experiment_name: str = "cancer_histopathology",
        tracking_uri: str = "experiments/mlruns",
        tags: Optional[Dict[str, str]] = None,
        log_model: bool = True,
    ):
        """Initialize MLflow callback.

        Args:
            experiment_name: MLflow experiment name
            tracking_uri: MLflow tracking URI
            tags: Optional tags for the run
            log_model: Whether to log model artifacts
        """
        super().__init__()

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.tags = tags or {}
        self.log_model = log_model

        # Set up MLflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        self.run = None

    def on_train_begin(self, trainer):
        """Start MLflow run."""
        self.run = mlflow.start_run(tags=self.tags)

        # Log parameters
        if hasattr(trainer, "config"):
            config_dict = trainer.config if isinstance(trainer.config, dict) else trainer.config.to_dict()

            # Flatten config for logging
            flat_config = self._flatten_dict(config_dict)
            mlflow.log_params(flat_config)

    def on_epoch_end(self, trainer, epoch, logs):
        """Log metrics at end of epoch."""
        if self.run:
            mlflow.log_metrics(logs, step=epoch)

    def on_train_end(self, trainer):
        """End MLflow run."""
        if self.run:
            # Log final model
            if self.log_model:
                # Save model architecture info
                model_info = {
                    "model_type": type(trainer.model).__name__,
                    "num_parameters": sum(p.numel() for p in trainer.model.parameters()),
                }

                mlflow.log_dict(model_info, "model_info.json")

            mlflow.end_run()

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                # Convert to string to avoid MLflow type errors
                items.append((new_key, str(v)))

        return dict(items)


class ProgressCallback(Callback):
    """Progress bar callback."""

    def __init__(self, verbose: bool = True):
        """Initialize progress callback.

        Args:
            verbose: Print progress
        """
        super().__init__()
        self.verbose = verbose

    def on_epoch_end(self, trainer, epoch, logs):
        """Print epoch progress."""
        if not self.verbose:
            return

        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
        print(f"Epoch {epoch + 1}/{trainer.max_epochs} - {metrics_str}")
