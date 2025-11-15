"""Training loop for cancer classification models."""

import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging_utils import get_logger
from ..utils.metrics_utils import compute_all_metrics
from .callbacks import Callback

logger = get_logger()


class Trainer:
    """Trainer class for model training."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        max_epochs: int = 50,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,
        callbacks: Optional[List[Callback]] = None,
        config: Optional[dict] = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            max_epochs: Maximum number of epochs
            scheduler: Learning rate scheduler
            use_amp: Use mixed precision training
            gradient_clip_val: Gradient clipping value
            callbacks: List of callbacks
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.max_epochs = max_epochs
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()
        self.gradient_clip_val = gradient_clip_val
        self.callbacks = callbacks or []
        self.config = config

        # Initialize scaler for mixed precision
        self.scaler = GradScaler() if self.use_amp else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_history = []
        self.val_history = []

    def fit(self):
        """Run full training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")

        # Callback: on_train_begin
        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Callback: on_epoch_begin
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)

            # Train epoch
            train_metrics = self.train_epoch()

            # Validation epoch
            val_metrics = {}
            if self.val_loader:
                val_metrics = self.validate()

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Store history
            self.train_history.append(train_metrics)
            if val_metrics:
                self.val_history.append(val_metrics)

            # Callback: on_epoch_end
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, epoch_metrics)

            # Check early stopping
            early_stop = any(
                getattr(cb, "early_stop", False) for cb in self.callbacks
            )
            if early_stop:
                logger.info("Early stopping triggered")
                break

            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau needs a metric
                    self.scheduler.step(epoch_metrics.get("val_loss", train_metrics["train_loss"]))
                else:
                    self.scheduler.step()

        # Callback: on_train_end
        for callback in self.callbacks:
            callback.on_train_end(self)

        logger.info("Training completed")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.current_epoch + 1}")

        for batch_idx, (images, labels, _) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Callback: on_batch_begin
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch_idx)

            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Callback: on_batch_end
            batch_logs = {"batch_loss": loss.item()}
            for callback in self.callbacks:
                callback.on_batch_end(self, batch_idx, batch_logs)

            self.global_step += 1

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)

        import numpy as np

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = compute_all_metrics(
            all_labels,
            all_preds,
            all_probs,
            num_classes=all_probs.shape[1],
            compute_per_class=False,
        )

        return {
            "train_loss": avg_loss,
            "train_accuracy": metrics["accuracy"],
            "train_f1": metrics["f1_score"],
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.val_loader, desc=f"Val Epoch {self.current_epoch + 1}")

        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            # Accumulate metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)

        import numpy as np

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        metrics = compute_all_metrics(
            all_labels,
            all_preds,
            all_probs,
            num_classes=all_probs.shape[1],
            compute_per_class=False,
        )

        return {
            "val_loss": avg_loss,
            "val_accuracy": metrics["accuracy"],
            "val_f1": metrics["f1_score"],
            "val_auc": metrics.get("auc_roc", 0.0),
        }

    def save_checkpoint(self, filepath: str):
        """Save checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "train_history": self.train_history,
            "val_history": self.val_history,
            "config": self.config,
        }

        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load checkpoint.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.train_history = checkpoint.get("train_history", [])
        self.val_history = checkpoint.get("val_history", [])

        logger.info(f"Loaded checkpoint from {filepath}")
