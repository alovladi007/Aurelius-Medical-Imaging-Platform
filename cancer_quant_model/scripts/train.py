#!/usr/bin/env python
"""Training script for cancer histopathology models."""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from cancer_quant_model.config import Config
from cancer_quant_model.data.datamodule import HistopathDataModule
from cancer_quant_model.models.resnet import build_resnet_model
from cancer_quant_model.models.efficientnet import build_efficientnet_model
from cancer_quant_model.models.vit import build_vit_model
from cancer_quant_model.training.train_loop import Trainer
from cancer_quant_model.utils.logging_utils import setup_logger
from cancer_quant_model.utils.seed_utils import set_seed

logger = setup_logger("train")


def build_model(model_config: dict, architecture: str):
    """Build model based on architecture."""
    if architecture == "resnet":
        return build_resnet_model({"model": model_config})
    elif architecture == "efficientnet":
        return build_efficientnet_model({"model": model_config})
    elif architecture == "vit":
        return build_vit_model({"model": model_config})
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def get_loss_function(loss_config: dict, class_weights=None):
    """Get loss function."""
    loss_type = loss_config.get("type", "cross_entropy")

    if loss_type == "cross_entropy":
        label_smoothing = loss_config.get("label_smoothing", 0.0)
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    elif loss_type == "focal":
        # Simple focal loss implementation
        from torch.nn import functional as F

        class FocalLoss(nn.Module):
            def __init__(self, alpha=None, gamma=2.0):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(inputs, targets, reduction="none")
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss

                if self.alpha is not None:
                    focal_loss = self.alpha[targets] * focal_loss

                return focal_loss.mean()

        alpha = loss_config.get("focal_alpha")
        if alpha:
            alpha = torch.tensor(alpha)
        gamma = loss_config.get("focal_gamma", 2.0)

        return FocalLoss(alpha=alpha, gamma=gamma)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_optimizer(model, optimizer_config: dict):
    """Get optimizer."""
    opt_type = optimizer_config.get("type", "adamw")
    lr = optimizer_config.get("lr", 0.001)
    weight_decay = optimizer_config.get("weight_decay", 0.0001)

    if opt_type == "adam":
        betas = tuple(optimizer_config.get("betas", [0.9, 0.999]))
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    elif opt_type == "adamw":
        betas = tuple(optimizer_config.get("betas", [0.9, 0.999]))
        return torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )

    elif opt_type == "sgd":
        momentum = optimizer_config.get("momentum", 0.9)
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )

    else:
        raise ValueError(f"Unknown optimizer type: {opt_type}")


def get_scheduler(optimizer, scheduler_config: dict, steps_per_epoch: int):
    """Get learning rate scheduler."""
    sched_type = scheduler_config.get("type", "cosine")

    if sched_type == "cosine":
        warmup_epochs = scheduler_config.get("warmup_epochs", 5)
        min_lr = scheduler_config.get("min_lr", 1e-6)

        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=min_lr
        )

    elif sched_type == "step":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif sched_type == "plateau":
        patience = scheduler_config.get("patience", 5)
        factor = scheduler_config.get("factor", 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience, factor=factor
        )

    elif sched_type == "onecycle":
        max_lr = optimizer.param_groups[0]["lr"]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=50
        )

    else:
        return None


def train(
    dataset_config_path: str = "dataset.yaml",
    model_config_path: str = "model_resnet.yaml",
    train_config_path: str = "train_default.yaml",
):
    """
    Train cancer histopathology model.

    Args:
        dataset_config_path: Path to dataset config
        model_config_path: Path to model config
        train_config_path: Path to training config
    """
    # Load configs
    config_manager = Config()
    dataset_config = config_manager.load_yaml(dataset_config_path)
    model_config = config_manager.load_yaml(model_config_path)
    train_config = config_manager.load_yaml(train_config_path)

    # Set seed
    seed = train_config.get("seed", 42)
    set_seed(seed, deterministic=train_config.get("deterministic", True))

    logger.info("Training cancer histopathology model...")

    # Load data splits
    splits_dir = Path(dataset_config["dataset"]["paths"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")

    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Create data module
    datamodule = HistopathDataModule(
        train_df=train_df,
        val_df=val_df,
        image_size=tuple(dataset_config["dataset"]["image"]["target_size"]),
        batch_size=dataset_config["dataset"]["dataloader"]["batch_size"],
        num_workers=dataset_config["dataset"]["dataloader"]["num_workers"],
        augmentation_config=dataset_config["dataset"].get("augmentation"),
        normalize_mean=dataset_config["dataset"]["preprocessing"]["normalize_mean"],
        normalize_std=dataset_config["dataset"]["preprocessing"]["normalize_std"],
        seed=seed,
    )

    logger.info(f"DataModule created: {datamodule.summary()}")

    # Build model
    architecture = model_config["model"].get("architecture", "resnet")
    model = build_model(model_config["model"], architecture)

    logger.info(f"Model: {model_config['model']['name']}")

    # Get device
    device = train_config.get("hardware", {}).get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    # Get class weights if needed
    class_weights = datamodule.get_class_weights()
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # Loss function
    criterion = get_loss_function(model_config["model"]["loss"], class_weights)

    # Optimizer
    optimizer = get_optimizer(model, model_config["model"]["optimizer"])

    # Scheduler
    train_loader = datamodule.train_dataloader()
    scheduler = get_scheduler(
        optimizer, model_config["model"]["scheduler"], len(train_loader)
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=datamodule.val_dataloader(),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        mixed_precision=train_config["training"].get("precision", "16-mixed") == "16-mixed",
        gradient_clip_val=train_config["training"].get("gradient_clip_val", 1.0),
        accumulate_grad_batches=train_config["training"].get("accumulate_grad_batches", 1),
        max_epochs=model_config["model"]["training"].get("max_epochs", 50),
        checkpoint_dir=Path(train_config["paths"]["checkpoint_dir"]),
        mlflow_tracking_uri=train_config["experiment"]["mlflow"]["tracking_uri"],
        experiment_name=train_config["experiment"]["mlflow"]["experiment_name"],
        run_name=train_config["experiment"].get("run_name"),
        log_every_n_steps=train_config["training"].get("log_every_n_steps", 10),
        save_top_k=train_config["training"].get("save_top_k", 3),
        monitor=train_config["training"].get("monitor", "val_auroc"),
        mode=train_config["training"].get("mode", "max"),
        early_stopping_patience=train_config["training"]["early_stopping"].get("patience", 10),
    )

    # Train
    trainer.train()

    logger.info("Training completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train cancer histopathology model")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="dataset.yaml",
        help="Dataset config file",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="model_resnet.yaml",
        help="Model config file (model_resnet.yaml, model_efficientnet.yaml, model_vit.yaml)",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default="train_default.yaml",
        help="Training config file",
    )

    args = parser.parse_args()

    train(args.dataset_config, args.model_config, args.train_config)


if __name__ == "__main__":
    main()
