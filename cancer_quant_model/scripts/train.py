"""Training script for cancer classification models."""
import argparse
from pathlib import Path

import torch
import yaml

from cancer_quant_model.config import load_config, merge_configs
from cancer_quant_model.data import HistopathDataModule
from cancer_quant_model.models import create_model
from cancer_quant_model.training import (Trainer, EarlyStoppingCallback,
                                          CheckpointCallback, MLflowLoggingCallback, 
                                          ProgressCallback)
from cancer_quant_model.utils.logging_utils import setup_logging
from cancer_quant_model.utils.seed_utils import seed_everything


def train_model(model_config, dataset_config, train_config):
    """Train cancer classification model."""
    # Setup logging
    logger = setup_logging(log_dir="experiments/logs", log_file="train.log")
    logger.info("Starting training...")

    # Set random seed
    seed = train_config.get("seed", 42)
    seed_everything(seed, deterministic=train_config.get("training", {}).get("deterministic", False))

    # Setup data module
    splits_root = dataset_config.get("splits_root", "data/splits")
    data_module = HistopathDataModule(
        config=dataset_config,
        train_csv=f"{splits_root}/train.csv",
        val_csv=f"{splits_root}/val.csv",
        batch_size=model_config.get("training", {}).get("batch_size", 32),
        num_workers=model_config.get("training", {}).get("num_workers", 4),
    )
    data_module.setup()

    # Create model
    model = create_model(model_config)
    logger.info(f"Created model: {model.__class__.__name__}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Loss function
    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=model_config.get("loss", {}).get("label_smoothing", 0.0)
    )

    # Optimizer
    opt_config = model_config.get("optimizer", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config.get("lr", 1e-4),
        weight_decay=opt_config.get("weight_decay", 1e-4),
    )

    # Scheduler
    sched_config = model_config.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=sched_config.get("T_max", 50),
        eta_min=sched_config.get("eta_min", 1e-6),
    )

    # Callbacks
    callbacks = [
        EarlyStoppingCallback(**model_config.get("early_stopping", {})),
        CheckpointCallback(**model_config.get("checkpoint", {})),
        MLflowLoggingCallback(**model_config.get("mlflow", {})),
        ProgressCallback(),
    ]

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=data_module.train_dataloader(),
        val_loader=data_module.val_dataloader(),
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=model_config.get("training", {}).get("num_epochs", 50),
        scheduler=scheduler,
        use_amp=model_config.get("training", {}).get("use_amp", True),
        gradient_clip_val=model_config.get("training", {}).get("gradient_clip_val", 1.0),
        callbacks=callbacks,
        config=model_config,
    )

    # Train
    trainer.fit()

    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(description="Train cancer classification model")
    parser.add_argument("--model_config", type=str, required=True, help="Path to model config")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config")
    parser.add_argument("--train_config", type=str, default="config/train_default.yaml", help="Path to training config")
    args = parser.parse_args()

    # Load configs
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)

    # Merge configs
    model_config.update(train_config.get("training", {}))

    train_model(model_config, dataset_config, train_config)


if __name__ == "__main__":
    main()
