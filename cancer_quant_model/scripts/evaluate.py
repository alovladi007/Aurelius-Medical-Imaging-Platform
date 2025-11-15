"""Evaluation script for cancer classification models."""
import argparse
from pathlib import Path

import torch
import yaml

from cancer_quant_model.data import HistopathDataModule
from cancer_quant_model.models import create_model
from cancer_quant_model.training import Evaluator
from cancer_quant_model.utils.logging_utils import setup_logging


def evaluate_model(checkpoint_path, model_config, dataset_config, output_dir):
    """Evaluate cancer classification model."""
    logger = setup_logging(log_dir=output_dir, log_file="evaluate.log")
    logger.info("Starting evaluation...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" in checkpoint and not model_config:
        model_config = checkpoint["config"]

    # Setup data module
    splits_root = dataset_config.get("splits_root", "data/splits")
    data_module = HistopathDataModule(
        config=dataset_config,
        test_csv=f"{splits_root}/test.csv",
        batch_size=32,
        num_workers=4,
    )
    data_module.setup()

    # Create model
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Get class names
    class_names = dataset_config.get("classes", {}).get("class_names", None)

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=data_module.test_dataloader(),
        device=device,
        num_classes=model_config.get("model", {}).get("num_classes", 2),
        class_names=class_names,
    )

    # Evaluate
    metrics = evaluator.evaluate(
        save_dir=output_dir,
        save_predictions=True,
        create_visualizations=True,
    )

    logger.info("Evaluation completed!")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate cancer classification model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, help="Path to model config (optional if in checkpoint)")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config")
    parser.add_argument("--output_dir", type=str, default="experiments/logs/evaluation", help="Output directory")
    args = parser.parse_args()

    # Load configs
    model_config = None
    if args.config:
        with open(args.config) as f:
            model_config = yaml.safe_load(f)

    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)

    evaluate_model(args.checkpoint, model_config, dataset_config, args.output_dir)


if __name__ == "__main__":
    main()
