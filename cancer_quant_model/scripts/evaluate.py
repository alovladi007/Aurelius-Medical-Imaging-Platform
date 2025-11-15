#!/usr/bin/env python
"""Evaluation script for trained models."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from cancer_quant_model.config import Config
from cancer_quant_model.data.datamodule import HistopathDataModule
from cancer_quant_model.models.resnet import build_resnet_model
from cancer_quant_model.models.efficientnet import build_efficientnet_model
from cancer_quant_model.models.vit import build_vit_model
from cancer_quant_model.training.eval_loop import Evaluator
from cancer_quant_model.utils.logging_utils import setup_logger
from cancer_quant_model.utils.viz_utils import plot_confusion_matrix, plot_roc_curve

logger = setup_logger("evaluate")


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


def evaluate(
    checkpoint_path: str,
    dataset_config_path: str = "dataset.yaml",
    model_config_path: str = "model_resnet.yaml",
    output_dir: str = "experiments/eval_results",
):
    """
    Evaluate trained model.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset_config_path: Path to dataset config
        model_config_path: Path to model config
        output_dir: Output directory for results
    """
    # Load configs
    config_manager = Config()
    dataset_config = config_manager.load_yaml(dataset_config_path)
    model_config = config_manager.load_yaml(model_config_path)

    logger.info("Evaluating model...")

    # Load test data
    splits_dir = Path(dataset_config["dataset"]["paths"]["splits_dir"])
    test_df = pd.read_csv(splits_dir / "test.csv")

    logger.info(f"Test samples: {len(test_df)}")

    # Create data module
    datamodule = HistopathDataModule(
        train_df=pd.DataFrame(),  # Not needed for evaluation
        test_df=test_df,
        image_size=tuple(dataset_config["dataset"]["image"]["target_size"]),
        batch_size=dataset_config["dataset"]["dataloader"]["batch_size"],
        num_workers=dataset_config["dataset"]["dataloader"]["num_workers"],
        normalize_mean=dataset_config["dataset"]["preprocessing"]["normalize_mean"],
        normalize_std=dataset_config["dataset"]["preprocessing"]["normalize_std"],
    )

    # Build model
    architecture = model_config["model"].get("architecture", "resnet")
    model = build_model(model_config["model"], architecture)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create evaluator
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(
        model=model,
        test_loader=datamodule.test_dataloader(),
        device=device,
        save_predictions=True,
        output_dir=output_path,
    )

    # Evaluate
    class_names = dataset_config["dataset"]["labels"]["class_names"]
    results = evaluator.evaluate(class_names=class_names)

    # Error analysis
    error_analysis = evaluator.analyze_errors(results, class_names=class_names)

    # Save visualizations
    logger.info("Creating visualizations...")

    # Confusion matrix
    cm_path = output_path / "confusion_matrix.png"
    plot_confusion_matrix(
        results["confusion_matrix_normalized"],
        class_names=class_names,
        save_path=cm_path,
    )
    logger.info(f"Saved confusion matrix: {cm_path}")

    # ROC curve (for binary classification)
    if len(class_names) == 2:
        roc_path = output_path / "roc_curve.png"
        plot_roc_curve(
            results["labels"],
            results["probabilities"][:, 1],
            save_path=roc_path,
        )
        logger.info(f"Saved ROC curve: {roc_path}")

    logger.info(f"Evaluation completed. Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
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
        help="Model config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/eval_results",
        help="Output directory",
    )

    args = parser.parse_args()

    evaluate(
        args.checkpoint,
        args.dataset_config,
        args.model_config,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
