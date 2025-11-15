#!/usr/bin/env python
"""Inference script for single image."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cancer_quant_model.config import Config
from cancer_quant_model.data.transforms import get_val_transforms
from cancer_quant_model.explainability.grad_cam import GradCAM, overlay_cam_on_image, get_target_layer
from cancer_quant_model.models.resnet import build_resnet_model
from cancer_quant_model.models.efficientnet import build_efficientnet_model
from cancer_quant_model.models.vit import build_vit_model
from cancer_quant_model.utils.logging_utils import setup_logger

logger = setup_logger("infer")


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


def infer(
    image_path: str,
    checkpoint_path: str,
    model_config_path: str = "model_resnet.yaml",
    dataset_config_path: str = "dataset.yaml",
    save_gradcam: bool = True,
    output_dir: str = "experiments/inference",
):
    """
    Run inference on a single image.

    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        model_config_path: Path to model config
        dataset_config_path: Path to dataset config
        save_gradcam: Save Grad-CAM visualization
        output_dir: Output directory
    """
    # Load configs
    config_manager = Config()
    model_config = config_manager.load_yaml(model_config_path)
    dataset_config = config_manager.load_yaml(dataset_config_path)

    logger.info(f"Running inference on: {image_path}")

    # Build model
    architecture = model_config["model"].get("architecture", "resnet")
    model = build_model(model_config["model"], architecture)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    image = np.array(Image.open(image_path).convert("RGB"))

    transform = get_val_transforms(
        image_size=tuple(dataset_config["dataset"]["image"]["target_size"]),
        normalize_mean=dataset_config["dataset"]["preprocessing"]["normalize_mean"],
        normalize_std=dataset_config["dataset"]["preprocessing"]["normalize_std"],
    )

    transformed = transform(image=image)
    input_tensor = transformed["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output

        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()

    # Get class names
    class_names = dataset_config["dataset"]["labels"]["class_names"]

    logger.info(f"Prediction: {class_names[pred_class]}")
    logger.info(f"Probabilities: {probs[0].cpu().numpy()}")

    for i, class_name in enumerate(class_names):
        logger.info(f"  {class_name}: {probs[0, i].item():.4f}")

    # Generate Grad-CAM if requested
    if save_gradcam:
        logger.info("Generating Grad-CAM...")

        try:
            target_layers = get_target_layer(model, architecture)
            gradcam = GradCAM(model, target_layers, device=device)

            cam = gradcam(input_tensor, target_class=pred_class)

            # Overlay on original image
            overlay = overlay_cam_on_image(image, cam)

            # Save
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            gradcam_path = output_path / f"{Path(image_path).stem}_gradcam.png"
            Image.fromarray(overlay).save(gradcam_path)

            logger.info(f"Saved Grad-CAM: {gradcam_path}")

        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inference on single image")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="model_resnet.yaml",
        help="Model config file",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="dataset.yaml",
        help="Dataset config file",
    )
    parser.add_argument(
        "--save-gradcam",
        action="store_true",
        default=True,
        help="Save Grad-CAM visualization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/inference",
        help="Output directory",
    )

    args = parser.parse_args()

    infer(
        args.image,
        args.checkpoint,
        args.model_config,
        args.dataset_config,
        args.save_gradcam,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
