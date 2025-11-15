"""Inference script for single image with Grad-CAM visualization."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from cancer_quant_model.data.transforms import get_inference_transforms
from cancer_quant_model.explainability import get_gradcam_for_model, save_gradcam_visualization
from cancer_quant_model.models import create_model
from cancer_quant_model.utils.logging_utils import setup_logging


def infer_image(image_path, checkpoint_path, model_config, output_dir):
    """Run inference on single image."""
    logger = setup_logging()
    logger.info(f"Running inference on {image_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" in checkpoint and not model_config:
        model_config = checkpoint["config"]

    # Create model
    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # Load and transform image
    img_size = model_config.get("image_settings", {}).get("target_size", 224)
    transform = get_inference_transforms(img_size=img_size)

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()

    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    # Get class names
    class_names = model_config.get("classes", {}).get("class_names", None)
    if class_names:
        pred_class_name = class_names[pred_class]
    else:
        pred_class_name = f"Class {pred_class}"

    logger.info(f"Prediction: {pred_class_name} (confidence: {confidence:.4f})")
    logger.info(f"Probabilities: {probs[0].cpu().numpy()}")

    # Generate Grad-CAM
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_type = "resnet" if "resnet" in model_config.get("model", {}).get("backbone", "") else "efficientnet"

        try:
            grad_cam = get_gradcam_for_model(model, model_type=model_type)
            cam = grad_cam(image_tensor, target_class=pred_class)

            # Save Grad-CAM
            save_gradcam_visualization(
                original_image,
                cam,
                save_path=str(output_dir / "gradcam_overlay.png"),
                alpha=0.4,
            )
            logger.info(f"Grad-CAM saved to {output_dir / 'gradcam_overlay.png'}")

        except Exception as e:
            logger.warning(f"Could not generate Grad-CAM: {e}")

        # Save original image
        cv2.imwrite(str(output_dir / "original.png"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))

    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser(description="Infer single image")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, help="Path to model config (optional)")
    parser.add_argument("--output_dir", type=str, default="experiments/inference", help="Output directory")
    args = parser.parse_args()

    model_config = None
    if args.config:
        with open(args.config) as f:
            model_config = yaml.safe_load(f)

    infer_image(args.image_path, args.checkpoint, model_config, args.output_dir)


if __name__ == "__main__":
    main()
