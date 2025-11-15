"""Batch inference utilities."""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cancer_quant_model.api.inference_api import InferenceAPI
from cancer_quant_model.utils.logging_utils import get_logger

logger = get_logger(__name__)


def batch_inference_from_csv(
    api: InferenceAPI,
    input_csv: Union[str, Path],
    output_csv: Union[str, Path],
    image_column: str = "image_path",
    batch_size: int = 32,
    save_features: bool = False,
    features_output: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run batch inference from a CSV file.

    Args:
        api: InferenceAPI instance
        input_csv: Path to CSV with image paths
        output_csv: Path to save predictions
        image_column: Column name containing image paths
        batch_size: Batch size for inference
        save_features: Save deep features
        features_output: Path to save features (if save_features=True)

    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    if image_column not in df.columns:
        raise ValueError(f"Column '{image_column}' not found in CSV")

    image_paths = df[image_column].tolist()
    logger.info(f"Found {len(image_paths)} images to process")

    # Run inference
    results = []
    all_features = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch inference"):
        batch_paths = image_paths[i : i + batch_size]

        for img_path in batch_paths:
            try:
                result = api.predict(
                    img_path,
                    return_features=save_features,
                    return_gradcam=False,
                )

                pred_data = {
                    "image_path": img_path,
                    "predicted_class": result["predicted_class"],
                    "predicted_label": result["predicted_label"],
                    "confidence": result["confidence"],
                }

                # Add probabilities
                for class_name, prob in result["probabilities"].items():
                    pred_data[f"prob_{class_name}"] = prob

                results.append(pred_data)

                # Save features if requested
                if save_features and "deep_features" in result:
                    all_features.append(result["deep_features"])

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                # Add placeholder result
                results.append(
                    {
                        "image_path": img_path,
                        "predicted_class": -1,
                        "predicted_label": "ERROR",
                        "confidence": 0.0,
                    }
                )

    # Create predictions DataFrame
    pred_df = pd.DataFrame(results)

    # Merge with original DataFrame
    output_df = df.merge(pred_df, left_on=image_column, right_on="image_path", how="left")

    # Save predictions
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    # Save features if requested
    if save_features and all_features:
        if features_output is None:
            features_output = output_path.parent / "features.npy"

        features_array = np.array(all_features)
        np.save(features_output, features_array)
        logger.info(f"Saved features to {features_output}")

    return output_df


def batch_inference_from_folder(
    api: InferenceAPI,
    input_dir: Union[str, Path],
    output_csv: Union[str, Path],
    extensions: List[str] = [".png", ".jpg", ".jpeg", ".tif"],
    batch_size: int = 32,
    save_gradcam: bool = False,
    gradcam_output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run batch inference on all images in a folder.

    Args:
        api: InferenceAPI instance
        input_dir: Directory containing images
        output_csv: Path to save predictions
        extensions: Image file extensions to process
        batch_size: Batch size for inference
        save_gradcam: Generate and save Grad-CAM
        gradcam_output_dir: Directory to save Grad-CAM images

    Returns:
        DataFrame with predictions
    """
    input_path = Path(input_dir)
    logger.info(f"Scanning directory: {input_path}")

    # Collect image paths
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.rglob(f"*{ext}"))

    logger.info(f"Found {len(image_paths)} images to process")

    if save_gradcam and gradcam_output_dir is None:
        gradcam_output_dir = Path(output_csv).parent / "gradcam"

    if save_gradcam:
        Path(gradcam_output_dir).mkdir(parents=True, exist_ok=True)

    # Run inference
    results = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch inference"):
        batch_paths = image_paths[i : i + batch_size]

        for img_path in batch_paths:
            try:
                result = api.predict(
                    str(img_path),
                    return_features=False,
                    return_gradcam=save_gradcam,
                )

                pred_data = {
                    "image_path": str(img_path),
                    "image_name": img_path.name,
                    "predicted_class": result["predicted_class"],
                    "predicted_label": result["predicted_label"],
                    "confidence": result["confidence"],
                }

                # Add probabilities
                for class_name, prob in result["probabilities"].items():
                    pred_data[f"prob_{class_name}"] = prob

                results.append(pred_data)

                # Save Grad-CAM if requested
                if save_gradcam and "gradcam" in result and result["gradcam"] is not None:
                    from PIL import Image

                    gradcam_path = (
                        Path(gradcam_output_dir) / f"{img_path.stem}_gradcam.png"
                    )

                    # Overlay Grad-CAM on original image
                    from cancer_quant_model.explainability.grad_cam import (
                        overlay_cam_on_image,
                    )

                    original_image = np.array(Image.open(img_path).convert("RGB"))
                    overlay = overlay_cam_on_image(original_image, result["gradcam"])

                    Image.fromarray(overlay).save(gradcam_path)

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append(
                    {
                        "image_path": str(img_path),
                        "image_name": img_path.name,
                        "predicted_class": -1,
                        "predicted_label": "ERROR",
                        "confidence": 0.0,
                    }
                )

    # Create DataFrame
    pred_df = pd.DataFrame(results)

    # Save predictions
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    return pred_df


def main():
    """CLI for batch inference."""
    import argparse

    from cancer_quant_model.config import Config
    from cancer_quant_model.models.resnet import build_resnet_model

    parser = argparse.ArgumentParser(description="Batch inference")
    parser.add_argument("--input", type=str, required=True, help="Input CSV or directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument(
        "--model-config", type=str, default="config/model_resnet.yaml", help="Model config"
    )
    parser.add_argument(
        "--dataset-config", type=str, default="config/dataset.yaml", help="Dataset config"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--save-gradcam", action="store_true", help="Save Grad-CAM")

    args = parser.parse_args()

    # Load configs
    config_manager = Config()
    model_config = config_manager.load_yaml(args.model_config)
    dataset_config = config_manager.load_yaml(args.dataset_config)

    # Build model
    architecture = model_config["model"].get("architecture", "resnet")
    if architecture == "resnet":
        from cancer_quant_model.models.resnet import build_resnet_model

        model = build_resnet_model(model_config)
    elif architecture == "efficientnet":
        from cancer_quant_model.models.efficientnet import build_efficientnet_model

        model = build_efficientnet_model(model_config)
    elif architecture == "vit":
        from cancer_quant_model.models.vit import build_vit_model

        model = build_vit_model(model_config)

    # Create API
    class_names = dataset_config["dataset"]["labels"]["class_names"]

    api = InferenceAPI.from_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
        class_names=class_names,
        architecture=architecture,
    )

    # Run inference
    input_path = Path(args.input)
    if input_path.is_file():
        # CSV input
        batch_inference_from_csv(
            api=api,
            input_csv=input_path,
            output_csv=args.output,
            batch_size=args.batch_size,
        )
    elif input_path.is_dir():
        # Directory input
        batch_inference_from_folder(
            api=api,
            input_dir=input_path,
            output_csv=args.output,
            batch_size=args.batch_size,
            save_gradcam=args.save_gradcam,
        )
    else:
        raise ValueError(f"Input must be a file or directory: {input_path}")


if __name__ == "__main__":
    main()
