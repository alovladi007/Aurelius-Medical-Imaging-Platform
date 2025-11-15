"""Script to extract quantitative features from images."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from cancer_quant_model.data import HistopathDataModule
from cancer_quant_model.models import create_model
from cancer_quant_model.utils.feature_utils import extract_all_features
from cancer_quant_model.utils.logging_utils import setup_logging


def extract_features(checkpoint_path, split, model_config, dataset_config, output_path):
    """Extract quantitative features."""
    logger = setup_logging()
    logger.info(f"Extracting features for {split} split...")

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

    # Setup data module
    splits_root = dataset_config.get("splits_root", "data/splits")
    csv_path = f"{splits_root}/{split}.csv"

    # Read split CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Processing {len(df)} images from {split} split")

    all_features = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        try:
            # Load image
            import cv2
            image_path = row["image_path"]
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Failed to load {image_path}")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Extract features
            features = extract_all_features(image, model=model, device=device)

            # Add metadata
            features["image_path"] = image_path
            features["label"] = row.get("label", -1)
            if "class_name" in row:
                features["class_name"] = row["class_name"]

            all_features.append(features)

        except Exception as e:
            logger.error(f"Error processing {row.get('image_path', 'unknown')}: {e}")

    # Create dataframe
    features_df = pd.DataFrame(all_features)

    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    logger.info(f"Features saved to {output_path}")
    logger.info(f"Feature shape: {features_df.shape}")

    return features_df


def main():
    parser = argparse.ArgumentParser(description="Extract quantitative features")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to process")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--config", type=str, help="Path to model config (optional)")
    parser.add_argument("--dataset_config", type=str, required=True, help="Path to dataset config")
    args = parser.parse_args()

    model_config = None
    if args.config:
        with open(args.config) as f:
            model_config = yaml.safe_load(f)

    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)

    extract_features(args.checkpoint, args.split, model_config, dataset_config, args.output)


if __name__ == "__main__":
    main()
