#!/usr/bin/env python
"""Extract quantitative features from images."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from cancer_quant_model.utils.feature_utils import QuantitativeFeatureExtractor
from cancer_quant_model.utils.logging_utils import setup_logger

logger = setup_logger("extract_features")


def extract_features(
    input_dir: str,
    output_path: str,
    extensions: list = [".png", ".jpg", ".jpeg"],
):
    """
    Extract quantitative features from all images in a directory.

    Args:
        input_dir: Input directory containing images
        output_path: Output CSV path for features
        extensions: Image file extensions to process
    """
    input_path = Path(input_dir)
    logger.info(f"Extracting features from: {input_path}")

    # Collect image paths
    image_paths = []
    for ext in extensions:
        image_paths.extend(input_path.rglob(f"*{ext}"))

    logger.info(f"Found {len(image_paths)} images")

    # Initialize feature extractor
    extractor = QuantitativeFeatureExtractor()

    # Extract features
    all_features = []

    for image_path in tqdm(image_paths, desc="Extracting features"):
        try:
            # Load image
            image = np.array(Image.open(image_path).convert("RGB"))

            # Extract features
            features = extractor.extract_all_features(image)

            # Add metadata
            features["image_path"] = str(image_path)
            features["image_name"] = image_path.name

            all_features.append(features)

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.suffix == ".csv":
        df.to_csv(output_file, index=False)
    elif output_file.suffix == ".parquet":
        df.to_parquet(output_file, index=False)
    else:
        # Default to parquet
        output_file = output_file.with_suffix(".parquet")
        df.to_parquet(output_file, index=False)

    logger.info(f"Saved features to: {output_file}")
    logger.info(f"Feature dimensions: {df.shape}")
    logger.info(f"Feature columns: {len(df.columns) - 2}")  # Excluding metadata columns


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Extract quantitative features")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/features/features.parquet",
        help="Output file path (CSV or Parquet)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Image file extensions",
    )

    args = parser.parse_args()

    extract_features(args.input_dir, args.output, args.extensions)


if __name__ == "__main__":
    main()
