"""Script to prepare histopathology data for training."""

import argparse
from pathlib import Path

import yaml

from cancer_quant_model.config import Config
from cancer_quant_model.utils.logging_utils import setup_logger
from cancer_quant_model.utils.tiling_utils import ImageTiler

logger = setup_logger("prepare_data")


def prepare_data(config_path: str, create_tiles: bool = False):
    """
    Prepare data for training.

    Args:
        config_path: Path to dataset config file
        create_tiles: Whether to create tiles from large images
    """
    # Load config
    config_manager = Config()
    config = config_manager.load_yaml(config_path)
    dataset_config = config.get("dataset", {})

    logger.info("Preparing histopathology data...")
    logger.info(f"Dataset type: {dataset_config.get('type')}")

    # Create output directories
    processed_dir = Path(dataset_config["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create tiles if requested
    if create_tiles or dataset_config.get("image", {}).get("create_patches", False):
        logger.info("Creating tiles from large images...")

        tile_size = dataset_config["image"].get("patch_size", 512)
        overlap = dataset_config["image"].get("patch_overlap", 0)
        min_tissue_ratio = dataset_config["image"].get("min_tissue_ratio", 0.5)

        tiler = ImageTiler(
            tile_size=tile_size,
            overlap=overlap,
            min_tissue_ratio=min_tissue_ratio,
        )

        raw_dir = Path(dataset_config["paths"]["raw_data_dir"])

        # Process all images
        extensions = dataset_config["image"].get("extensions", [".png", ".jpg", ".jpeg"])
        image_paths = []

        for ext in extensions:
            image_paths.extend(raw_dir.rglob(f"*{ext}"))

        logger.info(f"Found {len(image_paths)} images to process")

        for image_path in image_paths:
            try:
                # Determine output directory (preserve folder structure)
                relative_path = image_path.relative_to(raw_dir)
                output_subdir = processed_dir / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)

                # Create tiles
                tiles, coords = tiler.tile_from_file(
                    image_path,
                    output_dir=output_subdir,
                    filter_background=True,
                    save_tiles=True,
                )

                logger.info(f"Processed {image_path.name}: created {len(tiles)} tiles")

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")

        logger.info("Tiling completed")

    else:
        logger.info("No tiling requested. Data is ready in raw directory.")

    logger.info("Data preparation completed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Prepare histopathology data")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset.yaml",
        help="Path to dataset config file",
    )
    parser.add_argument(
        "--create-tiles",
        action="store_true",
        help="Create tiles from large images",
    )

    args = parser.parse_args()

    prepare_data(args.config, args.create_tiles)


if __name__ == "__main__":
    main()
