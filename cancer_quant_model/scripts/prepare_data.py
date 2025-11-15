"""Script to prepare and preprocess cancer histopathology data."""

import argparse
import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from cancer_quant_model.utils.logging_utils import setup_logging
from cancer_quant_model.utils.tiling_utils import create_tiles, save_tiles

logger = setup_logging(log_dir="experiments/logs", log_file="prepare_data.log")


def prepare_data(config_path: str):
    """Prepare dataset according to configuration.

    Args:
        config_path: Path to dataset configuration file
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    data_root = Path(config.get("data_root", "data/raw"))
    processed_root = Path(config.get("processed_root", "data/processed"))
    processed_root.mkdir(parents=True, exist_ok=True)

    img_settings = config.get("image_settings", {})
    create_tiles_flag = img_settings.get("create_tiles", False)
    tile_size = img_settings.get("tile_size", 512)
    tile_overlap = img_settings.get("tile_overlap", 0)
    extensions = tuple(img_settings.get("extensions", [".png", ".jpg", ".jpeg"]))

    dataset_type = config.get("dataset_type", "folder_binary")

    if dataset_type == "folder_binary":
        prepare_folder_dataset(
            data_root,
            processed_root,
            create_tiles_flag,
            tile_size,
            tile_overlap,
            extensions,
        )
    elif dataset_type == "csv_labels":
        prepare_csv_dataset(
            data_root,
            processed_root,
            config,
            create_tiles_flag,
            tile_size,
            tile_overlap,
        )
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
        return

    logger.info("Data preparation complete!")


def prepare_folder_dataset(
    data_root: Path,
    processed_root: Path,
    create_tiles: bool,
    tile_size: int,
    tile_overlap: int,
    extensions: tuple,
):
    """Prepare folder-based dataset.

    Args:
        data_root: Root directory with raw data
        processed_root: Directory for processed data
        create_tiles: Whether to create tiles from large images
        tile_size: Tile size
        tile_overlap: Tile overlap
        extensions: Allowed file extensions
    """
    logger.info("Preparing folder-based dataset...")

    # Find all class directories
    class_dirs = [d for d in data_root.iterdir() if d.is_dir()]

    if not class_dirs:
        logger.warning(f"No subdirectories found in {data_root}")
        return

    total_images = 0

    for class_dir in class_dirs:
        class_name = class_dir.name
        output_class_dir = processed_root / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing class: {class_name}")

        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(class_dir.glob(f"*{ext}"))

        logger.info(f"Found {len(image_files)} images in {class_name}")

        for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                if create_tiles:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        logger.warning(f"Failed to load {img_path}")
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Create tiles
                    tiles = create_tiles(img, tile_size=tile_size, overlap=tile_overlap)

                    if len(tiles) > 0:
                        save_tiles(
                            tiles,
                            output_class_dir,
                            base_name=img_path.stem,
                            format="png",
                        )
                        total_images += len(tiles)
                else:
                    # Copy image
                    output_path = output_class_dir / img_path.name
                    shutil.copy(img_path, output_path)
                    total_images += 1

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

    logger.info(f"Processed {total_images} images total")


def prepare_csv_dataset(
    data_root: Path,
    processed_root: Path,
    config: dict,
    create_tiles: bool,
    tile_size: int,
    tile_overlap: int,
):
    """Prepare CSV-based dataset.

    Args:
        data_root: Root directory
        processed_root: Processed data directory
        config: Configuration dictionary
        create_tiles: Whether to create tiles
        tile_size: Tile size
        tile_overlap: Tile overlap
    """
    import pandas as pd

    logger.info("Preparing CSV-based dataset...")

    csv_structure = config.get("csv_structure", {})
    csv_path = csv_structure.get("csv_path", "data/raw/labels.csv")
    image_dir = Path(csv_structure.get("image_dir", "data/raw/images"))

    # Read CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded CSV with {len(df)} entries")

    # Process images
    processed_root.mkdir(parents=True, exist_ok=True)

    total_images = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        img_id = row[csv_structure.get("image_id_column", "image_id")]
        img_path = image_dir / img_id

        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue

        try:
            if create_tiles:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                tiles = create_tiles(img, tile_size=tile_size, overlap=tile_overlap)

                if len(tiles) > 0:
                    save_tiles(
                        tiles,
                        processed_root,
                        base_name=img_path.stem,
                        format="png",
                    )
                    total_images += len(tiles)
            else:
                output_path = processed_root / img_path.name
                shutil.copy(img_path, output_path)
                total_images += 1

        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    logger.info(f"Processed {total_images} images total")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare cancer histopathology data")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset.yaml",
        help="Path to dataset configuration file",
    )

    args = parser.parse_args()

    prepare_data(args.config)


if __name__ == "__main__":
    main()
