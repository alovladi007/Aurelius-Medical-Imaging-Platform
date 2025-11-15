"""Script to create train/val/test splits from dataset."""

import argparse
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from cancer_quant_model.utils.logging_utils import setup_logging
from cancer_quant_model.utils.seed_utils import seed_everything

logger = setup_logging(log_dir="experiments/logs", log_file="create_splits.log")


def create_splits(config_path: str):
    """Create train/val/test splits.

    Args:
        config_path: Path to dataset configuration
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")

    # Set seed
    seed = config.get("split", {}).get("seed", 42)
    seed_everything(seed)

    dataset_type = config.get("dataset_type", "folder_binary")
    splits_root = Path(config.get("splits_root", "data/splits"))
    splits_root.mkdir(parents=True, exist_ok=True)

    if dataset_type == "folder_binary":
        create_splits_from_folders(config, splits_root)
    elif dataset_type == "csv_labels":
        create_splits_from_csv(config, splits_root)
    else:
        logger.error(f"Unknown dataset type: {dataset_type}")
        return

    logger.info("Split creation complete!")


def create_splits_from_folders(config: dict, splits_root: Path):
    """Create splits from folder structure.

    Args:
        config: Configuration dictionary
        splits_root: Directory to save splits
    """
    logger.info("Creating splits from folder structure...")

    processed_root = Path(config.get("processed_root", "data/processed"))

    if not processed_root.exists():
        processed_root = Path(config.get("data_root", "data/raw"))

    # Find all class directories
    class_dirs = [d for d in processed_root.iterdir() if d.is_dir()]

    if not class_dirs:
        logger.error(f"No class directories found in {processed_root}")
        return

    # Collect all images
    all_data = []

    extensions = tuple(config.get("image_settings", {}).get("extensions", [".png", ".jpg", ".jpeg"]))

    for class_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name

        # Find all images
        images = []
        for ext in extensions:
            images.extend(list(class_dir.glob(f"*{ext}")))

        logger.info(f"Class {class_name}: {len(images)} images")

        for img_path in images:
            all_data.append(
                {
                    "image_path": str(img_path.relative_to(processed_root.parent)),
                    "label": class_idx,
                    "class_name": class_name,
                }
            )

    # Create dataframe
    df = pd.DataFrame(all_data)
    logger.info(f"Total images: {len(df)}")

    # Split configuration
    split_config = config.get("split", {})
    train_ratio = split_config.get("train_ratio", 0.7)
    val_ratio = split_config.get("val_ratio", 0.15)
    test_ratio = split_config.get("test_ratio", 0.15)
    stratify = split_config.get("stratify", True)

    # Create splits
    stratify_col = df["label"] if stratify else None

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=stratify_col,
        random_state=split_config.get("seed", 42),
    )

    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df["label"] if stratify else None,
        random_state=split_config.get("seed", 42),
    )

    # Save splits
    train_df.to_csv(splits_root / "train.csv", index=False)
    val_df.to_csv(splits_root / "val.csv", index=False)
    test_df.to_csv(splits_root / "test.csv", index=False)

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"Splits saved to {splits_root}")


def create_splits_from_csv(config: dict, splits_root: Path):
    """Create splits from CSV labels.

    Args:
        config: Configuration dictionary
        splits_root: Directory to save splits
    """
    logger.info("Creating splits from CSV labels...")

    csv_structure = config.get("csv_structure", {})
    csv_path = csv_structure.get("csv_path", "data/raw/labels.csv")

    # Load CSV
    df = pd.read_csv(csv_path)

    label_col = csv_structure.get("label_column", "label")
    image_id_col = csv_structure.get("image_id_column", "image_id")

    logger.info(f"Loaded {len(df)} entries from {csv_path}")

    # Rename columns for consistency
    df = df.rename(columns={image_id_col: "image_path", label_col: "label"})

    # Split configuration
    split_config = config.get("split", {})
    train_ratio = split_config.get("train_ratio", 0.7)
    val_ratio = split_config.get("val_ratio", 0.15)
    test_ratio = split_config.get("test_ratio", 0.15)
    stratify = split_config.get("stratify", True)

    # Create splits
    stratify_col = df["label"] if stratify else None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=stratify_col,
        random_state=split_config.get("seed", 42),
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df["label"] if stratify else None,
        random_state=split_config.get("seed", 42),
    )

    # Save splits
    train_df.to_csv(splits_root / "train.csv", index=False)
    val_df.to_csv(splits_root / "val.csv", index=False)
    test_df.to_csv(splits_root / "test.csv", index=False)

    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    logger.info(f"Splits saved to {splits_root}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument(
        "--config",
        type=str,
        default="config/dataset.yaml",
        help="Path to dataset configuration file",
    )

    args = parser.parse_args()

    create_splits(args.config)


if __name__ == "__main__":
    main()
