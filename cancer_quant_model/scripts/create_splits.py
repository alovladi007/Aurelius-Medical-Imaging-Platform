"""Script to create train/val/test splits."""

import argparse
from pathlib import Path

from cancer_quant_model.config import Config
from cancer_quant_model.data.dataset import create_split_dataframes
from cancer_quant_model.utils.logging_utils import setup_logger
from cancer_quant_model.utils.seed_utils import set_seed

logger = setup_logger("create_splits")


def create_splits(config_path: str):
    """
    Create train/val/test splits.

    Args:
        config_path: Path to dataset config file
    """
    # Load config
    config_manager = Config()
    config = config_manager.load_yaml(config_path)
    dataset_config = config.get("dataset", {})

    # Set seed
    seed = dataset_config.get("split", {}).get("seed", 42)
    set_seed(seed)

    logger.info("Creating train/val/test splits...")

    # Get paths
    dataset_type = dataset_config.get("type", "folder_binary")
    splits_dir = Path(dataset_config["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    if dataset_type == "folder_binary":
        # Check if processed or raw data
        processed_dir = Path(dataset_config["paths"]["processed_data_dir"])
        raw_dir = Path(dataset_config["paths"]["raw_data_dir"])

        # Use processed if available, otherwise raw
        if any(processed_dir.iterdir()):
            data_dir = processed_dir
            logger.info(f"Using processed data from: {data_dir}")
        else:
            data_dir = raw_dir / "train"  # Assume folder_binary has train folder
            logger.info(f"Using raw data from: {data_dir}")

        # Create splits
        train_df, val_df, test_df = create_split_dataframes(
            data_dir=data_dir,
            dataset_type=dataset_type,
            train_ratio=dataset_config["split"]["train_ratio"],
            val_ratio=dataset_config["split"]["val_ratio"],
            test_ratio=dataset_config["split"]["test_ratio"],
            stratify=dataset_config["split"].get("stratify", True),
            seed=seed,
        )

    elif dataset_type == "csv_labels":
        csv_path = Path(dataset_config["paths"]["labels_csv"])
        images_folder = Path(dataset_config["paths"]["images_folder"])

        # Create splits
        train_df, val_df, test_df = create_split_dataframes(
            data_dir=images_folder,
            dataset_type=dataset_type,
            csv_path=csv_path,
            csv_image_col=dataset_config["paths"]["csv_image_col"],
            csv_label_col=dataset_config["paths"]["csv_label_col"],
            train_ratio=dataset_config["split"]["train_ratio"],
            val_ratio=dataset_config["split"]["val_ratio"],
            test_ratio=dataset_config["split"]["test_ratio"],
            stratify=dataset_config["split"].get("stratify", True),
            seed=seed,
        )

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Save splits
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    logger.info(f"Saved splits to {splits_dir}")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Log class distribution
    logger.info("\nClass distribution:")
    logger.info(f"Train:\n{train_df['label'].value_counts()}")
    logger.info(f"Val:\n{val_df['label'].value_counts()}")
    logger.info(f"Test:\n{test_df['label'].value_counts()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument(
        "--config",
        type=str,
        default="dataset.yaml",
        help="Path to dataset config file",
    )

    args = parser.parse_args()

    create_splits(args.config)


if __name__ == "__main__":
    main()
