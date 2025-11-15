"""
Dataset Preparation Script for Advanced Cancer Detection AI
Helps prepare and organize medical imaging datasets
"""

import argparse
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Helper class for preparing cancer detection datasets"""

    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize dataset preparer.

        Args:
            source_dir: Source directory containing raw data
            output_dir: Output directory for organized data
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_metadata_template(self, num_samples: int = 10):
        """Create a metadata template CSV file"""
        logger.info(f"Creating metadata template with {num_samples} example entries...")

        template_data = {
            'image_path': [f'images/sample_{i:04d}.dcm' for i in range(num_samples)],
            'cancer_type': [i % 4 for i in range(num_samples)],  # 0-3 for 4 cancer types
            'cancer_stage': [i % 5 for i in range(num_samples)],  # 0-4 for 5 stages
            'risk_score': [0.5 + (i * 0.05) for i in range(num_samples)],
            'age': [45 + i * 2 for i in range(num_samples)],
            'gender': [i % 2 for i in range(num_samples)],  # 0: female, 1: male
            'smoking_history': [i % 2 for i in range(num_samples)],  # 0: no, 1: yes
            'family_history': [i % 2 for i in range(num_samples)],
            'bmi': [22.0 + i * 0.5 for i in range(num_samples)],
            'tumor_size': [1.0 + i * 0.3 for i in range(num_samples)],
            'genomic_sequence': ['ATCGATCG' * 125 for _ in range(num_samples)]
        }

        df = pd.DataFrame(template_data)

        # Save metadata file
        metadata_file = self.output_dir / 'metadata_template.csv'
        df.to_csv(metadata_file, index=False)

        logger.info(f"Metadata template saved to {metadata_file}")
        logger.info("\nColumn descriptions:")
        logger.info("  - image_path: Path to medical image (DICOM, NIfTI, PNG, etc.)")
        logger.info("  - cancer_type: 0=Lung, 1=Breast, 2=Prostate, 3=Colorectal")
        logger.info("  - cancer_stage: 0-4 (0=Stage 0, 4=Stage IV)")
        logger.info("  - risk_score: Risk assessment score (0.0-1.0)")
        logger.info("  - age: Patient age in years")
        logger.info("  - gender: 0=Female, 1=Male")
        logger.info("  - smoking_history: 0=No, 1=Yes")
        logger.info("  - family_history: 0=No, 1=Yes")
        logger.info("  - bmi: Body Mass Index")
        logger.info("  - tumor_size: Tumor size in cm")
        logger.info("  - genomic_sequence: DNA sequence data")

        return metadata_file

    def validate_dataset(self, metadata_file: str):
        """Validate dataset structure and metadata"""
        logger.info(f"Validating dataset from {metadata_file}...")

        metadata_file = Path(metadata_file)
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False

        # Load metadata
        df = pd.read_csv(metadata_file)
        logger.info(f"Found {len(df)} samples in metadata")

        # Check required columns
        required_columns = ['image_path', 'cancer_type']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        # Check for missing values in required columns
        for col in required_columns:
            missing = df[col].isna().sum()
            if missing > 0:
                logger.warning(f"Column '{col}' has {missing} missing values")

        # Validate image paths
        base_dir = metadata_file.parent
        missing_images = []
        for idx, row in df.iterrows():
            img_path = base_dir / row['image_path']
            if not img_path.exists():
                missing_images.append(row['image_path'])

        if missing_images:
            logger.warning(f"Found {len(missing_images)} missing image files")
            if len(missing_images) <= 5:
                for img in missing_images:
                    logger.warning(f"  - {img}")
        else:
            logger.info("All image files found!")

        # Validate label ranges
        if 'cancer_type' in df.columns:
            unique_types = df['cancer_type'].unique()
            logger.info(f"Cancer types: {sorted(unique_types)}")
            if max(unique_types) > 3 or min(unique_types) < 0:
                logger.warning("Cancer type labels should be in range [0, 3]")

        if 'cancer_stage' in df.columns:
            unique_stages = df['cancer_stage'].unique()
            logger.info(f"Cancer stages: {sorted(unique_stages)}")

        logger.info("Validation complete!")
        return True

    def split_dataset(self, metadata_file: str, train_ratio: float = 0.7,
                     val_ratio: float = 0.15, test_ratio: float = 0.15):
        """Split dataset into train/val/test sets"""
        logger.info(f"Splitting dataset (train={train_ratio}, val={val_ratio}, test={test_ratio})...")

        df = pd.read_csv(metadata_file)
        total_samples = len(df)

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Calculate split indices
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)

        # Split
        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df = df[val_end:]

        # Save splits
        output_dir = Path(metadata_file).parent
        train_df.to_csv(output_dir / 'train_metadata.csv', index=False)
        val_df.to_csv(output_dir / 'val_metadata.csv', index=False)
        test_df.to_csv(output_dir / 'test_metadata.csv', index=False)

        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Val samples: {len(val_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        logger.info("Split files saved!")


def main():
    parser = argparse.ArgumentParser(description='Prepare cancer detection dataset')
    parser.add_argument('--source_dir', type=str, help='Source directory with raw data')
    parser.add_argument('--output_dir', type=str, default='./data',
                       help='Output directory for organized data')
    parser.add_argument('--create_template', action='store_true',
                       help='Create metadata template')
    parser.add_argument('--validate', type=str, help='Validate dataset from metadata file')
    parser.add_argument('--split', type=str, help='Split dataset from metadata file')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation data ratio')

    args = parser.parse_args()

    preparer = DatasetPreparer(
        source_dir=args.source_dir or './data',
        output_dir=args.output_dir
    )

    if args.create_template:
        preparer.create_metadata_template()

    if args.validate:
        preparer.validate_dataset(args.validate)

    if args.split:
        test_ratio = 1.0 - args.train_ratio - args.val_ratio
        preparer.split_dataset(args.split, args.train_ratio, args.val_ratio, test_ratio)


if __name__ == "__main__":
    main()
