"""
Data manager for organizing and loading cancer detection datasets.
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

from .datasets import (
    CancerImageDataset,
    MultimodalCancerDataset,
    PublicCancerDataset,
    create_dataset_from_directory,
    collate_multimodal
)
from .preprocessing import (
    MedicalImagePreprocessor,
    ClinicalDataPreprocessor,
    GenomicDataPreprocessor
)
from .augmentation import (
    get_training_augmentation,
    get_validation_augmentation
)

logger = logging.getLogger(__name__)


class DataManager:
    """Manager for cancer detection datasets and data loaders."""

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 image_size: Tuple[int, int] = (224, 224),
                 train_val_split: float = 0.8,
                 augmentation: bool = True,
                 multimodal: bool = True,
                 dataset_type: str = 'custom'):
        """
        Initialize data manager.

        Args:
            data_dir: Root directory containing data
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            image_size: Target image size
            train_val_split: Ratio for train/validation split
            augmentation: Whether to apply data augmentation
            multimodal: Whether to use multimodal data
            dataset_type: Type of dataset ('custom', 'tcga', 'lidc', etc.)
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train_val_split = train_val_split
        self.augmentation = augmentation
        self.multimodal = multimodal
        self.dataset_type = dataset_type

        # Initialize preprocessors
        self.image_preprocessor = MedicalImagePreprocessor(
            target_size=image_size,
            normalize_method='min_max',
            apply_clahe=True
        )

        self.clinical_preprocessor = ClinicalDataPreprocessor(
            numerical_features=['age', 'bmi', 'tumor_size'],
            normalize_numerical=True
        )

        self.genomic_preprocessor = GenomicDataPreprocessor(
            sequence_length=1000,
            encoding='one_hot'
        )

        # Get transformations
        self.train_transform = get_training_augmentation() if augmentation else None
        self.val_transform = get_validation_augmentation()

        logger.info(f"DataManager initialized for {dataset_type} dataset")

    def prepare_datasets(self) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
        """
        Prepare train, validation, and test datasets.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.multimodal:
            return self._prepare_multimodal_datasets()
        else:
            return self._prepare_image_datasets()

    def _prepare_multimodal_datasets(self) -> Tuple:
        """Prepare multimodal datasets."""
        logger.info("Preparing multimodal datasets...")

        # Check for split-specific metadata files
        train_file = self.data_dir / "train_metadata.csv"
        val_file = self.data_dir / "val_metadata.csv"
        test_file = self.data_dir / "test_metadata.csv"

        if train_file.exists() and val_file.exists():
            # Use pre-split data
            train_dataset = MultimodalCancerDataset(
                data_file=train_file,
                root_dir=self.data_dir,
                image_transform=self.train_transform,
                image_preprocessor=self.image_preprocessor,
                clinical_preprocessor=self.clinical_preprocessor,
                genomic_preprocessor=self.genomic_preprocessor
            )

            val_dataset = MultimodalCancerDataset(
                data_file=val_file,
                root_dir=self.data_dir,
                image_transform=self.val_transform,
                image_preprocessor=self.image_preprocessor,
                clinical_preprocessor=self.clinical_preprocessor,
                genomic_preprocessor=self.genomic_preprocessor
            )

            test_dataset = None
            if test_file.exists():
                test_dataset = MultimodalCancerDataset(
                    data_file=test_file,
                    root_dir=self.data_dir,
                    image_transform=self.val_transform,
                    image_preprocessor=self.image_preprocessor,
                    clinical_preprocessor=self.clinical_preprocessor,
                    genomic_preprocessor=self.genomic_preprocessor
                )

        else:
            # Look for single metadata file and split
            metadata_file = self.data_dir / "metadata.csv"
            if not metadata_file.exists():
                logger.warning(
                    f"No metadata file found. Creating template at {metadata_file}"
                )
                self._create_metadata_template(metadata_file)

            # Load full dataset
            full_dataset = MultimodalCancerDataset(
                data_file=metadata_file,
                root_dir=self.data_dir,
                image_transform=self.train_transform,
                image_preprocessor=self.image_preprocessor,
                clinical_preprocessor=self.clinical_preprocessor,
                genomic_preprocessor=self.genomic_preprocessor
            )

            # Split dataset
            train_size = int(self.train_val_split * len(full_dataset))
            val_size = len(full_dataset) - train_size

            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            test_dataset = None

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        if test_dataset:
            logger.info(f"Test samples: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def _prepare_image_datasets(self) -> Tuple:
        """Prepare image-only datasets."""
        logger.info("Preparing image-only datasets...")

        # Try to create from directory structure
        try:
            full_dataset = create_dataset_from_directory(
                self.data_dir,
                preprocessor=self.image_preprocessor,
                transform=self.train_transform
            )

            # Split dataset
            train_size = int(self.train_val_split * len(full_dataset))
            val_size = len(full_dataset) - train_size

            train_dataset, val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Val samples: {len(val_dataset)}")

            return train_dataset, val_dataset, None

        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        """
        Get data loaders for train, validation, and test sets.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()

        # Determine collate function
        collate_fn = collate_multimodal if self.multimodal else None

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        test_loader = None
        if test_dataset:
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )

        return train_loader, val_loader, test_loader

    def _create_metadata_template(self, output_file: Path):
        """Create a metadata template file."""
        import pandas as pd

        template_data = {
            'image_path': ['path/to/image1.dcm', 'path/to/image2.dcm'],
            'cancer_type': [0, 1],
            'cancer_stage': [2, 3],
            'risk_score': [0.5, 0.75],
            'age': [55, 62],
            'gender': [0, 1],  # 0: female, 1: male
            'smoking_history': [1, 0],
            'bmi': [25.5, 28.3],
            'genomic_sequence': ['ATCGATCG' * 125, 'GCTAGCTA' * 125]  # Example sequences
        }

        df = pd.DataFrame(template_data)
        df.to_csv(output_file, index=False)

        logger.info(f"Created metadata template at {output_file}")
        logger.info("Please populate this file with your actual data.")

    def get_dataset_statistics(self) -> Dict:
        """
        Compute and return dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        train_dataset, val_dataset, test_dataset = self.prepare_datasets()

        stats = {
            'num_train': len(train_dataset),
            'num_val': len(val_dataset),
            'num_test': len(test_dataset) if test_dataset else 0,
            'total': len(train_dataset) + len(val_dataset) + (len(test_dataset) if test_dataset else 0),
            'batch_size': self.batch_size,
            'image_size': self.image_size,
            'multimodal': self.multimodal,
        }

        return stats
