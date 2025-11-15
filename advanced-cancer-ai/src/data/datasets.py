"""
PyTorch Dataset classes for multimodal cancer detection.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable, Union
import logging
import json

from .loaders import create_loader, MedicalImageLoader
from .preprocessing import (
    MedicalImagePreprocessor,
    ClinicalDataPreprocessor,
    GenomicDataPreprocessor
)
from .augmentation import MedicalImageAugmentation

logger = logging.getLogger(__name__)


class CancerImageDataset(Dataset):
    """Dataset for cancer detection from medical images only."""

    def __init__(self,
                 image_paths: List[Union[str, Path]],
                 labels: List[int],
                 transform: Optional[Callable] = None,
                 preprocessor: Optional[MedicalImagePreprocessor] = None,
                 loader_kwargs: Optional[Dict] = None):
        """
        Initialize cancer image dataset.

        Args:
            image_paths: List of paths to medical images
            labels: List of cancer type labels
            transform: Optional transformation function
            preprocessor: Optional preprocessor
            loader_kwargs: Optional arguments for image loader
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.transform = transform
        self.preprocessor = preprocessor
        self.loader_kwargs = loader_kwargs or {}

        assert len(self.image_paths) == len(self.labels), \
            "Number of images must match number of labels"

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Tuple of (image tensor, label)
        """
        # Load image
        image_path = self.image_paths[idx]
        loader = create_loader(image_path, **self.loader_kwargs)
        image = loader.load(image_path)

        # Preprocess
        if self.preprocessor:
            image = self.preprocessor(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Ensure tensor format
        if not isinstance(image, torch.Tensor):
            if image.ndim == 2:
                image = image[np.newaxis, ...]  # Add channel dimension
            image = torch.from_numpy(image).float()

        label = self.labels[idx]

        return image, label


class MultimodalCancerDataset(Dataset):
    """Dataset for multimodal cancer detection (imaging + clinical + genomic)."""

    def __init__(self,
                 data_file: Union[str, Path],
                 root_dir: Optional[Union[str, Path]] = None,
                 image_transform: Optional[Callable] = None,
                 image_preprocessor: Optional[MedicalImagePreprocessor] = None,
                 clinical_preprocessor: Optional[ClinicalDataPreprocessor] = None,
                 genomic_preprocessor: Optional[GenomicDataPreprocessor] = None,
                 require_all_modalities: bool = False,
                 loader_kwargs: Optional[Dict] = None):
        """
        Initialize multimodal cancer dataset.

        Args:
            data_file: Path to CSV/JSON file with dataset information
            root_dir: Root directory for relative paths
            image_transform: Optional image transformation
            image_preprocessor: Optional image preprocessor
            clinical_preprocessor: Optional clinical data preprocessor
            genomic_preprocessor: Optional genomic data preprocessor
            require_all_modalities: Whether all modalities are required
            loader_kwargs: Optional arguments for image loader
        """
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.image_transform = image_transform
        self.image_preprocessor = image_preprocessor
        self.clinical_preprocessor = clinical_preprocessor
        self.genomic_preprocessor = genomic_preprocessor
        self.require_all_modalities = require_all_modalities
        self.loader_kwargs = loader_kwargs or {}

        # Load dataset metadata
        self.data = self._load_data_file(data_file)

        logger.info(f"Loaded {len(self.data)} samples from {data_file}")

    def _load_data_file(self, data_file: Union[str, Path]) -> pd.DataFrame:
        """Load dataset metadata from file."""
        data_file = Path(data_file)

        if data_file.suffix == '.csv':
            data = pd.read_csv(data_file)
        elif data_file.suffix == '.json':
            data = pd.read_json(data_file)
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")

        # Verify required columns
        required_columns = ['image_path', 'cancer_type']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        return data

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.

        Args:
            idx: Index

        Returns:
            Dictionary with modality data and labels
        """
        row = self.data.iloc[idx]
        sample = {}

        # Load image
        image_path = self.root_dir / row['image_path']
        if image_path.exists():
            try:
                loader = create_loader(image_path, **self.loader_kwargs)
                image = loader.load(image_path)

                # Preprocess
                if self.image_preprocessor:
                    image = self.image_preprocessor(image)

                # Apply transformations
                if self.image_transform:
                    image = self.image_transform(image)

                # Ensure tensor format
                if not isinstance(image, torch.Tensor):
                    if image.ndim == 2:
                        image = image[np.newaxis, ...]
                    image = torch.from_numpy(image).float()

                sample['image'] = image
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}")
                if self.require_all_modalities:
                    raise
                sample['image'] = torch.zeros(1, 224, 224)

        # Load clinical data
        clinical_features = self._extract_clinical_features(row)
        if clinical_features is not None:
            if self.clinical_preprocessor:
                clinical_features = self.clinical_preprocessor.transform(clinical_features)

            # Convert to tensor
            clinical_tensor = self._dict_to_tensor(clinical_features)
            sample['clinical'] = clinical_tensor
        elif not self.require_all_modalities:
            sample['clinical'] = torch.zeros(10)  # Default size

        # Load genomic data
        if 'genomic_sequence' in row and pd.notna(row['genomic_sequence']):
            if self.genomic_preprocessor:
                genomic_data = self.genomic_preprocessor(row['genomic_sequence'])
                sample['genomic'] = torch.from_numpy(genomic_data).float()
        elif not self.require_all_modalities:
            sample['genomic'] = torch.zeros(1000, 5)  # Default size

        # Add labels
        sample['cancer_type'] = torch.tensor(row['cancer_type'], dtype=torch.long)

        if 'cancer_stage' in row and pd.notna(row['cancer_stage']):
            sample['cancer_stage'] = torch.tensor(row['cancer_stage'], dtype=torch.long)

        if 'risk_score' in row and pd.notna(row['risk_score']):
            sample['risk_score'] = torch.tensor(row['risk_score'], dtype=torch.float)

        return sample

    def _extract_clinical_features(self, row: pd.Series) -> Optional[Dict[str, np.ndarray]]:
        """Extract clinical features from data row."""
        clinical_columns = [
            'age', 'gender', 'smoking_history', 'family_history',
            'bmi', 'blood_pressure', 'tumor_size', 'lymph_nodes'
        ]

        features = {}
        for col in clinical_columns:
            if col in row and pd.notna(row[col]):
                features[col] = np.array([row[col]])

        return features if features else None

    def _dict_to_tensor(self, data_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """Convert dictionary of features to single tensor."""
        values = []
        for key in sorted(data_dict.keys()):
            values.extend(data_dict[key].flatten().tolist())
        return torch.tensor(values, dtype=torch.float)


class PublicCancerDataset(MultimodalCancerDataset):
    """
    Dataset wrapper for public cancer datasets.
    Supports common datasets like TCGA, LIDC-IDRI, etc.
    """

    SUPPORTED_DATASETS = {
        'tcga': 'The Cancer Genome Atlas',
        'lidc': 'Lung Image Database Consortium',
        'cbis_ddsm': 'Curated Breast Imaging Subset of DDSM',
        'custom': 'Custom dataset format'
    }

    def __init__(self,
                 dataset_name: str,
                 data_dir: Union[str, Path],
                 split: str = 'train',
                 **kwargs):
        """
        Initialize public cancer dataset.

        Args:
            dataset_name: Name of dataset ('tcga', 'lidc', 'cbis_ddsm', 'custom')
            data_dir: Root directory containing dataset
            split: Dataset split ('train', 'val', 'test')
            **kwargs: Additional arguments passed to MultimodalCancerDataset
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.split = split

        if self.dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )

        # Find or create metadata file
        metadata_file = self._prepare_metadata_file()

        super().__init__(
            data_file=metadata_file,
            root_dir=self.data_dir,
            **kwargs
        )

    def _prepare_metadata_file(self) -> Path:
        """Prepare metadata file for the dataset."""
        metadata_file = self.data_dir / f"{self.split}_metadata.csv"

        if metadata_file.exists():
            return metadata_file

        logger.info(f"Metadata file not found, creating from dataset structure...")

        # Dataset-specific logic would go here
        # For now, create a template
        self._create_metadata_template(metadata_file)

        return metadata_file

    def _create_metadata_template(self, output_file: Path):
        """Create a metadata template file."""
        # This would be implemented based on the specific dataset structure
        logger.warning(
            f"Creating empty metadata template at {output_file}. "
            "Please populate it with your dataset information."
        )

        template_data = {
            'image_path': [],
            'cancer_type': [],
            'cancer_stage': [],
            'risk_score': []
        }

        df = pd.DataFrame(template_data)
        df.to_csv(output_file, index=False)


def create_dataset_from_directory(data_dir: Union[str, Path],
                                 file_pattern: str = "**/*.dcm",
                                 **kwargs) -> CancerImageDataset:
    """
    Create dataset from directory structure.

    Expected structure:
        data_dir/
            lung_cancer/
                image1.dcm
                image2.dcm
            breast_cancer/
                image1.dcm
                image2.dcm

    Args:
        data_dir: Root directory
        file_pattern: Glob pattern for finding images
        **kwargs: Additional arguments for CancerImageDataset

    Returns:
        CancerImageDataset instance
    """
    data_dir = Path(data_dir)
    image_paths = []
    labels = []

    # Map class names to indices
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

    logger.info(f"Found classes: {class_to_idx}")

    # Find all images
    for class_dir in class_dirs:
        class_idx = class_to_idx[class_dir.name]
        class_images = list(class_dir.glob(file_pattern))

        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    logger.info(f"Found {len(image_paths)} images across {len(class_to_idx)} classes")

    return CancerImageDataset(
        image_paths=image_paths,
        labels=labels,
        **kwargs
    )


def collate_multimodal(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for multimodal data.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary
    """
    collated = {}

    # Get all keys from first sample
    keys = batch[0].keys()

    for key in keys:
        # Stack all values for this key
        values = [sample[key] for sample in batch]

        # Handle different data types
        if all(isinstance(v, torch.Tensor) for v in values):
            # Pad if necessary for variable-length sequences
            if key == 'genomic' and any(v.shape != values[0].shape for v in values):
                max_len = max(v.shape[0] for v in values)
                padded = []
                for v in values:
                    if v.shape[0] < max_len:
                        pad_size = max_len - v.shape[0]
                        v = torch.nn.functional.pad(v, (0, 0, 0, pad_size))
                    padded.append(v)
                collated[key] = torch.stack(padded)
            else:
                try:
                    collated[key] = torch.stack(values)
                except:
                    # If stacking fails, keep as list
                    collated[key] = values
        else:
            collated[key] = values

    return collated
