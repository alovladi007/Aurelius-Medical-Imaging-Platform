#!/usr/bin/env python
"""Setup script for Kaggle Brain Cancer dataset."""

import argparse
import zipfile
from pathlib import Path
import shutil
import random

def setup_brain_cancer_dataset(
    zip_path: str,
    extract_to: str = "data/raw",
    create_sample: bool = False,
    sample_size: int = 200,
    seed: int = 42
):
    """
    Setup the Kaggle Brain Cancer dataset.

    Args:
        zip_path: Path to the zip file
        extract_to: Where to extract
        create_sample: Create a small sample for quick testing
        sample_size: Number of images per class for sample
        seed: Random seed
    """
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    print(f"Found dataset: {zip_path}")
    print(f"Size: {zip_path.stat().st_size / (1024**3):.2f} GB")

    # Extract
    print(f"\nExtracting to: {extract_to}")
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List contents first
        file_list = zip_ref.namelist()
        print(f"Archive contains {len(file_list)} files")

        # Show structure
        print("\nArchive structure (first 20 items):")
        for item in file_list[:20]:
            print(f"  {item}")

        # Extract
        print("\nExtracting... (this may take a few minutes)")
        zip_ref.extractall(extract_to)

    print("✓ Extraction complete")

    # Analyze structure
    print("\nAnalyzing dataset structure...")
    extracted_path = extract_to

    # Find the actual data directory
    subdirs = [d for d in extracted_path.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        extracted_path = subdirs[0]
        print(f"Data is in: {extracted_path}")

    # Count images per class
    for class_dir in sorted(extracted_path.iterdir()):
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + \
                    list(class_dir.glob("*.png")) + \
                    list(class_dir.glob("*.jpeg"))
            print(f"  {class_dir.name}: {len(images)} images")

    # Create sample if requested
    if create_sample:
        print(f"\nCreating sample dataset ({sample_size} images per class)...")
        sample_dir = Path("data/raw/sample")
        sample_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)

        for class_dir in sorted(extracted_path.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name

                # Get all images
                images = list(class_dir.glob("*.jpg")) + \
                        list(class_dir.glob("*.png")) + \
                        list(class_dir.glob("*.jpeg"))

                # Sample
                sample = random.sample(images, min(sample_size, len(images)))

                # Copy to sample directory
                sample_class_dir = sample_dir / class_name
                sample_class_dir.mkdir(parents=True, exist_ok=True)

                for img in sample:
                    shutil.copy2(img, sample_class_dir / img.name)

                print(f"  {class_name}: copied {len(sample)} images")

        print(f"\n✓ Sample dataset created at: {sample_dir}")
        print(f"  Use this for quick testing before full training")

    print("\n" + "="*60)
    print("DATASET READY!")
    print("="*60)
    print(f"\nFull dataset: {extracted_path}")
    if create_sample:
        print(f"Sample dataset: data/raw/sample")

    print("\nNext steps:")
    print("1. Update config/dataset.yaml with the correct path")
    print("2. Run: python scripts/create_splits.py --config config/dataset.yaml")
    print("3. Run: python scripts/train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Kaggle Brain Cancer dataset")
    parser.add_argument(
        "--zip-path",
        required=True,
        help="Path to 'Kaggle Brain Cancer Data.zip'"
    )
    parser.add_argument(
        "--extract-to",
        default="data/raw",
        help="Where to extract (default: data/raw)"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a small sample for testing (recommended for first run)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="Images per class in sample (default: 200)"
    )

    args = parser.parse_args()

    setup_brain_cancer_dataset(
        zip_path=args.zip_path,
        extract_to=args.extract_to,
        create_sample=args.create_sample,
        sample_size=args.sample_size
    )
