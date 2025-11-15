#!/usr/bin/env python
"""Generate synthetic histopathology-like images for testing the pipeline."""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random
from tqdm import tqdm


def generate_synthetic_histopathology_image(
    image_size=(224, 224),
    has_tumor=False,
    seed=None
):
    """
    Generate a synthetic histopathology-like image.

    Args:
        image_size: Tuple of (width, height)
        has_tumor: If True, add tumor-like features
        seed: Random seed for reproducibility

    Returns:
        PIL Image
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    width, height = image_size

    # Create base tissue-like background
    if has_tumor:
        # Tumor tissue - more purple/darker, irregular patterns
        base_color = (180, 150, 200)  # Purplish
        cell_density = 0.4  # Higher density
        cell_size_range = (3, 8)
        irregularity = 0.6
    else:
        # Normal tissue - lighter, more pink, regular patterns
        base_color = (230, 200, 220)  # Pinkish
        cell_density = 0.2  # Lower density
        cell_size_range = (4, 6)
        irregularity = 0.2

    # Create base image with noise
    img_array = np.random.normal(
        loc=base_color,
        scale=20,
        size=(height, width, 3)
    ).astype(np.uint8)

    img = Image.fromarray(img_array, mode='RGB')
    draw = ImageDraw.Draw(img)

    # Add cell-like structures
    num_cells = int(width * height * cell_density / 100)

    for _ in range(num_cells):
        # Random cell position
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # Cell size
        size = random.randint(*cell_size_range)

        # Cell shape (more irregular for tumor)
        if random.random() < irregularity:
            # Irregular shape
            points = []
            for angle in range(0, 360, 60):
                r = size + random.randint(-2, 2)
                px = x + int(r * np.cos(np.radians(angle)))
                py = y + int(r * np.sin(np.radians(angle)))
                points.append((px, py))
            draw.polygon(points, fill=(100, 50, 130) if has_tumor else (200, 150, 180))
        else:
            # Regular circular cell
            draw.ellipse(
                [x - size, y - size, x + size, y + size],
                fill=(120, 70, 150) if has_tumor else (210, 160, 190)
            )

    # Add texture
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Add some noise
    noise = np.random.normal(0, 5, (height, width, 3))
    img_array = np.array(img).astype(float)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    return img


def generate_synthetic_dataset(
    output_dir: str,
    num_classes: int = 2,
    samples_per_class: int = 200,
    image_size: tuple = (224, 224),
    class_names: list = None,
    seed: int = 42
):
    """
    Generate a complete synthetic dataset.

    Args:
        output_dir: Where to save the dataset
        num_classes: Number of classes (default: 2 for binary)
        samples_per_class: Images per class
        image_size: Image dimensions
        class_names: List of class names (or use defaults)
        seed: Random seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        if num_classes == 2:
            class_names = ["non_tumor", "tumor"]
        elif num_classes == 4:
            class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]
        else:
            class_names = [f"class_{i}" for i in range(num_classes)]

    print(f"Generating synthetic histopathology dataset...")
    print(f"Output: {output_path}")
    print(f"Classes: {class_names}")
    print(f"Samples per class: {samples_per_class}")
    print(f"Image size: {image_size}")
    print()

    total_images = num_classes * samples_per_class

    with tqdm(total=total_images, desc="Generating images") as pbar:
        for class_idx, class_name in enumerate(class_names):
            class_dir = output_path / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for img_idx in range(samples_per_class):
                # Generate image
                has_tumor = "tumor" in class_name.lower() or "glioma" in class_name.lower() or \
                           "meningioma" in class_name.lower() or "pituitary" in class_name.lower()

                img_seed = seed + class_idx * 10000 + img_idx
                img = generate_synthetic_histopathology_image(
                    image_size=image_size,
                    has_tumor=has_tumor,
                    seed=img_seed
                )

                # Save image
                img_path = class_dir / f"{class_name}_{img_idx:04d}.png"
                img.save(img_path)

                pbar.update(1)

    print("\n✓ Dataset generation complete!")
    print(f"\nDataset structure:")
    for class_name in class_names:
        class_dir = output_path / class_name
        num_images = len(list(class_dir.glob("*.png")))
        print(f"  {class_name}: {num_images} images")

    print(f"\nNext steps:")
    print(f"1. Update config/dataset.yaml:")
    print(f"   dataset:")
    print(f"     paths:")
    print(f"       raw_data_dir: \"{output_dir}\"")
    print(f"     labels:")
    print(f"       num_classes: {num_classes}")
    print(f"       class_names: {class_names}")
    print(f"\n2. Create splits:")
    print(f"   python scripts/create_splits.py --config config/dataset.yaml")
    print(f"\n3. Train:")
    print(f"   python scripts/train.py \\")
    print(f"       --dataset-config config/dataset.yaml \\")
    print(f"       --model-config config/model_resnet.yaml \\")
    print(f"       --train-config config/train_default.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic histopathology dataset for testing"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/synthetic",
        help="Output directory for synthetic dataset"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        choices=[2, 4],
        help="Number of classes (2 or 4)"
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=200,
        help="Number of images per class"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size (will be square)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    generate_synthetic_dataset(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        image_size=(args.image_size, args.image_size),
        seed=args.seed
    )
