"""Utilities for creating tiles/patches from large histopathology images."""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def create_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 0,
    min_tissue_ratio: float = 0.5,
) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """Create tiles from a large image.

    Args:
        image: Input image (H, W, C)
        tile_size: Size of each tile (square)
        overlap: Overlap between adjacent tiles in pixels
        min_tissue_ratio: Minimum ratio of tissue pixels to keep tile

    Returns:
        List of (tile_image, (x, y)) tuples
    """
    tiles = []
    height, width = image.shape[:2]

    stride = tile_size - overlap

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = image[y : y + tile_size, x : x + tile_size]

            # Check tissue content (simple background detection)
            if is_tissue_tile(tile, min_tissue_ratio):
                tiles.append((tile, (x, y)))

    return tiles


def is_tissue_tile(tile: np.ndarray, min_tissue_ratio: float = 0.5) -> bool:
    """Check if tile contains sufficient tissue (not mostly background).

    Args:
        tile: Image tile
        min_tissue_ratio: Minimum ratio of tissue pixels

    Returns:
        True if tile has sufficient tissue content
    """
    # Convert to grayscale
    if len(tile.shape) == 3:
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
    else:
        gray = tile

    # Simple thresholding: background is typically very bright (white)
    # Tissue has lower intensity values
    tissue_mask = gray < 200  # Adjust threshold as needed

    tissue_ratio = np.mean(tissue_mask)

    return tissue_ratio >= min_tissue_ratio


def save_tiles(
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
    output_dir: str,
    base_name: str,
    format: str = "png",
):
    """Save tiles to disk.

    Args:
        tiles: List of (tile_image, (x, y)) tuples
        output_dir: Output directory
        base_name: Base name for tile files
        format: Image format ('png', 'jpg', etc.)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (tile, (x, y)) in enumerate(tiles):
        filename = f"{base_name}_tile_{idx:04d}_x{x}_y{y}.{format}"
        filepath = output_dir / filename

        # Save tile
        if isinstance(tile, np.ndarray):
            Image.fromarray(tile).save(filepath)
        else:
            tile.save(filepath)


def reconstruct_from_tiles(
    tiles: List[Tuple[np.ndarray, Tuple[int, int]]],
    original_size: Tuple[int, int],
    tile_size: int = 512,
) -> np.ndarray:
    """Reconstruct image from tiles (for prediction aggregation).

    Args:
        tiles: List of (tile_image, (x, y)) tuples
        original_size: Original image size (height, width)
        tile_size: Size of tiles

    Returns:
        Reconstructed image
    """
    height, width = original_size
    n_channels = tiles[0][0].shape[2] if len(tiles[0][0].shape) == 3 else 1

    if n_channels == 1:
        reconstructed = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
    else:
        reconstructed = np.zeros((height, width, n_channels), dtype=np.float32)
        counts = np.zeros((height, width, n_channels), dtype=np.float32)

    for tile, (x, y) in tiles:
        tile_h, tile_w = tile.shape[:2]

        reconstructed[y : y + tile_h, x : x + tile_w] += tile
        counts[y : y + tile_h, x : x + tile_w] += 1

    # Average overlapping regions
    counts = np.maximum(counts, 1)  # Avoid division by zero
    reconstructed = reconstructed / counts

    return reconstructed.astype(np.uint8)


def load_large_image(image_path: str, max_size: int = 10000) -> np.ndarray:
    """Load large image with memory-efficient reading.

    Args:
        image_path: Path to image
        max_size: Maximum dimension (will resize if larger)

    Returns:
        Image as numpy array
    """
    # Try to load with PIL first
    try:
        img = Image.open(image_path)

        # Resize if too large
        width, height = img.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        return np.array(img)

    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}")
