"""Tiling utilities for processing large histopathology images."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


class ImageTiler:
    """Create tiles/patches from large histopathology images."""

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 0,
        min_tissue_ratio: float = 0.5,
        tissue_threshold: int = 220,
    ):
        """
        Initialize image tiler.

        Args:
            tile_size: Size of tiles (square)
            overlap: Overlap between tiles in pixels
            min_tissue_ratio: Minimum ratio of tissue (non-white) pixels
            tissue_threshold: Threshold for considering pixel as background (white)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_tissue_ratio = min_tissue_ratio
        self.tissue_threshold = tissue_threshold

    def is_tissue_tile(self, tile: np.ndarray) -> bool:
        """
        Check if tile contains sufficient tissue.

        Args:
            tile: Image tile (H, W, C)

        Returns:
            True if tile has enough tissue
        """
        # Convert to grayscale
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile

        # Count non-white pixels (tissue)
        tissue_pixels = np.sum(gray < self.tissue_threshold)
        total_pixels = gray.size
        tissue_ratio = tissue_pixels / total_pixels

        return tissue_ratio >= self.min_tissue_ratio

    def tile_image(
        self, image: np.ndarray, filter_background: bool = True
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Create tiles from a large image.

        Args:
            image: Input image (H, W, C)
            filter_background: Filter out tiles with mostly background

        Returns:
            Tuple of (list of tiles, list of (y, x) coordinates)
        """
        h, w = image.shape[:2]
        stride = self.tile_size - self.overlap

        tiles = []
        coords = []

        for y in range(0, h - self.tile_size + 1, stride):
            for x in range(0, w - self.tile_size + 1, stride):
                tile = image[y : y + self.tile_size, x : x + self.tile_size]

                # Check if tile has enough tissue
                if filter_background and not self.is_tissue_tile(tile):
                    continue

                tiles.append(tile)
                coords.append((y, x))

        return tiles, coords

    def save_tiles(
        self,
        tiles: List[np.ndarray],
        coords: List[Tuple[int, int]],
        output_dir: Path,
        base_name: str,
        format: str = "png",
    ):
        """
        Save tiles to disk.

        Args:
            tiles: List of tiles
            coords: List of coordinates
            output_dir: Output directory
            base_name: Base name for tile files
            format: Image format (png, jpg, etc.)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for idx, (tile, (y, x)) in enumerate(zip(tiles, coords)):
            filename = f"{base_name}_y{y}_x{x}_tile{idx}.{format}"
            filepath = output_dir / filename

            # Convert to PIL Image and save
            if tile.dtype != np.uint8:
                tile = (tile * 255).astype(np.uint8)

            img = Image.fromarray(tile)
            img.save(filepath)

    def tile_from_file(
        self,
        image_path: Path,
        output_dir: Path,
        filter_background: bool = True,
        save_tiles: bool = True,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Load image from file and create tiles.

        Args:
            image_path: Path to input image
            output_dir: Output directory for tiles
            filter_background: Filter background tiles
            save_tiles: Save tiles to disk

        Returns:
            Tuple of (list of tiles, list of coordinates)
        """
        # Load image
        image = np.array(Image.open(image_path))

        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Create tiles
        tiles, coords = self.tile_image(image, filter_background=filter_background)

        # Save tiles if requested
        if save_tiles:
            base_name = Path(image_path).stem
            self.save_tiles(tiles, coords, output_dir, base_name)

        return tiles, coords


def reconstruct_from_tiles(
    tiles: List[np.ndarray],
    coords: List[Tuple[int, int]],
    original_size: Tuple[int, int],
    tile_size: int,
    overlap: int = 0,
    aggregation: str = "average",
) -> np.ndarray:
    """
    Reconstruct image from tiles.

    Args:
        tiles: List of tiles
        coords: List of (y, x) coordinates
        original_size: Original image size (H, W)
        tile_size: Size of tiles
        overlap: Overlap between tiles
        aggregation: Aggregation method for overlapping regions ('average', 'max')

    Returns:
        Reconstructed image
    """
    h, w = original_size
    channels = tiles[0].shape[2] if len(tiles[0].shape) == 3 else 1

    if channels > 1:
        reconstructed = np.zeros((h, w, channels), dtype=np.float32)
        counts = np.zeros((h, w, channels), dtype=np.float32)
    else:
        reconstructed = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.float32)

    for tile, (y, x) in zip(tiles, coords):
        tile_h, tile_w = tile.shape[:2]

        if aggregation == "average":
            reconstructed[y : y + tile_h, x : x + tile_w] += tile
            if channels > 1:
                counts[y : y + tile_h, x : x + tile_w] += 1
            else:
                counts[y : y + tile_h, x : x + tile_w] += 1
        elif aggregation == "max":
            reconstructed[y : y + tile_h, x : x + tile_w] = np.maximum(
                reconstructed[y : y + tile_h, x : x + tile_w], tile
            )
            if channels > 1:
                counts[y : y + tile_h, x : x + tile_w] = 1
            else:
                counts[y : y + tile_h, x : x + tile_w] = 1

    # Average overlapping regions
    if aggregation == "average":
        mask = counts > 0
        reconstructed[mask] /= counts[mask]

    return reconstructed.astype(np.uint8)
