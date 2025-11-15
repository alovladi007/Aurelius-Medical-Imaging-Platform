"""
Medical image loaders for DICOM, NIfTI, and other medical imaging formats.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Tuple, Dict
import logging

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

from PIL import Image
import cv2


logger = logging.getLogger(__name__)


class MedicalImageLoader:
    """Base class for loading medical images."""

    def __init__(self, normalize: bool = True, resize: Optional[Tuple[int, int]] = None):
        """
        Initialize medical image loader.

        Args:
            normalize: Whether to normalize pixel values to [0, 1]
            resize: Target size for resizing (height, width)
        """
        self.normalize = normalize
        self.resize = resize

    def load(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load medical image from file.

        Args:
            path: Path to image file

        Returns:
            Numpy array of image data
        """
        raise NotImplementedError("Subclasses must implement load method")

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        if self.normalize:
            img_min = image.min()
            img_max = image.max()
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
        return image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        if self.resize:
            # Handle 2D and 3D images
            if image.ndim == 2:
                image = cv2.resize(image, self.resize[::-1], interpolation=cv2.INTER_LINEAR)
            elif image.ndim == 3:
                # For 3D, resize each slice
                resized_slices = []
                for i in range(image.shape[2]):
                    slice_2d = cv2.resize(
                        image[:, :, i],
                        self.resize[::-1],
                        interpolation=cv2.INTER_LINEAR
                    )
                    resized_slices.append(slice_2d)
                image = np.stack(resized_slices, axis=2)
        return image


class DICOMLoader(MedicalImageLoader):
    """Loader for DICOM medical imaging files."""

    def __init__(self, normalize: bool = True, resize: Optional[Tuple[int, int]] = None,
                 apply_windowing: bool = True, window_center: Optional[float] = None,
                 window_width: Optional[float] = None):
        """
        Initialize DICOM loader.

        Args:
            normalize: Whether to normalize pixel values
            resize: Target size for resizing
            apply_windowing: Whether to apply windowing
            window_center: Window center for windowing
            window_width: Window width for windowing
        """
        super().__init__(normalize, resize)

        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required for DICOM loading. Install with: pip install pydicom")

        self.apply_windowing = apply_windowing
        self.window_center = window_center
        self.window_width = window_width

    def load(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load DICOM image.

        Args:
            path: Path to DICOM file

        Returns:
            Numpy array of image data
        """
        try:
            # Read DICOM file
            dicom = pydicom.dcmread(str(path))
            image = dicom.pixel_array.astype(np.float32)

            # Apply rescale slope and intercept if available
            if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
                image = image * dicom.RescaleSlope + dicom.RescaleIntercept

            # Apply windowing
            if self.apply_windowing:
                window_center = self.window_center
                window_width = self.window_width

                # Use DICOM tags if not specified
                if window_center is None and hasattr(dicom, 'WindowCenter'):
                    window_center = float(dicom.WindowCenter) if not isinstance(
                        dicom.WindowCenter, (list, tuple)
                    ) else float(dicom.WindowCenter[0])

                if window_width is None and hasattr(dicom, 'WindowWidth'):
                    window_width = float(dicom.WindowWidth) if not isinstance(
                        dicom.WindowWidth, (list, tuple)
                    ) else float(dicom.WindowWidth[0])

                if window_center is not None and window_width is not None:
                    image = self._apply_window(image, window_center, window_width)

            # Resize if needed
            image = self._resize_image(image)

            # Normalize
            image = self._normalize_image(image)

            return image

        except Exception as e:
            logger.error(f"Error loading DICOM file {path}: {e}")
            raise

    def _apply_window(self, image: np.ndarray, center: float, width: float) -> np.ndarray:
        """Apply windowing to image."""
        img_min = center - width / 2
        img_max = center + width / 2
        image = np.clip(image, img_min, img_max)
        return image

    def load_series(self, directory: Union[str, Path]) -> np.ndarray:
        """
        Load a series of DICOM files from a directory.

        Args:
            directory: Path to directory containing DICOM files

        Returns:
            3D numpy array of stacked slices
        """
        if not SITK_AVAILABLE:
            logger.warning("SimpleITK not available, loading slices individually")
            # Fallback: load each file and stack
            directory = Path(directory)
            dicom_files = sorted(directory.glob("*.dcm"))
            slices = [self.load(f) for f in dicom_files]
            return np.stack(slices, axis=-1)

        # Use SimpleITK for proper series loading
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(directory))
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Convert to numpy array
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)

        # Apply preprocessing
        if self.normalize:
            image_array = self._normalize_image(image_array)

        return image_array


class NIfTILoader(MedicalImageLoader):
    """Loader for NIfTI medical imaging files."""

    def __init__(self, normalize: bool = True, resize: Optional[Tuple[int, int]] = None,
                 extract_slice: Optional[int] = None, axis: int = 2):
        """
        Initialize NIfTI loader.

        Args:
            normalize: Whether to normalize pixel values
            resize: Target size for resizing
            extract_slice: Which slice to extract from 3D volume (None for all)
            axis: Axis along which to extract slice
        """
        super().__init__(normalize, resize)

        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for NIfTI loading. Install with: pip install nibabel")

        self.extract_slice = extract_slice
        self.axis = axis

    def load(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load NIfTI image.

        Args:
            path: Path to NIfTI file (.nii or .nii.gz)

        Returns:
            Numpy array of image data
        """
        try:
            # Load NIfTI file
            nifti_img = nib.load(str(path))
            image = nifti_img.get_fdata().astype(np.float32)

            # Extract slice if specified
            if self.extract_slice is not None:
                if self.axis == 0:
                    image = image[self.extract_slice, :, :]
                elif self.axis == 1:
                    image = image[:, self.extract_slice, :]
                else:  # axis == 2
                    image = image[:, :, self.extract_slice]

            # Resize if needed
            image = self._resize_image(image)

            # Normalize
            image = self._normalize_image(image)

            return image

        except Exception as e:
            logger.error(f"Error loading NIfTI file {path}: {e}")
            raise

    def load_volume(self, path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Load entire NIfTI volume with metadata.

        Args:
            path: Path to NIfTI file

        Returns:
            Tuple of (image array, metadata dictionary)
        """
        nifti_img = nib.load(str(path))
        image = nifti_img.get_fdata().astype(np.float32)

        metadata = {
            'affine': nifti_img.affine,
            'header': dict(nifti_img.header),
            'shape': image.shape,
            'voxel_sizes': nifti_img.header.get_zooms(),
        }

        if self.normalize:
            image = self._normalize_image(image)

        return image, metadata


class StandardImageLoader(MedicalImageLoader):
    """Loader for standard image formats (PNG, JPEG, etc.)."""

    def __init__(self, normalize: bool = True, resize: Optional[Tuple[int, int]] = None,
                 grayscale: bool = False):
        """
        Initialize standard image loader.

        Args:
            normalize: Whether to normalize pixel values
            resize: Target size for resizing
            grayscale: Whether to convert to grayscale
        """
        super().__init__(normalize, resize)
        self.grayscale = grayscale

    def load(self, path: Union[str, Path]) -> np.ndarray:
        """
        Load standard image format.

        Args:
            path: Path to image file

        Returns:
            Numpy array of image data
        """
        try:
            # Load image
            image = Image.open(str(path))

            # Convert to grayscale if needed
            if self.grayscale:
                image = image.convert('L')
            else:
                image = image.convert('RGB')

            # Convert to numpy array
            image = np.array(image, dtype=np.float32)

            # Resize if needed
            image = self._resize_image(image)

            # Normalize
            image = self._normalize_image(image)

            return image

        except Exception as e:
            logger.error(f"Error loading image file {path}: {e}")
            raise


def create_loader(file_path: Union[str, Path], **kwargs) -> MedicalImageLoader:
    """
    Create appropriate loader based on file extension.

    Args:
        file_path: Path to image file
        **kwargs: Additional arguments for loader

    Returns:
        Appropriate image loader instance
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == '.dcm' or file_path.is_dir():
        return DICOMLoader(**kwargs)
    elif suffix in ['.nii', '.gz']:
        return NIfTILoader(**kwargs)
    elif suffix in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        return StandardImageLoader(**kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
