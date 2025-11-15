"""Inference API for cancer histopathology models."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from cancer_quant_model.data.transforms import get_val_transforms
from cancer_quant_model.explainability.grad_cam import GradCAM, get_target_layer
from cancer_quant_model.utils.feature_utils import QuantitativeFeatureExtractor
from cancer_quant_model.utils.logging_utils import get_logger

logger = get_logger(__name__)


class InferenceAPI:
    """API for running inference on histopathology images."""

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: str = "cuda",
        image_size: tuple = (224, 224),
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
        architecture: str = "resnet",
    ):
        """
        Initialize inference API.

        Args:
            model: Trained model
            class_names: List of class names
            device: Device for inference
            image_size: Input image size
            normalize_mean: Normalization mean
            normalize_std: Normalization std
            architecture: Model architecture for Grad-CAM
        """
        self.model = model.to(device)
        self.model.eval()
        self.class_names = class_names
        self.device = device
        self.architecture = architecture

        # Setup transform
        self.transform = get_val_transforms(
            image_size=image_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )

        # Feature extractor
        self.feature_extractor = QuantitativeFeatureExtractor()

        logger.info(f"Inference API initialized on device: {device}")

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        return_features: bool = False,
        return_gradcam: bool = False,
    ) -> Dict:
        """
        Run prediction on a single image.

        Args:
            image: Image path or numpy array (H, W, C)
            return_features: Return deep features
            return_gradcam: Return Grad-CAM heatmap

        Returns:
            Dictionary with prediction results
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))

        # Preprocess
        transformed = self.transform(image=image)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Forward pass
        output = self.model(input_tensor)

        if isinstance(output, dict):
            logits = output["logits"]
            deep_features = output.get("features")
        else:
            logits = output
            deep_features = None

        # Get predictions
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred_class].item()

        results = {
            "predicted_class": pred_class,
            "predicted_label": self.class_names[pred_class],
            "confidence": confidence,
            "probabilities": {
                self.class_names[i]: probs[0, i].item() for i in range(len(self.class_names))
            },
        }

        # Deep features
        if return_features and deep_features is not None:
            results["deep_features"] = deep_features.cpu().numpy()[0]

        # Grad-CAM
        if return_gradcam:
            try:
                target_layers = get_target_layer(self.model, self.architecture)
                gradcam = GradCAM(self.model, target_layers, self.device)

                self.model.train()  # Need gradients
                cam = gradcam(input_tensor, target_class=pred_class)
                self.model.eval()

                results["gradcam"] = cam

            except Exception as e:
                logger.error(f"Error generating Grad-CAM: {e}")
                results["gradcam"] = None

        return results

    def predict_batch(
        self, images: List[Union[str, Path, np.ndarray]], batch_size: int = 32
    ) -> List[Dict]:
        """
        Run prediction on a batch of images.

        Args:
            images: List of image paths or arrays
            batch_size: Batch size for processing

        Returns:
            List of prediction results
        """
        results = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            for img in batch:
                result = self.predict(img)
                results.append(result)

        return results

    def extract_quantitative_features(self, image: Union[str, Path, np.ndarray]) -> Dict:
        """
        Extract quantitative features from image.

        Args:
            image: Image path or numpy array

        Returns:
            Dictionary of quantitative features
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = np.array(Image.open(image).convert("RGB"))

        # Extract features
        features = self.feature_extractor.extract_all_features(image)

        return features

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model: nn.Module,
        class_names: List[str],
        device: str = "auto",
        **kwargs,
    ) -> "InferenceAPI":
        """
        Create API from a checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            model: Model architecture (uninitialized)
            class_names: List of class names
            device: Device
            **kwargs: Additional arguments for InferenceAPI

        Returns:
            InferenceAPI instance
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"Best metric: {checkpoint.get('best_metric', 'N/A')}")

        return cls(model=model, class_names=class_names, device=device, **kwargs)
