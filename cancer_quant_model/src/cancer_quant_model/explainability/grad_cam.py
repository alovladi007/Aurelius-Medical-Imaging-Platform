"""Grad-CAM implementation for model explainability."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cancer_quant_model.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)."""

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        device: str = "cuda",
    ):
        """
        Initialize Grad-CAM.

        Args:
            model: Model to explain
            target_layers: List of target layers for CAM
            device: Device
        """
        self.model = model.to(device)
        self.device = device
        self.target_layers = target_layers

        self.activations = []
        self.gradients = []

        # Register hooks
        self.handles = []
        for target_layer in self.target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self._save_activation)
            )
            self.handles.append(
                target_layer.register_full_backward_hook(self._save_gradient)
            )

    def _save_activation(self, module, input, output):
        """Save activation from forward pass."""
        self.activations.append(output.detach())

    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradient from backward pass."""
        self.gradients.append(grad_output[0].detach())

    def forward(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        """
        Forward pass to compute CAM.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class (None for predicted class)

        Returns:
            CAM heatmap (H, W)
        """
        # Clear previous activations and gradients
        self.activations = []
        self.gradients = []

        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        if isinstance(output, dict):
            logits = output["logits"]
        else:
            logits = output

        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()

        # Compute CAM
        cam = self._compute_cam()

        return cam

    def _compute_cam(self) -> np.ndarray:
        """
        Compute CAM from activations and gradients.

        Returns:
            CAM heatmap
        """
        # Get the last activation and gradient
        activations = self.activations[-1]  # (1, C, H, W)
        gradients = self.gradients[0]  # (1, C, H, W)

        # Global average pooling on gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination
        cam = (weights * activations).sum(dim=1).squeeze(0)  # (H, W)

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def __call__(self, input_tensor: torch.Tensor, target_class: Optional[int] = None):
        """Callable interface."""
        return self.forward(input_tensor, target_class)

    def __del__(self):
        """Remove hooks."""
        for handle in self.handles:
            handle.remove()


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ implementation."""

    def _compute_cam(self) -> np.ndarray:
        """
        Compute Grad-CAM++ from activations and gradients.

        Returns:
            CAM heatmap
        """
        activations = self.activations[-1]  # (1, C, H, W)
        gradients = self.gradients[0]  # (1, C, H, W)

        # Compute alpha (Grad-CAM++ weights)
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)

        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(
            alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom)
        )
        alpha = alpha_num / alpha_denom

        # Compute weights
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = (weights * activations).sum(dim=1).squeeze(0)  # (H, W)

        # ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay CAM heatmap on image.

    Args:
        image: Original image (H, W, C) in RGB, [0, 255], uint8
        cam: CAM heatmap (H, W) in [0, 1]
        colormap: OpenCV colormap
        alpha: Overlay transparency

    Returns:
        Overlaid image (H, W, C) in RGB, uint8
    """
    # Resize CAM to image size
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)

    return overlay


def get_target_layer(model: nn.Module, architecture: str = "resnet") -> List[nn.Module]:
    """
    Get target layer for Grad-CAM based on architecture.

    Args:
        model: Model
        architecture: Model architecture (resnet, efficientnet, vit)

    Returns:
        List of target layers
    """
    if architecture == "resnet":
        # For ResNet, use the last convolutional block
        if hasattr(model, "backbone"):
            if hasattr(model.backbone, "layer4"):
                return [model.backbone.layer4[-1]]
        elif hasattr(model, "layer4"):
            return [model.layer4[-1]]

    elif architecture == "efficientnet":
        # For EfficientNet, use the last block
        if hasattr(model, "backbone"):
            if hasattr(model.backbone, "blocks"):
                return [model.backbone.blocks[-1]]
        elif hasattr(model, "blocks"):
            return [model.blocks[-1]]

    elif architecture == "vit":
        # For ViT, use the last transformer block
        if hasattr(model, "backbone"):
            if hasattr(model.backbone, "blocks"):
                return [model.backbone.blocks[-1].norm1]
        elif hasattr(model, "blocks"):
            return [model.blocks[-1].norm1]

    # Default: try to find the last conv layer
    logger.warning(f"Unknown architecture '{architecture}', using default target layer")
    conv_layers = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)

    if conv_layers:
        return [conv_layers[-1]]

    raise ValueError("Could not find suitable target layer for Grad-CAM")
