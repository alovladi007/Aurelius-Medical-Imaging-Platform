"""Grad-CAM implementation for model explainability."""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM)."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM (if None, uses last conv layer)
        """
        self.model = model
        self.model.eval()

        # Find target layer
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _find_target_layer(self) -> nn.Module:
        """Find the last convolutional layer."""
        # Search for last conv layer
        for module in reversed(list(self.model.modules())):
            if isinstance(module, nn.Conv2d):
                return module

        raise ValueError("Could not find a convolutional layer in the model")

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate class activation map.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, uses predicted class)

        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)

        # Compute weights as global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate CAM (callable interface)."""
        return self.generate_cam(input_tensor, target_class)


def apply_colormap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Apply heatmap overlay on image.

    Args:
        image: Original image (H, W, 3) in [0, 255]
        heatmap: Heatmap (H, W) in [0, 1]
        alpha: Blending weight for heatmap
        colormap: OpenCV colormap

    Returns:
        Overlayed image
    """
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlayed


def save_gradcam_visualization(
    image: np.ndarray,
    cam: np.ndarray,
    save_path: str,
    alpha: float = 0.4,
):
    """Save Grad-CAM visualization.

    Args:
        image: Original image (H, W, 3)
        cam: CAM heatmap (H_cam, W_cam)
        save_path: Path to save visualization
        alpha: Blending weight
    """
    overlayed = apply_colormap_on_image(image, cam, alpha=alpha)

    from PIL import Image as PILImage

    PILImage.fromarray(overlayed).save(save_path)


class GuidedBackprop:
    """Guided Backpropagation for visualization."""

    def __init__(self, model: nn.Module):
        """Initialize Guided Backprop.

        Args:
            model: PyTorch model
        """
        self.model = model
        self.model.eval()

        self.gradients = None

        # Update ReLU backward hooks
        self._update_relus()

    def _update_relus(self):
        """Update ReLU layers to guided backprop."""

        def relu_backward_hook(module, grad_input, grad_output):
            """Modify ReLU backward pass."""
            return (F.relu(grad_input[0]),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_full_backward_hook(relu_backward_hook)

    def generate_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate guided gradients.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index

        Returns:
            Guided gradients
        """
        input_tensor.requires_grad = True

        # Forward pass
        output = self.model(input_tensor)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients
        gradients = input_tensor.grad.cpu().numpy()[0]

        return gradients


def get_gradcam_for_model(
    model: nn.Module,
    model_type: str = "resnet",
) -> GradCAM:
    """Get Grad-CAM instance with appropriate target layer for model type.

    Args:
        model: PyTorch model
        model_type: Type of model ('resnet', 'efficientnet', 'vit')

    Returns:
        GradCAM instance
    """
    if model_type == "resnet":
        # For ResNet, target the last conv layer before avgpool
        # Typically layer4 or the last block
        if hasattr(model, "backbone"):
            # Our custom ResNetClassifier
            backbone_modules = list(model.backbone.children())
            target_layer = backbone_modules[-2]  # Last layer before adaptive pooling
        else:
            # Standard ResNet
            target_layer = model.layer4[-1]

    elif model_type == "efficientnet":
        # For EfficientNet, target the last conv layer
        if hasattr(model, "backbone"):
            # Find last conv2d
            for module in reversed(list(model.backbone.modules())):
                if isinstance(module, nn.Conv2d):
                    target_layer = module
                    break
        else:
            raise ValueError("Could not find target layer for EfficientNet")

    elif model_type == "vit":
        # ViT doesn't have conv layers, Grad-CAM may not be suitable
        # Use attention rollout instead (not implemented here)
        raise NotImplementedError("Grad-CAM is not directly applicable to ViT. Use attention visualization instead.")

    else:
        # Auto-detect
        target_layer = None

    return GradCAM(model, target_layer=target_layer)
