"""FastAPI inference API for cancer classification."""

import base64
import io
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from ..config import load_config
from ..data.transforms import get_inference_transforms
from ..explainability.grad_cam import get_gradcam_for_model, save_gradcam_visualization
from ..models import create_model

# Initialize FastAPI app
app = FastAPI(
    title="Cancer Histopathology Classification API",
    description="API for cancer classification from histopathology images",
    version="0.1.0",
)

# Global model storage
MODEL = None
CONFIG = None
TRANSFORM = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = None


class PredictionRequest(BaseModel):
    """Prediction request model."""

    image_base64: str


class PredictionResponse(BaseModel):
    """Prediction response model."""

    predicted_class: int
    predicted_class_name: str
    probabilities: Dict[str, float]
    confidence: float


def load_model(checkpoint_path: str, config_path: Optional[str] = None):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
    """
    global MODEL, CONFIG, TRANSFORM, CLASS_NAMES

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # Load config
    if config_path:
        CONFIG = load_config(config_path).to_dict()
    elif "config" in checkpoint:
        CONFIG = checkpoint["config"]
    else:
        raise ValueError("Config not found in checkpoint and config_path not provided")

    # Create model
    model_type = CONFIG.get("model", {}).get("backbone", "resnet50")

    if "resnet" in model_type:
        MODEL = create_model(CONFIG, model_type="resnet")
    elif "efficientnet" in model_type:
        MODEL = create_model(CONFIG, model_type="efficientnet")
    elif "vit" in model_type:
        MODEL = create_model(CONFIG, model_type="vit")
    else:
        MODEL = create_model(CONFIG, model_type="resnet")

    # Load weights
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()

    # Get class names
    CLASS_NAMES = CONFIG.get("classes", {}).get("class_names", None)
    if CLASS_NAMES is None:
        num_classes = CONFIG.get("model", {}).get("num_classes", 2)
        CLASS_NAMES = [f"Class {i}" for i in range(num_classes)]

    # Get transform
    img_size = CONFIG.get("image_settings", {}).get("target_size", 224)
    TRANSFORM = get_inference_transforms(img_size=img_size)

    print(f"Model loaded successfully on {DEVICE}")
    print(f"Class names: {CLASS_NAMES}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    # This would be configured with environment variables or config file
    # For now, it's a placeholder
    pass


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Cancer Histopathology Classification API",
        "status": "running",
        "model_loaded": MODEL is not None,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict cancer classification from uploaded image.

    Args:
        file: Uploaded image file

    Returns:
        Prediction results
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Apply transforms
        transformed = TRANSFORM(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = MODEL(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        # Prepare response
        probabilities = {
            CLASS_NAMES[i]: float(probs[0, i])
            for i in range(len(CLASS_NAMES))
        }

        return PredictionResponse(
            predicted_class=pred_class,
            predicted_class_name=CLASS_NAMES[pred_class],
            probabilities=probabilities,
            confidence=confidence,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_with_explainability")
async def predict_with_explainability(file: UploadFile = File(...)):
    """Predict with Grad-CAM visualization.

    Args:
        file: Uploaded image file

    Returns:
        Prediction results with base64-encoded Grad-CAM overlay
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Apply transforms
        transformed = TRANSFORM(image=image_np)
        image_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = MODEL(image_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        # Generate Grad-CAM
        model_type = "resnet" if "resnet" in CONFIG.get("model", {}).get("backbone", "") else "efficientnet"

        try:
            grad_cam = get_gradcam_for_model(MODEL, model_type=model_type)
            cam = grad_cam(image_tensor, target_class=pred_class)

            # Overlay on image
            from ..explainability.grad_cam import apply_colormap_on_image

            overlayed = apply_colormap_on_image(image_np, cam, alpha=0.4)

            # Convert to base64
            img_pil = Image.fromarray(overlayed)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
            img_base64 = None

        # Prepare response
        probabilities = {
            CLASS_NAMES[i]: float(probs[0, i])
            for i in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": pred_class,
            "predicted_class_name": CLASS_NAMES[pred_class],
            "probabilities": probabilities,
            "confidence": confidence,
            "gradcam_overlay_base64": img_base64,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/load_model")
async def load_model_endpoint(checkpoint_path: str, config_path: Optional[str] = None):
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        config_path: Optional path to config

    Returns:
        Status message
    """
    try:
        load_model(checkpoint_path, config_path)
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


def run_server(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Run FastAPI server.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        host: Host address
        port: Port number
    """
    load_model(checkpoint_path, config_path)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference_api.py <checkpoint_path> [config_path]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    config_path = sys.argv[2] if len(sys.argv) > 2 else None

    run_server(checkpoint_path, config_path)
