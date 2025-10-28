"""Machine learning endpoints."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Any
from app.core.auth import get_current_user, User, require_any_role
import httpx
from app.core.config import settings

router = APIRouter()


class PredictionRequest(BaseModel):
    """Prediction request model."""
    model_name: str
    model_version: str = "latest"
    input_data: dict[str, Any]


class PredictionResponse(BaseModel):
    """Prediction response model."""
    prediction_id: str
    results: dict[str, Any]
    confidence: Optional[float]
    inference_time_ms: int


class ModelInfo(BaseModel):
    """Model information model."""
    model_name: str
    model_version: str
    model_type: str
    framework: str
    status: str


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    user: User = Depends(require_any_role(["clinician", "researcher", "ml-engineer"]))
):
    """
    Run inference on a study or image.
    
    Args:
        request: Prediction request
        user: Current user
        
    Returns:
        PredictionResponse: Prediction results
    """
    # Proxy to ML service
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ML_SVC_URL}/predict",
            json=request.dict(),
            headers={"X-User-ID": user.sub},
            timeout=60.0
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Prediction failed"
            )
        
        return response.json()


@router.post("/predict/async")
async def predict_async(
    request: PredictionRequest,
    user: User = Depends(require_any_role(["clinician", "researcher", "ml-engineer"]))
):
    """
    Run asynchronous inference (for large images/WSI).
    
    Args:
        request: Prediction request
        user: Current user
        
    Returns:
        Job ID for tracking
    """
    # Create background job
    return {
        "job_id": "mock-job-id",
        "status": "pending",
        "message": "Inference job queued"
    }


@router.get("/models", response_model=list[ModelInfo])
async def list_models(
    user: User = Depends(get_current_user)
):
    """
    List available ML models.
    
    Args:
        user: Current user
        
    Returns:
        List of available models
    """
    # Mock implementation
    return [
        ModelInfo(
            model_name="chest-xray-classifier",
            model_version="1.0.0",
            model_type="classification",
            framework="pytorch",
            status="active"
        ),
        ModelInfo(
            model_name="tumor-segmentation",
            model_version="2.1.0",
            model_type="segmentation",
            framework="pytorch",
            status="active"
        )
    ]


@router.get("/models/{model_name}")
async def get_model_info(
    model_name: str,
    user: User = Depends(get_current_user)
):
    """
    Get detailed model information.
    
    Args:
        model_name: Model name
        user: Current user
        
    Returns:
        Model details
    """
    # Mock implementation
    return {
        "model_name": model_name,
        "versions": ["1.0.0", "1.1.0"],
        "latest_version": "1.1.0",
        "description": "Example model",
        "metrics": {
            "accuracy": 0.95,
            "auc": 0.97
        }
    }


@router.get("/predictions/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get prediction result by ID.
    
    Args:
        prediction_id: Prediction ID
        user: Current user
        
    Returns:
        Prediction details
    """
    # Mock implementation
    return {
        "prediction_id": prediction_id,
        "status": "completed",
        "results": {}
    }
