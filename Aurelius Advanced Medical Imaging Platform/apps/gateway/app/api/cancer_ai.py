"""Cancer AI endpoints - Advanced multimodal cancer detection."""
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from app.core.auth import get_current_user, User, require_any_role
import httpx
from app.core.config import settings

router = APIRouter()


class CancerPredictionResponse(BaseModel):
    """Cancer prediction response model."""
    cancer_type: str
    risk_score: float
    confidence: float
    uncertainty: float
    recommendations: List[str]
    all_probabilities: Optional[Dict[str, float]] = None


class ModelInfoResponse(BaseModel):
    """Cancer AI model information."""
    model_path: str
    class_names: List[str]
    input_shape: List[int]
    framework: str


@router.get("/health")
async def cancer_ai_health():
    """
    Check Cancer AI service health.

    Returns:
        Health status
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.CANCER_AI_SVC_URL}/health",
                timeout=5.0
            )
            return response.json()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cancer AI service unavailable: {str(e)}"
        )


@router.post("/predict", response_model=CancerPredictionResponse)
async def predict_cancer(
    image: UploadFile = File(...),
    clinical_notes: Optional[str] = Form(""),
    patient_age: Optional[int] = Form(0),
    patient_gender: Optional[str] = Form(""),
    smoking_history: Optional[bool] = Form(False),
    family_history: Optional[bool] = Form(False),
    user: User = Depends(require_any_role(["clinician", "radiologist", "pathologist", "ml-engineer"]))
):
    """
    Predict cancer from medical image and clinical data.

    This endpoint integrates the advanced multimodal cancer detection AI.
    Supports: Lung, Breast, Prostate, and Colorectal cancer detection.

    Args:
        image: Medical image file (DICOM, PNG, JPG, etc.)
        clinical_notes: Clinical notes/observations
        patient_age: Patient age
        patient_gender: Patient gender
        smoking_history: Smoking history (true/false)
        family_history: Family history of cancer (true/false)
        user: Current authenticated user

    Returns:
        CancerPredictionResponse: Cancer prediction with recommendations
    """
    try:
        # Read image file
        image_content = await image.read()

        # Prepare multipart form data
        files = {"image": (image.filename, image_content, image.content_type)}
        data = {
            "clinical_notes": clinical_notes or "",
            "patient_age": patient_age or 0,
            "patient_gender": patient_gender or "",
            "smoking_history": smoking_history,
            "family_history": family_history
        }

        # Proxy to Cancer AI service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.CANCER_AI_SVC_URL}/predict",
                files=files,
                data=data,
                headers={"X-User-ID": user.sub},
                timeout=120.0  # Longer timeout for inference
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Cancer prediction failed: {response.text}"
                )

            return response.json()

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Cancer AI prediction timed out"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cancer AI service error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


@router.post("/predict/batch")
async def batch_predict_cancer(
    images: List[UploadFile] = File(...),
    user: User = Depends(require_any_role(["clinician", "radiologist", "pathologist", "ml-engineer", "researcher"]))
):
    """
    Batch cancer prediction for multiple images.

    Args:
        images: List of medical image files
        user: Current authenticated user

    Returns:
        List of predictions with results or errors
    """
    try:
        # Prepare files for batch upload
        files_list = []
        for img in images:
            content = await img.read()
            files_list.append(("images", (img.filename, content, img.content_type)))

        # Proxy to Cancer AI service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.CANCER_AI_SVC_URL}/batch_predict",
                files=files_list,
                headers={"X-User-ID": user.sub},
                timeout=300.0  # 5 minutes for batch
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Batch prediction failed"
                )

            return response.json()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_cancer_model_info(
    user: User = Depends(get_current_user)
):
    """
    Get Cancer AI model information.

    Args:
        user: Current authenticated user

    Returns:
        Model information including architecture and supported classes
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.CANCER_AI_SVC_URL}/model_info",
                timeout=5.0
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to retrieve model info"
                )

            return response.json()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving model info: {str(e)}"
        )


@router.post("/predict/dicom")
async def predict_from_dicom(
    study_instance_uid: str = Form(...),
    series_instance_uid: Optional[str] = Form(None),
    clinical_notes: Optional[str] = Form(""),
    patient_age: Optional[int] = Form(0),
    smoking_history: Optional[bool] = Form(False),
    family_history: Optional[bool] = Form(False),
    user: User = Depends(require_any_role(["clinician", "radiologist"]))
):
    """
    Predict cancer from DICOM study/series already in Orthanc.

    This endpoint fetches DICOM images from Orthanc and sends them
    to the Cancer AI service for prediction.

    Args:
        study_instance_uid: DICOM Study Instance UID
        series_instance_uid: Optional DICOM Series Instance UID
        clinical_notes: Clinical notes
        patient_age: Patient age
        smoking_history: Smoking history
        family_history: Family history
        user: Current authenticated user

    Returns:
        Cancer prediction result
    """
    # TODO: Implement DICOM retrieval from Orthanc and conversion
    # This will be implemented when we create the DICOM pipeline
    return {
        "message": "DICOM prediction endpoint - to be implemented",
        "study_uid": study_instance_uid,
        "series_uid": series_instance_uid,
        "status": "pending"
    }


@router.get("/statistics")
async def get_cancer_ai_statistics(
    user: User = Depends(require_any_role(["admin", "ml-engineer"]))
):
    """
    Get Cancer AI usage statistics and performance metrics.

    Args:
        user: Current authenticated user (admin or ml-engineer)

    Returns:
        Statistics and metrics
    """
    # TODO: Implement database queries for statistics
    return {
        "total_predictions": 0,
        "predictions_by_type": {
            "No Cancer": 0,
            "Lung Cancer": 0,
            "Breast Cancer": 0,
            "Prostate Cancer": 0,
            "Colorectal Cancer": 0
        },
        "average_confidence": 0.0,
        "average_inference_time_ms": 0
    }
