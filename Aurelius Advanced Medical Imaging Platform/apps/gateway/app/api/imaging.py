"""Imaging service proxy endpoints."""
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from app.core.auth import get_current_user, User
import httpx
from app.core.config import settings

router = APIRouter()


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user)
):
    """
    Upload medical image file.
    
    Args:
        file: Image file to upload
        user: Current user
        
    Returns:
        Upload result with job ID
    """
    # Proxy to imaging service
    async with httpx.AsyncClient() as client:
        files = {"file": (file.filename, await file.read(), file.content_type)}
        response = await client.post(
            f"{settings.IMAGING_SVC_URL}/ingest/file",
            files=files,
            headers={"X-User-ID": user.sub},
            timeout=300.0
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Upload failed"
            )
        
        return response.json()


@router.post("/dicomweb/stow")
async def stow_dicom(
    user: User = Depends(get_current_user)
):
    """
    STOW-RS endpoint for DICOM upload.
    
    Args:
        user: Current user
        
    Returns:
        STOW response
    """
    # Proxy to Orthanc
    return {"message": "STOW-RS endpoint - to be implemented"}


@router.get("/dicomweb/studies")
async def qido_studies(
    user: User = Depends(get_current_user)
):
    """
    QIDO-RS endpoint for study search.
    
    Args:
        user: Current user
        
    Returns:
        List of studies
    """
    # Proxy to Orthanc
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.ORTHANC_URL}/dicom-web/studies",
            auth=(settings.ORTHANC_USERNAME, settings.ORTHANC_PASSWORD)
        )
        
        return response.json()


@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    user: User = Depends(get_current_user)
):
    """
    Get status of background job.
    
    Args:
        job_id: Job ID
        user: Current user
        
    Returns:
        Job status
    """
    # Mock implementation
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100
    }
