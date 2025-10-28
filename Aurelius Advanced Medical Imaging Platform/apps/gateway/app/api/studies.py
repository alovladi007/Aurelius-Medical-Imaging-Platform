"""Study management endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from pydantic import BaseModel
from datetime import date
from app.core.database import get_db
from app.core.auth import get_current_user, User, require_any_role

router = APIRouter()


class StudyResponse(BaseModel):
    """Study response model."""
    id: str
    study_instance_uid: str
    patient_id: str
    accession_number: Optional[str]
    study_date: Optional[date]
    study_description: Optional[str]
    modality: Optional[str]
    number_of_series: int
    number_of_instances: int
    
    class Config:
        from_attributes = True


class StudyListResponse(BaseModel):
    """Study list response model."""
    total: int
    page: int
    page_size: int
    studies: list[StudyResponse]


@router.get("", response_model=StudyListResponse)
async def list_studies(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    patient_id: Optional[str] = None,
    modality: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    user: User = Depends(require_any_role(["clinician", "researcher", "admin"]))
):
    """
    List studies with filtering and pagination.
    
    Args:
        page: Page number
        page_size: Items per page
        patient_id: Filter by patient ID
        modality: Filter by modality
        date_from: Filter studies from this date
        date_to: Filter studies to this date
        search: Search in study description
        db: Database session
        user: Current user
        
    Returns:
        StudyListResponse: Paginated list of studies
    """
    # Build query (mock implementation - should query actual studies table)
    # For now, return empty list as placeholder
    
    return StudyListResponse(
        total=0,
        page=page,
        page_size=page_size,
        studies=[]
    )


@router.get("/{study_id}", response_model=StudyResponse)
async def get_study(
    study_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Get study by ID.
    
    Args:
        study_id: Study ID
        db: Database session
        user: Current user
        
    Returns:
        StudyResponse: Study details
        
    Raises:
        HTTPException: If study not found
    """
    # Mock implementation
    raise HTTPException(status_code=404, detail="Study not found")


@router.delete("/{study_id}")
async def delete_study(
    study_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_any_role(["admin", "clinician"]))
):
    """
    Delete study (soft delete).
    
    Args:
        study_id: Study ID
        db: Database session
        user: Current user
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If study not found or user lacks permission
    """
    # Mock implementation
    return {"message": f"Study {study_id} deleted"}


@router.post("/{study_id}/share")
async def share_study(
    study_id: str,
    recipient_email: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Share study with another user.
    
    Args:
        study_id: Study ID
        recipient_email: Email of recipient
        db: Database session
        user: Current user
        
    Returns:
        Success message
    """
    # Mock implementation
    return {"message": f"Study {study_id} shared with {recipient_email}"}
