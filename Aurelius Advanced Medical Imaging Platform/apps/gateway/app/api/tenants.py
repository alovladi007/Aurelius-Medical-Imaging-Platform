"""Tenant management API endpoints.

Provides endpoints for:
- Tenant CRUD
- User management
- Invitations
- Usage and billing
"""
from typing import List, Optional
from datetime import datetime, timedelta
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from app.db.session import get_db
from app.models.tenants import (
    Tenant, TenantTier, TenantStatus, TenantUserRole, BillingCycle
)
from app.services.tenant_service import TenantService
from app.services.billing_service import BillingService
from app.api.deps import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tenants", tags=["tenants"])


# ============================================================================
# REQUEST/RESPONSE SCHEMAS
# ============================================================================

class TenantCreate(BaseModel):
    """Request to create a new tenant."""
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=100, regex="^[a-z0-9-]+$")
    tier: TenantTier = TenantTier.FREE
    billing_email: Optional[EmailStr] = None


class TenantUpdate(BaseModel):
    """Request to update tenant."""
    name: Optional[str] = None
    domain: Optional[str] = None
    billing_email: Optional[EmailStr] = None
    settings: Optional[dict] = None


class TenantResponse(BaseModel):
    """Tenant response."""
    id: uuid.UUID
    name: str
    slug: str
    tier: TenantTier
    status: TenantStatus
    quota_api_calls: int
    quota_storage_gb: int
    quota_gpu_hours: int
    quota_users: int
    quota_studies: int
    created_at: datetime
    trial_ends_at: Optional[datetime]
    
    class Config:
        orm_mode = True


class UserInviteRequest(BaseModel):
    """Request to invite a user."""
    email: EmailStr
    role: TenantUserRole = TenantUserRole.MEMBER


class AcceptInviteRequest(BaseModel):
    """Request to accept an invitation."""
    token: str


class UsageResponse(BaseModel):
    """Current usage response."""
    tenant_id: uuid.UUID
    period_start: datetime
    period_end: datetime
    api_calls: int
    storage_gb: float
    gpu_hours: float
    quota_api_calls: int
    quota_storage_gb: int
    quota_gpu_hours: int
    percentage_used: dict


class SubscriptionRequest(BaseModel):
    """Request to create/update subscription."""
    tier: TenantTier
    billing_cycle: BillingCycle
    payment_method_id: str


# ============================================================================
# TENANT CRUD
# ============================================================================

@router.post("", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
def create_tenant(
    tenant_data: TenantCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new tenant.
    
    The current user will be set as the owner.
    """
    service = TenantService(db)
    
    # Check if slug is already taken
    existing = service.get_tenant_by_slug(tenant_data.slug)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Slug already taken"
        )
    
    # Create tenant
    tenant = service.create_tenant(
        name=tenant_data.name,
        slug=tenant_data.slug,
        owner_id=current_user.id,
        tier=tenant_data.tier,
        billing_email=tenant_data.billing_email or current_user.email
    )
    
    return tenant


@router.get("", response_model=List[TenantResponse])
def list_tenants(
    status: Optional[TenantStatus] = None,
    tier: Optional[TenantTier] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List all tenants the current user has access to.
    """
    service = TenantService(db)
    
    # Get user's tenants
    tenants = service.get_user_tenants(current_user.id)
    
    # Apply filters
    if status:
        tenants = [t for t in tenants if t.status == status]
    if tier:
        tenants = [t for t in tenants if t.tier == tier]
    
    return tenants[skip:skip+limit]


@router.get("/{tenant_id}", response_model=TenantResponse)
def get_tenant(
    tenant_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get tenant by ID."""
    service = TenantService(db)
    
    # Check access
    if not service.check_user_access(tenant_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No access to this tenant"
        )
    
    tenant = service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    return tenant


@router.patch("/{tenant_id}", response_model=TenantResponse)
def update_tenant(
    tenant_id: uuid.UUID,
    updates: TenantUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update tenant details.
    
    Requires ADMIN or OWNER role.
    """
    service = TenantService(db)
    
    # Check access (requires admin)
    if not service.check_user_access(tenant_id, current_user.id, TenantUserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Update tenant
    update_data = updates.dict(exclude_unset=True)
    tenant = service.update_tenant(tenant_id, **update_data)
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    return tenant


@router.delete("/{tenant_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_tenant(
    tenant_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a tenant.
    
    Requires OWNER role. This will CASCADE delete all resources.
    """
    service = TenantService(db)
    
    # Check access (requires owner)
    if not service.check_user_access(tenant_id, current_user.id, TenantUserRole.OWNER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners can delete tenants"
        )
    
    success = service.delete_tenant(tenant_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )


# ============================================================================
# USER MANAGEMENT
# ============================================================================

@router.get("/{tenant_id}/users")
def list_tenant_users(
    tenant_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all users in a tenant."""
    service = TenantService(db)
    
    # Check access
    if not service.check_user_access(tenant_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No access to this tenant"
        )
    
    users = service.get_tenant_users(tenant_id)
    
    return [
        {
            "user_id": str(tu.user_id),
            "email": tu.user.email,
            "role": tu.role,
            "joined_at": tu.joined_at,
        }
        for tu in users
    ]


@router.post("/{tenant_id}/invitations")
def invite_user(
    tenant_id: uuid.UUID,
    invite_data: UserInviteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Invite a user to join the tenant.
    
    Requires ADMIN or OWNER role.
    """
    service = TenantService(db)
    
    # Check access (requires admin)
    if not service.check_user_access(tenant_id, current_user.id, TenantUserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to invite users"
        )
    
    # Create invitation
    invitation = service.create_invitation(
        tenant_id=tenant_id,
        email=invite_data.email,
        role=invite_data.role,
        invited_by=current_user.id
    )
    
    # TODO: Send invitation email
    logger.info(f"Invitation created: {invitation.token}")
    
    return {
        "invitation_id": str(invitation.id),
        "email": invitation.email,
        "token": invitation.token,
        "expires_at": invitation.expires_at,
        "invitation_link": f"/invitations/accept?token={invitation.token}"
    }


@router.post("/invitations/accept")
def accept_invitation(
    accept_data: AcceptInviteRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Accept an invitation to join a tenant."""
    service = TenantService(db)
    
    try:
        tenant_user = service.accept_invitation(
            token=accept_data.token,
            user_id=current_user.id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    if not tenant_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found"
        )
    
    return {
        "message": "Invitation accepted",
        "tenant_id": str(tenant_user.tenant_id),
        "role": tenant_user.role
    }


@router.delete("/{tenant_id}/users/{user_id}")
def remove_user(
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Remove a user from the tenant.
    
    Requires ADMIN or OWNER role.
    """
    service = TenantService(db)
    
    # Check access (requires admin)
    if not service.check_user_access(tenant_id, current_user.id, TenantUserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    try:
        success = service.remove_user_from_tenant(tenant_id, user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in tenant"
        )
    
    return {"message": "User removed from tenant"}


# ============================================================================
# USAGE & QUOTAS
# ============================================================================

@router.get("/{tenant_id}/usage", response_model=UsageResponse)
def get_usage(
    tenant_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get current usage for the tenant."""
    service = TenantService(db)
    
    # Check access
    if not service.check_user_access(tenant_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No access to this tenant"
        )
    
    tenant = service.get_tenant(tenant_id)
    usage = service.get_current_usage(tenant_id)
    
    if not usage:
        # No usage yet this month
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        
        return UsageResponse(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            api_calls=0,
            storage_gb=0,
            gpu_hours=0,
            quota_api_calls=tenant.quota_api_calls,
            quota_storage_gb=tenant.quota_storage_gb,
            quota_gpu_hours=tenant.quota_gpu_hours,
            percentage_used={
                "api_calls": 0,
                "storage_gb": 0,
                "gpu_hours": 0
            }
        )
    
    # Calculate percentage used
    def calc_percentage(used, quota):
        if quota == -1:  # Unlimited
            return 0
        return (used / quota * 100) if quota > 0 else 0
    
    return UsageResponse(
        tenant_id=tenant_id,
        period_start=usage.period_start,
        period_end=usage.period_end,
        api_calls=usage.api_calls,
        storage_gb=float(usage.storage_gb),
        gpu_hours=float(usage.gpu_hours),
        quota_api_calls=tenant.quota_api_calls,
        quota_storage_gb=tenant.quota_storage_gb,
        quota_gpu_hours=tenant.quota_gpu_hours,
        percentage_used={
            "api_calls": calc_percentage(usage.api_calls, tenant.quota_api_calls),
            "storage_gb": calc_percentage(float(usage.storage_gb), tenant.quota_storage_gb),
            "gpu_hours": calc_percentage(float(usage.gpu_hours), tenant.quota_gpu_hours)
        }
    )


# ============================================================================
# BILLING & SUBSCRIPTIONS
# ============================================================================

@router.post("/{tenant_id}/subscribe")
def create_subscription(
    tenant_id: uuid.UUID,
    subscription_data: SubscriptionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Subscribe to a paid plan.
    
    Requires OWNER role.
    """
    tenant_service = TenantService(db)
    
    # Check access (requires owner)
    if not tenant_service.check_user_access(tenant_id, current_user.id, TenantUserRole.OWNER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners can manage subscriptions"
        )
    
    tenant = tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    # Create subscription via Stripe
    billing_service = BillingService(db)
    
    try:
        result = billing_service.create_subscription(
            tenant=tenant,
            tier=subscription_data.tier,
            billing_cycle=subscription_data.billing_cycle,
            payment_method_id=subscription_data.payment_method_id
        )
    except Exception as e:
        logger.error(f"Error creating subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create subscription: {str(e)}"
        )
    
    return result


@router.post("/{tenant_id}/cancel-subscription")
def cancel_subscription(
    tenant_id: uuid.UUID,
    immediate: bool = Query(False),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Cancel subscription.
    
    Requires OWNER role.
    """
    tenant_service = TenantService(db)
    
    # Check access (requires owner)
    if not tenant_service.check_user_access(tenant_id, current_user.id, TenantUserRole.OWNER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only owners can cancel subscriptions"
        )
    
    tenant = tenant_service.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    billing_service = BillingService(db)
    
    try:
        result = billing_service.cancel_subscription(tenant, immediate=immediate)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error cancelling subscription: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to cancel subscription: {str(e)}"
        )
    
    return result


@router.get("/{tenant_id}/billing")
def get_billing_summary(
    tenant_id: uuid.UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get billing summary for the tenant."""
    tenant_service = TenantService(db)
    
    # Check access (requires admin or owner)
    if not tenant_service.check_user_access(tenant_id, current_user.id, TenantUserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    billing_service = BillingService(db)
    summary = billing_service.get_tenant_billing_summary(tenant_id)
    
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    return summary
