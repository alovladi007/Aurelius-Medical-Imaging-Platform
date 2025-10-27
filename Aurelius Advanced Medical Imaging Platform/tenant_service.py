"""Tenant service for multi-tenancy operations.

This service handles tenant CRUD, user management, quota checking,
and usage tracking.
"""
import logging
from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
import secrets
import uuid

from app.models.tenants import (
    Tenant, TenantUser, TenantUserRole, TenantStatus, TenantTier,
    UsageRecord, TenantInvitation, TIER_PRICING
)
from app.models.user import User

logger = logging.getLogger(__name__)


class TenantService:
    """Service for tenant operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    # ============================================================================
    # TENANT CRUD
    # ============================================================================
    
    def create_tenant(
        self,
        name: str,
        slug: str,
        owner_id: uuid.UUID,
        tier: TenantTier = TenantTier.FREE,
        billing_email: Optional[str] = None
    ) -> Tenant:
        """
        Create a new tenant with an owner.
        
        Args:
            name: Tenant name
            slug: Unique slug for URL
            owner_id: User ID of the owner
            tier: Subscription tier
            billing_email: Billing email address
        
        Returns:
            Created tenant
        """
        # Get tier quotas
        pricing = TIER_PRICING[tier]
        
        # Create tenant
        tenant = Tenant(
            name=name,
            slug=slug,
            tier=tier,
            status=TenantStatus.TRIAL if tier == TenantTier.FREE else TenantStatus.ACTIVE,
            trial_ends_at=datetime.utcnow() + timedelta(days=14) if tier == TenantTier.FREE else None,
            quota_api_calls=pricing["api_calls"],
            quota_storage_gb=pricing["storage_gb"],
            quota_gpu_hours=pricing["gpu_hours"],
            quota_users=pricing["users"],
            quota_studies=pricing["studies"],
            billing_email=billing_email,
        )
        
        self.db.add(tenant)
        self.db.flush()
        
        # Add owner
        tenant_user = TenantUser(
            tenant_id=tenant.id,
            user_id=owner_id,
            role=TenantUserRole.OWNER,
            joined_at=datetime.utcnow()
        )
        self.db.add(tenant_user)
        
        self.db.commit()
        self.db.refresh(tenant)
        
        logger.info(f"Created tenant: {tenant.slug} (owner: {owner_id})")
        
        return tenant
    
    def get_tenant(self, tenant_id: uuid.UUID) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.db.query(Tenant).filter(Tenant.id == tenant_id).first()
    
    def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """Get tenant by slug."""
        return self.db.query(Tenant).filter(Tenant.slug == slug).first()
    
    def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        tier: Optional[TenantTier] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Tenant]:
        """List tenants with optional filters."""
        query = self.db.query(Tenant)
        
        if status:
            query = query.filter(Tenant.status == status)
        if tier:
            query = query.filter(Tenant.tier == tier)
        
        return query.offset(skip).limit(limit).all()
    
    def update_tenant(
        self,
        tenant_id: uuid.UUID,
        **kwargs
    ) -> Optional[Tenant]:
        """Update tenant attributes."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return None
        
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        tenant.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(tenant)
        
        logger.info(f"Updated tenant: {tenant.slug}")
        
        return tenant
    
    def delete_tenant(self, tenant_id: uuid.UUID) -> bool:
        """Delete a tenant (cascade deletes all associated resources)."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False
        
        slug = tenant.slug
        self.db.delete(tenant)
        self.db.commit()
        
        logger.warning(f"Deleted tenant: {slug}")
        
        return True
    
    # ============================================================================
    # USER MANAGEMENT
    # ============================================================================
    
    def add_user_to_tenant(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        role: TenantUserRole = TenantUserRole.MEMBER
    ) -> TenantUser:
        """Add a user to a tenant."""
        tenant_user = TenantUser(
            tenant_id=tenant_id,
            user_id=user_id,
            role=role,
            joined_at=datetime.utcnow()
        )
        
        self.db.add(tenant_user)
        self.db.commit()
        self.db.refresh(tenant_user)
        
        logger.info(f"Added user {user_id} to tenant {tenant_id} as {role}")
        
        return tenant_user
    
    def remove_user_from_tenant(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> bool:
        """Remove a user from a tenant."""
        tenant_user = self.db.query(TenantUser).filter(
            and_(
                TenantUser.tenant_id == tenant_id,
                TenantUser.user_id == user_id
            )
        ).first()
        
        if not tenant_user:
            return False
        
        # Don't allow removing the last owner
        if tenant_user.role == TenantUserRole.OWNER:
            owner_count = self.db.query(func.count(TenantUser.id)).filter(
                and_(
                    TenantUser.tenant_id == tenant_id,
                    TenantUser.role == TenantUserRole.OWNER
                )
            ).scalar()
            
            if owner_count <= 1:
                raise ValueError("Cannot remove the last owner from a tenant")
        
        self.db.delete(tenant_user)
        self.db.commit()
        
        logger.info(f"Removed user {user_id} from tenant {tenant_id}")
        
        return True
    
    def get_user_tenants(self, user_id: uuid.UUID) -> List[Tenant]:
        """Get all tenants a user belongs to."""
        return self.db.query(Tenant).join(TenantUser).filter(
            TenantUser.user_id == user_id,
            TenantUser.is_active == True
        ).all()
    
    def get_tenant_users(
        self,
        tenant_id: uuid.UUID,
        role: Optional[TenantUserRole] = None
    ) -> List[TenantUser]:
        """Get all users in a tenant."""
        query = self.db.query(TenantUser).filter(
            TenantUser.tenant_id == tenant_id,
            TenantUser.is_active == True
        )
        
        if role:
            query = query.filter(TenantUser.role == role)
        
        return query.all()
    
    def check_user_access(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        required_role: Optional[TenantUserRole] = None
    ) -> bool:
        """
        Check if a user has access to a tenant.
        
        Args:
            tenant_id: Tenant ID
            user_id: User ID
            required_role: Required role (if None, just check membership)
        
        Returns:
            True if user has access
        """
        tenant_user = self.db.query(TenantUser).filter(
            and_(
                TenantUser.tenant_id == tenant_id,
                TenantUser.user_id == user_id,
                TenantUser.is_active == True
            )
        ).first()
        
        if not tenant_user:
            return False
        
        if required_role is None:
            return True
        
        # Check role hierarchy: owner > admin > member > viewer
        role_hierarchy = {
            TenantUserRole.VIEWER: 0,
            TenantUserRole.MEMBER: 1,
            TenantUserRole.ADMIN: 2,
            TenantUserRole.OWNER: 3,
        }
        
        return role_hierarchy.get(tenant_user.role, 0) >= role_hierarchy.get(required_role, 0)
    
    # ============================================================================
    # INVITATIONS
    # ============================================================================
    
    def create_invitation(
        self,
        tenant_id: uuid.UUID,
        email: str,
        role: TenantUserRole,
        invited_by: uuid.UUID,
        expires_in_days: int = 7
    ) -> TenantInvitation:
        """Create an invitation to join a tenant."""
        invitation = TenantInvitation(
            tenant_id=tenant_id,
            email=email.lower(),
            role=role,
            invited_by=invited_by,
            token=secrets.token_urlsafe(32),
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days)
        )
        
        self.db.add(invitation)
        self.db.commit()
        self.db.refresh(invitation)
        
        logger.info(f"Created invitation for {email} to tenant {tenant_id}")
        
        return invitation
    
    def accept_invitation(
        self,
        token: str,
        user_id: uuid.UUID
    ) -> Optional[TenantUser]:
        """Accept an invitation and join the tenant."""
        invitation = self.db.query(TenantInvitation).filter(
            TenantInvitation.token == token
        ).first()
        
        if not invitation:
            return None
        
        if invitation.is_expired:
            raise ValueError("Invitation has expired")
        
        if invitation.is_accepted:
            raise ValueError("Invitation has already been accepted")
        
        # Add user to tenant
        tenant_user = self.add_user_to_tenant(
            tenant_id=invitation.tenant_id,
            user_id=user_id,
            role=invitation.role
        )
        
        # Mark invitation as accepted
        invitation.accepted_at = datetime.utcnow()
        self.db.commit()
        
        logger.info(f"User {user_id} accepted invitation to tenant {invitation.tenant_id}")
        
        return tenant_user
    
    # ============================================================================
    # USAGE TRACKING
    # ============================================================================
    
    def record_usage(
        self,
        tenant_id: uuid.UUID,
        api_calls: int = 0,
        storage_gb: float = 0,
        gpu_hours: float = 0
    ):
        """
        Record usage for a tenant (incremental).
        
        This should be called frequently to track usage in real-time.
        """
        # Get current month's usage record
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        
        usage = self.db.query(UsageRecord).filter(
            and_(
                UsageRecord.tenant_id == tenant_id,
                UsageRecord.period_start == period_start,
                UsageRecord.period_end == period_end
            )
        ).first()
        
        if not usage:
            # Create new usage record
            usage = UsageRecord(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=period_end,
                api_calls=api_calls,
                storage_gb=storage_gb,
                gpu_hours=gpu_hours
            )
            self.db.add(usage)
        else:
            # Update existing record
            usage.api_calls += api_calls
            usage.storage_gb += storage_gb
            usage.gpu_hours += gpu_hours
        
        self.db.commit()
    
    def get_current_usage(self, tenant_id: uuid.UUID) -> Optional[UsageRecord]:
        """Get current month's usage for a tenant."""
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        
        return self.db.query(UsageRecord).filter(
            and_(
                UsageRecord.tenant_id == tenant_id,
                UsageRecord.period_start == period_start,
                UsageRecord.period_end == period_end
            )
        ).first()
    
    def check_quota(
        self,
        tenant_id: uuid.UUID,
        resource: str,
        amount: float = 1
    ) -> tuple[bool, Optional[str]]:
        """
        Check if tenant has quota available for a resource.
        
        Args:
            tenant_id: Tenant ID
            resource: Resource type ("api_calls", "storage_gb", "gpu_hours", "users", "studies")
            amount: Amount to check
        
        Returns:
            (allowed, error_message)
        """
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return False, "Tenant not found"
        
        # Check tenant status
        if tenant.status != TenantStatus.ACTIVE and tenant.status != TenantStatus.TRIAL:
            return False, f"Tenant is {tenant.status}"
        
        # Check trial expiration
        if tenant.is_trial and tenant.trial_expired:
            return False, "Trial period has expired"
        
        # Get quota limit
        quota_field = f"quota_{resource}"
        if not hasattr(tenant, quota_field):
            return False, f"Unknown resource: {resource}"
        
        quota_limit = getattr(tenant, quota_field)
        
        # -1 means unlimited
        if quota_limit == -1:
            return True, None
        
        # Get current usage
        usage = self.get_current_usage(tenant_id)
        current_usage = 0
        
        if usage and hasattr(usage, resource):
            current_usage = getattr(usage, resource)
        
        # Check if would exceed
        if current_usage + amount > quota_limit:
            return False, f"{resource} quota exceeded: {current_usage}/{quota_limit}"
        
        return True, None
    
    # ============================================================================
    # SUBSCRIPTION MANAGEMENT
    # ============================================================================
    
    def upgrade_tenant(
        self,
        tenant_id: uuid.UUID,
        new_tier: TenantTier
    ) -> Tenant:
        """Upgrade (or downgrade) a tenant's subscription tier."""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError("Tenant not found")
        
        old_tier = tenant.tier
        
        # Update tier and quotas
        pricing = TIER_PRICING[new_tier]
        tenant.tier = new_tier
        tenant.quota_api_calls = pricing["api_calls"]
        tenant.quota_storage_gb = pricing["storage_gb"]
        tenant.quota_gpu_hours = pricing["gpu_hours"]
        tenant.quota_users = pricing["users"]
        tenant.quota_studies = pricing["studies"]
        
        # Update status
        if new_tier == TenantTier.FREE:
            tenant.status = TenantStatus.TRIAL
            tenant.trial_ends_at = datetime.utcnow() + timedelta(days=14)
        else:
            tenant.status = TenantStatus.ACTIVE
            tenant.subscription_started_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(tenant)
        
        logger.info(f"Upgraded tenant {tenant.slug} from {old_tier} to {new_tier}")
        
        return tenant
