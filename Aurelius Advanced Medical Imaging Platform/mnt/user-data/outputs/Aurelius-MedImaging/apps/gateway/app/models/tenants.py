"""Tenant models for multi-tenancy support.

This module defines the Tenant, TenantUser, and related models for 
implementing multi-tenant architecture with row-level security.
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Enum as SQLEnum,
    ForeignKey, Text, JSON, Numeric, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.db.base_class import Base


class TenantTier(str, Enum):
    """Tenant subscription tiers."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class TenantStatus(str, Enum):
    """Tenant account status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    CANCELLED = "cancelled"


class BillingCycle(str, Enum):
    """Billing cycle options."""
    MONTHLY = "monthly"
    ANNUAL = "annual"


class Tenant(Base):
    """
    Tenant (organization) model.
    
    Represents a customer organization using the platform.
    All resources (studies, users, etc.) belong to a tenant.
    """
    __tablename__ = "tenants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic Information
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    domain = Column(String(255), nullable=True)  # Custom domain
    
    # Subscription
    tier = Column(SQLEnum(TenantTier), nullable=False, default=TenantTier.FREE)
    status = Column(SQLEnum(TenantStatus), nullable=False, default=TenantStatus.TRIAL)
    billing_cycle = Column(SQLEnum(BillingCycle), default=BillingCycle.MONTHLY)
    
    # Quotas (monthly limits)
    quota_api_calls = Column(Integer, default=10000)
    quota_storage_gb = Column(Integer, default=10)
    quota_gpu_hours = Column(Integer, default=5)
    quota_users = Column(Integer, default=5)
    quota_studies = Column(Integer, default=1000)
    
    # Billing
    stripe_customer_id = Column(String(255), unique=True, nullable=True)
    stripe_subscription_id = Column(String(255), unique=True, nullable=True)
    billing_email = Column(String(255), nullable=True)
    
    # Dates
    trial_ends_at = Column(DateTime, nullable=True)
    subscription_started_at = Column(DateTime, nullable=True)
    subscription_ends_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Metadata
    settings = Column(JSON, default={})
    metadata = Column(JSON, default={})
    
    # Relationships
    users = relationship("TenantUser", back_populates="tenant", cascade="all, delete-orphan")
    usage_records = relationship("UsageRecord", back_populates="tenant", cascade="all, delete-orphan")
    invoices = relationship("Invoice", back_populates="tenant", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_tenant_status', 'status'),
        Index('idx_tenant_tier', 'tier'),
    )
    
    def __repr__(self):
        return f"<Tenant {self.name} ({self.slug})>"
    
    @property
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.status == TenantStatus.ACTIVE
    
    @property
    def is_trial(self) -> bool:
        """Check if tenant is in trial."""
        return self.status == TenantStatus.TRIAL
    
    @property
    def trial_expired(self) -> bool:
        """Check if trial has expired."""
        if not self.trial_ends_at:
            return False
        return datetime.utcnow() > self.trial_ends_at


class TenantUserRole(str, Enum):
    """Roles for tenant users."""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class TenantUser(Base):
    """
    Association between users and tenants.
    
    A user can belong to multiple tenants with different roles.
    """
    __tablename__ = "tenant_users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    role = Column(SQLEnum(TenantUserRole), nullable=False, default=TenantUserRole.MEMBER)
    is_active = Column(Boolean, default=True, nullable=False)
    
    invited_at = Column(DateTime, default=datetime.utcnow)
    joined_at = Column(DateTime, nullable=True)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="users")
    user = relationship("User")
    
    __table_args__ = (
        UniqueConstraint('tenant_id', 'user_id', name='uq_tenant_user'),
        Index('idx_tenant_user_tenant', 'tenant_id'),
        Index('idx_tenant_user_user', 'user_id'),
    )
    
    def __repr__(self):
        return f"<TenantUser tenant={self.tenant_id} user={self.user_id} role={self.role}>"


class UsageRecord(Base):
    """
    Usage tracking for billing.
    
    Tracks resource usage per tenant for billing purposes.
    """
    __tablename__ = "usage_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Usage metrics
    api_calls = Column(Integer, default=0)
    storage_gb = Column(Numeric(10, 2), default=0)
    gpu_hours = Column(Numeric(10, 2), default=0)
    active_users = Column(Integer, default=0)
    total_studies = Column(Integer, default=0)
    
    # Costs (in cents)
    cost_api_calls = Column(Integer, default=0)
    cost_storage = Column(Integer, default=0)
    cost_gpu = Column(Integer, default=0)
    cost_total = Column(Integer, default=0)
    
    # Metadata
    breakdown = Column(JSON, default={})  # Detailed breakdown
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    tenant = relationship("Tenant", back_populates="usage_records")
    
    __table_args__ = (
        Index('idx_usage_tenant_period', 'tenant_id', 'period_start', 'period_end'),
        Index('idx_usage_period', 'period_start', 'period_end'),
    )
    
    def __repr__(self):
        return f"<UsageRecord tenant={self.tenant_id} period={self.period_start}>"


class Invoice(Base):
    """
    Invoice for billing.
    
    Generated monthly for each tenant based on usage.
    """
    __tablename__ = "invoices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    
    # Invoice details
    invoice_number = Column(String(50), unique=True, nullable=False)
    amount_cents = Column(Integer, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    
    # Status
    status = Column(String(50), default="draft")  # draft, open, paid, void, uncollectible
    
    # Dates
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    due_date = Column(DateTime, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Stripe
    stripe_invoice_id = Column(String(255), unique=True, nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)
    
    # Invoice data
    line_items = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    
    # Relationships
    tenant = relationship("Tenant", back_populates="invoices")
    
    __table_args__ = (
        Index('idx_invoice_tenant', 'tenant_id'),
        Index('idx_invoice_period', 'period_start', 'period_end'),
        Index('idx_invoice_status', 'status'),
    )
    
    def __repr__(self):
        return f"<Invoice {self.invoice_number} tenant={self.tenant_id}>"


class TenantInvitation(Base):
    """
    Invitation to join a tenant.
    
    Used to invite new users to join an organization.
    """
    __tablename__ = "tenant_invitations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    email = Column(String(255), nullable=False)
    role = Column(SQLEnum(TenantUserRole), nullable=False, default=TenantUserRole.MEMBER)
    
    token = Column(String(255), unique=True, nullable=False, index=True)
    
    invited_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    accepted_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('idx_invitation_tenant', 'tenant_id'),
        Index('idx_invitation_email', 'email'),
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if invitation has expired."""
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_accepted(self) -> bool:
        """Check if invitation has been accepted."""
        return self.accepted_at is not None
    
    def __repr__(self):
        return f"<TenantInvitation {self.email} → tenant={self.tenant_id}>"


# Pricing configuration (can be moved to config/database)
TIER_PRICING = {
    TenantTier.FREE: {
        "monthly_price": 0,
        "annual_price": 0,
        "api_calls": 10000,
        "storage_gb": 10,
        "gpu_hours": 5,
        "users": 5,
        "studies": 1000,
    },
    TenantTier.STARTER: {
        "monthly_price": 49_00,  # $49 in cents
        "annual_price": 490_00,  # $490/year ($40.83/month)
        "api_calls": 100000,
        "storage_gb": 100,
        "gpu_hours": 50,
        "users": 10,
        "studies": 10000,
    },
    TenantTier.PROFESSIONAL: {
        "monthly_price": 199_00,  # $199
        "annual_price": 1990_00,  # $1990/year ($165.83/month)
        "api_calls": 1000000,
        "storage_gb": 1000,
        "gpu_hours": 200,
        "users": 50,
        "studies": 100000,
    },
    TenantTier.ENTERPRISE: {
        "monthly_price": 999_00,  # $999
        "annual_price": 9990_00,  # $9990/year ($832.50/month)
        "api_calls": -1,  # unlimited
        "storage_gb": -1,
        "gpu_hours": -1,
        "users": -1,
        "studies": -1,
    },
}

# Usage costs (per unit, in cents)
USAGE_COSTS = {
    "api_call": 0.1,  # $0.001 per call
    "storage_gb": 2.3,  # $0.023 per GB-month
    "gpu_hour": 250,  # $2.50 per hour
}
