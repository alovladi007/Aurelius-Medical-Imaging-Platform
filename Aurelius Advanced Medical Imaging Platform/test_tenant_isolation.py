"""Tests for tenant isolation and multi-tenancy.

These tests verify that:
1. Tenants cannot access each other's data
2. Users can only access authorized tenants
3. Quotas are enforced properly
4. Tenant context is isolated correctly
"""
import pytest
from datetime import datetime, timedelta
import uuid

from sqlalchemy.orm import Session
from fastapi import status
from fastapi.testclient import TestClient

from app.main import app
from app.models.tenants import (
    Tenant, TenantUser, TenantUserRole, TenantTier, TenantStatus, UsageRecord
)
from app.models.user import User
from app.models.study import Study
from app.services.tenant_service import TenantService
from app.db.session import SessionLocal


@pytest.fixture
def db_session():
    """Create a database session for tests."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def tenant_service(db_session):
    """Create tenant service."""
    return TenantService(db_session)


@pytest.fixture
def create_test_tenant(db_session, tenant_service):
    """Factory to create test tenants."""
    created_tenants = []
    
    def _create(name: str, slug: str, owner_email: str):
        # Create owner user
        user = User(
            email=owner_email,
            username=slug + "_owner",
            hashed_password="hashed"
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        # Create tenant
        tenant = tenant_service.create_tenant(
            name=name,
            slug=slug,
            owner_id=user.id,
            tier=TenantTier.PROFESSIONAL
        )
        
        created_tenants.append((tenant, user))
        return tenant, user
    
    yield _create
    
    # Cleanup
    for tenant, user in created_tenants:
        db_session.delete(tenant)
        db_session.delete(user)
        db_session.commit()


class TestTenantCRUD:
    """Test tenant CRUD operations."""
    
    def test_create_tenant(self, tenant_service, db_session):
        """Test creating a tenant."""
        # Create owner
        owner = User(email="owner@test.com", username="owner", hashed_password="test")
        db_session.add(owner)
        db_session.commit()
        
        # Create tenant
        tenant = tenant_service.create_tenant(
            name="Test Hospital",
            slug="test-hospital",
            owner_id=owner.id,
            tier=TenantTier.STARTER
        )
        
        assert tenant.id is not None
        assert tenant.name == "Test Hospital"
        assert tenant.slug == "test-hospital"
        assert tenant.tier == TenantTier.STARTER
        assert tenant.status == TenantStatus.ACTIVE
        
        # Check owner was added
        tenant_users = tenant_service.get_tenant_users(tenant.id)
        assert len(tenant_users) == 1
        assert tenant_users[0].role == TenantUserRole.OWNER
    
    def test_duplicate_slug_fails(self, tenant_service, db_session):
        """Test that duplicate slugs are rejected."""
        # Create first tenant
        owner1 = User(email="owner1@test.com", username="owner1", hashed_password="test")
        db_session.add(owner1)
        db_session.commit()
        
        tenant1 = tenant_service.create_tenant(
            name="Hospital 1",
            slug="hospital",
            owner_id=owner1.id
        )
        
        # Try to create second with same slug
        owner2 = User(email="owner2@test.com", username="owner2", hashed_password="test")
        db_session.add(owner2)
        db_session.commit()
        
        with pytest.raises(Exception):  # Should fail with unique constraint
            tenant2 = tenant_service.create_tenant(
                name="Hospital 2",
                slug="hospital",
                owner_id=owner2.id
            )


class TestTenantIsolation:
    """Test tenant data isolation."""
    
    def test_cannot_access_other_tenant_data(self, db_session, create_test_tenant):
        """Test that tenants cannot access each other's data."""
        # Create two tenants
        tenant1, owner1 = create_test_tenant("Hospital A", "hospital-a", "a@test.com")
        tenant2, owner2 = create_test_tenant("Hospital B", "hospital-b", "b@test.com")
        
        # Create studies for each tenant
        study1 = Study(
            tenant_id=tenant1.id,
            study_instance_uid=f"1.2.3.{uuid.uuid4()}",
            patient_id="PATIENT_A",
            patient_name="Patient A",
            modality="CT"
        )
        
        study2 = Study(
            tenant_id=tenant2.id,
            study_instance_uid=f"1.2.3.{uuid.uuid4()}",
            patient_id="PATIENT_B",
            patient_name="Patient B",
            modality="MR"
        )
        
        db_session.add_all([study1, study2])
        db_session.commit()
        
        # Query studies for tenant1
        tenant1_studies = db_session.query(Study).filter(
            Study.tenant_id == tenant1.id
        ).all()
        
        # Should only see tenant1's study
        assert len(tenant1_studies) == 1
        assert tenant1_studies[0].patient_id == "PATIENT_A"
        assert study2 not in tenant1_studies
        
        # Query studies for tenant2
        tenant2_studies = db_session.query(Study).filter(
            Study.tenant_id == tenant2.id
        ).all()
        
        # Should only see tenant2's study
        assert len(tenant2_studies) == 1
        assert tenant2_studies[0].patient_id == "PATIENT_B"
        assert study1 not in tenant2_studies
    
    def test_user_access_control(self, tenant_service, db_session, create_test_tenant):
        """Test that users can only access authorized tenants."""
        # Create tenant and users
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Create additional users
        member = User(email="member@test.com", username="member", hashed_password="test")
        outsider = User(email="outsider@test.com", username="outsider", hashed_password="test")
        db_session.add_all([member, outsider])
        db_session.commit()
        
        # Add member to tenant
        tenant_service.add_user_to_tenant(tenant.id, member.id, TenantUserRole.MEMBER)
        
        # Check access
        assert tenant_service.check_user_access(tenant.id, owner.id) is True
        assert tenant_service.check_user_access(tenant.id, member.id) is True
        assert tenant_service.check_user_access(tenant.id, outsider.id) is False
    
    def test_role_hierarchy(self, tenant_service, db_session, create_test_tenant):
        """Test tenant role hierarchy."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Create users with different roles
        admin = User(email="admin@test.com", username="admin", hashed_password="test")
        member = User(email="member@test.com", username="member", hashed_password="test")
        viewer = User(email="viewer@test.com", username="viewer", hashed_password="test")
        db_session.add_all([admin, member, viewer])
        db_session.commit()
        
        tenant_service.add_user_to_tenant(tenant.id, admin.id, TenantUserRole.ADMIN)
        tenant_service.add_user_to_tenant(tenant.id, member.id, TenantUserRole.MEMBER)
        tenant_service.add_user_to_tenant(tenant.id, viewer.id, TenantUserRole.VIEWER)
        
        # Owner should have all access
        assert tenant_service.check_user_access(tenant.id, owner.id, TenantUserRole.OWNER)
        assert tenant_service.check_user_access(tenant.id, owner.id, TenantUserRole.ADMIN)
        assert tenant_service.check_user_access(tenant.id, owner.id, TenantUserRole.MEMBER)
        
        # Admin should have admin and below
        assert not tenant_service.check_user_access(tenant.id, admin.id, TenantUserRole.OWNER)
        assert tenant_service.check_user_access(tenant.id, admin.id, TenantUserRole.ADMIN)
        assert tenant_service.check_user_access(tenant.id, admin.id, TenantUserRole.MEMBER)
        
        # Viewer should only have viewer access
        assert not tenant_service.check_user_access(tenant.id, viewer.id, TenantUserRole.MEMBER)
        assert tenant_service.check_user_access(tenant.id, viewer.id, TenantUserRole.VIEWER)


class TestQuotaEnforcement:
    """Test quota enforcement."""
    
    def test_api_call_quota(self, tenant_service, db_session, create_test_tenant):
        """Test API call quota enforcement."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Set low quota
        tenant.quota_api_calls = 10
        db_session.commit()
        
        # Record usage up to quota
        tenant_service.record_usage(tenant.id, api_calls=9)
        
        # Should be allowed
        allowed, error = tenant_service.check_quota(tenant.id, "api_calls", 1)
        assert allowed is True
        assert error is None
        
        # This would exceed quota
        allowed, error = tenant_service.check_quota(tenant.id, "api_calls", 2)
        assert allowed is False
        assert "exceeded" in error.lower()
    
    def test_storage_quota(self, tenant_service, db_session, create_test_tenant):
        """Test storage quota enforcement."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Set low quota
        tenant.quota_storage_gb = 100
        db_session.commit()
        
        # Record usage
        tenant_service.record_usage(tenant.id, storage_gb=90)
        
        # Should be allowed
        allowed, error = tenant_service.check_quota(tenant.id, "storage_gb", 5)
        assert allowed is True
        
        # Would exceed
        allowed, error = tenant_service.check_quota(tenant.id, "storage_gb", 20)
        assert allowed is False
    
    def test_unlimited_quota(self, tenant_service, db_session, create_test_tenant):
        """Test that -1 quota means unlimited."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Set unlimited
        tenant.quota_api_calls = -1
        db_session.commit()
        
        # Record large usage
        tenant_service.record_usage(tenant.id, api_calls=1000000)
        
        # Should still be allowed
        allowed, error = tenant_service.check_quota(tenant.id, "api_calls", 1000000)
        assert allowed is True
    
    def test_trial_expiration(self, tenant_service, db_session, create_test_tenant):
        """Test that expired trials are blocked."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Set trial to expired
        tenant.status = TenantStatus.TRIAL
        tenant.trial_ends_at = datetime.utcnow() - timedelta(days=1)
        db_session.commit()
        
        # Should be blocked
        allowed, error = tenant_service.check_quota(tenant.id, "api_calls", 1)
        assert allowed is False
        assert "trial" in error.lower() or "expired" in error.lower()


class TestUsageTracking:
    """Test usage tracking."""
    
    def test_record_usage(self, tenant_service, db_session, create_test_tenant):
        """Test recording usage."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Record usage
        tenant_service.record_usage(tenant.id, api_calls=100, storage_gb=50, gpu_hours=5)
        
        # Get usage
        usage = tenant_service.get_current_usage(tenant.id)
        
        assert usage is not None
        assert usage.api_calls == 100
        assert float(usage.storage_gb) == 50
        assert float(usage.gpu_hours) == 5
    
    def test_incremental_usage(self, tenant_service, db_session, create_test_tenant):
        """Test that usage is incremental."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Record usage multiple times
        tenant_service.record_usage(tenant.id, api_calls=50)
        tenant_service.record_usage(tenant.id, api_calls=30)
        tenant_service.record_usage(tenant.id, storage_gb=20)
        
        # Get usage
        usage = tenant_service.get_current_usage(tenant.id)
        
        assert usage.api_calls == 80  # 50 + 30
        assert float(usage.storage_gb) == 20


class TestInvitations:
    """Test tenant invitations."""
    
    def test_create_and_accept_invitation(self, tenant_service, db_session, create_test_tenant):
        """Test creating and accepting invitations."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Create invitation
        invitation = tenant_service.create_invitation(
            tenant_id=tenant.id,
            email="newuser@test.com",
            role=TenantUserRole.MEMBER,
            invited_by=owner.id
        )
        
        assert invitation.token is not None
        assert invitation.email == "newuser@test.com"
        assert not invitation.is_expired
        assert not invitation.is_accepted
        
        # Create user who will accept
        new_user = User(email="newuser@test.com", username="newuser", hashed_password="test")
        db_session.add(new_user)
        db_session.commit()
        
        # Accept invitation
        tenant_user = tenant_service.accept_invitation(invitation.token, new_user.id)
        
        assert tenant_user is not None
        assert tenant_user.tenant_id == tenant.id
        assert tenant_user.user_id == new_user.id
        assert tenant_user.role == TenantUserRole.MEMBER
        
        # Check user now has access
        assert tenant_service.check_user_access(tenant.id, new_user.id)
    
    def test_expired_invitation_fails(self, tenant_service, db_session, create_test_tenant):
        """Test that expired invitations cannot be accepted."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Create invitation
        invitation = tenant_service.create_invitation(
            tenant_id=tenant.id,
            email="newuser@test.com",
            role=TenantUserRole.MEMBER,
            invited_by=owner.id,
            expires_in_days=0  # Already expired
        )
        
        # Try to accept
        new_user = User(email="newuser@test.com", username="newuser", hashed_password="test")
        db_session.add(new_user)
        db_session.commit()
        
        with pytest.raises(ValueError, match="expired"):
            tenant_service.accept_invitation(invitation.token, new_user.id)


class TestTenantUpgrade:
    """Test tenant tier upgrades."""
    
    def test_upgrade_increases_quotas(self, tenant_service, db_session, create_test_tenant):
        """Test that upgrading increases quotas."""
        tenant, owner = create_test_tenant("Hospital", "hospital", "owner@test.com")
        
        # Start with free tier
        tenant_service.upgrade_tenant(tenant.id, TenantTier.FREE)
        tenant = tenant_service.get_tenant(tenant.id)
        
        original_quota = tenant.quota_api_calls
        
        # Upgrade to professional
        tenant = tenant_service.upgrade_tenant(tenant.id, TenantTier.PROFESSIONAL)
        
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.quota_api_calls > original_quota
        assert tenant.status == TenantStatus.ACTIVE


@pytest.mark.integration
class TestTenantAPI:
    """Integration tests for tenant API."""
    
    def test_create_tenant_via_api(self, client):
        """Test creating tenant via API."""
        # TODO: Implement with authentication
        pass
    
    def test_cross_tenant_access_blocked(self, client):
        """Test that cross-tenant access is blocked."""
        # TODO: Implement with authentication
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
