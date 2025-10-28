"""Tenant context middleware for row-level security.

This middleware ensures tenant isolation by:
1. Extracting tenant context from request (header, subdomain, or JWT)
2. Validating user access to the tenant
3. Setting tenant context for the request
4. Filtering database queries by tenant_id
"""
import logging
from typing import Optional, Callable
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uuid

from app.db.session import SessionLocal
from app.models.tenants import Tenant, TenantUser
from app.services.tenant_service import TenantService

logger = logging.getLogger(__name__)


class TenantContextMiddleware:
    """Middleware to manage tenant context."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, request: Request, call_next: Callable):
        """Process request and inject tenant context."""
        
        # Skip for certain paths
        if self._should_skip(request.url.path):
            return await call_next(request)
        
        # Extract tenant from request
        tenant_id = self._extract_tenant(request)
        
        if tenant_id:
            # Validate tenant access
            user_id = getattr(request.state, "user_id", None)
            
            if user_id:
                db = SessionLocal()
                try:
                    service = TenantService(db)
                    
                    # Check user has access to tenant
                    has_access = service.check_user_access(tenant_id, user_id)
                    
                    if not has_access:
                        return JSONResponse(
                            status_code=status.HTTP_403_FORBIDDEN,
                            content={
                                "detail": "You do not have access to this tenant"
                            }
                        )
                    
                    # Get tenant
                    tenant = service.get_tenant(tenant_id)
                    
                    if not tenant:
                        return JSONResponse(
                            status_code=status.HTTP_404_NOT_FOUND,
                            content={"detail": "Tenant not found"}
                        )
                    
                    # Check tenant is active
                    if not tenant.is_active and not tenant.is_trial:
                        return JSONResponse(
                            status_code=status.HTTP_403_FORBIDDEN,
                            content={
                                "detail": f"Tenant is {tenant.status}. Please contact support."
                            }
                        )
                    
                    # Set tenant context in request state
                    request.state.tenant_id = tenant_id
                    request.state.tenant = tenant
                    
                    logger.debug(f"Tenant context set: {tenant.slug} for user {user_id}")
                    
                finally:
                    db.close()
        
        # Process request
        response = await call_next(request)
        
        return response
    
    def _should_skip(self, path: str) -> bool:
        """Check if path should skip tenant middleware."""
        skip_paths = [
            "/health",
            "/ready",
            "/live",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/auth/",
            "/tenants",  # Tenant management endpoints
        ]
        
        for skip_path in skip_paths:
            if path.startswith(skip_path):
                return True
        
        return False
    
    def _extract_tenant(self, request: Request) -> Optional[uuid.UUID]:
        """
        Extract tenant ID from request.
        
        Priority:
        1. X-Tenant-ID header
        2. Subdomain (e.g., acme.aurelius.io)
        3. JWT token tenant_id claim
        
        Returns:
            Tenant UUID or None
        """
        # 1. Check header
        tenant_header = request.headers.get("X-Tenant-ID")
        if tenant_header:
            try:
                return uuid.UUID(tenant_header)
            except ValueError:
                logger.warning(f"Invalid tenant ID in header: {tenant_header}")
        
        # 2. Check subdomain
        host = request.headers.get("host", "")
        if "." in host:
            subdomain = host.split(".")[0]
            
            # Look up tenant by slug (subdomain)
            db = SessionLocal()
            try:
                from app.services.tenant_service import TenantService
                service = TenantService(db)
                tenant = service.get_tenant_by_slug(subdomain)
                if tenant:
                    return tenant.id
            finally:
                db.close()
        
        # 3. Check JWT token (if available)
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id:
            try:
                return uuid.UUID(str(tenant_id))
            except ValueError:
                pass
        
        return None


def get_tenant_context(request: Request) -> Optional[Tenant]:
    """
    Get tenant context from request state.
    
    Use this in endpoints to access the current tenant.
    """
    return getattr(request.state, "tenant", None)


def require_tenant_context(request: Request) -> Tenant:
    """
    Require tenant context in request.
    
    Raises HTTPException if no tenant context.
    Use as a dependency in endpoints that require tenant isolation.
    """
    tenant = get_tenant_context(request)
    
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tenant context required. Provide X-Tenant-ID header or use tenant subdomain."
        )
    
    return tenant


# Database query filter helpers

def add_tenant_filter(query, model, tenant_id: uuid.UUID):
    """
    Add tenant filter to a SQLAlchemy query.
    
    Usage:
        query = db.query(Study)
        query = add_tenant_filter(query, Study, tenant_id)
    """
    if hasattr(model, "tenant_id"):
        return query.filter(model.tenant_id == tenant_id)
    return query


def ensure_tenant_isolation(db: Session, tenant_id: uuid.UUID):
    """
    Configure session to automatically filter by tenant.
    
    This uses SQLAlchemy's query property to add automatic tenant filtering.
    
    Usage:
        db = SessionLocal()
        ensure_tenant_isolation(db, tenant_id)
        # All queries on models with tenant_id will be filtered
    """
    # Note: This is a placeholder for more advanced implementation
    # In production, consider using:
    # 1. SQLAlchemy events (before_compile)
    # 2. Custom Query class
    # 3. PostgreSQL Row-Level Security (RLS)
    pass


# Example: PostgreSQL Row-Level Security policy
"""
-- Enable RLS on tables
ALTER TABLE studies ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;

-- Create policy to filter by tenant_id
CREATE POLICY tenant_isolation_policy ON studies
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_policy ON annotations
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

CREATE POLICY tenant_isolation_policy ON ml_models
    USING (tenant_id = current_setting('app.current_tenant_id')::uuid);

-- Set tenant context in connection
-- In Python:
db.execute("SET LOCAL app.current_tenant_id = :tenant_id", {"tenant_id": str(tenant_id)})
"""


# Tenant-aware CRUD base class

class TenantAwareCRUD:
    """
    Base class for tenant-aware CRUD operations.
    
    All queries automatically filter by tenant_id.
    """
    
    def __init__(self, model, db: Session, tenant_id: uuid.UUID):
        self.model = model
        self.db = db
        self.tenant_id = tenant_id
    
    def _get_base_query(self):
        """Get base query with tenant filter."""
        query = self.db.query(self.model)
        
        if hasattr(self.model, "tenant_id"):
            query = query.filter(self.model.tenant_id == self.tenant_id)
        
        return query
    
    def get(self, id: uuid.UUID):
        """Get single record by ID (tenant-filtered)."""
        return self._get_base_query().filter(self.model.id == id).first()
    
    def list(self, skip: int = 0, limit: int = 100):
        """List records (tenant-filtered)."""
        return self._get_base_query().offset(skip).limit(limit).all()
    
    def create(self, obj_in: dict):
        """Create record with tenant_id."""
        if hasattr(self.model, "tenant_id"):
            obj_in["tenant_id"] = self.tenant_id
        
        db_obj = self.model(**obj_in)
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        
        return db_obj
    
    def update(self, id: uuid.UUID, obj_in: dict):
        """Update record (tenant-filtered)."""
        db_obj = self.get(id)
        if not db_obj:
            return None
        
        for key, value in obj_in.items():
            if hasattr(db_obj, key) and key != "tenant_id":  # Never allow changing tenant_id
                setattr(db_obj, key, value)
        
        self.db.commit()
        self.db.refresh(db_obj)
        
        return db_obj
    
    def delete(self, id: uuid.UUID):
        """Delete record (tenant-filtered)."""
        db_obj = self.get(id)
        if not db_obj:
            return False
        
        self.db.delete(db_obj)
        self.db.commit()
        
        return True


# Example usage in endpoints:
"""
from fastapi import Depends, Request
from app.middleware.tenant_context import require_tenant_context, TenantAwareCRUD

@router.get("/studies")
async def list_studies(
    request: Request,
    db: Session = Depends(get_db),
    tenant: Tenant = Depends(require_tenant_context)
):
    # All studies will be automatically filtered by tenant
    crud = TenantAwareCRUD(Study, db, tenant.id)
    studies = crud.list()
    return studies
"""
