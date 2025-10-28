"""Health check endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
import redis
import httpx
from app.core.database import get_db
from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "gateway",
        "version": settings.APP_VERSION
    }


@router.get("/health/detailed")
async def detailed_health_check(db: Session = Depends(get_db)):
    """Detailed health check with dependency status."""
    health_status = {
        "status": "healthy",
        "service": "gateway",
        "version": settings.APP_VERSION,
        "checks": {}
    }
    
    # Check database
    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check imaging service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.IMAGING_SVC_URL}/health",
                timeout=5.0
            )
            if response.status_code == 200:
                health_status["checks"]["imaging_service"] = "healthy"
            else:
                health_status["checks"]["imaging_service"] = f"unhealthy: status {response.status_code}"
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["imaging_service"] = f"unreachable: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check ML service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ML_SVC_URL}/health",
                timeout=5.0
            )
            if response.status_code == 200:
                health_status["checks"]["ml_service"] = "healthy"
            else:
                health_status["checks"]["ml_service"] = f"unhealthy: status {response.status_code}"
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["ml_service"] = f"unreachable: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Orthanc
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.ORTHANC_URL}/system",
                timeout=5.0,
                auth=(settings.ORTHANC_USERNAME, settings.ORTHANC_PASSWORD)
            )
            if response.status_code == 200:
                health_status["checks"]["orthanc"] = "healthy"
            else:
                health_status["checks"]["orthanc"] = f"unhealthy: status {response.status_code}"
                health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["orthanc"] = f"unreachable: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/ready")
async def readiness_check(db: Session = Depends(get_db)):
    """Kubernetes readiness probe."""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ready"}
    except Exception:
        return {"status": "not_ready"}, 503


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}
