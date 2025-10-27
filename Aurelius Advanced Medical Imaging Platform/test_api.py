"""Tests for API Gateway."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "version" in data


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_live():
    """Test liveness probe."""
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_ready():
    """Test readiness probe."""
    response = client.get("/ready")
    assert response.status_code in [200, 503]


@pytest.mark.asyncio
async def test_metrics():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content


def test_login_invalid_credentials():
    """Test login with invalid credentials."""
    response = client.post(
        "/auth/login",
        json={"username": "invalid", "password": "wrong"}
    )
    assert response.status_code == 401


def test_unauthorized_access():
    """Test unauthorized access to protected endpoint."""
    response = client.get("/studies")
    assert response.status_code == 403  # No auth header


def test_cors_headers():
    """Test CORS headers."""
    response = client.get("/", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
