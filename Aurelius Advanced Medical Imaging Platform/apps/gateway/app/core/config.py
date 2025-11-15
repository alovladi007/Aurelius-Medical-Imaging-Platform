"""Configuration for Gateway Service."""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # App
    APP_NAME: str = "Aurelius API Gateway"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@postgres:5432/aurelius",
        env="DATABASE_URL"
    )
    
    # Redis
    REDIS_URL: str = Field(
        default="redis://:redis123@redis:6379/0",
        env="REDIS_URL"
    )
    
    # Keycloak
    KEYCLOAK_URL: str = Field(
        default="http://keycloak:8080",
        env="KEYCLOAK_URL"
    )
    KEYCLOAK_REALM: str = Field(default="aurelius", env="KEYCLOAK_REALM")
    KEYCLOAK_CLIENT_ID: str = Field(default="gateway", env="KEYCLOAK_CLIENT_ID")
    KEYCLOAK_CLIENT_SECRET: str = Field(
        default="gateway-secret",
        env="KEYCLOAK_CLIENT_SECRET"
    )
    
    # Microservices
    IMAGING_SVC_URL: str = Field(
        default="http://imaging-svc:8001",
        env="IMAGING_SVC_URL"
    )
    ML_SVC_URL: str = Field(default="http://ml-svc:8002", env="ML_SVC_URL")
    CANCER_AI_SVC_URL: str = Field(
        default="http://cancer-ai-svc:8003",
        env="CANCER_AI_SVC_URL"
    )
    ETL_SVC_URL: str = Field(default="http://etl-svc:8004", env="ETL_SVC_URL")
    FHIR_SVC_URL: str = Field(
        default="http://fhir-server:8080/fhir",
        env="FHIR_SVC_URL"
    )
    
    # MinIO (S3)
    MINIO_ENDPOINT: str = Field(default="minio:9000", env="MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    MINIO_SECURE: bool = Field(default=False, env="MINIO_SECURE")
    
    # Orthanc
    ORTHANC_URL: str = Field(default="http://orthanc:8042", env="ORTHANC_URL")
    ORTHANC_USERNAME: str = Field(default="orthanc", env="ORTHANC_USERNAME")
    ORTHANC_PASSWORD: str = Field(default="orthanc", env="ORTHANC_PASSWORD")
    
    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    ENABLE_TRACING: bool = Field(default=True, env="ENABLE_TRACING")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
