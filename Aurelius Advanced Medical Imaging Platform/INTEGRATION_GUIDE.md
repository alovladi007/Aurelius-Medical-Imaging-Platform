# Aurelius Medical Imaging Platform - Integration Guide

## Overview

This guide explains how all the components of the Aurelius Medical Imaging Platform have been integrated to work together as a cohesive system.

## What Was Integrated

The integration brings together:

1. **Backend Microservices** (Python/FastAPI)
   - API Gateway
   - Imaging Service (DICOM processing)
   - ML Service (AI/ML inference)
   - Search Service (full-text search)

2. **Infrastructure Services**
   - PostgreSQL 16 + TimescaleDB (database)
   - Redis 7 (caching & queuing)
   - MinIO (S3-compatible object storage)
   - Keycloak (OAuth2/OIDC identity management)
   - Orthanc (DICOM server)
   - HAPI FHIR (HL7 FHIR server)

3. **Observability Stack**
   - Prometheus (metrics)
   - Grafana (dashboards)
   - Jaeger (distributed tracing)

4. **Additional Services**
   - Apache Kafka (message broker)
   - OpenSearch (search engine)
   - MLflow (ML model registry)

## Directory Structure

The integrated repository is organized as follows:

```
Aurelius-Medical-Imaging-Platform/
├── apps/                          # Microservices
│   ├── gateway/                   # API Gateway service
│   │   ├── app/
│   │   │   ├── core/             # Core modules (config, auth, database)
│   │   │   ├── api/              # API endpoints
│   │   │   ├── models/           # Data models
│   │   │   └── main.py           # FastAPI application
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── imaging-svc/              # Imaging service
│   │   ├── app/
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── ml-svc/                   # ML service
│   │   ├── app/
│   │   │   └── main.py
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── search-svc/               # Search service
│       ├── app/
│       │   └── main.py
│       ├── Dockerfile
│       └── requirements.txt
├── infra/                        # Infrastructure configuration
│   ├── postgres/                 # Database migrations
│   │   ├── 001_initial_schema.sql
│   │   └── 014_add_multitenancy.py
│   └── observability/            # Monitoring dashboards & alerts
│       ├── dashboards/
│       └── alerts/
├── compose.yaml                  # Docker Compose configuration
├── start.sh                      # Startup script
├── .env.example                  # Environment variables template
├── .gitignore                    # Git ignore rules
├── requirements.txt              # Python dependencies
├── package.json                  # Node.js dependencies
└── README.md                     # Main documentation

# Configuration files
├── prometheus.yml                # Prometheus configuration
├── grafana-datasources.yml       # Grafana data sources
├── keycloak-realm.json           # Keycloak realm configuration
├── init-db.sh                    # Database initialization script

# Documentation
├── ARCHITECTURE.md               # System architecture
├── API_CONTRACTS.md              # API specifications
├── DATA_MODEL.md                 # Database schema
├── SECURITY.md                   # Security & compliance
├── GETTING_STARTED.md            # Quick start guide
└── INTEGRATION_GUIDE.md          # This file
```

## How Services Communicate

### Service Communication Flow

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ↓
┌─────────────────────────────────────────────────────────┐
│  API Gateway (Port 8000)                                │
│  - Authentication via Keycloak                          │
│  - Request routing                                       │
│  - Rate limiting                                         │
│  - Observability (metrics, tracing)                     │
└──────┬──────────────────────────────────────────────────┘
       │
       ├──→ Imaging Service (8001) ──→ Orthanc (8042)
       │                            └──→ MinIO (9000)
       │
       ├──→ ML Service (8002) ──────→ MLflow (5000)
       │                         └──→ Triton (GPU inference)
       │
       ├──→ Search Service (8004) ──→ OpenSearch (9200)
       │
       └──→ FHIR Server (8083)

       ↓
┌─────────────────────────────────────────────────────────┐
│  Infrastructure Layer                                   │
│  - PostgreSQL (5432): Shared database                   │
│  - Redis (6379): Caching & job queue                    │
│  - Keycloak (8080): Identity management                 │
│  - Kafka (9092): Event streaming                        │
└─────────────────────────────────────────────────────────┘

       ↓
┌─────────────────────────────────────────────────────────┐
│  Observability Stack                                    │
│  - Prometheus (9090): Metrics collection                │
│  - Grafana (3001): Dashboards                           │
│  - Jaeger (16686): Distributed tracing                  │
└─────────────────────────────────────────────────────────┘
```

### Network Architecture

All services run in a Docker bridge network (`aurelius-net`) allowing:
- Service-to-service communication by service name
- Isolation from external networks
- Shared volumes for data persistence

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- At least 8GB RAM available for Docker
- 20GB free disk space

### Starting the Platform

1. **Clone and navigate to the repository:**
   ```bash
   cd "Aurelius Advanced Medical Imaging Platform"
   ```

2. **Create environment configuration:**
   ```bash
   cp .env.example .env
   # Edit .env if you need custom configuration
   ```

3. **Start all services:**
   ```bash
   ./start.sh
   ```

   This script will:
   - Build all Docker images
   - Start services in the correct order
   - Wait for health checks to pass
   - Display access URLs for all services

### Alternative: Manual Docker Compose

```bash
# Start all services
docker compose up -d

# Start specific services only
docker compose up -d postgres redis gateway

# View logs
docker compose logs -f gateway

# Stop all services
docker compose down

# Stop and remove volumes (clean slate)
docker compose down -v
```

## Service Health Checks

All services expose `/health` endpoints:

- **Gateway**: http://localhost:8000/health
- **Imaging**: http://localhost:8001/health
- **ML Service**: http://localhost:8002/health
- **Search**: http://localhost:8004/health

Infrastructure health:
- **PostgreSQL**: `docker exec aurelius-postgres pg_isready`
- **Redis**: `docker exec aurelius-redis redis-cli ping`
- **Keycloak**: http://localhost:8080/health/ready

## API Access

### Gateway API

The API Gateway is the main entry point for all API requests:

**Base URL**: http://localhost:8000

**Interactive Documentation**:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Key Endpoints

```
GET  /                   - Service info
GET  /health             - Health check
GET  /metrics            - Prometheus metrics

# Authentication
POST /auth/login         - User login
POST /auth/register      - User registration
GET  /auth/me            - Current user info

# Studies (DICOM)
GET  /studies            - List studies
GET  /studies/{id}       - Get study details
POST /studies            - Create study

# Imaging
POST /imaging/upload     - Upload image
GET  /imaging/{id}       - Get image

# ML Inference
POST /ml/inference       - Run inference
GET  /ml/models          - List models

# Worklists
GET  /worklists          - List worklists
POST /worklists          - Create worklist

# Multi-tenancy
GET  /tenants            - List tenants
POST /tenants            - Create tenant
GET  /tenants/{id}       - Get tenant details
```

## Database Integration

### Connection

All services connect to the shared PostgreSQL database:

```
Host: postgres (or localhost:5432 from host)
Database: aurelius
User: postgres
Password: postgres
```

### Schema

The database includes:
- **Core tables**: users, organizations, patients
- **Imaging tables**: studies, series, instances, slides
- **ML tables**: ml_models, predictions, annotations
- **Multi-tenancy**: tenants, tenant_users, usage_records
- **Audit**: audit_log (all PHI access)

### Migrations

Database migrations are in `infra/postgres/`:
- `001_initial_schema.sql` - Core schema
- `014_add_multitenancy.py` - Multi-tenancy setup

To run migrations manually:
```bash
docker exec -i aurelius-postgres psql -U postgres -d aurelius < infra/postgres/001_initial_schema.sql
```

## Authentication & Authorization

### Keycloak Integration

The platform uses Keycloak for OAuth2/OIDC authentication:

**Admin Console**: http://localhost:8080
- Username: `admin`
- Password: `admin`

**Realm**: `aurelius`

### Getting an Access Token

```bash
curl -X POST "http://localhost:8080/realms/aurelius/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=gateway" \
  -d "client_secret=gateway-secret" \
  -d "grant_type=password" \
  -d "username=your-username" \
  -d "password=your-password"
```

### Using the Token

Include the access token in API requests:

```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  http://localhost:8000/studies
```

## Object Storage (MinIO)

MinIO provides S3-compatible object storage:

**Console**: http://localhost:9001
- Access Key: `minioadmin`
- Secret Key: `minioadmin`

**Pre-configured Buckets**:
- `dicom-studies` - DICOM files
- `wsi-slides` - Whole slide images
- `ml-models` - ML model artifacts
- `processed-data` - Processed images

### S3 API Access

```python
from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# Upload file
client.fput_object("dicom-studies", "study.dcm", "/path/to/study.dcm")
```

## DICOM Integration

### Orthanc DICOM Server

**Web Interface**: http://localhost:8042
- Username: `orthanc`
- Password: `orthanc`

**DICOM Ports**:
- DICOM C-STORE: `4242`
- DICOMweb: `http://localhost:8042/dicom-web/`

### Sending DICOM Files

Using `dcmsend` (DCMTK):
```bash
dcmsend localhost 4242 study.dcm
```

Using `curl` (DICOMweb STOW-RS):
```bash
curl -X POST http://localhost:8042/dicom-web/studies \
  -H "Content-Type: multipart/related; type=application/dicom" \
  --data-binary @study.dcm
```

## Observability

### Prometheus Metrics

**URL**: http://localhost:9090

All services expose metrics at `/metrics`:
- Request counts
- Response times
- Error rates
- Custom business metrics

### Grafana Dashboards

**URL**: http://localhost:3001
- Username: `admin`
- Password: `admin`

Pre-configured dashboards are in `infra/observability/dashboards/`

### Distributed Tracing

**Jaeger UI**: http://localhost:16686

Traces show:
- Request flow across services
- Latency breakdown
- Error tracking
- Dependencies

## Troubleshooting

### Service Won't Start

1. Check Docker resources:
   ```bash
   docker info
   ```

2. Check service logs:
   ```bash
   docker compose logs [service-name]
   ```

3. Verify dependencies are healthy:
   ```bash
   docker compose ps
   ```

### Database Connection Issues

1. Verify PostgreSQL is running:
   ```bash
   docker exec aurelius-postgres pg_isready -U postgres
   ```

2. Check connection from service:
   ```bash
   docker exec aurelius-gateway curl postgres:5432
   ```

### Port Conflicts

If ports are already in use, you can modify them in `compose.yaml`:

```yaml
services:
  gateway:
    ports:
      - "8000:8000"  # Change to "8080:8000" to use port 8080 on host
```

### Reset Everything

To start fresh:
```bash
docker compose down -v  # Remove all containers and volumes
docker compose up -d    # Start fresh
```

## Development Workflow

### Local Development

1. **Edit code** in `apps/[service-name]/app/`

2. **Rebuild and restart service**:
   ```bash
   docker compose up -d --build gateway
   ```

3. **View logs**:
   ```bash
   docker compose logs -f gateway
   ```

### Adding a New Endpoint

1. Create route file in `apps/gateway/app/api/my_endpoint.py`
2. Add router in `apps/gateway/app/main.py`
3. Rebuild: `docker compose up -d --build gateway`

### Database Changes

1. Create migration SQL in `infra/postgres/`
2. Apply migration:
   ```bash
   docker exec -i aurelius-postgres psql -U postgres -d aurelius < infra/postgres/my_migration.sql
   ```

## Production Deployment

For production deployment, see:
- `KUBERNETES_QUICK_REFERENCE.md` - Kubernetes deployment
- `deploy.sh` - Deployment automation
- `values-prod.yaml` - Production Helm values

Key production considerations:
- Use external managed databases (RDS, Cloud SQL)
- Configure TLS/HTTPS
- Use secrets management (Vault, AWS Secrets Manager)
- Enable authentication on all services
- Configure backups
- Set up monitoring alerts

## Next Steps

1. **Review Documentation**:
   - [README.md](README.md) - Overview
   - [ARCHITECTURE.md](ARCHITECTURE.md) - System design
   - [API_CONTRACTS.md](API_CONTRACTS.md) - API specs
   - [SECURITY.md](SECURITY.md) - Security & HIPAA compliance

2. **Explore Services**:
   - Access API docs at http://localhost:8000/docs
   - Browse Grafana dashboards
   - Upload test DICOM files to Orthanc

3. **Customize**:
   - Modify `.env` for your environment
   - Add custom API endpoints
   - Configure additional services

## Support

For issues or questions:
- Check the documentation in this repository
- Review Docker Compose logs
- Consult the service-specific README files

## Summary

The Aurelius Medical Imaging Platform integration provides:
- ✅ Complete microservices architecture
- ✅ HIPAA-compliant infrastructure
- ✅ Production-ready observability
- ✅ Easy local development setup
- ✅ Comprehensive API documentation
- ✅ Multi-tenancy support
- ✅ AI/ML integration
- ✅ DICOM and FHIR support

All services are now integrated and can communicate seamlessly to provide a complete medical imaging platform.
