# Aurelius Medical Imaging Platform

[![CI](https://github.com/aurelius/med-imaging/actions/workflows/ci.yml/badge.svg)](https://github.com/aurelius/med-imaging/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A comprehensive, HIPAA-aware biomedical imaging platform for labs, hospitals, and universities supporting DICOM, whole slide imaging (WSI), multi-modal medical data, AI/ML inference, and clinical workflows.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                        │
│  DICOM Viewer • WSI Viewer • Study Browser • AI Dashboard       │
└─────────────────┬───────────────────────────────────────────────┘
                  │ REST + WebSocket
┌─────────────────▼───────────────────────────────────────────────┐
│                    API Gateway (FastAPI)                          │
│         Authentication • Rate Limiting • Routing                  │
└─────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
      │          │          │          │          │
   ┌──▼──┐   ┌──▼──┐   ┌───▼───┐  ┌──▼──┐   ┌───▼────┐
   │Image│   │ ML  │   │  ETL  │  │FHIR │   │ Orthanc│
   │ Svc │   │ Svc │   │  Svc  │  │ Svc │   │(DICOM) │
   └─────┘   └─────┘   └───────┘  └─────┘   └────────┘
      │          │          │          │          │
   ┌──▼──────────▼──────────▼──────────▼──────────▼────┐
   │  MinIO • Postgres • Redis • Keycloak • Prometheus │
   └──────────────────────────────────────────────────────┘
```

## 🚀 Quick Start (30 minutes)

### Prerequisites

- **Docker** 24.0+ & **Docker Compose** 2.20+
- **Node.js** 20+ (for frontend development)
- **Python** 3.11+ (for backend development)
- **pnpm** 8+ (Node package manager)
- **uv** or **pip-tools** (Python package manager)
- **make** or **just** (task runner)
- 16GB RAM minimum, 32GB recommended
- 50GB free disk space

### One-Command Setup

```bash
# Clone and enter the repository
git clone https://github.com/aurelius/med-imaging.git
cd med-imaging

# Bootstrap the entire platform
make bootstrap

# Start all services
make up

# Run end-to-end tests
make e2e
```

### Manual Setup

```bash
# 1. Install dependencies
make install

# 2. Generate API clients and proto stubs
make generate

# 3. Start infrastructure services
docker compose up -d postgres redis minio keycloak orthanc

# 4. Run database migrations
make migrate

# 5. Start application services
docker compose up -d gateway imaging-svc ml-svc

# 6. Start frontend
cd apps/frontend && pnpm dev

# 7. Visit http://localhost:3000
```

### Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | admin / admin123 |
| **API Gateway** | http://localhost:8000 | Bearer token from Keycloak |
| **API Docs** | http://localhost:8000/docs | - |
| **Keycloak** | http://localhost:8080 | admin / admin |
| **Orthanc** | http://localhost:8042 | orthanc / orthanc |
| **MinIO** | http://localhost:9001 | minioadmin / minioadmin |
| **Grafana** | http://localhost:3001 | admin / admin |
| **Prometheus** | http://localhost:9090 | - |

## 📋 Make Tasks

```bash
make help              # Show all available commands
make bootstrap         # Complete setup from scratch
make up                # Start all services
make down              # Stop all services
make restart           # Restart all services
make logs              # Tail logs from all services
make clean             # Remove all containers and volumes
make test              # Run all tests
make lint              # Run linters
make format            # Format code
make migrate           # Run database migrations
make seed              # Seed with sample data
make e2e               # Run end-to-end tests
```

## 🧪 Running Tests

```bash
# Backend tests
cd apps/gateway && pytest
cd apps/imaging-svc && pytest
cd apps/ml-svc && pytest

# Frontend tests
cd apps/frontend && pnpm test
cd apps/frontend && pnpm test:e2e

# Integration tests
make test-integration

# Full test suite
make test-all
```

## 📦 Project Structure

```
aurelius-med-imaging/
├── apps/
│   ├── frontend/              # Next.js 15 + TypeScript + Tailwind
│   ├── gateway/               # FastAPI API Gateway
│   ├── imaging-svc/           # DICOM/WSI/File ingestion service
│   ├── ml-svc/                # ML inference service (Triton client)
│   ├── etl-svc/               # Airflow/Prefect pipelines
│   └── fhir-svc/              # HAPI FHIR wrapper
├── packages/
│   ├── shared-types/          # Shared TypeScript/Pydantic schemas
│   └── ui/                    # Shared React UI components
├── infra/
│   ├── docker/                # Docker Compose configurations
│   ├── k8s/                   # Kubernetes Helm charts
│   └── terraform/             # Cloud infrastructure (AWS/GCP)
├── docs/                      # Architecture and API documentation
├── scripts/                   # Utility scripts
└── .github/workflows/         # CI/CD pipelines
```

## 🔐 Security & Compliance

This platform is designed with HIPAA compliance in mind:

- **Authentication**: OAuth2/OIDC via Keycloak
- **Authorization**: Role-based access control (RBAC) + OPA policies
- **Encryption**: TLS in transit, AES-256 at rest
- **Audit Logging**: All PHI access logged to append-only tables
- **De-identification**: DICOM tag anonymization with reversible mapping
- **Key Management**: HashiCorp Vault for secrets and PHI mapping keys

See [SECURITY.md](docs/SECURITY.md) and [COMPLIANCE.md](docs/COMPLIANCE.md) for details.

## 🤝 Development Workflow

1. **Create a feature branch**: `git checkout -b feature/my-feature`
2. **Make changes**: Follow code style guidelines
3. **Run tests**: `make test`
4. **Commit**: Use conventional commits (`feat:`, `fix:`, etc.)
5. **Push**: `git push origin feature/my-feature`
6. **Open PR**: CI will run tests and linting

### Code Style

- **Python**: Black, Ruff, mypy (strict)
- **TypeScript**: Prettier, ESLint
- **Commits**: Conventional Commits (commitlint)
- **Pre-commit hooks**: Enforced via husky/pre-commit

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Data Model](docs/DATA_MODEL.md)
- [API Contracts](docs/API_CONTRACTS.md)
- [Security & Compliance](docs/SECURITY.md)
- [Session Log](docs/SESSION_LOG.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🐛 Troubleshooting

### Services won't start

```bash
# Check Docker resources
docker system df
docker system prune -a  # Free up space

# Check service logs
make logs

# Reset everything
make clean && make up
```

### Database connection errors

```bash
# Ensure Postgres is healthy
docker compose ps postgres

# Run migrations
make migrate

# Check connection
docker compose exec postgres psql -U postgres -d aurelius
```

### Authentication issues

```bash
# Restart Keycloak
docker compose restart keycloak

# Import realm
make keycloak-import

# Get admin token
curl -X POST http://localhost:8080/realms/aurelius/protocol/openid-connect/token \
  -d "client_id=gateway" \
  -d "client_secret=..." \
  -d "grant_type=client_credentials"
```

## 🛠️ Technology Stack

### Frontend
- Next.js 15, React 18, TypeScript 5
- Tailwind CSS 3, shadcn/ui
- React Query (TanStack Query)
- Cornerstone3D (DICOM viewer)
- OpenSeadragon (WSI viewer)
- Zustand (state management)

### Backend
- FastAPI 0.109+ (Python 3.11+)
- SQLAlchemy 2.0 + Alembic
- Celery + Redis (async tasks)
- gRPC (inter-service communication)
- Pydantic v2 (data validation)

### Data & Storage
- PostgreSQL 16 + TimescaleDB
- Redis 7
- MinIO (S3-compatible object storage)
- Orthanc (DICOM server)
- HAPI FHIR (HL7 FHIR server)
- OpenSearch (full-text search)

### ML & AI
- PyTorch 2.1+, MONAI 1.3+
- NVIDIA Triton Inference Server
- MLflow (model registry)
- Feast (feature store)
- MONAI Label (interactive annotation)

### Infrastructure
- Docker & Docker Compose
- Kubernetes + Helm
- Terraform (IaC)
- GitHub Actions (CI/CD)
- Keycloak (identity management)
- OpenTelemetry + Prometheus + Grafana

## 📜 License

Apache License 2.0 - see [LICENSE](LICENSE)

## 🙏 Acknowledgments

Built on top of excellent open-source projects:
- Orthanc DICOM server
- MONAI medical imaging AI framework
- Cornerstone3D medical image viewer
- HAPI FHIR server
- And many others!

## 📧 Contact

- **Issues**: https://github.com/aurelius/med-imaging/issues
- **Discussions**: https://github.com/aurelius/med-imaging/discussions
- **Email**: support@aurelius-medical.io
