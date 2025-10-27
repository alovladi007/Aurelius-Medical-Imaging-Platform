# Aurelius Medical Imaging Platform - Architecture

## Overview

The Aurelius platform is a comprehensive, HIPAA-compliant biomedical imaging system designed for hospitals, research institutions, and clinical laboratories.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Next.js Frontend (React 18 + TypeScript + Tailwind)      │ │
│  │  - DICOM Viewer (Cornerstone3D)                           │ │
│  │  - WSI Viewer (OpenSeadragon)                             │ │
│  │  - Study Browser  │  Worklists  │  AI Dashboard          │ │
│  └────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTPS/WSS
┌───────────────────────────▼─────────────────────────────────────┐
│                     API Gateway Layer                            │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  FastAPI Gateway                                           │ │
│  │  - Authentication (Keycloak OIDC)                         │ │
│  │  - Authorization (RBAC + OPA)                             │ │
│  │  - Rate Limiting                                          │ │
│  │  - Request Routing                                        │ │
│  │  - Audit Logging                                          │ │
│  └────────────────────────────────────────────────────────────┘ │
└────┬──────────┬──────────┬──────────┬──────────┬───────────────┘
     │          │          │          │          │
     │          │          │          │          │
┌────▼────┐ ┌──▼───┐ ┌────▼────┐ ┌──▼───┐ ┌───▼────┐
│ Imaging │ │  ML  │ │   ETL   │ │ FHIR │ │Orthanc │
│ Service │ │  Svc │ │ Service │ │  Svc │ │ DICOM  │
└────┬────┘ └──┬───┘ └────┬────┘ └──┬───┘ └───┬────┘
     │          │          │          │          │
┌────▼──────────▼──────────▼──────────▼──────────▼────┐
│              Data & Storage Layer                     │
│  ┌──────────┐  ┌──────┐  ┌───────┐  ┌──────────┐   │
│  │PostgreSQL│  │ Redis│  │ MinIO │  │ Keycloak │   │
│  │+TimescaleDB │      │  │  (S3) │  │   IAM    │   │
│  └──────────┘  └──────┘  └───────┘  └──────────┘   │
└──────────────────────────────────────────────────────┘
                      │
            ┌─────────▼─────────┐
            │   Observability   │
            │  Prometheus       │
            │  Grafana          │
            │  OpenTelemetry    │
            └───────────────────┘
```

## Core Components

### 1. Frontend (Next.js)
- **Technology**: Next.js 15, React 18, TypeScript, Tailwind CSS
- **Authentication**: NextAuth.js with Keycloak provider
- **State Management**: Zustand + React Query
- **Viewers**:
  - **DICOM**: Cornerstone3D for 2D/3D medical imaging
  - **WSI**: OpenSeadragon/Viv for whole slide imaging
  - **Signals**: Custom viewers for ECG/EEG
- **Features**:
  - Role-based UI (clinician, researcher, admin)
  - Real-time worklists
  - AI model integration
  - Annotation tools
  - Study sharing

### 2. API Gateway
- **Technology**: FastAPI (Python 3.11)
- **Responsibilities**:
  - JWT token validation (Keycloak)
  - Request routing to microservices
  - Rate limiting and throttling
  - CORS handling
  - Audit logging
  - Prometheus metrics
- **Endpoints**:
  - `/auth` - Authentication
  - `/studies` - Study management
  - `/imaging` - Image ingestion
  - `/ml` - ML predictions
  - `/worklists` - Workflow management

### 3. Imaging Service
- **Technology**: FastAPI + Celery
- **Responsibilities**:
  - DICOM ingestion (C-STORE, STOW-RS)
  - WSI processing (OpenSlide, Bio-Formats)
  - File format conversion
  - Thumbnail generation
  - Metadata extraction
  - MinIO storage
- **Supported Formats**:
  - DICOM (.dcm)
  - WSI (SVS, NDPI, TIFF pyramids)
  - NIfTI (.nii, .nii.gz)
  - Standard images (PNG, JPEG, TIFF)
  - Video (MP4 for ultrasound)

### 4. ML Service
- **Technology**: FastAPI + PyTorch + MONAI
- **Responsibilities**:
  - Model serving (via Triton)
  - Inference execution
  - Result post-processing
  - Model versioning (MLflow)
- **Model Types**:
  - Classification (chest X-ray, skin lesions)
  - Segmentation (tumor, organ)
  - Detection (nodules, lesions)
  - Regression (biomarkers)

### 5. ETL Service
- **Technology**: Airflow/Prefect
- **Responsibilities**:
  - Data pipelines
  - De-identification
  - Quality control
  - Dataset preparation
  - Model training orchestration

### 6. FHIR Service
- **Technology**: HAPI FHIR Server
- **Responsibilities**:
  - HL7 FHIR R4 API
  - Clinical data integration
  - Patient demographics
  - Encounter management

## Data Model

### Core Entities

#### Patients
- Patient ID (MRN)
- Demographics
- De-identification status
- Consent records

#### Studies
- Study Instance UID (DICOM)
- Patient reference
- Modality (CT, MRI, X-Ray, etc.)
- Study date/time
- Series and instances
- Storage location

#### Slides (WSI)
- Slide ID
- Patient reference
- Specimen type
- Stain information
- Pyramid metadata
- Tile storage

#### Annotations
- Target reference (study/slide)
- User/author
- Coordinates/geometry
- Labels and properties
- Version history

#### ML Predictions
- Model reference
- Target reference
- Results (JSON)
- Confidence scores
- Inference metadata

## Security Architecture

### Authentication & Authorization
- **Keycloak**: OAuth2/OIDC identity provider
- **Roles**:
  - `admin`: Full system access
  - `clinician`: Patient data access
  - `researcher`: De-identified data access
  - `radiologist`: Imaging interpretation
  - `pathologist`: WSI access
  - `ml-engineer`: Model training/deployment
  - `student`: Limited educational access
- **Token Flow**:
  1. User authenticates with Keycloak
  2. Receives JWT access token
  3. Includes token in API requests
  4. Gateway validates token
  5. Service checks role permissions

### Data Protection
- **Encryption at Rest**: AES-256 (MinIO, Postgres)
- **Encryption in Transit**: TLS 1.3
- **PHI Handling**:
  - Dedicated PHI zones
  - Audit logging for all PHI access
  - De-identification pipelines
  - Reversible mapping (Vault)

### Audit Logging
- All API requests logged
- PHI access tracked
- Append-only audit table (TimescaleDB)
- Retention: 7 years (HIPAA requirement)

## Deployment Architecture

### Development
- Docker Compose
- Local volumes
- Hot reloading

### Production (Kubernetes)
- Helm charts
- Horizontal pod autoscaling
- Load balancing (Ingress)
- Persistent volumes (PVC)
- Secrets management (Vault)
- Certificate management (cert-manager)

### High Availability
- Multi-replica services
- PostgreSQL replication
- Redis Sentinel
- MinIO distributed mode
- Cross-AZ deployment

## Network Architecture

### Service Communication
- **External → Gateway**: HTTPS (443)
- **Gateway → Services**: HTTP/gRPC (internal network)
- **Services → Databases**: TCP (internal network)
- **Frontend → Gateway**: HTTPS + WebSocket

### Security Groups/Firewall
- Frontend: 3000 (dev only)
- Gateway: 8000
- Services: Not exposed externally
- Postgres: 5432 (internal only)
- Redis: 6379 (internal only)
- MinIO: 9000, 9001 (internal only)
- Keycloak: 8080 (internal + admin access)

## Scalability

### Horizontal Scaling
- **Stateless Services**: Gateway, Imaging, ML (scale freely)
- **Stateful Services**: Postgres (read replicas), Redis (cluster mode)
- **Object Storage**: MinIO (distributed mode)

### Performance Optimization
- **Caching**: Redis for sessions, query results
- **CDN**: Static assets (future)
- **Database**:
  - Connection pooling
  - Indexed queries
  - TimescaleDB compression
  - Partitioning by date
- **Object Storage**: Pre-signed URLs for direct access

## Disaster Recovery

### Backup Strategy
- **Databases**: Daily full + hourly incremental
- **Object Storage**: Cross-region replication
- **Configuration**: Git repository

### RPO/RTO
- **Recovery Point Objective**: 1 hour
- **Recovery Time Objective**: 4 hours

## Monitoring & Observability

### Metrics (Prometheus)
- Request rates, latencies, errors
- Resource utilization (CPU, memory, disk)
- Queue depths (Celery, Kafka)
- Custom business metrics

### Dashboards (Grafana)
- System health overview
- Service-specific dashboards
- Database performance
- Storage utilization

### Tracing (OpenTelemetry)
- Distributed request tracing
- Service dependency mapping
- Performance bottleneck identification

### Logging
- Structured JSON logging
- Centralized aggregation (future: ELK stack)
- Log retention: 90 days

## Future Enhancements

### Short Term (3-6 months)
- MONAI Label integration for interactive annotation
- Federated learning support
- Advanced search (OpenSearch)
- Mobile app (React Native)

### Medium Term (6-12 months)
- Multi-tenancy support
- Marketplace for AI models
- Advanced reporting templates
- HL7v2 integration (RIS/PACS)

### Long Term (12+ months)
- Real-time collaboration features
- 3D reconstruction and visualization
- Advanced analytics dashboard
- Blockchain for audit trail immutability

## Technology Stack Summary

| Component | Technology | Version |
|-----------|-----------|---------|
| Frontend | Next.js | 15.x |
| Backend | FastAPI | 0.109+ |
| Database | PostgreSQL + TimescaleDB | 16 |
| Cache | Redis | 7 |
| Object Storage | MinIO | Latest |
| DICOM Server | Orthanc | Latest |
| Identity | Keycloak | 23.x |
| ML Framework | PyTorch + MONAI | 2.1+ / 1.3+ |
| Model Serving | NVIDIA Triton | 23.12 |
| Model Registry | MLflow | 2.9+ |
| Workflow | Airflow/Prefect | Latest |
| Metrics | Prometheus | Latest |
| Visualization | Grafana | Latest |
| Container Runtime | Docker | 24+ |
| Orchestration | Kubernetes | 1.28+ |
| IaC | Terraform | 1.6+ |

## References

- [Data Model](./DATA_MODEL.md)
- [API Contracts](./API_CONTRACTS.md)
- [Security & Compliance](./SECURITY.md)
- [Deployment Guide](./DEPLOYMENT.md)
