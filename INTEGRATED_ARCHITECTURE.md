# Integrated Aurelius + Cancer AI Platform Architecture

## System Overview

This document describes the fully integrated architecture combining the Aurelius Medical Imaging Platform with the Advanced Cancer AI system.

---

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            UNIFIED FRONTEND                                  │
│                         Next.js 14 (Port 10100)                             │
│  ┌──────────────┬──────────────┬──────────────┬──────────────────────────┐ │
│  │   Studies    │   DICOM      │     ML       │      Cancer AI           │ │
│  │   Browser    │   Viewer     │  Inference   │      Dashboard           │ │
│  └──────────────┴──────────────┴──────────────┴──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY (Port 10200)                            │
│                            FastAPI + Keycloak                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Authentication  │  Rate Limiting  │  Audit Logging  │  Metrics       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Routes:                                                                    │
│    /studies        → Imaging Service                                       │
│    /imaging/*      → Imaging Service                                       │
│    /ml/*           → ML Service                                            │
│    /cancer-ai/*    → Cancer AI Service (NEW)                               │
│    /worklists/*    → Backend Services                                      │
└─────────────────────────────────────────────────────────────────────────────┘
              │              │              │                │
      ┌───────┘              │              │                └────────┐
      ▼                      ▼              ▼                         ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│  Imaging     │   │   ML         │   │  Search      │   │  Cancer AI       │
│  Service     │   │   Service    │   │  Service     │   │  Service (NEW)   │
│  (Port 8001) │   │  (Port 8002) │   │  (Port 8004) │   │  (Port 8003)     │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────────┘
      │                    │                  │                     │
      │                    │                  │                     │
      └────────────────────┴──────────────────┴─────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SHARED INFRASTRUCTURE                                │
│  ┌──────────────┬──────────────┬──────────────┬──────────────────────────┐ │
│  │ PostgreSQL   │    Redis     │    MinIO     │      Keycloak            │ │
│  │ (10400)      │   (6379)     │  (10700)     │      (10300)             │ │
│  └──────────────┴──────────────┴──────────────┴──────────────────────────┘ │
│  ┌──────────────┬──────────────┬──────────────┬──────────────────────────┐ │
│  │  Orthanc     │   Kafka      │  Prometheus  │     Grafana              │ │
│  │  (8042)      │   (9092)     │   (10600)    │      (10500)             │ │
│  └──────────────┴──────────────┴──────────────┴──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### 1. DICOM Upload → Automatic Cancer AI Analysis

```
┌─────────────┐
│  DICOM      │
│  C-STORE    │
│  Client     │
└──────┬──────┘
       │ DICOM Protocol (Port 4242)
       ▼
┌─────────────────────────────────────┐
│          Orthanc Server             │
│  • Receives DICOM                   │
│  • Stores in PostgreSQL             │
│  • Triggers Lua Hook                │
└──────┬──────────────────────────────┘
       │ Lua Script: cancer_ai_hook.lua
       │ (Auto-trigger for CT, MRI, X-Ray)
       ▼
┌─────────────────────────────────────┐
│       API Gateway                   │
│  POST /cancer-ai/predict/dicom      │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│     Cancer AI Service               │
│  • Fetches DICOM from Orthanc       │
│  • Runs inference (ONNX/PyTorch)    │
│  • Returns prediction                │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│      PostgreSQL Database            │
│  • Stores prediction results        │
│  • Links to study UID               │
└─────────────────────────────────────┘
```

### 2. Manual Cancer AI Prediction via UI

```
┌─────────────────┐
│  User uploads   │
│  medical image  │
│  via Frontend   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    Next.js Frontend                 │
│    /cancer-ai/predict               │
└────────┬────────────────────────────┘
         │ POST /api/cancer-ai/predict
         ▼
┌─────────────────────────────────────┐
│    API Gateway                      │
│    /cancer-ai/predict               │
│  • Validates auth token             │
│  • Checks user role                 │
│  • Logs request                     │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    Cancer AI Service                │
│  • Preprocesses image               │
│  • Runs ONNX inference              │
│  • Generates recommendations        │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│    Response to User                 │
│  • Cancer type                      │
│  • Confidence score                 │
│  • Risk assessment                  │
│  • Clinical recommendations         │
└─────────────────────────────────────┘
```

---

## Service Integration Matrix

| Service | Port | Purpose | Integrations |
|---------|------|---------|--------------|
| **Frontend** | 10100 | Unified Next.js UI | Gateway API, Keycloak |
| **Gateway** | 10200 | API routing & auth | All services, Keycloak |
| **Imaging Svc** | 8001 | DICOM processing | Orthanc, MinIO, Cancer AI |
| **ML Svc** | 8002 | General ML inference | MinIO, MLflow |
| **Cancer AI Svc** | 8003 | Cancer detection | MinIO, PostgreSQL |
| **Search Svc** | 8004 | Full-text search | OpenSearch, PostgreSQL |
| **PostgreSQL** | 10400 | Primary database | All services |
| **Redis** | 6379 | Cache & queue | All services |
| **MinIO** | 10700 | Object storage | Imaging, ML, Cancer AI |
| **Keycloak** | 10300 | Authentication | Gateway, Frontend |
| **Orthanc** | 8042 | DICOM server | Imaging Svc, Cancer AI |
| **Prometheus** | 10600 | Metrics | All services |
| **Grafana** | 10500 | Dashboards | Prometheus |

---

## Database Schema Integration

### New Cancer AI Tables

```sql
-- Cancer AI Predictions
CREATE TABLE cancer_ai_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    study_id UUID REFERENCES studies(id),  -- Links to existing Aurelius studies
    image_path VARCHAR(500),

    -- Prediction Results
    cancer_type VARCHAR(100),              -- Lung, Breast, Prostate, Colorectal, None
    risk_score FLOAT,                      -- 0.0 to 1.0
    confidence FLOAT,                      -- 0.0 to 1.0
    uncertainty FLOAT,                     -- Entropy measure

    -- Clinical Data
    patient_age INTEGER,
    patient_gender VARCHAR(10),
    smoking_history BOOLEAN,
    family_history BOOLEAN,
    clinical_notes TEXT,

    -- Model Info
    model_version VARCHAR(50),
    inference_time_ms INTEGER,

    -- Probabilities
    probabilities JSONB,                   -- All class probabilities
    recommendations TEXT[],                 -- Clinical recommendations

    -- Metadata
    auto_triggered BOOLEAN DEFAULT FALSE,  -- True if triggered by Orthanc hook
    reviewed_by_clinician UUID REFERENCES users(id),
    review_notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_cancer_predictions_study ON cancer_ai_predictions(study_id);
CREATE INDEX idx_cancer_predictions_user ON cancer_ai_predictions(user_id);
CREATE INDEX idx_cancer_predictions_type ON cancer_ai_predictions(cancer_type);
CREATE INDEX idx_cancer_predictions_created ON cancer_ai_predictions(created_at DESC);
```

---

## Authentication & Authorization Flow

```
┌──────────────┐
│    User      │
│   Browser    │
└──────┬───────┘
       │ 1. Access frontend
       ▼
┌──────────────────────┐
│  Next.js Frontend    │
└──────┬───────────────┘
       │ 2. Redirect to Keycloak if not authenticated
       ▼
┌──────────────────────┐
│    Keycloak          │
│  • User login        │
│  • MFA (optional)    │
│  • Returns JWT       │
└──────┬───────────────┘
       │ 3. JWT token
       ▼
┌──────────────────────┐
│  Frontend stores     │
│  token in session    │
└──────┬───────────────┘
       │ 4. API request with Bearer token
       ▼
┌──────────────────────┐
│   API Gateway        │
│  • Validates JWT     │
│  • Checks roles      │
│  • Extracts user ID  │
└──────┬───────────────┘
       │ 5. Authorized request
       ▼
┌──────────────────────┐
│  Backend Services    │
│  (Imaging, ML,       │
│   Cancer AI, etc.)   │
└──────────────────────┘
```

### Role-Based Access Control

| Role | Cancer AI Permissions |
|------|----------------------|
| **admin** | Full access, statistics, model management |
| **clinician** | Create predictions, view results, review |
| **radiologist** | Create predictions, DICOM integration |
| **pathologist** | Create predictions, WSI analysis |
| **ml-engineer** | Model info, statistics, batch processing |
| **researcher** | De-identified predictions only |
| **student** | View-only educational access |

---

## Storage Architecture

### MinIO Buckets

```
minio:9000/
├── dicom-studies/          # Original DICOM files
│   └── {study-uid}/
│       └── {series-uid}/
│           └── {instance-uid}.dcm
│
├── wsi-slides/             # Whole slide images
│   └── {slide-id}/
│
├── ml-models/              # General ML models (MLflow)
│   └── {model-name}/
│       └── {version}/
│
├── cancer-ai-models/       # Cancer AI specific models (NEW)
│   ├── cancer_detector.onnx
│   ├── cancer_detector.pt
│   └── config.yaml
│
└── processed-data/         # De-identified datasets
    └── {dataset-id}/
```

---

## Deployment Topology

### Docker Compose Services

```yaml
Services (20 containers):

Infrastructure (6):
  - postgres          PostgreSQL + TimescaleDB
  - redis             Cache and message broker
  - minio             S3-compatible object storage
  - keycloak          Identity and access management
  - kafka             Event streaming
  - orthanc           DICOM server

Application (6):
  - gateway           API Gateway + routing
  - imaging-svc       DICOM processing
  - ml-svc            General ML inference
  - cancer-ai-svc     Cancer detection AI (NEW)
  - search-svc        Full-text search
  - celery-worker     Background jobs

Frontend (1):
  - frontend          Unified Next.js application

Data & ML (3):
  - fhir-server       FHIR R4 server
  - mlflow            ML experiment tracking
  - opensearch        Search engine

Observability (4):
  - prometheus        Metrics collection
  - grafana           Dashboards
  - jaeger            Distributed tracing
  - opensearch-dashboards  Search UI
```

---

## Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network: aurelius-net             │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              External Access (Host Ports)              │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  10100  →  Frontend (Next.js)                         │ │
│  │  10200  →  Gateway (FastAPI)                          │ │
│  │  10300  →  Keycloak                                   │ │
│  │  10400  →  PostgreSQL                                 │ │
│  │  10500  →  Grafana                                    │ │
│  │  10600  →  Prometheus                                 │ │
│  │  10700  →  MinIO S3 API                               │ │
│  │  10701  →  MinIO Console                              │ │
│  │  10800  →  MLflow                                     │ │
│  │  10900  →  FHIR Server                                │ │
│  │  11000  →  OpenSearch                                 │ │
│  │  8042   →  Orthanc Web UI                             │ │
│  │  4242   →  Orthanc DICOM                              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            Internal Service Communication              │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  imaging-svc:8001      ←→  orthanc:8042               │ │
│  │  ml-svc:8002           ←→  minio:9000                 │ │
│  │  cancer-ai-svc:8003    ←→  postgres:5432              │ │
│  │  search-svc:8004       ←→  opensearch:9200            │ │
│  │  All services          ←→  redis:6379                 │ │
│  │  All services          ←→  keycloak:8080              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Integration Points

### 1. **Shared Authentication**
- All services use Keycloak for SSO
- JWT tokens validated at Gateway level
- Role-based access control enforced
- Session management in Redis

### 2. **Shared Database**
- PostgreSQL stores all application data
- Cancer AI predictions linked to existing studies
- TimescaleDB for time-series audit logs
- Unified user management

### 3. **Shared Storage**
- MinIO buckets for all services
- DICOM files accessible by both Imaging and Cancer AI
- Model artifacts stored centrally
- Automatic bucket creation on startup

### 4. **DICOM Pipeline Integration**
- Orthanc receives DICOM studies
- Lua hook automatically triggers Cancer AI analysis
- Results stored and linked to original study
- Async processing via Celery workers

### 5. **Unified Frontend**
- Single Next.js application
- Cancer AI dashboard integrated as `/cancer-ai/*` routes
- Seamless navigation between features
- Consistent UI/UX with shadcn/ui components

### 6. **Observability**
- Prometheus collects metrics from all services
- Grafana dashboards for system monitoring
- Jaeger for distributed tracing
- OpenTelemetry instrumentation

---

## Performance & Scaling

### Resource Allocation

| Service | CPU | Memory | Storage |
|---------|-----|--------|---------|
| PostgreSQL | 2 cores | 4 GB | 100 GB+ |
| Redis | 1 core | 2 GB | 10 GB |
| MinIO | 2 cores | 4 GB | 1 TB+ |
| Cancer AI | 4 cores* | 8 GB | 50 GB |
| Orthanc | 2 cores | 4 GB | 500 GB+ |
| Gateway | 1 core | 2 GB | 1 GB |
| Frontend | 1 core | 1 GB | 1 GB |

*With GPU: 1+ NVIDIA GPU with 8GB+ VRAM

### Scaling Strategies

1. **Horizontal Scaling**
   - Multiple gateway instances behind load balancer
   - Multiple Cancer AI workers for inference
   - Celery workers for async processing
   - Read replicas for PostgreSQL

2. **GPU Acceleration**
   - NVIDIA GPU for Cancer AI inference
   - CUDA support via Docker
   - Batch processing for efficiency

3. **Caching**
   - Redis for API response caching
   - Model predictions cached (optional)
   - Study metadata cached

---

## Security Considerations

### 1. **Data Protection**
- ✅ TLS 1.3 for external communication
- ✅ Encryption at rest (MinIO SSE, PostgreSQL TDE)
- ✅ PHI de-identification pipeline
- ✅ Audit logging for all access

### 2. **Authentication & Authorization**
- ✅ OAuth 2.0 / OpenID Connect (Keycloak)
- ✅ JWT token validation
- ✅ Role-based access control
- ✅ MFA support (optional)

### 3. **Network Security**
- ✅ Docker network isolation
- ✅ Service-to-service authentication
- ✅ Rate limiting at gateway
- ✅ CORS policies configured

### 4. **Compliance**
- ✅ HIPAA-ready architecture
- ✅ Audit trail (7-year retention)
- ✅ Data breach notification templates
- ✅ Consent tracking

---

## Monitoring & Alerts

### Prometheus Metrics

```yaml
# Cancer AI specific metrics
cancer_ai_predictions_total{type="lung|breast|prostate|colorectal"}
cancer_ai_inference_duration_seconds
cancer_ai_confidence_score{type="*"}
cancer_ai_model_load_time_seconds

# Gateway metrics
http_requests_total{service="cancer-ai", status="200|400|500"}
http_request_duration_seconds{endpoint="/cancer-ai/*"}
auth_failures_total
rate_limit_exceeded_total
```

### Grafana Dashboards

1. **System Overview**
   - Service health status
   - Request rates
   - Error rates
   - Resource utilization

2. **Cancer AI Performance**
   - Predictions per hour
   - Inference time distribution
   - Confidence score distribution
   - Cancer type breakdown

3. **DICOM Pipeline**
   - Studies received
   - Auto-analysis trigger rate
   - Processing time
   - Storage utilization

---

## Disaster Recovery

### Backup Strategy

```bash
# Automated daily backups
PostgreSQL:  pg_dump → S3 (7-day retention)
MinIO:       Bucket replication → S3 (30-day retention)
Redis:       RDB snapshots → S3 (3-day retention)

# Critical data
- DICOM studies (永久保存)
- Cancer AI predictions (7 years - HIPAA)
- Audit logs (7 years - HIPAA)
- User data (indefinite)
```

### Recovery Procedures

1. **Database Recovery**: Restore from latest pg_dump
2. **Storage Recovery**: Restore MinIO buckets from S3
3. **Service Recovery**: Redeploy containers from images
4. **Configuration Recovery**: GitOps approach (all config in Git)

---

## Future Enhancements

1. **Federated Learning**: Multi-institutional model training
2. **Real-time Collaboration**: WebRTC for telepathology
3. **Advanced Visualizations**: 3D reconstruction, heatmaps
4. **Mobile App**: React Native app for clinicians
5. **API Gateway**: GraphQL federation
6. **Multi-region**: Global deployment with data residency

---

## Contact & Support

For technical questions or issues:
- Create an issue in the GitHub repository
- Review documentation in `/docs`
- Check logs: `docker-compose logs -f [service-name]`

---

**Last Updated**: November 2025
**Version**: 1.0.0
