# 🏥 Aurelius Medical Imaging Platform + Advanced Cancer AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)
[![HIPAA](https://img.shields.io/badge/HIPAA-Compliant-green.svg)](https://www.hhs.gov/hipaa)

**Enterprise-grade medical imaging platform with integrated AI-powered cancer detection**

---

## 🌟 What's New: Integrated Cancer AI

This platform now combines:
- **Aurelius Medical Imaging Platform**: DICOM processing, PACS functionality, clinical workflows
- **Advanced Cancer AI**: State-of-the-art multimodal cancer detection (Lung, Breast, Prostate, Colorectal)

### Key Integration Features

✅ **Unified Authentication** - Single sign-on via Keycloak
✅ **Shared Infrastructure** - PostgreSQL, Redis, MinIO, Kafka
✅ **Automatic DICOM Analysis** - Cancer AI triggered on study upload
✅ **Unified Dashboard** - Seamless frontend with Cancer AI module
✅ **Production Ready** - Docker Compose & Kubernetes deployment

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Configuration](#-configuration)
- [Development](#-development)
- [Deployment](#-deployment)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Quick Start

### One-Command Startup

```bash
# Clone the repository
git clone https://github.com/alovladi007/Aurelius-Medical-Imaging-Platform.git
cd Aurelius-Medical-Imaging-Platform

# Copy environment template
cp .env.example .env

# Start all services
docker compose up -d

# Wait for services to be healthy (2-3 minutes)
docker compose ps

# Access the platform
open http://localhost:10100
```

### Default Credentials

| Service | URL | Username | Password |
|---------|-----|----------|----------|
| **Frontend** | http://localhost:10100 | admin | admin |
| **Keycloak** | http://localhost:10300 | admin | admin |
| **Grafana** | http://localhost:10500 | admin | admin |
| **MinIO Console** | http://localhost:10701 | minioadmin | minioadmin |
| **Orthanc** | http://localhost:8042 | orthanc | orthanc |

⚠️ **IMPORTANT**: Change all default passwords before production deployment!

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Frontend                          │
│                   (Next.js - Port 10100)                     │
│  Studies | DICOM Viewer | ML Inference | Cancer AI         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               API Gateway (Port 10200)                       │
│  Auth | Routing | Rate Limiting | Audit Logging            │
└─────────────────────────────────────────────────────────────┘
         │          │          │             │
    ┌────┘          │          │             └────┐
    ▼               ▼          ▼                  ▼
Imaging Svc    ML Service  Cancer AI     Search Service
(Port 8001)    (Port 8002) (Port 8003)   (Port 8004)
    │              │           │                │
    └──────────────┴───────────┴────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  PostgreSQL | Redis | MinIO | Kafka  │
    │  Keycloak | Orthanc | Prometheus     │
    └──────────────────────────────────────┘
```

For detailed architecture, see [INTEGRATED_ARCHITECTURE.md](./INTEGRATED_ARCHITECTURE.md)

---

## ✨ Features

### Medical Imaging (Aurelius Platform)

- ✅ **DICOM Support**: Full DICOM protocol (C-STORE, DICOMweb)
- ✅ **PACS Functionality**: Study browser, viewer, worklist management
- ✅ **Multi-Modality**: CT, MRI, X-Ray, Ultrasound, PET, SPECT, WSI
- ✅ **Viewers**: 2D/3D DICOM viewer (Cornerstone3D), WSI viewer (OpenSeadragon)
- ✅ **FHIR R4**: Clinical data interoperability
- ✅ **Multi-Tenant**: Complete data isolation with tenant management

### Cancer AI (Integrated)

- ✅ **Cancer Detection**: Lung, Breast, Prostate, Colorectal cancer
- ✅ **Multimodal AI**: Vision Transformers + EfficientNet ensemble
- ✅ **Clinical Data**: Integrates age, gender, smoking history, family history
- ✅ **Automatic Analysis**: DICOM uploads trigger AI analysis automatically
- ✅ **Batch Processing**: Analyze multiple images simultaneously
- ✅ **Confidence Scoring**: Risk assessment and uncertainty quantification
- ✅ **Recommendations**: Clinical recommendations based on predictions

### Infrastructure & Security

- ✅ **Authentication**: OAuth 2.0 / OpenID Connect via Keycloak
- ✅ **Role-Based Access**: Admin, Clinician, Radiologist, Pathologist, ML Engineer, Researcher, Student
- ✅ **HIPAA Compliance**: Audit logging, encryption, de-identification
- ✅ **Observability**: Prometheus metrics, Grafana dashboards, Jaeger tracing
- ✅ **Scalability**: Kubernetes-ready with Helm charts

---

## 📦 Prerequisites

### Required

- **Docker**: 24.0+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: 2.20+ ([Install Docker Compose](https://docs.docker.com/compose/install/))
- **System Resources**:
  - CPU: 8+ cores recommended
  - RAM: 16+ GB (32 GB for optimal performance)
  - Disk: 100+ GB free space
  - Network: Stable internet connection

### Optional (for development)

- **Node.js**: 18+ (for frontend development)
- **Python**: 3.11+ (for backend development)
- **NVIDIA GPU**: For GPU-accelerated AI inference

---

## 📥 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/alovladi007/Aurelius-Medical-Imaging-Platform.git
cd Aurelius-Medical-Imaging-Platform
```

### Step 2: Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env  # or your preferred editor
```

**Key settings to configure:**

```env
# Change passwords
POSTGRES_PASSWORD=your-secure-password
REDIS_PASSWORD=your-redis-password
KEYCLOAK_ADMIN_PASSWORD=your-admin-password

# Configure storage
MINIO_ROOT_PASSWORD=your-minio-password

# Security
SECRET_KEY=your-32-character-minimum-secret-key
JWT_SECRET=your-jwt-secret-key

# Frontend URL (if not localhost)
NEXT_PUBLIC_API_URL=https://your-domain.com
```

### Step 3: Start Services

```bash
# Start all services
docker compose up -d

# Check service health
docker compose ps

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f cancer-ai-svc
docker compose logs -f gateway
```

### Step 4: Initial Setup

```bash
# Wait for all services to be healthy (2-3 minutes)
# You can check status with:
watch docker compose ps

# Access the frontend
open http://localhost:10100

# Login with default credentials
# Username: admin
# Password: admin
```

---

## 💡 Usage

### Accessing the Platform

| Component | URL | Purpose |
|-----------|-----|---------|
| **Main Dashboard** | http://localhost:10100 | Unified frontend interface |
| **Cancer AI Dashboard** | http://localhost:10100/cancer-ai | Cancer AI module |
| **API Gateway** | http://localhost:10200 | REST API endpoints |
| **API Documentation** | http://localhost:10200/docs | Swagger/OpenAPI docs |
| **Orthanc DICOM Server** | http://localhost:8042 | DICOM web interface |
| **Grafana Monitoring** | http://localhost:10500 | System dashboards |
| **MinIO Console** | http://localhost:10701 | Object storage management |

### Using Cancer AI

#### Method 1: Web Interface

1. Navigate to http://localhost:10100/cancer-ai
2. Click "New Prediction"
3. Upload medical image (DICOM, PNG, JPG)
4. (Optional) Add clinical information
5. Click "Analyze with AI"
6. View results and recommendations

#### Method 2: API

```bash
# Predict from image file
curl -X POST http://localhost:10200/cancer-ai/predict \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "image=@scan.dcm" \
  -F "patient_age=55" \
  -F "smoking_history=true"

# Response:
{
  "cancer_type": "Lung Cancer",
  "confidence": 0.87,
  "risk_score": 0.75,
  "uncertainty": 0.12,
  "recommendations": [
    "Immediate consultation with oncologist recommended",
    "Additional diagnostic tests may be required",
    "Smoking cessation counseling strongly recommended"
  ]
}
```

#### Method 3: Automatic DICOM Analysis

1. Send DICOM study to Orthanc:
   ```bash
   # Using DICOM C-STORE
   dcmsend localhost 4242 study/*.dcm
   ```

2. Orthanc automatically triggers Cancer AI analysis (via Lua hook)

3. View results in the dashboard under "Cancer AI" → "History"

### Uploading DICOM Studies

```bash
# Install DICOM tools (if needed)
# Ubuntu/Debian
sudo apt-get install dcmtk

# macOS
brew install dcmtk

# Send DICOM files to Orthanc
dcmsend localhost 4242 /path/to/dicom/files/*.dcm

# Or use DICOMweb (STOW-RS)
curl -X POST http://localhost:8042/dicom-web/studies \
  -H "Content-Type: multipart/related" \
  --data-binary @study.dcm
```

---

## 📚 API Documentation

### Cancer AI Endpoints

#### POST /cancer-ai/predict

Predict cancer from uploaded medical image.

**Request:**
```http
POST /cancer-ai/predict
Content-Type: multipart/form-data
Authorization: Bearer {token}

image: <file>
clinical_notes: string (optional)
patient_age: integer (optional)
patient_gender: string (optional)
smoking_history: boolean (optional)
family_history: boolean (optional)
```

**Response:**
```json
{
  "cancer_type": "Lung Cancer",
  "risk_score": 0.75,
  "confidence": 0.87,
  "uncertainty": 0.12,
  "recommendations": ["..."],
  "all_probabilities": {
    "No Cancer": 0.05,
    "Lung Cancer": 0.87,
    "Breast Cancer": 0.03,
    "Prostate Cancer": 0.02,
    "Colorectal Cancer": 0.03
  }
}
```

#### POST /cancer-ai/predict/batch

Batch prediction for multiple images.

#### GET /cancer-ai/model/info

Get Cancer AI model information.

#### GET /cancer-ai/health

Health check for Cancer AI service.

For complete API documentation, visit: http://localhost:10200/docs

---

## ⚙️ Configuration

### Docker Compose Services

The platform runs 20 Docker containers:

**Infrastructure (6):**
- postgres, redis, minio, keycloak, kafka, orthanc

**Application (6):**
- gateway, imaging-svc, ml-svc, cancer-ai-svc, search-svc, celery-worker

**Frontend (1):**
- frontend

**Data & ML (3):**
- fhir-server, mlflow, opensearch

**Observability (4):**
- prometheus, grafana, jaeger, opensearch-dashboards

### Resource Limits

Edit `docker-compose.yml` to adjust resource limits:

```yaml
cancer-ai-svc:
  deploy:
    resources:
      limits:
        cpus: '4'
        memory: 8G
      reservations:
        cpus: '2'
        memory: 4G
```

### GPU Support (Optional)

To enable GPU acceleration for Cancer AI:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Edit `docker-compose.yml`:

```yaml
cancer-ai-svc:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  environment:
    CANCER_AI_GPU_ENABLED: "true"
```

3. Restart service:
```bash
docker compose up -d cancer-ai-svc
```

---

## 🛠️ Development

### Local Development Setup

```bash
# Backend development (FastAPI)
cd "Aurelius Advanced Medical Imaging Platform/apps/gateway"
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend development (Next.js)
cd "Aurelius Advanced Medical Imaging Platform/apps/frontend"
npm install
npm run dev  # Runs on http://localhost:3000
```

### Running Tests

```bash
# Backend tests
pytest apps/gateway/tests -v

# Frontend tests
cd apps/frontend
npm test
```

### Code Quality

```bash
# Python linting
flake8 apps/gateway/app
black apps/gateway/app

# TypeScript/JavaScript linting
cd apps/frontend
npm run lint
npm run format
```

---

## 🚢 Deployment

### Docker Compose (Production)

```bash
# Use production environment
cp .env.example .env.production
nano .env.production  # Configure for production

# Start with production settings
docker compose -f docker-compose.yml --env-file .env.production up -d

# Enable SSL/TLS
# Update docker-compose.yml with SSL certificates
```

### Kubernetes Deployment

```bash
# Navigate to Helm chart
cd "Aurelius Advanced Medical Imaging Platform/infra/k8s/helm/aurelius"

# Install with Helm
helm install aurelius . \
  --namespace aurelius \
  --create-namespace \
  --values values-prod.yaml

# Check deployment
kubectl get pods -n aurelius
kubectl get services -n aurelius
```

### Cloud Deployment

See deployment guides:
- [AWS Deployment](./docs/deployment/AWS.md)
- [Google Cloud Deployment](./docs/deployment/GCP.md)
- [Azure Deployment](./docs/deployment/Azure.md)

---

## 🔒 Security

### HIPAA Compliance

This platform includes HIPAA-ready features:

✅ **Audit Logging**: All PHI access logged for 7 years
✅ **Encryption**: At rest (AES-256) and in transit (TLS 1.3)
✅ **Access Control**: Role-based with MFA support
✅ **De-identification**: HIPAA Safe Harbor compliance
✅ **Data Breach Procedures**: Templates and notification workflows

⚠️ **Note**: This software provides HIPAA-ready infrastructure but requires proper configuration, policies, and procedures for full compliance.

### Security Checklist

- [ ] Change all default passwords
- [ ] Configure SSL/TLS certificates
- [ ] Enable MFA for admin accounts
- [ ] Set up regular backups
- [ ] Configure firewall rules
- [ ] Enable audit logging
- [ ] Review and restrict CORS origins
- [ ] Implement network security policies
- [ ] Set up monitoring and alerting
- [ ] Conduct security audit

### Authentication

All API requests require JWT authentication:

```bash
# Login to get token
curl -X POST http://localhost:10200/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# Use token in requests
curl -X GET http://localhost:10200/studies \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## 🐛 Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check service logs
docker compose logs [service-name]

# Restart specific service
docker compose restart [service-name]

# Rebuild and restart
docker compose up -d --build [service-name]
```

#### Database Connection Errors

```bash
# Check PostgreSQL is running
docker compose ps postgres

# Check database logs
docker compose logs postgres

# Restart database
docker compose restart postgres
```

#### Cancer AI Inference Errors

```bash
# Check Cancer AI logs
docker compose logs cancer-ai-svc

# Verify model file exists
docker compose exec cancer-ai-svc ls -la /app/models/

# Restart service
docker compose restart cancer-ai-svc
```

#### Out of Memory

```bash
# Check resource usage
docker stats

# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory
# Allocate at least 16 GB
```

### Getting Help

1. **Check logs**: `docker compose logs -f`
2. **Review documentation**: See `/docs` directory
3. **GitHub Issues**: https://github.com/alovladi007/Aurelius-Medical-Imaging-Platform/issues
4. **Contact**: Create an issue with logs and configuration details

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- Additional cancer types
- New AI models
- Performance optimizations
- Documentation improvements
- Bug fixes
- Test coverage
- Translations (i18n)

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for research and educational purposes only.

- NOT approved for clinical diagnosis or treatment decisions
- NOT a substitute for professional medical advice
- NOT validated for regulatory compliance (FDA, CE, etc.)
- Requires validation and approval before any clinical use
- Users assume all responsibility for any application

Always consult qualified healthcare professionals for medical decisions.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Aurelius Platform**: Original medical imaging platform architecture
- **Cancer AI**: Advanced multimodal cancer detection system
- **Open Source Libraries**:
  - FastAPI, Next.js, PyTorch, ONNX Runtime
  - Keycloak, PostgreSQL, Redis, MinIO
  - Orthanc, HAPI FHIR, MLflow
  - Cornerstone3D, OpenSeadragon

---

## 📧 Contact

For questions, issues, or collaboration opportunities:
- **GitHub Issues**: https://github.com/alovladi007/Aurelius-Medical-Imaging-Platform/issues
- **Documentation**: See `/docs` directory
- **Architecture**: See [INTEGRATED_ARCHITECTURE.md](./INTEGRATED_ARCHITECTURE.md)

---

## 📊 Project Status

- ✅ **Phase 1**: Platform integration (COMPLETE)
- ✅ **Phase 2**: Unified deployment (COMPLETE)
- ✅ **Phase 3**: Cancer AI integration (COMPLETE)
- 🚧 **Phase 4**: Production testing (IN PROGRESS)
- 📋 **Phase 5**: Clinical validation (PLANNED)

---

**Built with ❤️ for advancing cancer detection and medical imaging**

Last Updated: November 2025 | Version: 1.0.0
