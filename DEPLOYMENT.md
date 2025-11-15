# Deployment Guide

## ⚠️ IMPORTANT: Which Docker Compose File to Use

### ✅ Use This for Integrated Platform
```bash
# Root directory - UNIFIED DEPLOYMENT (recommended)
docker-compose.yml
```

This is the **main deployment file** that runs the complete integrated platform with all 22 services including:
- Aurelius Medical Imaging Platform
- Advanced Cancer AI system
- Shared infrastructure (PostgreSQL, Keycloak, MinIO, etc.)

### 📦 Legacy/Standalone Files (Not for Production)

The following files are kept for reference only:

1. **`Aurelius Advanced Medical Imaging Platform/compose.yaml.legacy`**
   - Old Aurelius-only deployment (before Cancer AI integration)
   - Use only if you need to run Aurelius standalone without Cancer AI
   - Not recommended - missing latest features

2. **`advanced-cancer-ai/docker-compose.yml.standalone`**
   - Standalone Cancer AI deployment
   - Use only for Cancer AI development/testing in isolation
   - Not recommended - missing integration with DICOM pipeline

## 🚀 Recommended Deployment

Always use the root `docker-compose.yml`:

```bash
cd "/Users/vladimirantoine/Aurelius Medical Imaging Platform/Aurelius-Medical-Imaging-Platform"

# Copy environment template
cp .env.example .env

# Start integrated platform
docker compose up -d

# Access at http://localhost:10100
```

## 📊 Service Architecture

The root `docker-compose.yml` deploys:
- **Infrastructure**: PostgreSQL, Redis, MinIO, Keycloak, Kafka, Orthanc, Jaeger
- **Application Services**: Gateway, Imaging, ML, Cancer AI, Search, Celery
- **Frontend**: Next.js unified dashboard
- **Data & ML**: FHIR, MLflow, OpenSearch
- **Observability**: Prometheus, Grafana, OpenSearch Dashboards

Total: 22 integrated microservices

## 🔍 Quick Reference

| File | Purpose | Status | Use Case |
|------|---------|--------|----------|
| `docker-compose.yml` | Unified platform | ✅ Active | **Production & Development** |
| `Aurelius.../compose.yaml.legacy` | Aurelius only | ⚠️ Legacy | Reference only |
| `advanced-cancer-ai/...standalone` | Cancer AI only | ⚠️ Legacy | Standalone development |

---

**Last Updated**: November 2025
**Platform Version**: 1.0.0
