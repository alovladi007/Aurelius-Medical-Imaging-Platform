# Aurelius Medical Imaging Platform - Quick Reference

## Quick Start Commands

```bash
# Verify setup
./verify-setup.sh

# Start all services
./start.sh

# Stop all services
docker compose down

# View logs
docker compose logs -f

# View specific service logs
docker compose logs -f gateway

# Restart a service
docker compose restart gateway

# Rebuild after code changes
docker compose up -d --build gateway

# Complete reset (removes all data)
docker compose down -v
./start.sh
```

## Service URLs

| Service | URL | Login |
|---------|-----|-------|
| API Gateway | http://localhost:8000 | - |
| API Docs (Swagger) | http://localhost:8000/docs | - |
| API Docs (ReDoc) | http://localhost:8000/redoc | - |
| Keycloak Admin | http://localhost:8080 | admin/admin |
| Grafana | http://localhost:3001 | admin/admin |
| Prometheus | http://localhost:9090 | - |
| Jaeger Tracing | http://localhost:16686 | - |
| MinIO Console | http://localhost:9001 | minioadmin/minioadmin |
| Orthanc DICOM | http://localhost:8042 | orthanc/orthanc |
| FHIR Server | http://localhost:8083/fhir | - |
| OpenSearch | http://localhost:9200 | - |
| MLflow | http://localhost:5000 | - |

## Service Ports

| Service | Port(s) | Purpose |
|---------|---------|---------|
| Gateway | 8000 | Main API |
| Imaging | 8001 | DICOM processing |
| ML Service | 8002 | AI inference |
| Search | 8004 | Full-text search |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache/Queue |
| Keycloak | 8080 | Auth |
| Orthanc | 8042, 4242 | DICOM |
| FHIR | 8083 | HL7 FHIR |
| MinIO | 9000, 9001 | S3 Storage |
| Prometheus | 9090 | Metrics |
| Grafana | 3001 | Dashboards |
| Jaeger | 16686 | Tracing |
| OpenSearch | 9200 | Search |
| Kafka | 9092 | Messaging |
| MLflow | 5000 | ML Registry |

## Health Check Endpoints

```bash
# Gateway
curl http://localhost:8000/health

# Imaging Service
curl http://localhost:8001/health

# ML Service
curl http://localhost:8002/health

# Search Service
curl http://localhost:8004/health

# Keycloak
curl http://localhost:8080/health/ready

# Orthanc
curl http://localhost:8042/system

# MinIO
curl http://localhost:9000/minio/health/live
```

## Common API Calls

### Get API Info
```bash
curl http://localhost:8000/
```

### List Studies
```bash
curl http://localhost:8000/studies
```

### Get Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

## Database Access

```bash
# PostgreSQL CLI
docker exec -it aurelius-postgres psql -U postgres -d aurelius

# Run SQL file
docker exec -i aurelius-postgres psql -U postgres -d aurelius < file.sql

# Backup database
docker exec aurelius-postgres pg_dump -U postgres aurelius > backup.sql

# Redis CLI
docker exec -it aurelius-redis redis-cli
```

## File Locations

### Application Code
- `apps/gateway/app/main.py` - Gateway entry point
- `apps/gateway/app/core/` - Core modules
- `apps/gateway/app/api/` - API endpoints

### Configuration
- `.env` - Environment variables
- `compose.yaml` - Docker Compose
- `prometheus.yml` - Metrics config

### Database
- `infra/postgres/001_initial_schema.sql` - Main schema
- `infra/postgres/014_add_multitenancy.py` - Tenancy setup

### Documentation
- `INTEGRATION_GUIDE.md` - Complete guide
- `INTEGRATION_COMPLETE.md` - Integration summary
- `README.md` - Project overview

## Docker Commands

```bash
# List running containers
docker ps

# List all containers
docker ps -a

# View container logs
docker logs aurelius-gateway

# Execute command in container
docker exec -it aurelius-gateway bash

# View resource usage
docker stats

# Remove stopped containers
docker compose rm

# Remove all (including volumes)
docker compose down -v

# Rebuild specific service
docker compose build gateway

# Scale a service
docker compose up -d --scale gateway=3
```

## Development Workflow

### 1. Make Code Changes
Edit files in `apps/gateway/app/`

### 2. Rebuild Service
```bash
docker compose up -d --build gateway
```

### 3. Check Logs
```bash
docker compose logs -f gateway
```

### 4. Test Changes
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Service Won't Start
```bash
# Check logs
docker compose logs gateway

# Check status
docker compose ps

# Restart service
docker compose restart gateway
```

### Port Already in Use
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Database Connection Issues
```bash
# Check PostgreSQL
docker exec aurelius-postgres pg_isready -U postgres

# Restart PostgreSQL
docker compose restart postgres
```

### Reset Everything
```bash
docker compose down -v
docker system prune -a
./start.sh
```

## Authentication

### Get Access Token
```bash
curl -X POST "http://localhost:8080/realms/aurelius/protocol/openid-connect/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id=gateway" \
  -d "client_secret=gateway-secret" \
  -d "grant_type=password" \
  -d "username=your-username" \
  -d "password=your-password"
```

### Use Token
```bash
curl -H "Authorization: Bearer <TOKEN>" \
  http://localhost:8000/studies
```

## MinIO S3 Access

```bash
# MinIO client
docker exec aurelius-minio-init mc alias set myminio http://minio:9000 minioadmin minioadmin

# List buckets
docker exec aurelius-minio-init mc ls myminio

# Upload file
docker exec aurelius-minio-init mc cp file.txt myminio/dicom-studies/
```

## DICOM Operations

### Send DICOM via dcmsend
```bash
dcmsend localhost 4242 study.dcm
```

### DICOMweb STOW-RS
```bash
curl -X POST http://localhost:8042/dicom-web/studies \
  -H "Content-Type: multipart/related; type=application/dicom" \
  --data-binary @study.dcm
```

### Query Studies (QIDO-RS)
```bash
curl http://localhost:8042/dicom-web/studies
```

## Monitoring

### View Metrics
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001 (pre-built dashboards)

### View Traces
- Jaeger: http://localhost:16686

### Custom Metrics
All services expose metrics at `/metrics` endpoint

## Environment Variables

Key variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/aurelius

# Redis
REDIS_URL=redis://:redis123@redis:6379/0

# Keycloak
KEYCLOAK_URL=http://localhost:8080
KEYCLOAK_CLIENT_ID=gateway
KEYCLOAK_CLIENT_SECRET=gateway-secret

# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Debug
DEBUG=true
LOG_LEVEL=INFO
```

## Production Deployment

See detailed guides:
- `KUBERNETES_QUICK_REFERENCE.md` - Kubernetes deployment
- `deploy.sh` - Deployment script
- `values-prod.yaml` - Production values

## Support

- Documentation: See all `.md` files in repository
- Logs: `docker compose logs [service]`
- Health: Check `/health` endpoints
- Issues: Review Docker Compose output

## Quick Links

- 📚 Full Guide: [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- 🎉 Summary: [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md)
- 📖 Overview: [README.md](README.md)
- 🏗️  Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- 🔒 Security: [SECURITY.md](SECURITY.md)

---

**Tip**: Bookmark this page for quick reference!
