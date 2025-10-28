# ⚠️ Port Configuration Changes

## Overview

Due to port conflicts on the system, all external ports have been updated to use available ports. **Internal container ports remain unchanged** - only the host machine ports have been modified.

## Port Mapping Changes

### Original vs New Ports

| Service | Old Port | New Port | Internal Port |
|---------|----------|----------|---------------|
| **Frontend** | 3000 | **3100** | 3000 |
| **API Gateway** | 8000 | **8200** | 8000 |
| **Keycloak** | 8080 | **8180** | 8080 |
| **PostgreSQL** | 5432 | **5434** | 5432 |
| **MLflow** | 5000 | **5100** | 5000 |
| **Grafana** | 3001 | **3101** | 3000 |
| **FHIR Server** | 8083 | **8183** | 8080 |
| **MinIO API** | 9000 | **9100** | 9000 |
| **MinIO Console** | 9001 | **9101** | 9001 |
| **Prometheus** | 9090 | **9190** | 9090 |
| **OpenSearch** | 9200 | **9300** | 9200 |

### Unchanged Ports

These ports did not have conflicts:

| Service | Port | Notes |
|---------|------|-------|
| Redis | 6379 | No conflict |
| Orthanc Web | 8042 | No conflict |
| Orthanc DICOM | 4242 | No conflict |
| Imaging Service | 8001 | No conflict |
| ML Service | 8002 | No conflict |
| Search Service | 8004 | No conflict |
| Jaeger UI | 16686 | No conflict |
| OpenSearch Dashboards | 5601 | No conflict |
| Kafka | 9092 | No conflict |

## Updated Access URLs

### Main Services

```
🌐 Frontend Application
   http://localhost:3100

📚 API Gateway & Documentation
   http://localhost:8200
   http://localhost:8200/docs

🔐 Authentication (Keycloak)
   http://localhost:8180
   Username: admin
   Password: admin

📊 Monitoring & Metrics
   Grafana:    http://localhost:3101 (admin/admin)
   Prometheus: http://localhost:9190
   Jaeger:     http://localhost:16686

🗂️ Storage & Data
   MinIO Console: http://localhost:9101 (minioadmin/minioadmin)
   PostgreSQL:    localhost:5434
   OpenSearch:    http://localhost:9300

🏥 Medical Imaging
   Orthanc:       http://localhost:8042 (orthanc/orthanc)
   DICOM Port:    localhost:4242

📦 Additional Services
   FHIR Server:   http://localhost:8183/fhir
   MLflow:        http://localhost:5100
```

## Configuration Updates

### Docker Compose

The `compose.yaml` file has been updated with all new port mappings. All services now use:

```yaml
ports:
  - "NEW_PORT:INTERNAL_PORT"
```

### Environment Variables

The `.env.example` file has been updated with new URLs:

```env
# Updated URLs
KEYCLOAK_URL=http://localhost:8180
MINIO_ENDPOINT=localhost:9100
DATABASE_URL=postgresql://postgres:postgres@localhost:5434/aurelius
FHIR_SVC_URL=http://localhost:8183/fhir
```

### Frontend Configuration

The frontend is configured to use the new Gateway port:

```env
NEXT_PUBLIC_API_URL=http://localhost:8200
NEXT_PUBLIC_KEYCLOAK_URL=http://localhost:8180
```

### Startup Script

The `start.sh` script has been updated with all new URLs for health checks and the final service list.

## Quick Start with New Ports

### 1. Start the Platform

```bash
./start.sh
```

The script will automatically use the new port configuration.

### 2. Access Main Services

After startup, visit:

```bash
# Frontend (React/Next.js Application)
open http://localhost:3100

# API Documentation
open http://localhost:8200/docs

# Keycloak Admin Console
open http://localhost:8180

# Grafana Dashboards
open http://localhost:3101
```

### 3. Database Connection

Update your database client:

```
Host: localhost
Port: 5434
Database: aurelius
Username: postgres
Password: postgres
```

### 4. API Calls

Update your API base URL in code:

```javascript
// Old
const API_URL = 'http://localhost:8000'

// New
const API_URL = 'http://localhost:8200'
```

## Impact on Existing Connections

### Breaking Changes

If you have existing applications or scripts connecting to the old ports, you must update:

1. **Database connections**: Change port from 5432 → 5434
2. **API calls**: Change base URL from 8000 → 8200
3. **Keycloak integration**: Change URL from 8080 → 8180
4. **MinIO clients**: Change endpoint from 9000 → 9100

### Docker Internal Communication

**No changes needed!** Services communicating within the Docker network continue to use internal ports:

```yaml
# Services still communicate internally on original ports
DATABASE_URL: postgresql://postgres:postgres@postgres:5432/aurelius
KEYCLOAK_URL: http://keycloak:8080
MINIO_ENDPOINT: minio:9000
```

## Troubleshooting

### Port Still in Use?

If you see errors about ports being in use:

```bash
# Check what's using a port
lsof -i :3100
lsof -i :8200

# Kill the process if needed
kill -9 <PID>
```

### Service Not Accessible?

1. **Check if containers are running:**
   ```bash
   docker ps
   ```

2. **Check service logs:**
   ```bash
   docker logs aurelius-frontend
   docker logs aurelius-gateway
   ```

3. **Verify port mapping:**
   ```bash
   docker port aurelius-frontend
   docker port aurelius-gateway
   ```

### Reset Everything

To start fresh:

```bash
# Stop all containers and remove volumes
docker compose down -v

# Start again
./start.sh
```

## Testing the New Configuration

### Health Check Commands

```bash
# Frontend
curl http://localhost:3100

# API Gateway
curl http://localhost:8200/health

# Keycloak
curl http://localhost:8180/health/ready

# MinIO
curl http://localhost:9100/minio/health/live

# Prometheus
curl http://localhost:9190/-/healthy

# Grafana
curl http://localhost:3101/api/health
```

### Docker Compose Verification

```bash
# Validate compose file
docker compose config

# Check service status
docker compose ps

# View logs
docker compose logs -f
```

## Summary

✅ **All port conflicts resolved**
✅ **Docker Compose updated**
✅ **Environment files updated**
✅ **Startup script updated**
✅ **Documentation updated**

### Key Changes:
- Frontend: **3100** (was 3000)
- API Gateway: **8200** (was 8000)
- Keycloak: **8180** (was 8080)
- PostgreSQL: **5434** (was 5432)
- All other services updated as needed

### No Changes to:
- Internal Docker network communication
- Service architecture
- API endpoints
- Authentication flows
- Data persistence

The platform functionality remains **exactly the same** - only external access ports have changed!

---

**Last Updated**: October 27, 2025
**Status**: ✅ All services operational on new ports
