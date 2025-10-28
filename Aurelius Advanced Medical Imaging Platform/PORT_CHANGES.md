# ⚠️ Port Configuration Changes

## Overview

Due to extensive port conflicts on the system, **all external ports have been moved to the high port range (10000-11200)** to ensure zero conflicts. **Internal container ports remain unchanged** - only the host machine ports have been modified.

## Port Mapping Changes

### Original vs New Ports (High Range)

| Service | Original Port | New Port | Internal Port |
|---------|---------------|----------|---------------|
| **Frontend** | 3000 | **10100** | 3000 |
| **API Gateway** | 8000 | **10200** | 8000 |
| **Imaging Service** | 8001 | **10201** | 8001 |
| **ML Service** | 8002 | **10202** | 8002 |
| **Search Service** | 8004 | **10203** | 8004 |
| **Redis** | 6379 | **10250** | 6379 |
| **Keycloak** | 8080 | **10300** | 8080 |
| **PostgreSQL** | 5432 | **10400** | 5432 |
| **Grafana** | 3001 | **10500** | 3000 |
| **Prometheus** | 9090 | **10600** | 9090 |
| **MinIO API** | 9000 | **10700** | 9000 |
| **MinIO Console** | 9001 | **10701** | 9001 |
| **MLflow** | 5000 | **10800** | 5000 |
| **Orthanc Web** | 8042 | **10850** | 8042 |
| **Orthanc DICOM** | 4242 | **10851** | 4242 |
| **FHIR Server** | 8083 | **10900** | 8080 |
| **Jaeger UI** | 16686 | **10950** | 16686 |
| **OpenSearch** | 9200 | **11000** | 9200 |
| **OpenSearch (Perf)** | 9600 | **11001** | 9600 |
| **OpenSearch Dashboards** | 5601 | **11100** | 5601 |
| **Kafka** | 9092 | **11200** | 9092 |

## Updated Access URLs

### Main Services

```
🌐 Frontend Application
   http://localhost:10100

📚 API Gateway & Documentation
   http://localhost:10200
   http://localhost:10200/docs

🔐 Authentication (Keycloak)
   http://localhost:10300
   Username: admin
   Password: admin

📊 Monitoring & Metrics
   Grafana:    http://localhost:10500 (admin/admin)
   Prometheus: http://localhost:10600
   Jaeger:     http://localhost:10950

🗂️ Storage & Data
   MinIO Console: http://localhost:10701 (minioadmin/minioadmin)
   PostgreSQL:    localhost:10400
   OpenSearch:    http://localhost:11000
   Redis:         localhost:10250

🏥 Medical Imaging
   Orthanc Web:   http://localhost:10850 (orthanc/orthanc)
   DICOM Port:    localhost:10851

📦 Additional Services
   FHIR Server:   http://localhost:10900/fhir
   MLflow:        http://localhost:10800
   OpenSearch Dashboards: http://localhost:11100
   Kafka:         localhost:11200
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
KEYCLOAK_URL=http://localhost:10300
MINIO_ENDPOINT=localhost:10700
DATABASE_URL=postgresql://postgres:postgres@localhost:10400/aurelius
FHIR_SVC_URL=http://localhost:10900/fhir
IMAGING_SVC_URL=http://localhost:10201
ML_SVC_URL=http://localhost:10202
SEARCH_SVC_URL=http://localhost:10203
ORTHANC_URL=http://localhost:10850
```

### Frontend Configuration

The frontend is configured to use the new Gateway port:

```env
NEXT_PUBLIC_API_URL=http://localhost:10200
NEXT_PUBLIC_KEYCLOAK_URL=http://localhost:10300
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
open http://localhost:10100

# API Documentation
open http://localhost:10200/docs

# Keycloak Admin Console
open http://localhost:10300

# Grafana Dashboards
open http://localhost:10500
```

### 3. Database Connection

Update your database client:

```
Host: localhost
Port: 10400
Database: aurelius
Username: postgres
Password: postgres
```

### 4. API Calls

Update your API base URL in code:

```javascript
// Old
const API_URL = 'http://localhost:8000'

// New (High Port Range)
const API_URL = 'http://localhost:10200'
```

## Impact on Existing Connections

### Breaking Changes

If you have existing applications or scripts connecting to the old ports, you must update:

1. **Database connections**: Change port from 5432 → **10400**
2. **API calls**: Change base URL from 8000 → **10200**
3. **Keycloak integration**: Change URL from 8080 → **10300**
4. **MinIO clients**: Change endpoint from 9000 → **10700**
5. **Frontend**: Change URL from 3000 → **10100**
6. **Monitoring**: Grafana 3001 → **10500**, Prometheus 9090 → **10600**

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
lsof -i :10100
lsof -i :10200

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
curl http://localhost:10100

# API Gateway
curl http://localhost:10200/health

# Keycloak
curl http://localhost:10300/health/ready

# MinIO
curl http://localhost:10700/minio/health/live

# Prometheus
curl http://localhost:10600/-/healthy

# Grafana
curl http://localhost:10500/api/health

# Orthanc
curl http://localhost:10850/system
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

✅ **All port conflicts resolved using high port range**
✅ **Docker Compose updated**
✅ **Environment files updated**
✅ **Startup script updated**
✅ **Documentation updated**
✅ **Zero conflicts with existing services**

### Key Changes:
- **All services now use port range 10000-11200**
- Frontend: **10100** (was 3000)
- API Gateway: **10200** (was 8000)
- Keycloak: **10300** (was 8080)
- PostgreSQL: **10400** (was 5432)
- Grafana: **10500** (was 3001)
- Prometheus: **10600** (was 9090)
- MinIO: **10700/10701** (was 9000/9001)
- MLflow: **10800** (was 5000)
- Orthanc: **10850/10851** (was 8042/4242)
- FHIR: **10900** (was 8083)
- Jaeger: **10950** (was 16686)
- OpenSearch: **11000/11001** (was 9200/9600)
- OpenSearch Dashboards: **11100** (was 5601)
- Kafka: **11200** (was 9092)

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
