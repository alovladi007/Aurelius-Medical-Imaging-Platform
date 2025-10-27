# Observability & Cost Controls - README

This document explains the comprehensive observability stack implemented in Session 13.

## Overview

Session 13 implements full-stack observability with:
- **Distributed Tracing**: OpenTelemetry + Jaeger
- **Metrics**: Prometheus with custom metrics
- **Dashboards**: Grafana with 3 comprehensive dashboards
- **Alerting**: SLO-based alerts and budget alerts
- **Rate Limiting**: Per-user and per-tenant quotas
- **Load Testing**: Locust and k6 scripts

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Gateway   │  │  Imaging   │  │   Search   │            │
│  │    +OTel   │  │    +OTel   │  │    +OTel   │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          │ Traces         │ Traces         │ Traces
          ▼                ▼                ▼
    ┌─────────────────────────────────────────────┐
    │          Jaeger (Trace Collector)           │
    │  - Trace ingestion (OTLP)                   │
    │  - Trace storage                            │
    │  - Trace UI (port 16686)                    │
    └─────────────────────────────────────────────┘
    
          │ Metrics        │ Metrics        │ Metrics
          ▼                ▼                ▼
    ┌─────────────────────────────────────────────┐
    │          Prometheus (Metrics)                │
    │  - Scrape metrics from services             │
    │  - Evaluate alert rules                     │
    │  - Store time-series data                   │
    └─────────────────┬───────────────────────────┘
                      │
                      ▼
          ┌───────────────────────┐
          │       Grafana         │
          │  - API Performance    │
          │  - Job Queues         │
          │  - GPU & Costs        │
          └───────────────────────┘
```

---

## Components

### 1. OpenTelemetry Tracing

**What It Does**:
- Instruments all FastAPI applications
- Captures distributed traces across services
- Tracks request flow through the system
- Measures latency at each service hop

**Files**:
- `apps/gateway/app/core/tracing.py` - Tracing initialization module

**Usage**:
```python
from app.core.tracing import setup_tracing, instrument_app

# In lifespan or startup:
setup_tracing("my-service")
instrument_app(app, db_engine=engine)

# For custom spans:
from app.core.tracing import trace_function

@trace_function("process_image")
async def process_image(image_id: str):
    # Your code here
    pass
```

**Automatic Instrumentation**:
- FastAPI requests/responses
- HTTP client calls (httpx)
- Database queries (SQLAlchemy)
- Redis operations

### 2. Jaeger UI

**Access**: http://localhost:16686

**Features**:
- View distributed traces
- Analyze request latency
- Identify bottlenecks
- Service dependency graph

**Example Queries**:
- All traces: service=gateway
- Slow requests: minDuration=1s
- Error traces: error=true

### 3. Rate Limiting

**Implementation**: `apps/gateway/app/middleware/rate_limit.py`

**Features**:
- Per-user rate limits:
  - 60 requests/minute
  - 1000 requests/hour
  - 10000 requests/day
- Per-tenant quotas:
  - 1M API calls/month
  - 1TB storage
  - 100 GPU hours/month
- Redis-backed state
- Automatic cleanup

**Response Headers**:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1706390400
```

**429 Response Example**:
```json
{
  "detail": "Rate limit exceeded. Try again in 45 seconds.",
  "window": "minute",
  "limit": 60,
  "retry_after": 45
}
```

### 4. Prometheus Metrics

**Access**: http://localhost:9090

**Custom Metrics** (exported by services):

**Gateway**:
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request latency histogram
- `http_active_connections` - Active connections
- `rate_limit_exceeded_total` - Rate limit hits
- `db_query_duration_seconds` - Database query latency
- `cache_hits_total` / `cache_misses_total` - Cache statistics

**ML Service**:
- `ml_inference_duration_seconds` - Inference latency
- `ml_inference_batch_size` - Batch size distribution
- `ml_predictions_total` - Total predictions
- `nvidia_smi_utilization_gpu_ratio` - GPU utilization
- `nvidia_smi_memory_used_bytes` - GPU memory usage
- `nvidia_smi_temperature_gpu_celsius` - GPU temperature
- `nvidia_smi_power_draw_watts` - GPU power consumption

**Search Service**:
- `search_request_duration_seconds` - Search latency
- `opensearch_cluster_health_status` - Cluster health

**Celery Workers**:
- `celery_workers_total` - Active workers
- `celery_queue_length` - Queue depth
- `celery_tasks_active_total` - Running tasks
- `celery_tasks_succeeded_total` - Successful tasks
- `celery_tasks_failed_total` - Failed tasks
- `celery_task_duration_seconds` - Task execution time

**Cost Metrics**:
- `api_calls_cost_usd_total` - API call costs
- `storage_cost_usd_total` - Storage costs
- `gpu_hours_total` - GPU usage hours
- `tenant_costs_usd_total` - Per-tenant costs
- `tenant_quota_usage_percent` - Quota usage percentage

### 5. Grafana Dashboards

**Access**: http://localhost:3001 (admin/admin)

**Dashboard 1: API Performance & Latency**
- Request rate by endpoint
- P95/P99 latency trends
- Error rate percentage
- Active connections
- Status code distribution
- Rate limit hits
- Service-to-service latency
- Database query performance
- Cache hit rate
- OpenTelemetry trace counts

**Dashboard 2: Job Queues & Workers**
- Active worker count
- Queue depth by queue
- Tasks running
- Task success rate gauge
- Task throughput (tasks/sec)
- Task duration by type (P95/P99)
- Queue length over time (with alert)
- Failed tasks breakdown
- Task retry rate
- Worker memory usage
- Worker CPU usage
- Indexing job progress
- ML inference queue depth
- Image processing queue depth

**Dashboard 3: GPU & Cost Tracking**
- GPU utilization percentage
- GPU memory usage
- GPU temperature
- GPU power consumption
- Inference batch size
- API call costs (monthly)
- Storage costs (monthly)
- GPU costs (monthly)
- Total monthly cost (with thresholds)
- Cost breakdown by tenant (pie chart)
- Cost projection over time
- Tenant quota usage table
- GPU hours by model
- Budget alerts list
- Cost per API call
- Cost per inference

### 6. Alert Rules (SLOs)

**File**: `infra/observability/alerts/slo-alerts.yaml`

**Alerts Configured**:

**API SLOs**:
- Availability < 99.9% → Critical
- P95 latency > 500ms → Warning
- Error rate > 1% → Critical

**ML Inference SLOs**:
- P95 latency > 2s → Warning
- GPU utilization < 20% for 30min → Info (cost optimization)
- GPU memory > 90% → Warning

**Job Queue Alerts**:
- Queue depth > 1000 → Warning
- No active workers → Critical
- Task failure rate > 10% → Warning

**Cost Budget Alerts**:
- Monthly cost > $8000 (80%) → Info
- Monthly cost > $9000 (90%) → Warning
- Monthly cost > $10000 (100%) → Critical
- Tenant quota > 80% → Warning
- Tenant quota > 100% → Critical

**Database Alerts**:
- Connection pool > 90% → Critical
- P95 query latency > 1s → Warning

**Storage Alerts**:
- Storage > 80% → Warning
- Storage > 90% → Critical

**Search Alerts**:
- OpenSearch cluster not green → Warning
- Search P95 latency > 1s → Warning

---

## Load Testing

### Option 1: Locust (Python-based)

**File**: `infra/observability/load_test.py`

**Installation**:
```bash
pip install locust
```

**Usage**:
```bash
# Web UI (recommended)
locust -f infra/observability/load_test.py --host=http://localhost:8000

# Visit http://localhost:8089
# Set users: 100, spawn rate: 10

# Headless mode
locust -f infra/observability/load_test.py \
  --host=http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 5m \
  --headless

# Distributed (master + workers)
locust -f infra/observability/load_test.py --host=http://localhost:8000 --master
locust -f infra/observability/load_test.py --host=http://localhost:8000 --worker --master-host=<ip>
```

**User Classes**:
1. `ClinicianUser` (50%) - Browse studies, search, view details
2. `ResearcherUser` (30%) - Semantic search, ML predictions
3. `AdminUser` (10%) - Health checks, metrics, job status
4. `RadiologyWorkflowUser` (20%) - Complete workflows

### Option 2: k6 (Go-based, higher performance)

**File**: `infra/observability/k6_load_test.js`

**Installation**:
```bash
# macOS
brew install k6

# Ubuntu
sudo apt install k6
```

**Usage**:
```bash
# Basic smoke test
k6 run infra/observability/k6_load_test.js

# Load test (10 VUs, 5 minutes)
k6 run --vus 10 --duration 5m infra/observability/k6_load_test.js

# Stress test (ramp up)
k6 run --stage 1m:10,5m:50,2m:100,5m:100,2m:0 infra/observability/k6_load_test.js

# With results export
k6 run --vus 50 --duration 10m --out json=results.json infra/observability/k6_load_test.js
```

**Test Scenarios**:
1. Browse Studies
2. Search - Keyword
3. Search - Filtered
4. Search - Semantic
5. ML Inference
6. Health Check

**SLO Thresholds**:
- P95 latency < 500ms
- P99 latency < 1000ms
- Error rate < 1%
- HTTP failure rate < 5%
- Search P95 < 1000ms
- ML P95 < 2000ms

---

## Quick Start

### 1. Start Services with Observability

```bash
cd Aurelius-MedImaging
make up

# Wait for services to start (check Jaeger takes ~30s)
docker compose logs -f jaeger
```

### 2. Verify Tracing

```bash
# Check Jaeger is running
curl http://localhost:16686

# Visit Jaeger UI
open http://localhost:16686

# Make some API calls to generate traces
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# View traces in Jaeger UI:
# - Service: gateway
# - Look for /auth/login operation
# - Click trace to see spans
```

### 3. Verify Metrics

```bash
# Check Prometheus
curl http://localhost:9090/-/healthy

# Query API request rate
curl 'http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m])'

# Visit Prometheus UI
open http://localhost:9090

# Try queries:
# - http_requests_total
# - rate(http_requests_total[5m])
# - histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

### 4. View Dashboards

```bash
# Visit Grafana
open http://localhost:3001

# Login: admin / admin

# Navigate to Dashboards:
# - Aurelius API Performance & Latency
# - Aurelius Job Queues & Workers
# - Aurelius GPU & Cost Tracking

# Each dashboard auto-refreshes every 30s
```

### 5. Test Rate Limiting

```bash
# Make rapid requests to trigger rate limit
for i in {1..100}; do
  curl -X GET http://localhost:8000/studies
  sleep 0.1
done

# Should eventually get 429 response:
# {"detail": "Rate limit exceeded. Try again in XX seconds."}

# Check headers:
curl -I http://localhost:8000/studies
# X-RateLimit-Limit: 60
# X-RateLimit-Remaining: 45
# X-RateLimit-Reset: 1706390400
```

### 6. Run Load Test

```bash
# Install Locust
pip install locust

# Run load test
locust -f infra/observability/load_test.py --host=http://localhost:8000

# Visit http://localhost:8089
# Enter:
#   - Number of users: 50
#   - Spawn rate: 5
# Click "Start Swarming"

# Watch in real-time:
# - Grafana dashboards update
# - Prometheus metrics increase
# - Jaeger traces appear

# Stop after 5 minutes and review results
```

### 7. Check Alerts

```bash
# View active alerts in Prometheus
curl http://localhost:9090/api/v1/alerts

# Or visit UI:
open http://localhost:9090/alerts

# Trigger an alert by generating load:
# - Run load test with 200 users
# - Watch queue depth alert trigger
# - Check Grafana "Budget Alerts" panel
```

---

## Cost Tracking

### How It Works

**Cost Calculation**:
```python
# API Calls: $0.001 per call
api_cost = api_calls_total * 0.001

# Storage: $0.023 per GB-month
storage_cost = storage_gb * 0.023

# GPU: $2.50 per hour
gpu_cost = gpu_hours * 2.50

# Total Monthly Cost
total_cost = api_cost + storage_cost + gpu_cost
```

**Tracking in Prometheus**:
```promql
# Monthly cost query
sum(
  api_calls_cost_usd_total{period="month"} +
  storage_cost_usd_total{period="month"} +
  (gpu_hours_total * 2.5)
)
```

**Budget Alerts**:
- 80% budget ($8,000) → Info alert
- 90% budget ($9,000) → Warning alert
- 100% budget ($10,000) → Critical alert

### Per-Tenant Tracking

**Metrics**:
```promql
# Cost by tenant
sum(tenant_costs_usd_total{period="month"}) by (tenant_id)

# Quota usage by tenant
tenant_quota_usage_percent{tenant_id="acme-hospital"}
```

**Dashboard**:
- Pie chart: Cost breakdown by tenant
- Table: Tenant quota usage (API calls, storage, GPU hours)
- Alerts: Tenant exceeding quota

---

## Performance Benchmarks

**Expected Performance** (with load test):

| Metric | Target | Typical |
|--------|--------|---------|
| P95 API Latency | < 500ms | 200-400ms |
| P99 API Latency | < 1s | 400-800ms |
| Error Rate | < 1% | 0.1-0.5% |
| Search P95 | < 1s | 300-700ms |
| ML Inference P95 | < 2s | 800-1500ms |
| Throughput | N/A | 100-500 req/s |
| GPU Utilization | 60-80% | 70% |
| Cost per API Call | N/A | $0.001 |
| Cost per Inference | N/A | $0.025 |

---

## Troubleshooting

### Jaeger Not Showing Traces

```bash
# Check Jaeger is running
docker compose ps jaeger

# Check logs
docker compose logs jaeger

# Verify OTLP endpoint
curl http://localhost:4317

# Check service environment
docker compose exec gateway env | grep OTEL
```

### Prometheus Not Scraping

```bash
# Check Prometheus targets
open http://localhost:9090/targets

# All targets should be "UP"
# If "DOWN", check service health

# Check service metrics endpoint
curl http://localhost:8000/metrics
```

### Grafana Dashboards Not Loading

```bash
# Check dashboard files exist
ls -la infra/observability/dashboards/

# Check Grafana logs
docker compose logs grafana

# Restart Grafana
docker compose restart grafana
```

### Rate Limiting Not Working

```bash
# Check Redis is running
docker compose ps redis

# Check Redis data
docker compose exec redis redis-cli
> KEYS ratelimit:*
> GET ratelimit:user:admin:minute:12345

# Check middleware is loaded
# Should see rate limit headers in response:
curl -I http://localhost:8000/studies
```

### Load Test Fails

```bash
# Check all services are healthy
make health

# Check you're authenticated
# Some endpoints require JWT token

# Reduce load (start with 10 users)
locust -f load_test.py --host=http://localhost:8000 --users 10

# Check logs for errors
make logs
```

---

## Production Considerations

### Security

**Current (Development)**:
- [x] Tracing enabled
- [x] Metrics collection
- [x] Rate limiting
- [ ] TLS for OTLP
- [ ] Authentication for Prometheus
- [ ] Authentication for Grafana
- [ ] Authentication for Jaeger

**Production Checklist**:
- [ ] Enable TLS for Jaeger OTLP endpoint
- [ ] Add authentication to Prometheus (basic auth)
- [ ] Configure Grafana LDAP/OAuth
- [ ] Restrict Jaeger UI access
- [ ] Set up Alertmanager with notification channels
- [ ] Configure log aggregation (ELK/Loki)

### Scalability

**Current Limits**:
- Jaeger: In-memory storage (not for production)
- Prometheus: Local storage (15 day retention)
- Grafana: SQLite database

**Production Setup**:
- Jaeger: Elasticsearch or Cassandra backend
- Prometheus: Remote write to Thanos/Cortex/Mimir
- Grafana: PostgreSQL database
- Add Alertmanager cluster for HA

### Cost Optimization

**Tips**:
1. Monitor GPU utilization - scale down if < 30%
2. Use spot instances for batch jobs
3. Implement caching to reduce API calls
4. Archive old traces/metrics
5. Use tiered storage (hot/warm/cold)
6. Set up budget alerts per team

---

## References

- OpenTelemetry Docs: https://opentelemetry.io/docs/
- Jaeger Docs: https://www.jaegertracing.io/docs/
- Prometheus Docs: https://prometheus.io/docs/
- Grafana Docs: https://grafana.com/docs/
- Locust Docs: https://docs.locust.io/
- k6 Docs: https://k6.io/docs/

---

**Session 13 Status**: ✅ Complete  
**Observability**: Full-stack tracing, metrics, dashboards  
**Rate Limiting**: User and tenant quotas  
**Cost Tracking**: Real-time budget monitoring  
**Load Testing**: Locust + k6 scripts  
**Alerts**: SLO-based + budget alerts
