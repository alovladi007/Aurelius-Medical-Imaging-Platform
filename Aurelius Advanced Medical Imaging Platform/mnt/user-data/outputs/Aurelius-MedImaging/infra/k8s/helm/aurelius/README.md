# Aurelius Medical Imaging Platform - Helm Chart

Production-ready Kubernetes deployment for the Aurelius Medical Imaging Platform.

## Features

- ✅ **Production-Ready**: Configured for high availability with autoscaling, health checks, and resource limits
- ✅ **Security by Default**: OIDC authentication, TLS, network policies, and non-root containers
- ✅ **GPU Support**: Native support for ML workloads with NVIDIA GPUs
- ✅ **Observability**: Integrated Prometheus, Grafana, and Jaeger for monitoring and tracing
- ✅ **Multi-Tenant**: Built-in tenant isolation with row-level security
- ✅ **Disaster Recovery**: Automated backups with Velero and database snapshots
- ✅ **Idempotent**: Deploy and upgrade safely with Helm

## Prerequisites

- Kubernetes 1.24+ cluster
- Helm 3.8+
- kubectl configured
- At least 16 CPU cores and 64 GB RAM available
- Storage classes configured (standard, fast-ssd)
- (Optional) GPU nodes with NVIDIA GPU Operator for ML workloads
- (Optional) Ingress controller (nginx) for external access
- (Optional) cert-manager for automatic TLS certificates

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/aurelius/aurelius-medimaging.git
cd aurelius-medimaging/infra/k8s/helm/aurelius
```

### 2. Update Values

Copy and customize the values file:

```bash
cp values.yaml my-values.yaml
# Edit my-values.yaml to set:
# - domain names
# - passwords (IMPORTANT!)
# - resource limits
# - storage sizes
```

### 3. Deploy with Script

```bash
cd ../scripts
./deploy.sh -f ../helm/aurelius/my-values.yaml
```

Or manually:

```bash
# Add Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo update

# Install chart
helm install aurelius . \
  --namespace aurelius \
  --create-namespace \
  --values my-values.yaml \
  --wait \
  --timeout 15m
```

### 4. Verify Deployment

```bash
# Run validation tests
cd ../scripts
./test-deployment.sh

# Check pod status
kubectl get pods -n aurelius

# Check services
kubectl get svc -n aurelius

# View logs
kubectl logs -n aurelius -l app.kubernetes.io/component=gateway --tail=100 -f
```

### 5. Access Application

```bash
# Get ingress IP/hostname
kubectl get ingress -n aurelius

# Or port-forward for local access
kubectl port-forward -n aurelius svc/aurelius-gateway 8000:8000
kubectl port-forward -n aurelius svc/aurelius-web-ui 3000:3000

# Open in browser
open http://localhost:3000
```

## Configuration

### Essential Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.domain` | Base domain for all services | `aurelius.io` |
| `global.security.oidc.enabled` | Enable OIDC authentication | `true` |
| `global.security.tls.enabled` | Enable TLS/HTTPS | `true` |
| `gateway.replicaCount` | Number of gateway replicas | `3` |
| `mlService.enabled` | Enable ML service with GPU | `true` |
| `postgresql.auth.password` | PostgreSQL password | `CHANGE_ME` |
| `redis.auth.password` | Redis password | `CHANGE_ME` |

### Service Configuration

Each service has the following configurable parameters:

- `enabled` - Enable/disable the service
- `replicaCount` - Number of replicas
- `image.repository` - Docker image repository
- `image.tag` - Docker image tag
- `resources.limits` - CPU/memory limits
- `resources.requests` - CPU/memory requests
- `autoscaling.enabled` - Enable HorizontalPodAutoscaler
- `autoscaling.minReplicas` - Minimum replicas
- `autoscaling.maxReplicas` - Maximum replicas

### Storage Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mlService.persistence.enabled` | Enable persistent storage for ML models | `true` |
| `mlService.persistence.size` | ML model storage size | `200Gi` |
| `dicomService.persistence.size` | DICOM cache size | `100Gi` |
| `postgresql.primary.persistence.size` | Database storage | `100Gi` |

### Security Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.security.networkPolicy.enabled` | Enable network policies | `true` |
| `podSecurityPolicy.enabled` | Enable pod security policies | `true` |
| `secrets.create` | Create secrets (dev only) | `true` |

## Production Deployment

For production, use the production values override:

```bash
helm install aurelius . \
  --namespace aurelius \
  --create-namespace \
  --values values.yaml \
  --values values-prod.yaml \
  --wait \
  --timeout 15m
```

### Production Checklist

- [ ] Update all passwords and secrets
- [ ] Configure external secrets management (Vault, AWS Secrets Manager, etc.)
- [ ] Set up DNS records for ingress
- [ ] Configure TLS certificates (cert-manager)
- [ ] Enable backup schedule (Velero)
- [ ] Configure monitoring alerts
- [ ] Set up log aggregation (ELK/Loki)
- [ ] Configure OIDC with Keycloak
- [ ] Test disaster recovery procedures
- [ ] Set up CI/CD pipeline
- [ ] Configure autoscaling policies
- [ ] Enable network policies
- [ ] Set resource quotas and limits
- [ ] Configure pod disruption budgets
- [ ] Test upgrade procedures

## Architecture

### Components

1. **Gateway** - FastAPI REST API gateway
2. **DICOM Service** - DICOM protocol handling (C-STORE, WADO)
3. **ML Service** - AI/ML inference with GPU support
4. **Render Service** - 3D visualization and rendering
5. **Annotation Service** - Medical image annotations
6. **Worklist Service** - Radiology worklist management
7. **Celery Workers** - Background job processing
8. **Web UI** - React-based user interface
9. **PostgreSQL** - Primary database
10. **Redis** - Caching and job queue
11. **MinIO** - S3-compatible object storage
12. **Keycloak** - Identity and access management
13. **Prometheus** - Metrics collection
14. **Grafana** - Monitoring dashboards
15. **Jaeger** - Distributed tracing

### Resource Requirements

**Minimum (Development)**:
- 8 CPU cores
- 32 GB RAM
- 500 GB storage

**Recommended (Production)**:
- 64 CPU cores
- 256 GB RAM
- 5 TB storage
- GPU nodes (4x NVIDIA T4 or better)

### High Availability

- All stateless services run with minimum 2 replicas
- PostgreSQL with streaming replication
- Redis with sentinel/cluster mode
- MinIO in distributed mode (8 nodes)
- Pod anti-affinity rules
- Pod disruption budgets
- Cross-zone deployment

## Monitoring

### Prometheus Metrics

Access Prometheus:
```bash
kubectl port-forward -n aurelius svc/aurelius-prometheus-server 9090:80
```

### Grafana Dashboards

Access Grafana:
```bash
kubectl port-forward -n aurelius svc/aurelius-grafana 3000:80
```

Default credentials:
- Username: `admin`
- Password: See secret `aurelius-grafana`

### Jaeger Tracing

Access Jaeger UI:
```bash
kubectl port-forward -n aurelius svc/aurelius-jaeger-query 16686:16686
```

## Backup and Recovery

### Automated Backups

Backups are configured with Velero:

```bash
# View backup schedule
kubectl get schedule -n velero

# Trigger manual backup
velero backup create aurelius-manual --include-namespaces aurelius

# List backups
velero backup get

# Restore from backup
velero restore create --from-backup aurelius-manual
```

### Database Backups

PostgreSQL backups run automatically:

```bash
# View backup cronjob
kubectl get cronjob -n aurelius

# View backup logs
kubectl logs -n aurelius -l app=postgres-backup
```

## Upgrades

### Helm Upgrade

```bash
helm upgrade aurelius . \
  --namespace aurelius \
  --values values.yaml \
  --wait \
  --timeout 15m
```

### Database Migrations

```bash
# Run migrations
kubectl exec -n aurelius deployment/aurelius-gateway -- alembic upgrade head
```

### Rolling Updates

All deployments use `RollingUpdate` strategy:
- `maxSurge: 1` - One additional pod during update
- `maxUnavailable: 0` - Zero downtime

### Rollback

```bash
# View release history
helm history aurelius -n aurelius

# Rollback to previous version
helm rollback aurelius -n aurelius

# Rollback to specific revision
helm rollback aurelius 3 -n aurelius
```

## Troubleshooting

### Pods Not Starting

```bash
# Describe pod
kubectl describe pod -n aurelius <pod-name>

# View logs
kubectl logs -n aurelius <pod-name>

# Check events
kubectl get events -n aurelius --sort-by='.lastTimestamp'
```

### Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints -n aurelius

# Test internal connectivity
kubectl run test-curl --image=curlimages/curl --rm -i --restart=Never -- \
  curl http://aurelius-gateway:8000/health/live
```

### Database Connection Issues

```bash
# Check PostgreSQL pod
kubectl logs -n aurelius -l app.kubernetes.io/name=postgresql

# Test connection
kubectl exec -n aurelius -it deployment/aurelius-gateway -- \
  psql postgresql://user:pass@postgresql:5432/aurelius
```

### GPU Not Available

```bash
# Check GPU operator
kubectl get pods -n gpu-operator

# Check GPU nodes
kubectl get nodes -l accelerator=nvidia-tesla-t4

# Describe ML service pod
kubectl describe pod -n aurelius -l app.kubernetes.io/component=ml-service
```

## Uninstalling

```bash
# Delete Helm release
helm uninstall aurelius -n aurelius

# Delete namespace (includes all resources)
kubectl delete namespace aurelius

# Delete PVCs (optional, data will be lost)
kubectl delete pvc -n aurelius --all
```

## Support

- Documentation: https://docs.aurelius.io
- GitHub Issues: https://github.com/aurelius/aurelius-medimaging/issues
- Slack: https://aurelius.slack.com
- Email: support@aurelius.io

## License

Apache 2.0 - See LICENSE file for details
