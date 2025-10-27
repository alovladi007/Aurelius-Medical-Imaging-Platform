# Aurelius Kubernetes - Quick Reference Card

## 🚀 Deployment Commands

### Initial Setup
```bash
# 1. Navigate to Helm chart
cd infra/k8s/helm/aurelius

# 2. Copy and edit values
cp values.yaml my-values.yaml
vim my-values.yaml  # Update passwords, domains, resources

# 3. Deploy using script (recommended)
cd ../scripts
./deploy.sh -f ../helm/aurelius/my-values.yaml

# OR deploy manually
helm install aurelius ../helm/aurelius \
  -n aurelius --create-namespace \
  -f my-values.yaml --wait --timeout 15m
```

### Production Deployment
```bash
./deploy.sh -n production \
  -f ../helm/aurelius/values.yaml \
  -f ../helm/aurelius/values-prod.yaml
```

### Dry Run (Test)
```bash
./deploy.sh --dry-run -f ../helm/aurelius/my-values.yaml
```

## 🧪 Testing

### Run Validation Tests
```bash
cd infra/k8s/scripts
./test-deployment.sh -n aurelius
```

### Manual Tests
```bash
# Check pod status
kubectl get pods -n aurelius

# Check services
kubectl get svc -n aurelius

# Check ingress
kubectl get ingress -n aurelius

# View logs
kubectl logs -n aurelius -l app.kubernetes.io/component=gateway -f

# Health check
kubectl run test-curl --image=curlimages/curl --rm -i --restart=Never -n aurelius -- \
  curl http://aurelius-gateway:8000/health/live
```

## 🔍 Monitoring

### Access Prometheus
```bash
kubectl port-forward -n aurelius svc/aurelius-prometheus-server 9090:80
open http://localhost:9090
```

### Access Grafana
```bash
# Get password
kubectl get secret aurelius-grafana -n aurelius \
  -o jsonpath="{.data.admin-password}" | base64 --decode

kubectl port-forward -n aurelius svc/aurelius-grafana 3000:80
open http://localhost:3000
```

### Access Jaeger
```bash
kubectl port-forward -n aurelius svc/aurelius-jaeger-query 16686:16686
open http://localhost:16686
```

## 📊 Status Commands

### Helm Status
```bash
helm list -n aurelius
helm status aurelius -n aurelius
helm history aurelius -n aurelius
```

### Resource Status
```bash
# All resources
kubectl get all -n aurelius

# Pods
kubectl get pods -n aurelius -o wide

# Deployments
kubectl get deployments -n aurelius

# Services
kubectl get svc -n aurelius

# HPA
kubectl get hpa -n aurelius

# PVC
kubectl get pvc -n aurelius
```

### Resource Usage
```bash
# Node resources
kubectl top nodes

# Pod resources
kubectl top pods -n aurelius

# Describe pod
kubectl describe pod <pod-name> -n aurelius
```

## 🔧 Maintenance

### Update Configuration
```bash
# Edit values
vim my-values.yaml

# Upgrade
helm upgrade aurelius . -n aurelius -f my-values.yaml --wait

# Restart pods (if ConfigMap changed)
kubectl rollout restart deployment/aurelius-gateway -n aurelius
```

### Scale Services
```bash
# Scale manually
kubectl scale deployment/aurelius-gateway -n aurelius --replicas=10

# Or update HPA
kubectl edit hpa aurelius-gateway -n aurelius
```

### Database Migrations
```bash
kubectl exec -n aurelius deployment/aurelius-gateway -- \
  alembic upgrade head
```

### Logs
```bash
# Gateway logs
kubectl logs -n aurelius -l app.kubernetes.io/component=gateway --tail=100 -f

# All service logs
kubectl logs -n aurelius -l app.kubernetes.io/instance=aurelius --tail=50

# Specific pod
kubectl logs -n aurelius <pod-name>

# Previous container (after crash)
kubectl logs -n aurelius <pod-name> --previous
```

## 🔄 Upgrades & Rollbacks

### Upgrade
```bash
# Backup first
velero backup create pre-upgrade-$(date +%Y%m%d)

# Upgrade
helm upgrade aurelius . -n aurelius -f my-values.yaml --wait

# Verify
kubectl get pods -n aurelius
./test-deployment.sh -n aurelius
```

### Rollback
```bash
# View history
helm history aurelius -n aurelius

# Rollback to previous
helm rollback aurelius -n aurelius

# Rollback to specific revision
helm rollback aurelius 3 -n aurelius
```

## 💾 Backup & Restore

### Create Backup
```bash
# Cluster backup (Velero)
velero backup create aurelius-$(date +%Y%m%d) --include-namespaces aurelius

# Database backup
kubectl exec -n aurelius -l app.kubernetes.io/name=postgresql -- \
  pg_dump -U postgres aurelius > backup.sql
```

### Restore
```bash
# Restore from Velero
velero restore create --from-backup aurelius-20250127

# Restore database
kubectl exec -n aurelius -i -l app.kubernetes.io/name=postgresql -- \
  psql -U postgres aurelius < backup.sql
```

### List Backups
```bash
velero backup get
```

## 🐛 Troubleshooting

### Pod Issues
```bash
# Describe pod
kubectl describe pod <pod-name> -n aurelius

# Events
kubectl get events -n aurelius --sort-by='.lastTimestamp' | tail -20

# Pod logs
kubectl logs <pod-name> -n aurelius

# Execute in pod
kubectl exec -n aurelius <pod-name> -it -- /bin/bash
```

### Network Issues
```bash
# Check endpoints
kubectl get endpoints -n aurelius

# Test connectivity
kubectl run test-curl --image=curlimages/curl --rm -i --restart=Never -n aurelius -- \
  curl -v http://aurelius-gateway:8000/health/live

# Check network policies
kubectl get networkpolicies -n aurelius
kubectl describe networkpolicy <policy-name> -n aurelius
```

### Resource Issues
```bash
# Check node capacity
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check resource quotas
kubectl describe resourcequota -n aurelius

# Check pod resource usage
kubectl top pods -n aurelius --sort-by=memory
```

## 🗑️ Cleanup

### Uninstall
```bash
# Delete Helm release
helm uninstall aurelius -n aurelius

# Delete namespace
kubectl delete namespace aurelius

# Delete PVCs (data will be lost!)
kubectl delete pvc -n aurelius --all
```

### Delete Specific Resources
```bash
# Delete deployment
kubectl delete deployment aurelius-gateway -n aurelius

# Delete service
kubectl delete svc aurelius-gateway -n aurelius

# Delete pod
kubectl delete pod <pod-name> -n aurelius
```

## 🔐 Security

### View Secrets
```bash
# List secrets
kubectl get secrets -n aurelius

# Get secret value
kubectl get secret aurelius-secrets -n aurelius \
  -o jsonpath="{.data.database-url}" | base64 --decode
```

### Update Secret
```bash
# Edit secret
kubectl edit secret aurelius-secrets -n aurelius

# Or recreate
kubectl delete secret aurelius-secrets -n aurelius
kubectl create secret generic aurelius-secrets -n aurelius \
  --from-literal=database-url="postgresql://..." \
  --from-literal=redis-url="redis://..."
```

## 📍 Port Forwarding

### Access Services Locally
```bash
# Gateway API
kubectl port-forward -n aurelius svc/aurelius-gateway 8000:8000

# Web UI
kubectl port-forward -n aurelius svc/aurelius-web-ui 3000:3000

# PostgreSQL
kubectl port-forward -n aurelius svc/aurelius-postgresql 5432:5432

# Redis
kubectl port-forward -n aurelius svc/aurelius-redis-master 6379:6379

# Prometheus
kubectl port-forward -n aurelius svc/aurelius-prometheus-server 9090:80

# Grafana
kubectl port-forward -n aurelius svc/aurelius-grafana 3000:80
```

## 🔑 Common Paths

| Component | Path |
|-----------|------|
| Helm Chart | `infra/k8s/helm/aurelius/` |
| Scripts | `infra/k8s/scripts/` |
| Values | `infra/k8s/helm/aurelius/values.yaml` |
| Prod Values | `infra/k8s/helm/aurelius/values-prod.yaml` |
| Templates | `infra/k8s/helm/aurelius/templates/` |

## 📱 Quick Checks

### Is Everything Running?
```bash
kubectl get pods -n aurelius | grep -v Running
# Empty output = all pods running
```

### Are Services Accessible?
```bash
kubectl get endpoints -n aurelius
# All services should have endpoints
```

### Is Database Connected?
```bash
kubectl exec -n aurelius deployment/aurelius-gateway -- \
  python -c "from app.db.session import engine; print(engine.execute('SELECT 1').scalar())"
# Output: 1
```

### Check Recent Errors
```bash
kubectl logs -n aurelius -l app.kubernetes.io/instance=aurelius \
  --since=1h | grep -i error
```

---

## 🆘 Emergency Commands

### Delete Stuck Pod
```bash
kubectl delete pod <pod-name> -n aurelius --grace-period=0 --force
```

### Restart All Pods
```bash
kubectl rollout restart deployment -n aurelius
```

### Emergency Scale Down
```bash
kubectl scale deployment --all -n aurelius --replicas=1
```

### Get All Errors
```bash
kubectl get events -n aurelius --field-selector type=Warning
```

---

**For detailed documentation, see:**
- Full Guide: `/infra/k8s/helm/aurelius/README.md`
- Session Summary: `/SESSION_15_COMPLETE.md`
- Helm Chart: `https://github.com/aurelius/aurelius-medimaging`
