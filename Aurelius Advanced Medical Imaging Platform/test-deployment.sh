#!/bin/bash
set -euo pipefail

# Aurelius - Kubernetes Deployment Validation Script
# Tests deployed services for readiness and connectivity

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
NAMESPACE="${NAMESPACE:-aurelius}"
RELEASE_NAME="${RELEASE_NAME:-aurelius}"

# Counters
PASSED=0
FAILED=0
WARNINGS=0

print_success() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED++))
}

print_failure() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED++))
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

print_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Test: Helm release exists
test_helm_release() {
    print_section "Testing Helm Release"
    
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        print_success "Helm release '$RELEASE_NAME' found"
        
        # Get release status
        STATUS=$(helm status "$RELEASE_NAME" -n "$NAMESPACE" -o json | jq -r '.info.status')
        if [ "$STATUS" = "deployed" ]; then
            print_success "Release status: deployed"
        else
            print_failure "Release status: $STATUS (expected: deployed)"
        fi
    else
        print_failure "Helm release '$RELEASE_NAME' not found"
    fi
}

# Test: Namespace exists
test_namespace() {
    print_section "Testing Namespace"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_success "Namespace '$NAMESPACE' exists"
    else
        print_failure "Namespace '$NAMESPACE' not found"
    fi
}

# Test: All pods are running
test_pods() {
    print_section "Testing Pods"
    
    # Get all pods
    PODS=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" --no-headers)
    
    if [ -z "$PODS" ]; then
        print_failure "No pods found"
        return
    fi
    
    # Count total pods
    TOTAL=$(echo "$PODS" | wc -l)
    print_success "Found $TOTAL pods"
    
    # Check each pod
    while IFS= read -r line; do
        POD_NAME=$(echo "$line" | awk '{print $1}')
        STATUS=$(echo "$line" | awk '{print $3}')
        READY=$(echo "$line" | awk '{print $2}')
        
        if [ "$STATUS" = "Running" ]; then
            # Check if ready
            READY_COUNT=$(echo "$READY" | cut -d'/' -f1)
            TOTAL_COUNT=$(echo "$READY" | cut -d'/' -f2)
            
            if [ "$READY_COUNT" = "$TOTAL_COUNT" ]; then
                print_success "Pod $POD_NAME is Running and Ready ($READY)"
            else
                print_warning "Pod $POD_NAME is Running but not Ready ($READY)"
            fi
        else
            print_failure "Pod $POD_NAME status: $STATUS"
            
            # Show recent events
            echo "  Recent events:"
            kubectl get events -n "$NAMESPACE" --field-selector involvedObject.name="$POD_NAME" \
                --sort-by='.lastTimestamp' | tail -3 | sed 's/^/    /'
        fi
    done <<< "$PODS"
}

# Test: Services are accessible
test_services() {
    print_section "Testing Services"
    
    SERVICES=$(kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" --no-headers)
    
    if [ -z "$SERVICES" ]; then
        print_failure "No services found"
        return
    fi
    
    while IFS= read -r line; do
        SVC_NAME=$(echo "$line" | awk '{print $1}')
        TYPE=$(echo "$line" | awk '{print $2}')
        CLUSTER_IP=$(echo "$line" | awk '{print $3}')
        
        if [ "$CLUSTER_IP" != "None" ] && [ "$CLUSTER_IP" != "<none>" ]; then
            print_success "Service $SVC_NAME ($TYPE) has ClusterIP: $CLUSTER_IP"
        else
            print_warning "Service $SVC_NAME has no ClusterIP"
        fi
    done <<< "$SERVICES"
}

# Test: Ingress configuration
test_ingress() {
    print_section "Testing Ingress"
    
    INGRESS=$(kubectl get ingress -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" --no-headers 2>/dev/null)
    
    if [ -z "$INGRESS" ]; then
        print_warning "No ingress found (may be intentional)"
        return
    fi
    
    while IFS= read -r line; do
        ING_NAME=$(echo "$line" | awk '{print $1}')
        HOSTS=$(echo "$line" | awk '{print $3}')
        ADDRESS=$(echo "$line" | awk '{print $4}')
        
        if [ -n "$ADDRESS" ] && [ "$ADDRESS" != "<pending>" ]; then
            print_success "Ingress $ING_NAME has address: $ADDRESS"
        else
            print_warning "Ingress $ING_NAME is pending address assignment"
        fi
        
        echo "  Hosts: $HOSTS"
    done <<< "$INGRESS"
}

# Test: PVC status
test_pvcs() {
    print_section "Testing Persistent Volume Claims"
    
    PVCS=$(kubectl get pvc -n "$NAMESPACE" --no-headers 2>/dev/null)
    
    if [ -z "$PVCS" ]; then
        print_warning "No PVCs found (may be intentional)"
        return
    fi
    
    while IFS= read -r line; do
        PVC_NAME=$(echo "$line" | awk '{print $1}')
        STATUS=$(echo "$line" | awk '{print $2}')
        VOLUME=$(echo "$line" | awk '{print $3}')
        
        if [ "$STATUS" = "Bound" ]; then
            print_success "PVC $PVC_NAME is Bound to $VOLUME"
        else
            print_failure "PVC $PVC_NAME status: $STATUS"
        fi
    done <<< "$PVCS"
}

# Test: ConfigMaps and Secrets exist
test_config() {
    print_section "Testing Configuration"
    
    # Check ConfigMap
    if kubectl get configmap "$RELEASE_NAME" -n "$NAMESPACE" &> /dev/null; then
        print_success "ConfigMap exists"
    else
        print_warning "ConfigMap not found"
    fi
    
    # Check Secrets
    if kubectl get secret "$RELEASE_NAME-secrets" -n "$NAMESPACE" &> /dev/null; then
        print_success "Secrets exist"
    else
        print_failure "Secrets not found"
    fi
}

# Test: Service endpoints
test_endpoints() {
    print_section "Testing Service Endpoints"
    
    # Test gateway health endpoint
    print_success "Testing Gateway health endpoint..."
    
    if kubectl run test-curl --image=curlimages/curl:latest --rm -i --restart=Never -n "$NAMESPACE" -- \
        curl -s -o /dev/null -w "%{http_code}" "http://$RELEASE_NAME-gateway:8000/health/live" | grep -q "200"; then
        print_success "Gateway /health/live returns 200 OK"
    else
        print_failure "Gateway /health/live check failed"
    fi
    
    # Test readiness endpoint
    if kubectl run test-curl --image=curlimages/curl:latest --rm -i --restart=Never -n "$NAMESPACE" -- \
        curl -s -o /dev/null -w "%{http_code}" "http://$RELEASE_NAME-gateway:8000/health/ready" | grep -q "200"; then
        print_success "Gateway /health/ready returns 200 OK"
    else
        print_warning "Gateway /health/ready check failed (may not be fully initialized)"
    fi
}

# Test: Database connectivity
test_database() {
    print_section "Testing Database Connectivity"
    
    # Check if PostgreSQL pod exists
    PG_POD=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=postgresql --no-headers | head -1 | awk '{print $1}')
    
    if [ -z "$PG_POD" ]; then
        print_warning "PostgreSQL pod not found (may use external database)"
        return
    fi
    
    print_success "PostgreSQL pod found: $PG_POD"
    
    # Check if PostgreSQL is ready
    if kubectl exec -n "$NAMESPACE" "$PG_POD" -- pg_isready -U postgres &> /dev/null; then
        print_success "PostgreSQL is accepting connections"
    else
        print_failure "PostgreSQL is not ready"
    fi
}

# Test: Redis connectivity
test_redis() {
    print_section "Testing Redis Connectivity"
    
    # Check if Redis pod exists
    REDIS_POD=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=redis --no-headers | head -1 | awk '{print $1}')
    
    if [ -z "$REDIS_POD" ]; then
        print_warning "Redis pod not found (may use external Redis)"
        return
    fi
    
    print_success "Redis pod found: $REDIS_POD"
    
    # Check if Redis is ready
    if kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli ping | grep -q "PONG"; then
        print_success "Redis is responding to PING"
    else
        print_failure "Redis is not responding"
    fi
}

# Test: HPA configuration
test_hpa() {
    print_section "Testing HorizontalPodAutoscaler"
    
    HPAS=$(kubectl get hpa -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" --no-headers 2>/dev/null)
    
    if [ -z "$HPAS" ]; then
        print_warning "No HPAs found (autoscaling may be disabled)"
        return
    fi
    
    while IFS= read -r line; do
        HPA_NAME=$(echo "$line" | awk '{print $1}')
        print_success "HPA $HPA_NAME is configured"
    done <<< "$HPAS"
}

# Test: Resource limits
test_resource_limits() {
    print_section "Testing Resource Limits"
    
    PODS=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME" -o json)
    
    # Check if pods have resource limits
    PODS_WITHOUT_LIMITS=$(echo "$PODS" | jq -r '.items[] | select(.spec.containers[].resources.limits == null) | .metadata.name')
    
    if [ -z "$PODS_WITHOUT_LIMITS" ]; then
        print_success "All pods have resource limits configured"
    else
        print_warning "Some pods don't have resource limits:"
        echo "$PODS_WITHOUT_LIMITS" | sed 's/^/    /'
    fi
}

# Test: Network policies
test_network_policies() {
    print_section "Testing Network Policies"
    
    NETPOLS=$(kubectl get networkpolicies -n "$NAMESPACE" --no-headers 2>/dev/null)
    
    if [ -z "$NETPOLS" ]; then
        print_warning "No network policies found (network isolation may be disabled)"
        return
    fi
    
    while IFS= read -r line; do
        NETPOL_NAME=$(echo "$line" | awk '{print $1}')
        print_success "Network policy $NETPOL_NAME exists"
    done <<< "$NETPOLS"
}

# Test: GPU availability (if ML service enabled)
test_gpu() {
    print_section "Testing GPU Availability"
    
    ML_PODS=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/component=ml-service --no-headers 2>/dev/null)
    
    if [ -z "$ML_PODS" ]; then
        print_warning "ML service not found (GPU tests skipped)"
        return
    fi
    
    # Check if GPU operator is installed
    if kubectl get pods -n gpu-operator &> /dev/null; then
        print_success "NVIDIA GPU Operator is installed"
    else
        print_warning "NVIDIA GPU Operator not found"
    fi
    
    # Check GPU nodes
    GPU_NODES=$(kubectl get nodes -l accelerator=nvidia-tesla-t4 --no-headers 2>/dev/null | wc -l)
    if [ "$GPU_NODES" -gt 0 ]; then
        print_success "Found $GPU_NODES GPU node(s)"
    else
        print_warning "No GPU nodes found"
    fi
}

# Generate summary report
generate_summary() {
    print_section "Test Summary"
    
    TOTAL=$((PASSED + FAILED + WARNINGS))
    
    echo "Total Tests: $TOTAL"
    echo -e "${GREEN}Passed: $PASSED${NC}"
    echo -e "${RED}Failed: $FAILED${NC}"
    echo -e "${YELLOW}Warnings: $WARNINGS${NC}"
    echo ""
    
    if [ $FAILED -eq 0 ]; then
        echo -e "${GREEN}✓ All critical tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}✗ Some tests failed. Please review the output above.${NC}"
        exit 1
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "Aurelius Deployment Validation"
    echo "=========================================="
    echo "Namespace: $NAMESPACE"
    echo "Release: $RELEASE_NAME"
    echo ""
    
    # Run all tests
    test_helm_release
    test_namespace
    test_pods
    test_services
    test_ingress
    test_pvcs
    test_config
    test_endpoints
    test_database
    test_redis
    test_hpa
    test_resource_limits
    test_network_policies
    test_gpu
    
    # Generate summary
    generate_summary
}

# Handle arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --namespace|-n)
            NAMESPACE="$2"
            shift 2
            ;;
        --release|-r)
            RELEASE_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --namespace, -n <name>  Kubernetes namespace (default: aurelius)"
            echo "  --release, -r <name>    Helm release name (default: aurelius)"
            echo "  --help, -h              Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main
main
