#!/bin/bash
set -euo pipefail

# Aurelius Medical Imaging Platform - Kubernetes Deployment Script
# This script deploys the complete platform to Kubernetes using Helm

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-aurelius}"
RELEASE_NAME="${RELEASE_NAME:-aurelius}"
VALUES_FILE="${VALUES_FILE:-values.yaml}"
DRY_RUN="${DRY_RUN:-false}"
WAIT="${WAIT:-true}"
TIMEOUT="${TIMEOUT:-10m}"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        print_error "helm not found. Please install Helm 3."
        exit 1
    fi
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi
    
    # Check Helm version
    HELM_VERSION=$(helm version --short | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$HELM_VERSION" -lt 3 ]; then
        print_error "Helm 3 or higher is required."
        exit 1
    fi
    
    print_info "Prerequisites check passed ✓"
}

# Function to create namespace
create_namespace() {
    print_info "Creating namespace: $NAMESPACE"
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_warn "Namespace $NAMESPACE already exists"
    else
        kubectl create namespace "$NAMESPACE"
        
        # Add labels
        kubectl label namespace "$NAMESPACE" \
            name="$NAMESPACE" \
            app.kubernetes.io/managed-by=helm \
            --overwrite
        
        print_info "Namespace created ✓"
    fi
}

# Function to add Helm repositories
add_helm_repos() {
    print_info "Adding Helm repositories..."
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    
    helm repo update
    
    print_info "Helm repositories added ✓"
}

# Function to install or upgrade cert-manager (for TLS)
install_cert_manager() {
    print_info "Checking cert-manager..."
    
    if ! kubectl get namespace cert-manager &> /dev/null; then
        print_info "Installing cert-manager..."
        
        kubectl create namespace cert-manager
        helm repo add jetstack https://charts.jetstack.io
        helm repo update
        
        helm install cert-manager jetstack/cert-manager \
            --namespace cert-manager \
            --version v1.13.0 \
            --set installCRDs=true \
            --wait
        
        print_info "cert-manager installed ✓"
    else
        print_info "cert-manager already installed ✓"
    fi
}

# Function to install NVIDIA GPU operator (if needed)
install_gpu_operator() {
    if grep -q "mlService.enabled: true" "$VALUES_FILE"; then
        print_info "Checking NVIDIA GPU Operator..."
        
        if ! kubectl get pods -n gpu-operator &> /dev/null; then
            print_warn "NVIDIA GPU Operator not found. ML service requires GPU support."
            print_warn "Install it with: helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace"
            print_warn "Continuing without GPU operator..."
        else
            print_info "GPU operator detected ✓"
        fi
    fi
}

# Function to build Helm dependencies
build_dependencies() {
    print_info "Building Helm dependencies..."
    
    cd "$(dirname "$0")"
    
    helm dependency update
    
    print_info "Dependencies built ✓"
}

# Function to validate values file
validate_values() {
    print_info "Validating values file: $VALUES_FILE"
    
    if [ ! -f "$VALUES_FILE" ]; then
        print_error "Values file not found: $VALUES_FILE"
        exit 1
    fi
    
    # Check for placeholder passwords
    if grep -q "CHANGE_ME_IN_PRODUCTION" "$VALUES_FILE"; then
        print_warn "⚠️  WARNING: Found placeholder passwords in values file!"
        print_warn "Please update passwords before deploying to production."
        
        if [ "$DRY_RUN" = "false" ]; then
            read -p "Continue anyway? (y/N) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    print_info "Values file validated ✓"
}

# Function to deploy with Helm
deploy() {
    print_info "Deploying Aurelius to Kubernetes..."
    
    cd "$(dirname "$0")"
    
    HELM_CMD="helm upgrade --install $RELEASE_NAME . \
        --namespace $NAMESPACE \
        --create-namespace \
        --values $VALUES_FILE"
    
    if [ "$DRY_RUN" = "true" ]; then
        HELM_CMD="$HELM_CMD --dry-run --debug"
        print_info "Running in DRY RUN mode..."
    fi
    
    if [ "$WAIT" = "true" ]; then
        HELM_CMD="$HELM_CMD --wait --timeout $TIMEOUT"
    fi
    
    print_info "Executing: $HELM_CMD"
    
    if eval "$HELM_CMD"; then
        print_info "Deployment successful ✓"
    else
        print_error "Deployment failed!"
        exit 1
    fi
}

# Function to wait for pods
wait_for_pods() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi
    
    print_info "Waiting for pods to be ready..."
    
    # Wait for gateway
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/component=gateway \
        -n "$NAMESPACE" \
        --timeout=5m || print_warn "Gateway pods not ready yet"
    
    # Wait for services
    kubectl wait --for=condition=ready pod \
        -l app.kubernetes.io/instance="$RELEASE_NAME" \
        -n "$NAMESPACE" \
        --timeout=10m || print_warn "Some pods not ready yet"
    
    print_info "Pods are ready ✓"
}

# Function to run post-deployment checks
post_deployment_checks() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi
    
    print_info "Running post-deployment checks..."
    
    # Check pod status
    print_info "Pod Status:"
    kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME"
    
    # Check services
    print_info "Service Status:"
    kubectl get svc -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME"
    
    # Check ingress
    print_info "Ingress Status:"
    kubectl get ingress -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME"
    
    # Check HPA
    print_info "HPA Status:"
    kubectl get hpa -n "$NAMESPACE" -l app.kubernetes.io/instance="$RELEASE_NAME"
    
    # Check PVC
    print_info "PVC Status:"
    kubectl get pvc -n "$NAMESPACE"
    
    print_info "Post-deployment checks complete ✓"
}

# Function to display access information
display_access_info() {
    if [ "$DRY_RUN" = "true" ]; then
        return
    fi
    
    print_info ""
    print_info "=========================================="
    print_info "🎉 Deployment Complete!"
    print_info "=========================================="
    print_info ""
    print_info "Release Name: $RELEASE_NAME"
    print_info "Namespace: $NAMESPACE"
    print_info ""
    print_info "To view status:"
    print_info "  helm status $RELEASE_NAME -n $NAMESPACE"
    print_info ""
    print_info "To view resources:"
    print_info "  kubectl get all -n $NAMESPACE"
    print_info ""
    print_info "To view logs:"
    print_info "  kubectl logs -n $NAMESPACE -l app.kubernetes.io/component=gateway --tail=100 -f"
    print_info ""
    print_info "To port-forward (local access):"
    print_info "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME-gateway 8000:8000"
    print_info "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME-web-ui 3000:3000"
    print_info ""
    print_info "Next steps:"
    print_info "  1. Run database migrations"
    print_info "  2. Create admin tenant"
    print_info "  3. Configure Keycloak"
    print_info "  4. Update DNS records"
    print_info ""
    print_info "For detailed instructions, run:"
    print_info "  helm get notes $RELEASE_NAME -n $NAMESPACE"
    print_info ""
}

# Main execution
main() {
    print_info "=========================================="
    print_info "Aurelius Kubernetes Deployment"
    print_info "=========================================="
    print_info ""
    
    check_prerequisites
    create_namespace
    add_helm_repos
    install_cert_manager
    install_gpu_operator
    build_dependencies
    validate_values
    deploy
    wait_for_pods
    post_deployment_checks
    display_access_info
    
    print_info ""
    print_info "✅ All done!"
    print_info ""
}

# Handle script arguments
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
        --values|-f)
            VALUES_FILE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-wait)
            WAIT=false
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --namespace, -n <name>     Kubernetes namespace (default: aurelius)"
            echo "  --release, -r <name>       Helm release name (default: aurelius)"
            echo "  --values, -f <file>        Values file (default: values.yaml)"
            echo "  --dry-run                  Run Helm in dry-run mode"
            echo "  --no-wait                  Don't wait for resources to be ready"
            echo "  --timeout <duration>       Wait timeout (default: 10m)"
            echo "  --help, -h                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Deploy with defaults"
            echo "  $0 --dry-run                         # Test deployment"
            echo "  $0 -n prod -f values-prod.yaml      # Deploy to production"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
