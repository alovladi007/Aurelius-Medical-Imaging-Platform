#!/bin/bash

# Aurelius Medical Imaging Platform - Startup Script
# This script starts all services and performs health checks

set -e

echo "=========================================="
echo "Aurelius Medical Imaging Platform"
echo "Starting all services..."
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    exit 1
fi

# Use docker compose or docker-compose based on what's available
DOCKER_COMPOSE="docker compose"
if ! docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from .env.example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}.env file created successfully${NC}"
    else
        echo -e "${RED}Error: .env.example not found${NC}"
        exit 1
    fi
fi

echo "Step 1: Building Docker images..."
echo "This may take a few minutes on first run..."
$DOCKER_COMPOSE build

echo ""
echo "Step 2: Starting infrastructure services..."
echo "(PostgreSQL, Redis, MinIO, Keycloak, Orthanc)"
$DOCKER_COMPOSE up -d postgres redis minio minio-init keycloak orthanc

echo ""
echo "Step 3: Waiting for infrastructure services to be healthy..."
echo "This may take 1-2 minutes..."

# Wait for PostgreSQL
echo -n "Waiting for PostgreSQL..."
for i in {1..30}; do
    if docker exec aurelius-postgres pg_isready -U postgres > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for Redis
echo -n "Waiting for Redis..."
for i in {1..30}; do
    if docker exec aurelius-redis redis-cli ping > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 2
done

# Wait for Keycloak
echo -n "Waiting for Keycloak..."
for i in {1..60}; do
    if curl -sf http://localhost:10300/health/ready > /dev/null 2>&1; then
        echo -e " ${GREEN}✓${NC}"
        break
    fi
    echo -n "."
    sleep 3
done

echo ""
echo "Step 4: Starting observability stack..."
echo "(Prometheus, Grafana, Jaeger)"
$DOCKER_COMPOSE up -d prometheus grafana jaeger

echo ""
echo "Step 5: Starting application services..."
echo "(Gateway, Imaging, ML, Search)"
$DOCKER_COMPOSE up -d gateway imaging-svc ml-svc search-svc

echo ""
echo "Step 6: Starting optional services..."
echo "(FHIR Server, OpenSearch, MLflow, Kafka)"
$DOCKER_COMPOSE up -d fhir-server opensearch opensearch-dashboards mlflow kafka

echo ""
echo "Step 7: Performing health checks..."
sleep 10

# Health check function
check_service() {
    local service=$1
    local url=$2
    local name=$3

    if curl -sf "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $name is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $name is not responding (this may be normal if still starting)"
        return 1
    fi
}

echo ""
check_service "postgres" "http://localhost:10400" "PostgreSQL"
check_service "redis" "http://localhost:10250" "Redis"
check_service "gateway" "http://localhost:10200/health" "API Gateway"
check_service "imaging-svc" "http://localhost:10201/health" "Imaging Service"
check_service "ml-svc" "http://localhost:10202/health" "ML Service"
check_service "search-svc" "http://localhost:10203/health" "Search Service"
check_service "keycloak" "http://localhost:10300/health/ready" "Keycloak"
check_service "orthanc" "http://localhost:10850/system" "Orthanc"
check_service "minio" "http://localhost:10700/minio/health/live" "MinIO"
check_service "prometheus" "http://localhost:10600/-/healthy" "Prometheus"
check_service "grafana" "http://localhost:10500/api/health" "Grafana"
check_service "jaeger" "http://localhost:10950/" "Jaeger"
check_service "frontend" "http://localhost:10100" "Frontend"

echo ""
echo "=========================================="
echo -e "${GREEN}Aurelius Platform Started Successfully!${NC}"
echo "=========================================="
echo ""
echo "Access the services at:"
echo ""
echo "  🌐 Frontend:           http://localhost:10100"
echo "  📚 API Gateway:        http://localhost:10200"
echo "  📖 API Documentation:  http://localhost:10200/docs"
echo "  🔐 Keycloak Admin:     http://localhost:10300 (admin/admin)"
echo "  📊 Grafana:            http://localhost:10500 (admin/admin)"
echo "  📈 Prometheus:         http://localhost:10600"
echo "  🔍 Jaeger Tracing:     http://localhost:10950"
echo "  🗂️  MinIO Console:      http://localhost:10701 (minioadmin/minioadmin)"
echo "  🏥 Orthanc:            http://localhost:10850 (orthanc/orthanc)"
echo "  📦 FHIR Server:        http://localhost:10900/fhir"
echo "  🔎 OpenSearch:         http://localhost:11000"
echo "  🤖 MLflow:             http://localhost:10800"
echo ""
echo "To view logs:"
echo "  $DOCKER_COMPOSE logs -f [service-name]"
echo ""
echo "To stop all services:"
echo "  $DOCKER_COMPOSE down"
echo ""
echo "To restart a service:"
echo "  $DOCKER_COMPOSE restart [service-name]"
echo ""
echo "=========================================="
