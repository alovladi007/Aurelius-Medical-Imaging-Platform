#!/bin/bash

# Aurelius Medical Imaging Platform - Setup Verification Script
# This script verifies that all files and configurations are in place

set -e

echo "=========================================="
echo "Aurelius Setup Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 - MISSING"
        ((ERRORS++))
    fi
}

# Function to check if directory exists
check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
    else
        echo -e "${RED}✗${NC} $1/ - MISSING"
        ((ERRORS++))
    fi
}

# Function to check optional file
check_optional() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${YELLOW}⚠${NC} $1 - Optional (not critical)"
        ((WARNINGS++))
    fi
}

echo "Checking core configuration files..."
check_file "compose.yaml"
check_file "requirements.txt"
check_file "package.json"
check_file ".env.example"
check_file ".gitignore"
check_file "start.sh"
check_file "init-db.sh"
check_file "Dockerfile"

echo ""
echo "Checking configuration files..."
check_file "prometheus.yml"
check_file "grafana-datasources.yml"
check_file "keycloak-realm.json"

echo ""
echo "Checking documentation..."
check_file "README.md"
check_file "INTEGRATION_GUIDE.md"
check_file "ARCHITECTURE.md"
check_file "API_CONTRACTS.md"
check_file "DATA_MODEL.md"
check_file "SECURITY.md"
check_file "GETTING_STARTED.md"

echo ""
echo "Checking application services..."
check_dir "apps"
check_dir "apps/gateway"
check_dir "apps/gateway/app"
check_dir "apps/gateway/app/core"
check_dir "apps/gateway/app/api"
check_dir "apps/imaging-svc"
check_dir "apps/ml-svc"
check_dir "apps/search-svc"

echo ""
echo "Checking service files..."
check_file "apps/gateway/Dockerfile"
check_file "apps/gateway/requirements.txt"
check_file "apps/gateway/app/main.py"
check_file "apps/gateway/app/core/config.py"
check_file "apps/gateway/app/core/auth.py"
check_file "apps/gateway/app/core/database.py"

check_file "apps/imaging-svc/Dockerfile"
check_file "apps/imaging-svc/app/main.py"

check_file "apps/ml-svc/Dockerfile"
check_file "apps/ml-svc/app/main.py"

check_file "apps/search-svc/Dockerfile"
check_file "apps/search-svc/app/main.py"

echo ""
echo "Checking infrastructure files..."
check_dir "infra"
check_dir "infra/postgres"
check_dir "infra/observability"

check_file "infra/postgres/001_initial_schema.sql"
check_file "infra/postgres/014_add_multitenancy.py"

echo ""
echo "Checking Docker and Docker Compose..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker is installed"
    if docker info > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Docker daemon is running"
    else
        echo -e "${RED}✗${NC} Docker daemon is not running"
        ((ERRORS++))
    fi
else
    echo -e "${RED}✗${NC} Docker is not installed"
    ((ERRORS++))
fi

if command -v docker-compose &> /dev/null || docker compose version &> /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Docker Compose is available"
else
    echo -e "${RED}✗${NC} Docker Compose is not available"
    ((ERRORS++))
fi

echo ""
echo "Checking Docker Compose configuration..."
if docker compose config --quiet 2>&1; then
    echo -e "${GREEN}✓${NC} Docker Compose configuration is valid"
else
    echo -e "${RED}✗${NC} Docker Compose configuration has errors"
    ((ERRORS++))
fi

echo ""
echo "Checking .env file..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} .env file exists"
else
    echo -e "${YELLOW}⚠${NC} .env file not found (will be created from .env.example on first run)"
    ((WARNINGS++))
fi

echo ""
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Setup verification PASSED${NC}"
    echo ""
    echo "All critical files and configurations are in place!"
    echo ""
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s) found (non-critical)${NC}"
        echo ""
    fi
    echo "You can now start the platform with:"
    echo "  ./start.sh"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Setup verification FAILED${NC}"
    echo ""
    echo -e "${RED}$ERRORS critical error(s) found${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}$WARNINGS warning(s) found${NC}"
    fi
    echo ""
    echo "Please fix the errors above before starting the platform."
    exit 1
fi
