#!/bin/bash

# Advanced Cancer AI System Startup Script

set -e

echo "========================================="
echo "  Advanced Cancer AI Detection System  "
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python and Node.js found"
echo ""

# Function to check if port is available
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

# Check backend dependencies
echo "Checking backend dependencies..."
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

echo "Installing backend dependencies..."
pip install -q -r requirements.txt

echo -e "${GREEN}✓${NC} Backend dependencies installed"
echo ""

# Check frontend dependencies
echo "Checking frontend dependencies..."
cd frontend

if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
else
    echo -e "${GREEN}✓${NC} Frontend dependencies already installed"
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

cd ..
echo ""

# Start backend server
echo "========================================="
echo "Starting Backend API Server..."
echo "========================================="

check_port 8000

echo "Backend will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""

# Start backend in background
python -m src.deployment.inference_server &
BACKEND_PID=$!

# Wait for backend to be ready
echo "Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Backend is ready!"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Start frontend server
echo "========================================="
echo "Starting Frontend Dashboard..."
echo "========================================="

check_port 5173

cd frontend

echo "Frontend will be available at: http://localhost:5173"
echo ""

# Start frontend in background
npm run dev &
FRONTEND_PID=$!

cd ..

echo ""
echo "========================================="
echo -e "${GREEN}System Started Successfully!${NC}"
echo "========================================="
echo ""
echo "Backend API:     http://localhost:8000"
echo "Frontend App:    http://localhost:5173"
echo "API Docs:        http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Services stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT TERM

# Wait for processes
wait
