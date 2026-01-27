#!/bin/bash

# AirType - Local Development Script
# This script helps you run the frontend and backend locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_help() {
    echo "AirType Development Script"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  backend     Start the backend server"
    echo "  frontend    Start the frontend server"
    echo "  both        Start both servers (in separate terminals)"
    echo "  install     Install dependencies"
    echo "  test        Run tests"
    echo "  docker      Start with Docker Compose"
    echo "  help        Show this help message"
    echo ""
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Python 3 is not installed${NC}"
        exit 1
    fi
}

check_venv() {
    if [ ! -d "$SCRIPT_DIR/backend/venv" ]; then
        echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
        cd "$SCRIPT_DIR/backend"
        python3 -m venv venv
        echo -e "${GREEN}Virtual environment created${NC}"
    fi
}

install_deps() {
    check_python
    check_venv
    
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    cd "$SCRIPT_DIR/backend"
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo -e "${GREEN}Backend dependencies installed${NC}"
}

start_backend() {
    check_python
    check_venv
    
    echo -e "${GREEN}Starting backend server...${NC}"
    cd "$SCRIPT_DIR/backend"
    source venv/bin/activate
    
    # Set environment variables if not already set
    export FLASK_ENV=${FLASK_ENV:-development}
    export DATABASE_URL=${DATABASE_URL:-sqlite:///airtype.db}
    export REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}
    export SECRET_KEY=${SECRET_KEY:-dev-secret-key}
    export JWT_SECRET_KEY=${JWT_SECRET_KEY:-dev-jwt-secret}
    
    echo -e "${YELLOW}Backend running at http://localhost:5000${NC}"
    python wsgi.py
}

start_frontend() {
    echo -e "${GREEN}Starting frontend server...${NC}"
    cd "$SCRIPT_DIR/frontend"
    
    echo -e "${YELLOW}Frontend running at http://localhost:8080${NC}"
    python3 -m http.server 8080
}

start_both() {
    echo -e "${GREEN}Starting both servers...${NC}"
    echo ""
    echo "Please run the following in separate terminal windows:"
    echo ""
    echo -e "${YELLOW}Terminal 1 (Backend):${NC}"
    echo "  cd $SCRIPT_DIR && ./run.sh backend"
    echo ""
    echo -e "${YELLOW}Terminal 2 (Frontend):${NC}"
    echo "  cd $SCRIPT_DIR && ./run.sh frontend"
    echo ""
}

run_tests() {
    check_python
    check_venv
    
    echo -e "${YELLOW}Running tests...${NC}"
    cd "$SCRIPT_DIR/backend"
    source venv/bin/activate
    pytest tests/ -v
}

start_docker() {
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Docker Compose is not installed${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Starting with Docker Compose...${NC}"
    cd "$SCRIPT_DIR"
    docker-compose up -d
    
    echo ""
    echo -e "${GREEN}Services started:${NC}"
    echo "  Frontend:   http://localhost"
    echo "  Backend:    http://localhost:5000"
    echo "  Grafana:    http://localhost:3000"
    echo "  Prometheus: http://localhost:9090"
    echo ""
}

# Main script
case "${1:-help}" in
    backend)
        start_backend
        ;;
    frontend)
        start_frontend
        ;;
    both)
        start_both
        ;;
    install)
        install_deps
        ;;
    test)
        run_tests
        ;;
    docker)
        start_docker
        ;;
    help|*)
        print_help
        ;;
esac
