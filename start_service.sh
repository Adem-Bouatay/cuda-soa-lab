#!/bin/bash

# GPU Matrix Addition Service - Build and Run Script
# This script helps you build and run the GPU-accelerated matrix addition service

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}GPU Matrix Addition Service${NC}"
echo -e "${BLUE}================================${NC}"

# Set your student port (CHANGE THIS!)
export STUDENT_PORT=${STUDENT_PORT:-8020}

echo -e "\n${YELLOW}Using port: ${STUDENT_PORT}${NC}\n"

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker is installed${NC}"
}

# Function to check NVIDIA Docker runtime
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}Warning: NVIDIA Docker runtime may not be properly configured${NC}"
        echo -e "${YELLOW}The build will continue, but GPU access may not work${NC}"
    else
        echo -e "${GREEN}✓ NVIDIA Docker runtime is working${NC}"
    fi
}

# Function to build the Docker image
build_image() {
    echo -e "\n${BLUE}Building Docker image...${NC}"
    docker build -t gpu-service:latest .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Docker image built successfully${NC}"
    else
        echo -e "${RED}✗ Failed to build Docker image${NC}"
        exit 1
    fi
}

# Function to run the container
run_container() {
    echo -e "\n${BLUE}Starting container...${NC}"

    # Stop existing container if running
    docker stop gpu-matrix-service 2>/dev/null
    docker rm gpu-matrix-service 2>/dev/null

    docker run --gpus all \
        -p ${STUDENT_PORT}:${STUDENT_PORT} \
        -p 8000:8000 \
        --name gpu-matrix-service \
        --rm \
        gpu-service:latest
}

# Function to run in detached mode
run_detached() {
    echo -e "\n${BLUE}Starting container in detached mode...${NC}"

    # Stop existing container if running
    docker stop gpu-matrix-service 2>/dev/null
    docker rm gpu-matrix-service 2>/dev/null

    docker run --gpus all \
        -d \
        -p ${STUDENT_PORT}:${STUDENT_PORT} \
        -p 8000:8000 \
        --name gpu-matrix-service \
        --restart unless-stopped \
        gpu-service:latest

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Container started successfully${NC}"
        echo -e "\n${YELLOW}View logs with: docker logs -f gpu-matrix-service${NC}"
        echo -e "${YELLOW}Stop with: docker stop gpu-matrix-service${NC}"
    fi
}

# Function to test the service
test_service() {
    echo -e "\n${BLUE}Testing service...${NC}"
    sleep 2

    echo -e "\n${YELLOW}Testing /health endpoint...${NC}"
    curl -s http://localhost:${STUDENT_PORT}/health | python3 -m json.tool

    echo -e "\n${YELLOW}Testing /gpu-info endpoint...${NC}"
    curl -s http://localhost:${STUDENT_PORT}/gpu-info | python3 -m json.tool

    if [ -f "matrix1.npz" ] && [ -f "matrix2.npz" ]; then
        echo -e "\n${YELLOW}Testing /add endpoint with sample matrices...${NC}"
        curl -s -X POST "http://localhost:${STUDENT_PORT}/add" \
            -F "file_a=@matrix1.npz" \
            -F "file_b=@matrix2.npz" | python3 -m json.tool
    fi
}

# Main script
case "$1" in
    build)
        check_docker
        build_image
        ;;
    run)
        check_docker
        check_nvidia_docker
        run_container
        ;;
    start)
        check_docker
        check_nvidia_docker
        run_detached
        ;;
    test)
        test_service
        ;;
    all)
        check_docker
        check_nvidia_docker
        build_image
        run_detached
        sleep 5
        test_service
        ;;
    stop)
        echo -e "${BLUE}Stopping container...${NC}"
        docker stop gpu-matrix-service
        ;;
    logs)
        docker logs -f gpu-matrix-service
        ;;
    *)
        echo "Usage: $0 {build|run|start|test|all|stop|logs}"
        echo ""
        echo "Commands:"
        echo "  build  - Build the Docker image"
        echo "  run    - Build and run container in foreground"
        echo "  start  - Build and run container in background (detached)"
        echo "  test   - Test the running service endpoints"
        echo "  all    - Build, start, and test"
        echo "  stop   - Stop the running container"
        echo "  logs   - View container logs"
        echo ""
        echo "Set STUDENT_PORT environment variable to use a different port:"
        echo "  export STUDENT_PORT=8021"
        exit 1
        ;;
esac

