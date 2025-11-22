# Docker Deployment Guide - Task 3

## Overview

This guide provides complete instructions for containerizing and running the GPU Matrix Addition Service using Docker.

## Prerequisites

- Docker installed (version 20.10+)
- NVIDIA Docker runtime (`nvidia-docker2`)
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed on host

## Dockerfile Configuration

The provided `Dockerfile` includes:

- **Base Image**: `nvidia/cuda:12.0.0-runtime-ubuntu22.04` - Provides CUDA runtime
- **Python**: Python 3.10 with pip
- **Dependencies**: FastAPI, Numba, NumPy, Uvicorn
- **Ports Exposed**:
  - `8001` (or your STUDENT_PORT) - FastAPI service
  - `8000` - Prometheus metrics (for Task 5)
- **Health Check**: Automatic health monitoring via `/health` endpoint

## Quick Start

### Method 1: Using the Helper Script (Recommended)

The `start_service.sh` script simplifies all Docker operations:

```bash
# Set your student port (IMPORTANT: Change this!)
export STUDENT_PORT=8020

# Build and run everything
./start_service.sh all

# Or step by step:
./start_service.sh build   # Build Docker image
./start_service.sh start   # Run in background
./start_service.sh test    # Test endpoints
./start_service.sh logs    # View logs
./start_service.sh stop    # Stop container
```

### Method 2: Manual Docker Commands

```bash
# Set your student port
export STUDENT_PORT=8020

# Build the image
docker build -t gpu-service:latest .

# Run the container
docker run --gpus all \
  -p ${STUDENT_PORT}:${STUDENT_PORT} \
  -p 8000:8000 \
  --name gpu-matrix-service \
  --rm \
  gpu-service:latest

# Run in detached mode (background)
docker run --gpus all \
  -d \
  -p ${STUDENT_PORT}:${STUDENT_PORT} \
  -p 8000:8000 \
  --name gpu-matrix-service \
  --restart unless-stopped \
  gpu-service:latest
```

### Method 3: Using Docker Compose

```bash
# Set your student port
export STUDENT_PORT=8020

# Build and run
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Testing the Deployed Service

### 1. Health Check
```bash
curl http://localhost:${STUDENT_PORT}/health
```

Expected output:
```json
{"status": "ok"}
```

### 2. GPU Info
```bash
curl http://localhost:${STUDENT_PORT}/gpu-info
```

Expected output:
```json
{
  "gpus": [
    {
      "gpu": "0",
      "memory_used_MB": 312,
      "memory_total_MB": 4096
    }
  ]
}
```

### 3. Matrix Addition
```bash
curl -X POST "http://localhost:${STUDENT_PORT}/add" \
  -F "file_a=@matrix1.npz" \
  -F "file_b=@matrix2.npz"
```

Expected output:
```json
{
  "matrix_shape": [512, 512],
  "elapsed_time": 0.002134,
  "device": "GPU"
}
```

## Monitoring GPU Usage

While the service is running, monitor GPU activity:

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or check once
nvidia-smi
```

## Troubleshooting

### Issue: "could not select device driver"

**Solution**: Ensure NVIDIA Docker runtime is installed:
```bash
# Check if nvidia runtime is available
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# Install nvidia-docker2 if needed (Ubuntu)
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Issue: "Cannot connect to GPU"

**Solution**: Verify NVIDIA drivers:
```bash
nvidia-smi
# Should show your GPU information
```

### Issue: "Port already in use"

**Solution**: Change your STUDENT_PORT or stop the conflicting service:
```bash
# Find what's using the port
sudo lsof -i :8020

# Stop existing container
docker stop gpu-matrix-service

# Or use a different port
export STUDENT_PORT=8021
```

### Issue: Container exits immediately

**Solution**: Check logs:
```bash
docker logs gpu-matrix-service
```

Common causes:
- Python dependencies not installed correctly
- CUDA driver mismatch
- Port already bound

## Container Management

```bash
# View running containers
docker ps

# View all containers (including stopped)
docker ps -a

# Stop container
docker stop gpu-matrix-service

# Remove container
docker rm gpu-matrix-service

# Remove image
docker rmi gpu-service:latest

# View logs
docker logs gpu-matrix-service

# Follow logs in real-time
docker logs -f gpu-matrix-service

# Execute command inside running container
docker exec -it gpu-matrix-service bash

# Check resource usage
docker stats gpu-matrix-service
```

## Important Notes for Task 3

1. **STUDENT_PORT**: Make sure to change the port in `main.py` to your assigned port number
2. **GPU Access**: The `--gpus all` flag is mandatory for GPU access
3. **Health Check**: The Dockerfile includes automatic health monitoring
4. **Multi-stage**: For production, consider multi-stage builds to reduce image size
5. **Security**: Don't expose ports unnecessarily in production environments

## Validation Checklist

- [x] Dockerfile uses CUDA base image
- [x] Python and dependencies installed correctly
- [x] Ports exposed (STUDENT_PORT and 8000)
- [x] Health check configured
- [x] Service starts with `python3 main.py`
- [x] GPU access works with `--gpus all`
- [x] All endpoints respond correctly

## Next Steps

After completing Task 3:
- **Task 4**: Set up Jenkins pipeline for automated deployment
- **Task 5**: Configure Prometheus metrics and Grafana dashboards

## Additional Resources

- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [FastAPI Docker Deployment](https://fastapi.tiangolo.com/deployment/docker/)

