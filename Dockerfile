FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    nvidia-utils-535 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY matrix1.npz .
COPY matrix2.npz .

# Expose ports
# 8001 for FastAPI service (change to your student port)
# 8000 for Prometheus metrics (if needed later)
EXPOSE 8001 8000

# Set environment variable for CUDA
ENV NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# Run the application
CMD ["python3", "main.py"]
