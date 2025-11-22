FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Upgrade pip
RUN python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project files
COPY pyproject.toml .
COPY main.py .
COPY cuda_test.py .

# Install Python dependencies from pyproject.toml
RUN pip install --no-cache-dir -e .

# Expose port for FastAPI (default 8001, can be overridden)
EXPOSE 8001
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["python", "main.py"]
