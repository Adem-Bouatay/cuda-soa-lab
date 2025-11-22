# GPU Matrix Addition Service - Task 1 Implementation

This is a GPU-accelerated matrix addition microservice built with FastAPI and Numba CUDA.

## 🚀 Features

- **GPU-accelerated matrix addition** using CUDA kernels via Numba
- **FastAPI REST API** with three endpoints:
  - `GET /health` - Service health check
  - `POST /add` - Matrix addition on GPU
  - `GET /gpu-info` - GPU memory information
- **Input validation** for matrix shape compatibility
- **Performance timing** for GPU operations
- **Docker support** for containerized deployment

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed
- Docker with NVIDIA Container Toolkit (for containerized deployment)

## 🛠️ Installation

### Option 1: Virtual Environment (Recommended for Development)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn numpy numba-cuda prometheus-client python-multipart

# Note: numba-cuda installation requires CUDA toolkit
# For CUDA 12.x:
pip install numba-cuda[cu12]
# For CUDA 13.x:
pip install numba-cuda[cu13]
```

### Option 2: Using requirements.txt

```bash
pip install -r requirements.txt
```

## 🎯 How It Works

### CUDA Kernel Implementation

The service uses a **2D CUDA kernel** for parallel matrix addition:

```python
@cuda.jit
def matrix_add_kernel(a, b, c):
    """Each thread processes one matrix element"""
    i, j = cuda.grid(2)  # Get 2D thread position
    if i < a.shape[0] and j < a.shape[1]:
        c[i, j] = a[i, j] + b[i, j]
```

**Key concepts:**
- **Thread Grid**: The computation is distributed across a 2D grid of threads
- **Block Configuration**: Uses 16×16 threads per block (256 threads)
- **Grid Configuration**: Automatically calculated based on matrix size
- **Memory Management**: Data is transferred to GPU, processed, then copied back

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8001/health
```
**Response:**
```json
{"status": "ok"}
```

#### 2. Matrix Addition
```bash
curl -X POST http://localhost:8001/add \
  -F "file_a=@matrix1.npz" \
  -F "file_b=@matrix2.npz"
```
**Response:**
```json
{
  "matrix_shape": [512, 512],
  "elapsed_time": 0.002134,
  "device": "GPU"
}
```

#### 3. GPU Information
```bash
curl http://localhost:8001/gpu-info
```
**Response:**
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

## 🧪 Testing

### Run Validation Tests
```bash
python3 test_service.py
```

This will:
- ✅ Check for sample matrix files
- ✅ Create test matrices
- ✅ Verify CPU computation logic
- ✅ Check installed dependencies
- ✅ Print usage instructions

### Start the Service
```bash
# With virtual environment
source venv/bin/activate
python3 main.py

# Or directly
python3 main.py
```

The service will start on port **8001** (configurable in `main.py`).

### Test Matrix Addition

Using the provided test matrices:
```bash
# Small test matrices (100×100)
curl -X POST http://localhost:8001/add \
  -F "file_a=@test_matrix_a.npz" \
  -F "file_b=@test_matrix_b.npz"

# Larger matrices (512×512)
curl -X POST http://localhost:8001/add \
  -F "file_a=@matrix1.npz" \
  -F "file_b=@matrix2.npz"
```

### Test Error Handling

Test with mismatched matrix shapes:
```bash
curl -X POST http://localhost:8001/add \
  -F "file_a=@test_matrix_a.npz" \
  -F "file_b=@test_matrix_mismatch.npz"
```

Expected: HTTP 400 error with message about shape mismatch.

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t gpu-matrix-service .
```

### Run the Container
```bash
docker run --gpus all -p 8001:8001 gpu-matrix-service
```

### Test the Containerized Service
```bash
curl http://localhost:8001/health
```

## 📊 Performance Notes

- **Data Transfer Overhead**: For small matrices, CPU might be faster due to GPU memory transfer overhead
- **Optimal Performance**: GPU acceleration shines with larger matrices (>1000×1000)
- **Thread Configuration**: 16×16 block size is optimized for most modern GPUs
- **Memory Usage**: Matrices are temporarily stored on GPU during computation

## 🔧 Configuration

### Change Port Number
Edit the `STUDENT_PORT` variable in `main.py`:
```python
STUDENT_PORT = 8001  # Change to your assigned port
```

### Adjust Thread Block Size
Modify the `threads_per_block` in `gpu_matrix_add()` function:
```python
threads_per_block = (16, 16)  # Default: 256 threads per block
```

## 📝 Project Structure

```
cuda-soa-lab/
├── main.py                    # FastAPI service with CUDA kernel
├── test_service.py           # Validation and testing script
├── matrix1.npz               # Sample matrix 1 (512×512)
├── matrix2.npz               # Sample matrix 2 (512×512)
├── test_matrix_a.npz         # Test matrix A (100×100)
├── test_matrix_b.npz         # Test matrix B (100×100)
├── test_matrix_mismatch.npz  # Mismatch test (50×50)
├── Dockerfile                # Container configuration
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project metadata
└── README.md                # This file
```

## 🎓 Learning Points

### CUDA Concepts Demonstrated

1. **Kernel Functions**: GPU functions decorated with `@cuda.jit`
2. **Thread Indexing**: Using `cuda.grid(2)` for 2D indexing
3. **Memory Management**: Explicit data transfer with `cuda.to_device()` and `copy_to_host()`
4. **Grid/Block Configuration**: Calculating optimal thread organization
5. **Boundary Checking**: Preventing out-of-bounds memory access

### When to Use GPU Acceleration

✅ **Use GPU when:**
- Matrix size is large (>1000×1000)
- Computation per element is non-trivial
- Multiple operations on same data
- Real-time processing needed

❌ **Avoid GPU when:**
- Matrices are very small
- Single operation with high overhead
- Data transfer dominates computation time

## 🔍 Troubleshooting

### "No GPU detected" Error
- Verify NVIDIA drivers: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- Install numba-cuda: `pip install numba-cuda[cu12]`

### "Module not found" Errors
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Port Already in Use
- Change `STUDENT_PORT` in `main.py`
- Or kill existing process: `lsof -ti:8001 | xargs kill -9`

## 📚 Next Steps (Lab Tasks)

- ✅ **Task 1**: GPU Matrix Addition Service (COMPLETED)
- ⏭️ **Task 2**: Add `/gpu-info` endpoint (COMPLETED)
- ⏭️ **Task 3**: Containerize the application (Dockerfile ready)
- ⏭️ **Task 4**: Jenkins CI/CD pipeline
- ⏭️ **Task 5**: Prometheus monitoring and Grafana visualization

## 👨‍💻 Author

Lab implementation for SOA Course - GPU-Accelerated Microservices

## 📄 License

Educational use only

