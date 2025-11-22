"""
GPU Matrix Addition Service with FastAPI
Path A: Python + Numba Implementation

This service provides GPU-accelerated matrix addition using CUDA/Numba.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from numba import cuda
import time
import io
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Student port - CHANGE THIS TO YOUR ASSIGNED PORT
STUDENT_PORT = 8001  # Each student must use a different port

# Check CUDA availability at startup
try:
    GPU_AVAILABLE = cuda.is_available()
    if GPU_AVAILABLE:
        logger.info("✓ CUDA is available - GPU mode enabled")
    else:
        logger.warning("⚠ CUDA not available - CPU fallback mode enabled")
except Exception as e:
    logger.warning(f"⚠ CUDA check failed: {e} - CPU fallback mode enabled")
    GPU_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="GPU Matrix Addition Service",
    description="GPU-accelerated matrix addition using CUDA/Numba",
    version="1.0.0"
)

# Prometheus metrics
REQUEST_COUNT = Counter('matrix_add_requests_total', 'Total matrix addition requests', ['status'])
COMPUTATION_TIME = Histogram('matrix_add_duration_seconds', 'Time spent on GPU computation')


@cuda.jit
def matrix_add_kernel(A, B, C):
    """
    CUDA kernel for matrix addition.
    Each thread computes one element of the result matrix.
    
    Grid/Block structure:
    - 2D grid of 2D blocks
    - Each thread processes one matrix element at position (i, j)
    """
    i, j = cuda.grid(2)
    
    # Boundary check to avoid out-of-bounds access
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = A[i, j] + B[i, j]


def cpu_matrix_add(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple:
    """
    Perform CPU matrix addition (fallback when GPU is not available).
    
    Args:
        matrix_a: First input matrix (NumPy array)
        matrix_b: Second input matrix (NumPy array)
    
    Returns:
        tuple: (result_matrix, elapsed_time_seconds)
    """
    start_time = time.perf_counter()
    
    # Ensure matrices are float32 for consistency
    A = matrix_a.astype(np.float32)
    B = matrix_b.astype(np.float32)
    
    # Perform addition on CPU
    result = A + B
    
    elapsed_time = time.perf_counter() - start_time
    
    logger.info(f"CPU matrix addition completed: shape={A.shape}, time={elapsed_time:.6f}s")
    
    return result, elapsed_time


def gpu_matrix_add(matrix_a: np.ndarray, matrix_b: np.ndarray) -> tuple:
    """
    Perform GPU-accelerated matrix addition.
    
    Args:
        matrix_a: First input matrix (NumPy array)
        matrix_b: Second input matrix (NumPy array)
    
    Returns:
        tuple: (result_matrix, elapsed_time_seconds)
    """
    start_time = time.perf_counter()
    
    try:
        # Ensure matrices are float32 for GPU compatibility
        A = matrix_a.astype(np.float32)
        B = matrix_b.astype(np.float32)
        
        # Allocate device memory and transfer data to GPU
        d_A = cuda.to_device(A)
        d_B = cuda.to_device(B)
        d_C = cuda.device_array_like(d_A)
        
        # Configure grid and block dimensions
        # Using 16x16 threads per block (common configuration)
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(A.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        # Launch kernel
        matrix_add_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)
        
        # Copy result back to host
        result = d_C.copy_to_host()
        
        elapsed_time = time.perf_counter() - start_time
        
        logger.info(f"GPU matrix addition completed: shape={A.shape}, time={elapsed_time:.6f}s")
        
        return result, elapsed_time
        
    except Exception as e:
        logger.error(f"GPU computation failed: {e}, falling back to CPU")
        # If GPU fails, fallback to CPU
        return cpu_matrix_add(matrix_a, matrix_b)


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "GPU Matrix Addition Service",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/add",
            "/gpu-info",
            "/metrics"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ok", 
                "cuda_available": GPU_AVAILABLE,
                "mode": "GPU" if GPU_AVAILABLE else "CPU"
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )


@app.post("/add")
async def matrix_addition(
    file_a: UploadFile = File(..., description="First matrix (.npz file)"),
    file_b: UploadFile = File(..., description="Second matrix (.npz file)")
):
    """
    GPU-accelerated matrix addition endpoint.
    
    Accepts two .npz files containing NumPy matrices and returns the sum.
    
    Args:
        file_a: First matrix as .npz file
        file_b: Second matrix as .npz file
    
    Returns:
        JSON with matrix_shape, elapsed_time, and device
    """
    try:
        # Validate file extensions
        if not file_a.filename.endswith('.npz'):
            raise HTTPException(status_code=400, detail="file_a must be a .npz file")
        if not file_b.filename.endswith('.npz'):
            raise HTTPException(status_code=400, detail="file_b must be a .npz file")
        
        # Read uploaded files
        content_a = await file_a.read()
        content_b = await file_b.read()
        
        # Load matrices from .npz files
        with np.load(io.BytesIO(content_a)) as data:
            # Try common keys: arr_0, matrix, data
            if 'arr_0' in data:
                matrix_a = data['arr_0']
            elif 'matrix' in data:
                matrix_a = data['matrix']
            else:
                # Take the first array
                matrix_a = data[list(data.keys())[0]]
        
        with np.load(io.BytesIO(content_b)) as data:
            if 'arr_0' in data:
                matrix_b = data['arr_0']
            elif 'matrix' in data:
                matrix_b = data['matrix']
            else:
                matrix_b = data[list(data.keys())[0]]
        
        # Validate matrix shapes
        if matrix_a.shape != matrix_b.shape:
            REQUEST_COUNT.labels(status='shape_mismatch').inc()
            raise HTTPException(
                status_code=400,
                detail=f"Matrix shapes must match. Got {matrix_a.shape} and {matrix_b.shape}"
            )
        
        # Check if matrices are 2D
        if len(matrix_a.shape) != 2 or len(matrix_b.shape) != 2:
            REQUEST_COUNT.labels(status='invalid_dimensions').inc()
            raise HTTPException(
                status_code=400,
                detail="Matrices must be 2-dimensional arrays"
            )
        
        # Perform matrix addition (GPU if available, otherwise CPU)
        if GPU_AVAILABLE:
            result, elapsed_time = gpu_matrix_add(matrix_a, matrix_b)
            device_used = "GPU"
        else:
            result, elapsed_time = cpu_matrix_add(matrix_a, matrix_b)
            device_used = "CPU"
            logger.warning("Using CPU fallback - CUDA not available")
        
        # Update metrics
        REQUEST_COUNT.labels(status='success').inc()
        COMPUTATION_TIME.observe(elapsed_time)
        
        # Return response (without the result matrix data as per spec)
        return JSONResponse(
            status_code=200,
            content={
                "matrix_shape": list(result.shape),
                "elapsed_time": round(elapsed_time, 6),
                "device": device_used
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(status='error').inc()
        logger.error(f"Error in matrix addition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gpu-info")
async def gpu_info():
    """
    Get GPU information using nvidia-smi.
    Returns GPU memory usage and device information.
    """
    try:
        import subprocess
        
        # Run nvidia-smi command
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                gpus.append({
                    "gpu": parts[0],
                    "memory_used_MB": int(parts[1]),
                    "memory_total_MB": int(parts[2]),
                    "utilization_percent": int(parts[3]),
                    "temperature_celsius": int(parts[4])
                })
        
        return {"gpus": gpus}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get GPU information")
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="nvidia-smi not found")
    except Exception as e:
        logger.error(f"Error getting GPU info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting GPU Matrix Addition Service on port {STUDENT_PORT}")
    logger.info(f"CUDA Available: {GPU_AVAILABLE}")
    logger.info(f"Mode: {'GPU' if GPU_AVAILABLE else 'CPU fallback'}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=STUDENT_PORT,
        log_level="info"
    )
