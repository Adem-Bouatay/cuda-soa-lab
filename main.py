from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from numba import cuda
import time
import io
import subprocess
import re

app = FastAPI(title="GPU Matrix Addition Service")

# Student port - CHANGE THIS TO YOUR ASSIGNED PORT
STUDENT_PORT = 8020

@cuda.jit
def matrix_add_kernel(a, b, c):
    """
    CUDA kernel for matrix addition.
    Each thread processes one element of the matrix.
    """
    # Get the 2D position of the current thread
    i, j = cuda.grid(2)

    # Boundary check to ensure we don't go out of bounds
    if i < a.shape[0] and j < a.shape[1]:
        c[i, j] = a[i, j] + b[i, j]


def gpu_matrix_add(matrix_a: np.ndarray, matrix_b: np.ndarray):
    """
    Perform matrix addition on GPU using CUDA.

    Args:
        matrix_a: First input matrix (NumPy array)
        matrix_b: Second input matrix (NumPy array)

    Returns:
        tuple: (result_matrix, elapsed_time)
    """
    # Start timing
    start_time = time.perf_counter()

    # Transfer data to GPU
    d_a = cuda.to_device(matrix_a)
    d_b = cuda.to_device(matrix_b)
    d_c = cuda.device_array_like(d_a)

    # Configure the blocks and threads
    threads_per_block = (16, 16)  # 16x16 = 256 threads per block
    blocks_per_grid_x = (matrix_a.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (matrix_a.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch kernel
    matrix_add_kernel[blocks_per_grid, threads_per_block](d_a, d_b, d_c)

    # Copy result back to host
    result = d_c.copy_to_host()

    # End timing
    elapsed_time = time.perf_counter() - start_time

    return result, elapsed_time


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/add")
async def add_matrices(
    file_a: UploadFile = File(..., description="First matrix (.npz file)"),
    file_b: UploadFile = File(..., description="Second matrix (.npz file)")
):
    """
    Add two matrices on GPU.

    Accepts two .npz files containing NumPy arrays and returns their sum computed on GPU.
    """
    try:
        # Read the uploaded files
        content_a = await file_a.read()
        content_b = await file_b.read()

        # Load matrices from .npz files
        with np.load(io.BytesIO(content_a)) as data:
            # Get the first array from the .npz file
            matrix_a = data[data.files[0]]

        with np.load(io.BytesIO(content_b)) as data:
            matrix_b = data[data.files[0]]

        # Validate shapes
        if matrix_a.shape != matrix_b.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Matrix shapes do not match: {matrix_a.shape} vs {matrix_b.shape}"
            )

        # Convert to float32 if needed (GPU works better with float32)
        if matrix_a.dtype != np.float32:
            matrix_a = matrix_a.astype(np.float32)
        if matrix_b.dtype != np.float32:
            matrix_b = matrix_b.astype(np.float32)

        # Perform GPU addition
        result, elapsed_time = gpu_matrix_add(matrix_a, matrix_b)

        # Return response (without the actual result matrix, only metadata)
        return JSONResponse(content={
            "matrix_shape": list(result.shape),
            "elapsed_time": round(elapsed_time, 6),
            "device": "GPU"
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing matrices: {str(e)}")


@app.get("/gpu-info")
async def get_gpu_info():
    """
    Get GPU memory information using nvidia-smi.

    Returns information about available GPUs including memory usage.
    """
    try:
        # Run nvidia-smi command
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 3:
                    gpus.append({
                        "gpu": parts[0],
                        "memory_used_MB": int(parts[1]),
                        "memory_total_MB": int(parts[2])
                    })

        return {"gpus": gpus}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to query GPU: {str(e)}")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="nvidia-smi not found. Is NVIDIA driver installed?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting GPU info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print(f"Starting GPU Matrix Addition Service on port {STUDENT_PORT}")
    print(f"Available endpoints:")
    print(f"  - GET  /health")
    print(f"  - POST /add")
    print(f"  - GET  /gpu-info")
    uvicorn.run(app, host="0.0.0.0", port=STUDENT_PORT)
