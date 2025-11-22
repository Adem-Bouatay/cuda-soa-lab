# CUDA Kernel Explanation & Implementation Guide

## Understanding GPU Parallelism

### What is a CUDA Kernel?

A **CUDA kernel** is a function that runs on the GPU and executes in parallel across many threads. Unlike CPU functions that run sequentially, a GPU kernel is designed to have thousands of threads execute the same code simultaneously on different data.

```python
@cuda.jit
def matrix_add_kernel(A, B, C):
    """
    This function runs on the GPU.
    Each thread computes ONE element of the result matrix.
    """
    i, j = cuda.grid(2)  # Get this thread's position
    if i < C.shape[0] and j < C.shape[1]:
        C[i, j] = A[i, j] + B[i, j]
```

## Thread Hierarchy

### 1. Threads

- The smallest unit of execution
- Each thread executes the kernel function
- Threads are identified by indices

### 2. Blocks

- Threads are grouped into **blocks**
- Threads within a block can synchronize and share memory
- Typical block size: 16×16 = 256 threads

### 3. Grid

- Blocks are organized into a **grid**
- The grid covers the entire problem space
- For a 512×512 matrix with 16×16 blocks: Grid = 32×32 blocks

### Visual Representation

```
Grid (32 x 32 blocks)
┌─────────────────────────────────┐
│ Block(0,0)  Block(0,1)  ...     │
│   [16x16]     [16x16]           │
│                                  │
│ Block(1,0)  Block(1,1)  ...     │
│   [16x16]     [16x16]           │
│                                  │
│   ...         ...       ...     │
└─────────────────────────────────┘

Each Block (16 x 16 threads)
┌─────────────────────────────────┐
│ T(0,0) T(0,1) ... T(0,15)      │
│ T(1,0) T(1,1) ... T(1,15)      │
│  ...    ...    ...    ...       │
│ T(15,0) T(15,1) ... T(15,15)   │
└─────────────────────────────────┘
```

## Computing Global Thread Index

The `cuda.grid(2)` function computes the global 2D index for each thread:

```python
i, j = cuda.grid(2)

# Equivalent to:
# i = blockIdx.x * blockDim.x + threadIdx.x
# j = blockIdx.y * blockDim.y + threadIdx.y
```

**Example:**

- Block (2, 3) with 16×16 threads
- Thread (5, 7) within that block
- Global position: i = 2×16 + 5 = 37, j = 3×16 + 7 = 55
- This thread processes matrix element C[37, 55]

## Memory Management

### Host (CPU) ↔ Device (GPU) Data Flow

```python
# 1. Create data on host (CPU)
matrix_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
matrix_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

# 2. Transfer to device (GPU)
d_A = cuda.to_device(matrix_a)  # Copy host → device
d_B = cuda.to_device(matrix_b)

# 3. Allocate memory on device for result
d_C = cuda.device_array_like(d_A)  # Allocate GPU memory

# 4. Launch kernel (computation happens here)
matrix_add_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)

# 5. Copy result back to host
result = d_C.copy_to_host()  # Copy device → host
```

### Memory Hierarchy

```
┌────────────────────────────────────┐
│         CPU (Host)                 │
│  - System RAM (large, slow)        │
│  - Accessible to CPU only          │
└────────────────────────────────────┘
           ↕ PCIe Transfer (slow!)
┌────────────────────────────────────┐
│         GPU (Device)               │
│                                    │
│  Global Memory (large, slower)     │
│  ├─ Accessible by all threads      │
│  └─ Data must be copied here       │
│                                    │
│  Shared Memory (small, fast)       │
│  ├─ Shared within a block          │
│  └─ Requires explicit management   │
│                                    │
│  Registers (tiny, fastest)         │
│  └─ Private to each thread         │
└────────────────────────────────────┘
```

## Complete Matrix Addition Implementation

### Step-by-Step Breakdown

```python
from numba import cuda
import numpy as np
import time

# Step 1: Define the CUDA kernel
@cuda.jit
def matrix_add_kernel(A, B, C):
    """
    GPU kernel for element-wise matrix addition.

    Each thread computes: C[i,j] = A[i,j] + B[i,j]

    Args:
        A: Input matrix 1 (device array)
        B: Input matrix 2 (device array)
        C: Output matrix (device array)
    """
    # Get global thread position in 2D grid
    i, j = cuda.grid(2)

    # Boundary check: ensure thread is within matrix bounds
    if i < C.shape[0] and j < C.shape[1]:
        # Perform addition for this element
        C[i, j] = A[i, j] + B[i, j]


# Step 2: Setup and launch
def gpu_matrix_add(A_host, B_host):
    """Perform matrix addition on GPU"""

    # Ensure float32 type (GPU-friendly)
    A = A_host.astype(np.float32)
    B = B_host.astype(np.float32)

    # Start timing
    start = time.perf_counter()

    # Transfer data to GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array_like(d_A)

    # Configure launch parameters
    threadsperblock = (16, 16)  # 256 threads per block

    # Calculate grid dimensions
    blockspergrid_x = int(np.ceil(A.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(A.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    matrix_add_kernel[blockspergrid, threadsperblock](d_A, d_B, d_C)

    # Copy result back to host
    result = d_C.copy_to_host()

    elapsed = time.perf_counter() - start

    return result, elapsed


# Step 3: Test the implementation
if __name__ == "__main__":
    # Create test matrices
    size = 512
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    # GPU computation
    result_gpu, time_gpu = gpu_matrix_add(A, B)

    # CPU computation (for comparison)
    start = time.perf_counter()
    result_cpu = A + B
    time_cpu = time.perf_counter() - start

    # Verify correctness
    if np.allclose(result_gpu, result_cpu):
        print("✓ Results match!")
        print(f"GPU time: {time_gpu*1000:.2f} ms")
        print(f"CPU time: {time_cpu*1000:.2f} ms")
        print(f"Speedup: {time_cpu/time_gpu:.2f}x")
    else:
        print("✗ Results don't match!")
```

## Choosing Block and Grid Sizes

### Block Size Selection

**Common configurations:**

- 1D: 256 or 512 threads
- 2D: 16×16 (256) or 32×32 (1024) threads
- 3D: 8×8×8 (512) threads

**Constraints:**

- Maximum threads per block: typically 1024
- Should be multiple of 32 (warp size)
- Consider shared memory requirements

### Grid Size Calculation

```python
# For 512×512 matrix with 16×16 blocks:
threadsperblock = (16, 16)

# Need ceil(512/16) = 32 blocks in each dimension
blockspergrid_x = (512 + 16 - 1) // 16  # = 32
blockspergrid_y = (512 + 16 - 1) // 16  # = 32
blockspergrid = (32, 32)  # 1024 blocks total

# Total threads launched: 32 × 32 × 16 × 16 = 262,144 threads
# Each thread processes 1 matrix element
```

## Performance Considerations

### When GPU is Faster

✓ **Large matrices** (e.g., 512×512 or larger)
✓ **Compute-intensive** operations
✓ **Many parallel operations**
✓ **Data reuse** on GPU across multiple operations

### When CPU is Faster

✗ **Small matrices** (e.g., 10×10)
✗ **Simple operations** (overhead > computation)
✗ **Single operations** (data transfer overhead)
✗ **Sequential algorithms**

### Optimization Tips

1. **Minimize data transfers**: Keep data on GPU when possible
2. **Coalesce memory accesses**: Access memory in contiguous patterns
3. **Use shared memory**: For data reused within a block
4. **Choose appropriate block size**: Balance occupancy and resources
5. **Avoid divergence**: Minimize if/else branches that differ per thread

## Common Patterns

### Pattern 1: Element-wise Operations

```python
@cuda.jit
def element_wise(A, B, C):
    i = cuda.grid(1)
    if i < C.shape[0]:
        C[i] = A[i] + B[i]  # or *, -, /, etc.
```

### Pattern 2: Reduction (Sum)

```python
@cuda.jit
def sum_kernel(arr, result):
    shared = cuda.shared.array(256, dtype=float32)
    tid = cuda.threadIdx.x
    i = cuda.grid(1)

    # Load data into shared memory
    if i < arr.shape[0]:
        shared[tid] = arr[i]
    else:
        shared[tid] = 0
    cuda.syncthreads()

    # Reduction in shared memory
    s = 256 // 2
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2

    # Write result
    if tid == 0:
        cuda.atomic.add(result, 0, shared[0])
```

### Pattern 3: Matrix Multiplication

```python
@cuda.jit
def matmul_kernel(A, B, C):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp
```

## Debugging Tips

### 1. Print from Kernel (Limited)

```python
@cuda.jit
def debug_kernel(arr):
    i = cuda.grid(1)
    if i == 0:  # Only thread 0 prints
        print("Thread 0 executing")
```

### 2. Check for Errors

```python
cuda.synchronize()  # Wait for kernel to complete
# Any errors will be raised here
```

### 3. Verify on CPU First

```python
# Always test CPU version first
result_cpu = A + B
# Then compare GPU result
result_gpu = gpu_add(A, B)
assert np.allclose(result_gpu, result_cpu)
```

### 4. Start Small

```python
# Test with tiny arrays first
A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[5, 6], [7, 8]], dtype=np.float32)
# Expected: [[6, 8], [10, 12]]
```

## Resources

- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [GPU Gems (Advanced Techniques)](https://developer.nvidia.com/gpugems/gpugems3/contributors)

## Summary

**Key Takeaways:**

1. CUDA kernels run in parallel across thousands of threads
2. Threads are organized in blocks, blocks in grids
3. Each thread computes its position with `cuda.grid()`
4. Data must be explicitly transferred between CPU and GPU
5. GPU excels at large-scale parallel computations
6. Always include boundary checks in kernels
7. Choose block sizes that are multiples of 32 (warp size)
