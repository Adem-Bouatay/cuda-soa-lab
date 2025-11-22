"""
Minimal CUDA kernel test to verify GPU availability and basic CUDA functionality.
This is run during Jenkins pipeline to ensure GPU is accessible.
"""

from numba import cuda
import numpy as np
import sys


@cuda.jit
def test_add_kernel(x, out):
    """Simple CUDA kernel that adds 10 to each element"""
    idx = cuda.grid(1)
    if idx < x.shape[0]:
        out[idx] = x[idx] + 10


def test_cuda_basic():
    """Test basic CUDA functionality"""
    print("=" * 60)
    print("CUDA Sanity Test")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"✓ CUDA Available: {cuda.is_available()}")
    
    if not cuda.is_available():
        print("✗ CUDA is not available!")
        sys.exit(1)
    
    # List GPUs
    print(f"✓ Number of GPUs: {len(cuda.gpus)}")
    
    # Get GPU details
    for gpu in cuda.gpus:
        with gpu:
            print(f"  - GPU: {gpu.name}")
            print(f"    Compute Capability: {gpu.compute_capability}")
    
    # Test simple kernel
    print("\nTesting simple CUDA kernel...")
    
    # Create test data
    n = 10
    x = np.arange(n, dtype=np.float32)
    expected = x + 10
    
    # Allocate device memory
    d_x = cuda.to_device(x)
    d_out = cuda.device_array(n, dtype=np.float32)
    
    # Launch kernel
    threads_per_block = 32
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    test_add_kernel[blocks_per_grid, threads_per_block](d_x, d_out)
    
    # Copy result back
    result = d_out.copy_to_host()
    
    # Verify result
    if np.allclose(result, expected):
        print("✓ Kernel execution successful!")
        print(f"  Input:    {x[:5]}...")
        print(f"  Output:   {result[:5]}...")
        print(f"  Expected: {expected[:5]}...")
    else:
        print("✗ Kernel execution failed!")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All CUDA tests passed!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = test_cuda_basic()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n✗ CUDA test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
