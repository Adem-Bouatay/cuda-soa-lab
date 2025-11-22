#!/usr/bin/env python3
"""
Minimal CUDA kernel test to verify GPU is accessible and working.
This script tests basic CUDA functionality before deployment.
"""
from numba import cuda
import numpy as np
import sys

@cuda.jit
def test_kernel(arr, out):
    """Simple kernel that adds 1 to each element"""
    idx = cuda.grid(1)
    if idx < arr.size:
        out[idx] = arr[idx] + 1

def main():
    print("=" * 50)
    print("CUDA GPU Sanity Check")
    print("=" * 50)

    # Check if CUDA is available
    try:
        print(f"\n✓ CUDA available: {cuda.is_available()}")
        if not cuda.is_available():
            print("✗ CUDA is not available on this system")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Error checking CUDA availability: {e}")
        sys.exit(1)

    # List available GPUs
    try:
        gpus = cuda.list_devices()
        print(f"✓ Number of GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name.decode()}")
    except Exception as e:
        print(f"✗ Error listing GPUs: {e}")
        sys.exit(1)

    # Test simple kernel execution
    try:
        print("\n" + "-" * 50)
        print("Testing kernel execution...")

        # Create test data
        size = 1000
        test_data = np.arange(size, dtype=np.float32)

        # Allocate device memory
        d_in = cuda.to_device(test_data)
        d_out = cuda.device_array_like(d_in)

        # Configure and launch kernel
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
        test_kernel[blocks_per_grid, threads_per_block](d_in, d_out)

        # Copy result back
        result = d_out.copy_to_host()

        # Verify result
        expected = test_data + 1
        if np.allclose(result, expected):
            print("✓ Kernel execution successful!")
            print(f"✓ Data transfer and computation verified")
        else:
            print("✗ Kernel produced incorrect results")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Error during kernel execution: {e}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("✓ All GPU tests passed!")
    print("=" * 50)
    return 0

if __name__ == "__main__":
    sys.exit(main())

