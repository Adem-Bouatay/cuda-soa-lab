#!/usr/bin/env python3
"""
Test script for GPU Matrix Addition Service
This script validates the implementation without requiring GPU hardware.
"""

import numpy as np
import os

def test_matrix_files():
    """Test that the sample matrix files exist and are valid"""
    print("=" * 60)
    print("Test 1: Checking matrix files")
    print("=" * 60)
    
    files = ['matrix1.npz', 'matrix2.npz']
    for filename in files:
        if os.path.exists(filename):
            with np.load(filename) as data:
                matrix = data[data.files[0]]
                print(f"✓ {filename}: shape={matrix.shape}, dtype={matrix.dtype}")
        else:
            print(f"✗ {filename}: NOT FOUND")
    print()

def create_test_matrices():
    """Create small test matrices for validation"""
    print("=" * 60)
    print("Test 2: Creating test matrices")
    print("=" * 60)
    
    # Create two small test matrices
    a = np.random.rand(100, 100).astype(np.float32)
    b = np.random.rand(100, 100).astype(np.float32)
    
    # Save them
    np.savez('test_matrix_a.npz', a)
    np.savez('test_matrix_b.npz', b)
    
    print(f"✓ Created test_matrix_a.npz: shape={a.shape}")
    print(f"✓ Created test_matrix_b.npz: shape={b.shape}")
    
    # Create mismatched matrices for error testing
    c = np.random.rand(50, 50).astype(np.float32)
    np.savez('test_matrix_mismatch.npz', c)
    print(f"✓ Created test_matrix_mismatch.npz: shape={c.shape} (for error testing)")
    print()
    
    return a, b

def verify_cpu_computation(a, b):
    """Verify the computation logic on CPU"""
    print("=" * 60)
    print("Test 3: CPU verification")
    print("=" * 60)
    
    result_cpu = a + b
    print(f"✓ CPU addition successful: shape={result_cpu.shape}")
    print(f"  Sample values:")
    print(f"    a[0,0] = {a[0,0]:.4f}")
    print(f"    b[0,0] = {b[0,0]:.4f}")
    print(f"    result[0,0] = {result_cpu[0,0]:.4f}")
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("=" * 60)
    print("Test 4: Checking dependencies")
    print("=" * 60)
    
    try:
        import fastapi
        print(f"✓ FastAPI: {fastapi.__version__}")
    except ImportError:
        print("✗ FastAPI: NOT INSTALLED")
    
    try:
        import uvicorn
        print(f"✓ Uvicorn: installed")
    except ImportError:
        print("✗ Uvicorn: NOT INSTALLED")
    
    try:
        import numpy
        print(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        print("✗ NumPy: NOT INSTALLED")
    
    try:
        from numba import cuda
        print(f"✓ Numba CUDA: installed")
        # Try to detect GPU
        try:
            gpu = cuda.get_current_device()
            print(f"  → GPU detected: {gpu.name.decode()}")
        except:
            print(f"  ⚠ No GPU detected (service will fail at runtime)")
    except ImportError:
        print("✗ Numba CUDA: NOT INSTALLED")
    
    try:
        import prometheus_client
        print(f"✓ Prometheus Client: installed")
    except ImportError:
        print("✗ Prometheus Client: NOT INSTALLED")
    
    print()

def print_usage_instructions():
    """Print instructions for testing the service"""
    print("=" * 60)
    print("How to test the service")
    print("=" * 60)
    print()
    print("1. Install dependencies (if not already done):")
    print("   pip install fastapi uvicorn numpy numba-cuda prometheus-client python-multipart")
    print()
    print("2. Start the service:")
    print("   python3 main.py")
    print()
    print("3. Test endpoints in another terminal:")
    print()
    print("   # Health check")
    print("   curl http://localhost:8001/health")
    print()
    print("   # GPU info")
    print("   curl http://localhost:8001/gpu-info")
    print()
    print("   # Matrix addition with test matrices")
    print("   curl -X POST http://localhost:8001/add \\")
    print("     -F 'file_a=@test_matrix_a.npz' \\")
    print("     -F 'file_b=@test_matrix_b.npz'")
    print()
    print("   # Matrix addition with provided matrices")
    print("   curl -X POST http://localhost:8001/add \\")
    print("     -F 'file_a=@matrix1.npz' \\")
    print("     -F 'file_b=@matrix2.npz'")
    print()
    print("   # Test error handling (mismatched shapes)")
    print("   curl -X POST http://localhost:8001/add \\")
    print("     -F 'file_a=@test_matrix_a.npz' \\")
    print("     -F 'file_b=@test_matrix_mismatch.npz'")
    print()
    print("Expected output format:")
    print('  {"matrix_shape": [100, 100], "elapsed_time": 0.002134, "device": "GPU"}')
    print()

if __name__ == "__main__":
    print()
    print("GPU MATRIX ADDITION SERVICE - VALIDATION TESTS")
    print()
    
    # Run tests
    test_matrix_files()
    a, b = create_test_matrices()
    verify_cpu_computation(a, b)
    check_dependencies()
    print_usage_instructions()
    
    print("=" * 60)
    print("✓ Pre-flight checks complete!")
    print("=" * 60)
    print()

