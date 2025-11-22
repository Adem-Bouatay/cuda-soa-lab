#!/usr/bin/env python3
"""
Script to create test matrices for the GPU Matrix Addition Service.
Generates .npz files with different sizes for testing.
"""

import numpy as np
import sys


def create_matrices(size=512, dtype=np.float32, filename_prefix="matrix"):
    """
    Create test matrices and save as .npz files.
    
    Args:
        size: Matrix dimension (creates size x size matrix)
        dtype: Data type for the matrices
        filename_prefix: Prefix for output files
    """
    print(f"Creating {size}x{size} test matrices...")
    
    # Create random matrices
    matrix1 = np.random.rand(size, size).astype(dtype)
    matrix2 = np.random.rand(size, size).astype(dtype)
    
    # Save as .npz files
    filename1 = f"{filename_prefix}1.npz"
    filename2 = f"{filename_prefix}2.npz"
    
    np.savez(filename1, arr_0=matrix1)
    np.savez(filename2, arr_0=matrix2)
    
    print(f"✓ Created {filename1} - Shape: {matrix1.shape}, Size: {matrix1.nbytes / 1024:.2f} KB")
    print(f"✓ Created {filename2} - Shape: {matrix2.shape}, Size: {matrix2.nbytes / 1024:.2f} KB")
    
    # Calculate expected result for verification
    expected_sum = matrix1 + matrix2
    print(f"\nSample values:")
    print(f"  Matrix 1 [0,0]: {matrix1[0,0]:.6f}")
    print(f"  Matrix 2 [0,0]: {matrix2[0,0]:.6f}")
    print(f"  Expected sum [0,0]: {expected_sum[0,0]:.6f}")
    
    return filename1, filename2


def create_test_suite():
    """Create a suite of test matrices with different sizes"""
    print("=" * 60)
    print("Creating Test Matrix Suite")
    print("=" * 60)
    print()
    
    # Small matrices (fast testing)
    print("1. Small matrices (64x64):")
    create_matrices(64, filename_prefix="small_matrix")
    print()
    
    # Medium matrices (standard)
    print("2. Medium matrices (512x512):")
    create_matrices(512, filename_prefix="matrix")
    print()
    
    # Large matrices (performance testing)
    print("3. Large matrices (2048x2048):")
    create_matrices(2048, filename_prefix="large_matrix")
    print()
    
    # Create mismatched matrices for error testing
    print("4. Mismatched matrices for error testing:")
    matrix_a = np.random.rand(512, 512).astype(np.float32)
    matrix_b = np.random.rand(256, 256).astype(np.float32)
    
    np.savez("mismatch_a.npz", arr_0=matrix_a)
    np.savez("mismatch_b.npz", arr_0=matrix_b)
    
    print(f"✓ Created mismatch_a.npz - Shape: {matrix_a.shape}")
    print(f"✓ Created mismatch_b.npz - Shape: {matrix_b.shape}")
    print("  (These should trigger a 400 error when used together)")
    print()
    
    print("=" * 60)
    print("✓ Test suite created successfully!")
    print("=" * 60)
    print()
    print("Test your API with:")
    print("  curl -X POST http://localhost:8001/add \\")
    print("    -F 'file_a=@matrix1.npz' \\")
    print("    -F 'file_b=@matrix2.npz'")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom size specified
        try:
            size = int(sys.argv[1])
            create_matrices(size)
        except ValueError:
            print(f"Error: Invalid size '{sys.argv[1]}'. Please provide an integer.")
            sys.exit(1)
    else:
        # Create full test suite
        create_test_suite()
