#!/usr/bin/env python3
"""
Simple test client for the GPU Matrix Addition Service
Tests all endpoints without requiring the service to be running
"""

import numpy as np
import requests
import sys
import time

# Configuration
BASE_URL = "http://localhost:8001"

def test_health():
    """Test the health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✓ Status: {response.status_code}")
            print(f"✓ Response: {response.json()}")
            return True
        else:
            print(f"✗ Failed with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed. Is the service running?")
        print(f"  Start it with: python3 main.py")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_gpu_info():
    """Test the GPU info endpoint"""
    print("\n" + "="*60)
    print("TEST 2: GPU Information")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/gpu-info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Status: {response.status_code}")
            print(f"✓ Response:")
            for gpu in data.get('gpus', []):
                print(f"    GPU {gpu['gpu']}: {gpu['memory_used_MB']} MB / {gpu['memory_total_MB']} MB")
            return True
        else:
            print(f"✗ Failed with status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_matrix_addition(file_a, file_b, expected_success=True):
    """Test matrix addition endpoint"""
    print("\n" + "="*60)
    print(f"TEST 3: Matrix Addition - {file_a} + {file_b}")
    print("="*60)
    
    try:
        with open(file_a, 'rb') as fa, open(file_b, 'rb') as fb:
            files = {
                'file_a': (file_a, fa, 'application/octet-stream'),
                'file_b': (file_b, fb, 'application/octet-stream')
            }
            
            start = time.time()
            response = requests.post(f"{BASE_URL}/add", files=files, timeout=30)
            elapsed = time.time() - start
            
            print(f"  Request took: {elapsed:.4f} seconds")
            
            if expected_success:
                if response.status_code == 200:
                    data = response.json()
                    print(f"✓ Status: {response.status_code}")
                    print(f"✓ Matrix shape: {data['matrix_shape']}")
                    print(f"✓ GPU elapsed time: {data['elapsed_time']} seconds")
                    print(f"✓ Device: {data['device']}")
                    return True
                else:
                    print(f"✗ Failed with status: {response.status_code}")
                    print(f"  Response: {response.text}")
                    return False
            else:
                # Expecting an error
                if response.status_code != 200:
                    print(f"✓ Correctly rejected with status: {response.status_code}")
                    print(f"✓ Error message: {response.json().get('detail', 'N/A')}")
                    return True
                else:
                    print(f"✗ Should have failed but got status: {response.status_code}")
                    return False
                    
    except FileNotFoundError as e:
        print(f"✗ File not found: {e}")
        print("  Run 'python3 test_service.py' to create test matrices")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GPU MATRIX ADDITION SERVICE - API TESTS")
    print("="*60)
    print(f"Target: {BASE_URL}")
    print()
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health()))
    
    # Test 2: GPU info
    results.append(("GPU Info", test_gpu_info()))
    
    # Test 3: Valid matrix addition (small)
    results.append(("Matrix Add (100x100)", 
                   test_matrix_addition("test_matrix_a.npz", "test_matrix_b.npz")))
    
    # Test 4: Valid matrix addition (large)
    results.append(("Matrix Add (512x512)", 
                   test_matrix_addition("matrix1.npz", "matrix2.npz")))
    
    # Test 5: Invalid - mismatched shapes
    results.append(("Error Handling", 
                   test_matrix_addition("test_matrix_a.npz", "test_matrix_mismatch.npz", 
                                       expected_success=False)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Error: 'requests' library not found")
        print("Install it with: pip install requests")
        sys.exit(1)
    
    sys.exit(main())

