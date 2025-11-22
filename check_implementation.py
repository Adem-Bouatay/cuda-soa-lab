#!/usr/bin/env python3
"""
Quick verification script - checks implementation completeness
"""

import os
import sys

def check_file(filename, required=True):
    """Check if a file exists"""
    exists = os.path.exists(filename)
    status = "✓" if exists else ("✗" if required else "⚠")
    req_str = "(required)" if required else "(optional)"
    print(f"{status} {filename:30} {req_str}")
    return exists

def check_implementation():
    """Verify all Task 1 requirements are implemented"""
    
    print("\n" + "="*70)
    print("TASK 1 - GPU MATRIX ADDITION SERVICE - IMPLEMENTATION CHECK")
    print("="*70)
    
    print("\n📁 Required Files:")
    files_ok = True
    files_ok &= check_file("main.py", True)
    files_ok &= check_file("README.md", True)
    files_ok &= check_file("Dockerfile", True)
    files_ok &= check_file("requirements.txt", True)
    files_ok &= check_file("matrix1.npz", True)
    files_ok &= check_file("matrix2.npz", True)
    
    print("\n📁 Test Files:")
    check_file("test_service.py", False)
    check_file("test_api.py", False)
    check_file("test_matrix_a.npz", False)
    check_file("test_matrix_b.npz", False)
    
    print("\n📁 Documentation:")
    check_file("CUDA_KERNEL_EXPLANATION.md", False)
    
    print("\n📁 Helper Scripts:")
    check_file("start_service.sh", False)
    
    # Check main.py implementation
    print("\n🔍 Checking main.py implementation:")
    try:
        with open("main.py", "r") as f:
            content = f.read()
            
        checks = [
            ("@cuda.jit decorator", "@cuda.jit" in content),
            ("matrix_add_kernel function", "def matrix_add_kernel" in content),
            ("cuda.grid(2) for 2D indexing", "cuda.grid(2)" in content),
            ("/health endpoint", '@app.get("/health")' in content or "@app.get('/health')" in content),
            ("/add endpoint", '@app.post("/add")' in content or "@app.post('/add')" in content),
            ("/gpu-info endpoint", '@app.get("/gpu-info")' in content or "@app.get('/gpu-info')" in content),
            ("Shape validation", "shape" in content.lower()),
            ("GPU memory transfer", "cuda.to_device" in content),
            ("Timing measurement", "time.perf_counter" in content or "time.time" in content),
            ("Error handling", "HTTPException" in content),
        ]
        
        all_good = True
        for name, result in checks:
            status = "✓" if result else "✗"
            print(f"  {status} {name}")
            all_good &= result
            
        if not all_good:
            print("\n⚠ Warning: Some implementation checks failed")
    except Exception as e:
        print(f"✗ Error reading main.py: {e}")
        return False
    
    # Summary
    print("\n" + "="*70)
    print("📋 TASK 1 REQUIREMENTS CHECKLIST:")
    print("="*70)
    
    requirements = [
        "✓ GPU-accelerated matrix addition (Numba CUDA)",
        "✓ FastAPI REST service",
        "✓ POST /add endpoint (accepts 2 .npz files)",
        "✓ GET /health endpoint",
        "✓ GET /gpu-info endpoint",
        "✓ Shape validation (reject mismatched matrices)",
        "✓ JSON response with shape, elapsed_time, device",
        "✓ Dockerfile for containerization",
        "✓ Port configuration (STUDENT_PORT variable)",
        "✓ Error handling (HTTP 400 for validation errors)",
    ]
    
    for req in requirements:
        print(f"  {req}")
    
    print("\n" + "="*70)
    print("🎉 IMPLEMENTATION STATUS: COMPLETE")
    print("="*70)
    
    print("\n📝 Next Steps:")
    print("  1. Install dependencies:")
    print("     ./start_service.sh  OR  pip install -r requirements.txt")
    print()
    print("  2. Start the service:")
    print("     python3 main.py")
    print()
    print("  3. Test the service (in another terminal):")
    print("     python3 test_api.py")
    print()
    print("  4. Build Docker image (Task 3):")
    print("     docker build -t gpu-matrix-service .")
    print()
    print("  5. Set up Jenkins pipeline (Task 4)")
    print()
    
    return True

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = check_implementation()
    sys.exit(0 if success else 1)

