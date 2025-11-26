#!/bin/bash
# Clean build script for libortho

echo "Cleaning build artifacts..."

# Remove Python build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf libortho.egg-info/

# Remove Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove CUDA object files
find . -type f -name "*.o" -delete 2>/dev/null || true
find . -type f -name "*.so" -delete 2>/dev/null || true

# Remove test binaries
rm -f tests/test_cuda_kernel
rm -f tests/test_tensor_core
rm -f tests/test_cpu_forward
rm -f tests/benchmark_null_test

echo "✅ Clean complete!"

