#!/bin/bash
# Check CUDA version and supported architectures

echo "Checking CUDA version and supported architectures..."
echo ""

if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found. CUDA Toolkit not installed."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
echo "CUDA Version: $CUDA_VERSION"

CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

echo ""
echo "Supported Architectures:"
echo "  ✅ sm_75  (Turing) - CUDA 10.0+"
echo "  ✅ sm_80  (Ampere) - CUDA 11.0+"
echo "  ✅ sm_86  (Ampere consumer) - CUDA 11.0+"
echo "  ✅ sm_89  (Ada Lovelace) - CUDA 11.8+"

if [ $CUDA_MAJOR -gt 12 ] || ([ $CUDA_MAJOR -eq 12 ] && [ $CUDA_MINOR -ge 8 ]); then
    echo "  ✅ sm_100 (Blackwell) - CUDA 12.8+"
    echo ""
    echo "✅ Your CUDA version supports all architectures including Blackwell (sm_100)"
else
    echo "  ❌ sm_100 (Blackwell) - Requires CUDA 12.8+"
    echo ""
    echo "⚠️  Your CUDA version does not support sm_100 (Blackwell)"
    echo "   To use RTX 5060, upgrade to CUDA 12.8 or higher"
    echo "   Current Makefile will automatically exclude sm_100"
fi

