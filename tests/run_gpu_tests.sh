#!/bin/bash
#
# libortho - GPU Test Runner
#
# This script runs GPU tests in WSL environment
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "libortho - GPU Test Suite"
echo "============================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    echo "------------------------------------------------------------"
    echo "Test: $test_name"
    echo "------------------------------------------------------------"
    
    if eval "$test_cmd"; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# 1. GPU Environment Check
echo "1. GPU Environment Check"
echo "============================================================"
if command -v python3 &> /dev/null; then
    run_test "GPU Environment" "python3 check_gpu.py"
else
    echo -e "${YELLOW}⚠️  Python3 not found, skipping GPU check${NC}"
fi

# 2. CUDA Kernel Test
echo ""
echo "2. CUDA Kernel Test"
echo "============================================================"
if command -v nvcc &> /dev/null; then
    echo "Compiling CUDA test..."
    if make cuda-test 2>&1 | tail -20; then
        run_test "CUDA Kernel" "./test_cuda_kernel"
    else
        echo -e "${YELLOW}⚠️  CUDA test compilation failed${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  nvcc not found, skipping CUDA test${NC}"
fi

# 3. Tensor Core Test
echo ""
echo "3. Tensor Core Test"
echo "============================================================"
if command -v nvcc &> /dev/null; then
    echo "Compiling Tensor Core test..."
    if make tensor-core-test 2>&1 | tail -30; then
        run_test "Tensor Core" "./test_tensor_core"
    else
        echo -e "${YELLOW}⚠️  Tensor Core test compilation failed${NC}"
        echo "This may be due to:"
        echo "  - Missing CUDA Toolkit"
        echo "  - Missing Tensor Core support (requires compute capability >= 7.0)"
        echo "  - Compilation errors (check output above)"
    fi
else
    echo -e "${YELLOW}⚠️  nvcc not found, skipping Tensor Core test${NC}"
fi

# Summary
echo ""
echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi

