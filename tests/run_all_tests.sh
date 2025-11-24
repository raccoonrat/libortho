#!/bin/bash
#
# libortho - Run All Tests
#
# This script runs all available tests:
# 1. GPU environment check
# 2. CUDA kernel test (if CUDA available)
# 3. CPU forward test
# 4. Performance benchmark (Null Test)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "libortho - Test Suite"
echo "============================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
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
echo ""
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

# 3. CPU Forward Test
echo ""
echo "3. CPU Forward Test"
echo "============================================================"
echo "Compiling CPU test..."
if make test 2>&1 | tail -20; then
    run_test "CPU Forward" "./test_cpu_forward"
else
    echo -e "${RED}❌ CPU test compilation failed${NC}"
    ((TESTS_FAILED++))
fi

# 4. Performance Benchmark
echo ""
echo "4. Performance Benchmark (Null Test)"
echo "============================================================"
echo "Compiling benchmark..."
if make benchmark 2>&1 | tail -20; then
    run_test "Null Test Benchmark" "./benchmark_null_test"
else
    echo -e "${RED}❌ Benchmark compilation failed${NC}"
    ((TESTS_FAILED++))
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

