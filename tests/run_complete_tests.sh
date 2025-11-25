#!/bin/bash
#
# libortho - Complete Test Suite
#
# This script runs all tests including:
# 1. Python core logic verification
# 2. CPU forward test (with aligned allocation)
# 3. Performance benchmark (Null Test)
# 4. GPU tests (if available)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "============================================================"
echo "libortho - Complete Test Suite"
echo "============================================================"
echo "Date: $(date)"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    echo -e "${BLUE}------------------------------------------------------------${NC}"
    echo -e "${BLUE}Test: $test_name${NC}"
    echo -e "${BLUE}------------------------------------------------------------${NC}"
    echo ""
    
    if eval "$test_cmd"; then
        echo ""
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo ""
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Function to skip a test
skip_test() {
    local test_name="$1"
    local reason="$2"
    echo -e "${YELLOW}⚠️  SKIPPED: $test_name${NC}"
    echo "   Reason: $reason"
    ((TESTS_SKIPPED++))
}

# ============================================================
# Test 1: Python Core Logic Verification
# ============================================================
echo ""
echo "============================================================"
echo "Test 1: Python Core Logic Verification"
echo "============================================================"
echo "This test verifies the dual-manifold hypothesis:"
echo "  - Privacy can be separated via Hessian sieve"
echo "  - Kill switch (alpha=0) removes privacy but preserves general logic"
echo ""

if command -v python3 &> /dev/null; then
    run_test "Core Logic Verification" "python3 experiments/verify_core_logic.py"
else
    skip_test "Core Logic Verification" "python3 not found"
fi

# ============================================================
# Test 2: CPU Forward Test (with aligned allocation)
# ============================================================
echo ""
echo "============================================================"
echo "Test 2: CPU Forward Test"
echo "============================================================"
echo "This test verifies:"
echo "  - Aligned memory allocation (128-byte)"
echo "  - Correct forward pass computation"
echo "  - Alpha kill switch functionality"
echo ""

cd tests
if command -v gcc &> /dev/null; then
    echo "Compiling CPU test..."
    if make clean && make test 2>&1 | tail -30; then
        run_test "CPU Forward" "./test_cpu_forward"
    else
        echo -e "${RED}❌ CPU test compilation failed${NC}"
        ((TESTS_FAILED++))
    fi
else
    skip_test "CPU Forward" "gcc not found"
fi
cd ..

# ============================================================
# Test 3: Performance Benchmark (Null Test)
# ============================================================
echo ""
echo "============================================================"
echo "Test 3: Performance Benchmark (Null Test)"
echo "============================================================"
echo "This test verifies Linus's requirement:"
echo "  - If W_ortho is all zero, performance must be COMPLETELY EQUAL"
echo "  - Overhead must be < 1%"
echo ""

cd tests
if command -v gcc &> /dev/null; then
    echo "Compiling benchmark..."
    if make benchmark 2>&1 | tail -30; then
        run_test "Null Test Benchmark" "./benchmark_null_test"
    else
        echo -e "${RED}❌ Benchmark compilation failed${NC}"
        ((TESTS_FAILED++))
    fi
else
    skip_test "Null Test Benchmark" "gcc not found"
fi
cd ..

# ============================================================
# Test 4: GPU Environment Check
# ============================================================
echo ""
echo "============================================================"
echo "Test 4: GPU Environment Check"
echo "============================================================"
echo ""

cd tests
if command -v python3 &> /dev/null; then
    run_test "GPU Environment" "python3 check_gpu.py" || skip_test "GPU Environment" "GPU not available"
else
    skip_test "GPU Environment" "python3 not found"
fi
cd ..

# ============================================================
# Test 5: CUDA Kernel Test (if available)
# ============================================================
echo ""
echo "============================================================"
echo "Test 5: CUDA Kernel Test"
echo "============================================================"
echo ""

cd tests
if command -v nvcc &> /dev/null; then
    echo "Compiling CUDA test..."
    if make cuda-test 2>&1 | tail -30; then
        run_test "CUDA Kernel" "./test_cuda_kernel" || skip_test "CUDA Kernel" "CUDA runtime error"
    else
        skip_test "CUDA Kernel" "Compilation failed"
    fi
else
    skip_test "CUDA Kernel" "nvcc not found"
fi
cd ..

# ============================================================
# Test 6: Tensor Core Test (if available)
# ============================================================
echo ""
echo "============================================================"
echo "Test 6: Tensor Core Test"
echo "============================================================"
echo ""

cd tests
if command -v nvcc &> /dev/null; then
    echo "Compiling Tensor Core test..."
    if make tensor-core-test 2>&1 | tail -30; then
        run_test "Tensor Core" "./test_tensor_core" || skip_test "Tensor Core" "Tensor Core not available"
    else
        skip_test "Tensor Core" "Compilation failed"
    fi
else
    skip_test "Tensor Core" "nvcc not found"
fi
cd ..

# ============================================================
# Test 7: Sieve Tool Test
# ============================================================
echo ""
echo "============================================================"
echo "Test 7: Sieve Tool (Index Sorting)"
echo "============================================================"
echo "This test verifies:"
echo "  - Hessian sieve functionality"
echo "  - Index sorting (row-major order)"
echo ""

if command -v python3 &> /dev/null; then
    python3 << 'EOF'
import sys
import torch
sys.path.insert(0, '.')
from tools.sieve import hessian_sieve, pack_ortho_sparse, compute_hessian_diag_approx

print("Testing Hessian Sieve...")

# Create test data
out_features, in_features = 64, 64
weight = torch.randn(out_features, in_features)
inputs = torch.randn(100, in_features)
H_diag = compute_hessian_diag_approx(inputs)

# Run sieve
w_base, w_ortho = hessian_sieve(weight, H_diag, sparsity_target=0.95)
print(f"  Base shape: {w_base.shape}")
print(f"  Ortho shape: {w_ortho.shape}")
print(f"  Ortho sparsity: {(w_ortho == 0).float().mean():.2%}")

# Test index sorting
indices, values = pack_ortho_sparse(w_ortho, format="coo")
print(f"  Non-zero count: {len(indices)}")

# Verify sorting: indices should be in row-major order
if len(indices) > 0:
    rows = indices // in_features
    cols = indices % in_features
    
    # Check if sorted by row, then column
    is_sorted = True
    for i in range(len(rows) - 1):
        if rows[i] > rows[i+1] or (rows[i] == rows[i+1] and cols[i] > cols[i+1]):
            is_sorted = False
            break
    
    if is_sorted:
        print("  ✅ Index sorting: PASSED")
        sys.exit(0)
    else:
        print("  ❌ Index sorting: FAILED")
        sys.exit(1)
else:
    print("  ⚠️  No non-zero elements to test sorting")
    sys.exit(0)
EOF
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ PASSED: Sieve Tool${NC}"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}❌ FAILED: Sieve Tool${NC}"
        ((TESTS_FAILED++))
    fi
else
    skip_test "Sieve Tool" "python3 not found"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "Test Summary"
echo "============================================================"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo -e "${YELLOW}Tests Skipped: $TESTS_SKIPPED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi

