#!/bin/bash
# Quick test script for libortho experiments
# Run this in WSL: bash test_experiments.sh

set -e

echo "=========================================="
echo "libortho - Testing Experiments"
echo "=========================================="
echo ""

# Check Python
echo "1. Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "   ✅ $PYTHON_VERSION"
else
    echo "   ❌ python3 not found"
    exit 1
fi
echo ""

# Check dependencies
echo "2. Checking dependencies..."
if python3 -c "import torch; import numpy" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null)
    echo "   ✅ PyTorch: $TORCH_VERSION"
    echo "   ✅ NumPy: $NUMPY_VERSION"
else
    echo "   ❌ Missing dependencies (torch, numpy)"
    echo "   Install with: pip3 install torch numpy"
    exit 1
fi
echo ""

# Check we're in the right directory
echo "3. Checking project structure..."
if [ -f "experiments/verify_core_logic.py" ] && [ -f "experiments/saving_genius.py" ] && [ -f "experiments/dual_dp.py" ]; then
    echo "   ✅ All experiment files found"
else
    echo "   ❌ Experiment files not found"
    echo "   Current directory: $(pwd)"
    exit 1
fi
echo ""

# Run experiments
echo "=========================================="
echo "Running Experiments"
echo "=========================================="
echo ""

EXPERIMENTS=(
    "experiments/verify_core_logic.py:实验1: 隐私开关测试"
    "experiments/saving_genius.py:实验2: 天才的保留"
    "experiments/dual_dp.py:实验3: 对偶差分隐私"
)

SUCCESS=0
FAIL=0

for exp_info in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r script_path exp_num exp_name <<< "$exp_info"
    
    echo "----------------------------------------"
    echo "$exp_num: $exp_name"
    echo "----------------------------------------"
    echo "Running: $script_path"
    echo ""
    
    if python3 "$script_path" 2>&1; then
        echo ""
        echo "✅ $exp_name: PASSED"
        SUCCESS=$((SUCCESS + 1))
    else
        EXIT_CODE=$?
        echo ""
        echo "❌ $exp_name: FAILED (exit code: $EXIT_CODE)"
        FAIL=$((FAIL + 1))
    fi
    
    echo ""
    echo ""
done

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "✅ Passed: $SUCCESS"
echo "❌ Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "🎉 All experiments passed!"
    exit 0
else
    echo "⚠️  Some experiments failed. Check output above."
    exit 1
fi

