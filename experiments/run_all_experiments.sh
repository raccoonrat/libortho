#!/bin/bash
# libortho - Run All Experiments in WSL
# 
# This script runs all three verification experiments
# in the WSL environment.

set -e  # Exit on error

echo "=========================================="
echo "libortho - Running All Experiments"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Python: $PYTHON_VERSION"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; import numpy" 2>/dev/null || {
    echo "❌ Error: Missing dependencies (torch, numpy)"
    echo "Please install: pip install torch numpy"
    exit 1
}
echo "✅ Dependencies OK"
echo ""

# Run experiments
EXPERIMENTS=(
    "verify_core_logic.py:Experiment 1: Privacy Kill Switch"
    "saving_genius.py:Experiment 2: Saving the Genius"
    "dual_dp.py:Experiment 3: Dual Differential Privacy"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

for exp_info in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r script_name exp_title <<< "$exp_info"
    script_path="$SCRIPT_DIR/$script_name"
    
    if [ ! -f "$script_path" ]; then
        echo "❌ Error: $script_name not found"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi
    
    echo "=========================================="
    echo "$exp_title"
    echo "=========================================="
    echo "Running: $script_name"
    echo ""
    
    if python3 "$script_path"; then
        echo ""
        echo "✅ $exp_title: PASSED"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "❌ $exp_title: FAILED"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo ""
    echo ""
done

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Passed: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✅ All experiments passed!"
    exit 0
else
    echo "❌ Some experiments failed"
    exit 1
fi

