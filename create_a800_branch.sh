#!/bin/bash
# Script to create and setup A800 (Ampere) support branch

set -e

echo "============================================================"
echo "Creating A800 (Ampere) Support Branch"
echo "============================================================"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"
echo ""

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  Warning: You have uncommitted changes"
    echo "   Consider committing or stashing them before creating a new branch"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create new branch
BRANCH_NAME="feature/a800-ampere-support"
echo "Creating branch: $BRANCH_NAME"
git checkout -b $BRANCH_NAME

echo ""
echo "✅ Branch created successfully!"
echo ""
echo "Branch: $BRANCH_NAME"
echo "GPU: NVIDIA A800"
echo "Architecture: Ampere"
echo "CUDA Compute Capability: 8.0 (sm_80)"
echo "Memory: 80GB (typical)"
echo ""
echo "Note: A800 uses Ampere architecture (sm_80),"
echo "      which is already supported in the main configuration."
echo "      This branch ensures explicit support and optimization for A800."
echo ""
echo "Changes made:"
echo "  - docs/A800_AMPERE_SUPPORT.md: Added documentation"
echo "  - QUICKSTART_A800.md: Added quick start guide"
echo "  - Memory optimizations for large models (80GB support)"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Test compilation: cd tests && make clean && make cuda-test"
echo "  3. Check GPU: cd tests && python3 check_gpu.py"
echo "  4. Test with large model: python3 experiments/complete_real_model_experiments.py"
echo "  5. Commit changes: git add . && git commit -m 'Add A800 (Ampere) support'"
echo ""
echo "For detailed information, see: docs/A800_AMPERE_SUPPORT.md"

