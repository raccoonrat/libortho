#!/bin/bash
# Script to create and setup RTX 5060 (Blackwell) support branch

set -e

echo "============================================================"
echo "Creating RTX 5060 (Blackwell) Support Branch"
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
BRANCH_NAME="feature/rtx5060-blackwell-support"
echo "Creating branch: $BRANCH_NAME"
git checkout -b $BRANCH_NAME

echo ""
echo "✅ Branch created successfully!"
echo ""
echo "Branch: $BRANCH_NAME"
echo "GPU: NVIDIA GeForce RTX 5060"
echo "Architecture: Blackwell"
echo "CUDA Compute Capability: 12.0 (sm_100)"
echo ""
echo "Changes made:"
echo "  - setup.py: Added -arch=sm_100 support"
echo "  - tests/Makefile: Added -arch=sm_100 support"
echo "  - docs/RTX5060_BLACKWELL_SUPPORT.md: Added documentation"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Test compilation: cd tests && make clean && make cuda-test"
echo "  3. Check GPU: cd tests && python3 check_gpu.py"
echo "  4. Commit changes: git add . && git commit -m 'Add RTX 5060 (Blackwell) support'"
echo ""
echo "For detailed information, see: docs/RTX5060_BLACKWELL_SUPPORT.md"

