#!/bin/bash
# Script to create and setup GTX 4050 (Ada Lovelace) support branch

set -e

echo "============================================================"
echo "Creating GTX 4050 (Ada Lovelace) Support Branch"
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
BRANCH_NAME="feature/gtx4050-ada-lovelace-support"
echo "Creating branch: $BRANCH_NAME"
git checkout -b $BRANCH_NAME

echo ""
echo "✅ Branch created successfully!"
echo ""
echo "Branch: $BRANCH_NAME"
echo "GPU: NVIDIA GeForce GTX 4050"
echo "Architecture: Ada Lovelace"
echo "CUDA Compute Capability: 8.9 (sm_89)"
echo ""
echo "Note: GTX 4050 uses Ada Lovelace architecture (sm_89),"
echo "      which is already supported in the main configuration."
echo "      This branch ensures explicit support and documentation."
echo ""
echo "Changes made:"
echo "  - docs/GTX4050_ADA_LOVELACE_SUPPORT.md: Added documentation"
echo "  - QUICKSTART_GTX4050.md: Added quick start guide"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git diff"
echo "  2. Test compilation: cd tests && make clean && make cuda-test"
echo "  3. Check GPU: cd tests && python3 check_gpu.py"
echo "  4. Commit changes: git add . && git commit -m 'Add GTX 4050 (Ada Lovelace) support'"
echo ""
echo "For detailed information, see: docs/GTX4050_ADA_LOVELACE_SUPPORT.md"

