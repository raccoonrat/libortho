#!/usr/bin/env python3
"""
Quick check script to verify all experiment files are ready.
Run this before running the actual experiments.
"""

import os
import sys

def check_file(path, description):
    """Check if a file exists and is readable."""
    if os.path.isfile(path):
        size = os.path.getsize(path)
        print(f"✅ {description}: {path} ({size} bytes)")
        return True
    else:
        print(f"❌ {description}: {path} NOT FOUND")
        return False

def check_imports():
    """Check if required modules can be imported."""
    try:
        import torch
        import numpy
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ NumPy: {numpy.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Install with: pip3 install torch numpy")
        return False

def main():
    print("=" * 60)
    print("libortho - Experiment Files Check")
    print("=" * 60)
    print()
    
    # Check experiment files
    print("Checking experiment files...")
    experiments = [
        ("experiments/verify_core_logic.py", "Experiment 1: Privacy Kill Switch"),
        ("experiments/saving_genius.py", "Experiment 2: Saving the Genius"),
        ("experiments/dual_dp.py", "Experiment 3: Dual Differential Privacy"),
    ]
    
    all_files_ok = True
    for path, desc in experiments:
        if not check_file(path, desc):
            all_files_ok = False
    
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    deps_ok = check_imports()
    
    print()
    print("=" * 60)
    
    if all_files_ok and deps_ok:
        print("✅ All checks passed! Ready to run experiments.")
        print()
        print("Run experiments with:")
        print("  python3 experiments/verify_core_logic.py")
        print("  python3 experiments/saving_genius.py")
        print("  python3 experiments/dual_dp.py")
        print()
        print("Or use the test script:")
        print("  bash test_experiments.sh")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

