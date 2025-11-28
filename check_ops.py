#!/usr/bin/env python3
"""
libortho - Check C++ Extension Availability

Linus: 别在那猜了。写个脚本 check_ops.py 打印一下 libortho._C_ops 是否存在。
如果存在，你的"法拉利引擎"就就绪了。
"""

import sys
import os

def check_ops():
    """Check if libortho._C_ops is available."""
    print("=" * 60)
    print("libortho C++ Extension Check")
    print("=" * 60)
    
    # Try primary import
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    # Try to import C++ extension
    print("\n[Step 1] Trying to import libortho._C_ops...")
    try:
        import libortho._C_ops as _C
        print(f"✅ Successfully imported: {_C}")
        print(f"   Module location: {_C.__file__ if hasattr(_C, '__file__') else 'builtin'}")
        print(f"   Available functions: {[x for x in dir(_C) if not x.startswith('_')]}")
        
        # Check if forward function exists
        if hasattr(_C, 'forward'):
            print(f"✅ forward() function found")
            return True
        else:
            print(f"❌ forward() function not found")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        
        # Try fallback import
        print("\n[Step 2] Trying fallback import (_C_ops)...")
        try:
            import _C_ops as _C
            print(f"✅ Successfully imported via fallback: {_C}")
            print(f"   Available functions: {[x for x in dir(_C) if not x.startswith('_')]}")
            return True
        except ImportError as e2:
            print(f"❌ Fallback import also failed: {e2}")
            print("\n💡 Troubleshooting:")
            print("   1. Make sure you ran 'pip install -e .'")
            print("   2. Check if .so file exists: find . -name '*.so'")
            print("   3. Ensure you're in the project root directory")
            return False

def check_so_file():
    """Check if .so file exists."""
    print("\n[Step 3] Checking for .so files...")
    import subprocess
    try:
        result = subprocess.run(
            ['find', '.', '-name', '*.so', '-type', 'f'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            print("✅ Found .so files:")
            for line in result.stdout.strip().split('\n'):
                if line:
                    print(f"   {line}")
            return True
        else:
            print("❌ No .so files found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Windows or find not available
        print("⚠️  Could not check for .so files (Windows or find not available)")
        return None

if __name__ == "__main__":
    success = check_ops()
    check_so_file()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ RESULT: C++ extension is available. Your 'Ferrari engine' is ready!")
    else:
        print("❌ RESULT: C++ extension is NOT available. Using Python fallback.")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

