#!/usr/bin/env python3
"""
libortho - GPU Environment Check

Check CUDA availability, GPU capabilities, and Tensor Core support.
"""

import sys
import subprocess

def check_nvcc():
    """Check if nvcc is available."""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ nvcc found")
            print(f"   {result.stdout.split(chr(10))[0]}")
            return True
        else:
            print("❌ nvcc not found or error")
            return False
    except FileNotFoundError:
        print("❌ nvcc not found in PATH")
        return False
    except Exception as e:
        print(f"❌ Error checking nvcc: {e}")
        return False

def check_pytorch_cuda():
    """Check PyTorch CUDA support."""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n   GPU {i}: {props.name}")
                print(f"     Compute Capability: {props.major}.{props.minor}")
                print(f"     Total Memory: {props.total_memory / 1024**3:.2f} GB")
                
                # Check Tensor Core support
                has_tensor_cores = props.major >= 7
                if has_tensor_cores:
                    print(f"     ✅ Tensor Cores: Supported (sm_{props.major}{props.minor})")
                else:
                    print(f"     ❌ Tensor Cores: Not supported (requires >= 7.0)")
            
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            print("   Install CUDA-enabled PyTorch: https://pytorch.org/")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking PyTorch CUDA: {e}")
        return False

def check_nvidia_smi():
    """Check nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n✅ nvidia-smi output:")
            # Print first few lines
            lines = result.stdout.split('\n')[:10]
            for line in lines:
                print(f"   {line}")
            return True
        else:
            print("❌ nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        print("   GPU may not be available or drivers not installed")
        return False
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
        return False

def main():
    print("=" * 60)
    print("libortho - GPU Environment Check")
    print("=" * 60)
    print()
    
    print("1. Checking nvcc (CUDA Compiler)...")
    nvcc_ok = check_nvcc()
    print()
    
    print("2. Checking PyTorch CUDA support...")
    pytorch_ok = check_pytorch_cuda()
    print()
    
    print("3. Checking nvidia-smi (GPU Info)...")
    nvidia_smi_ok = check_nvidia_smi()
    print()
    
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    all_ok = nvcc_ok and pytorch_ok
    
    if all_ok:
        print("✅ GPU environment is ready!")
        print("   You can compile and test CUDA kernels.")
        if pytorch_ok:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                if props.major >= 7:
                    print("   ✅ Tensor Core support available!")
                else:
                    print("   ⚠️  Tensor Cores not available (requires compute capability >= 7.0)")
    else:
        print("❌ GPU environment not ready")
        if not nvcc_ok:
            print("   - Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
        if not pytorch_ok:
            print("   - Install CUDA-enabled PyTorch")
    
    print()
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())

