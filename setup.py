"""
libortho - Setup Configuration
"""

from setuptools import setup
from pybind11 import get_cmake_dir
import pybind11

# Check if CUDA is available
import subprocess
import sys

def check_cuda():
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def get_cuda_archs():
    """Get CUDA architectures based on CUDA version."""
    base_archs = [
        '-arch=sm_75',  # Turing (Tensor Core INT8)
        '-arch=sm_80',  # Ampere
        '-arch=sm_86',  # Ampere (consumer)
        '-arch=sm_89',  # Ada Lovelace
    ]
    
    # Check if CUDA version supports sm_100 (Blackwell)
    # sm_100 requires CUDA 12.8+
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            import re
            version_match = re.search(r'release (\d+)\.(\d+)', result.stdout)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2))
                if major > 12 or (major == 12 and minor >= 8):
                    base_archs.append('-arch=sm_100')  # Blackwell (RTX 5060)
                    print(f"CUDA {major}.{minor} detected: Including sm_100 (Blackwell) support")
                else:
                    print(f"CUDA {major}.{minor} detected: sm_100 not supported (requires CUDA 12.8+)")
    except:
        pass  # If version detection fails, just use base architectures
    
    return base_archs

HAS_CUDA = check_cuda()

ext_modules = []

if HAS_CUDA:
    # FIXED: Use PyTorch's CUDAExtension which properly handles .cu files
    # This is the recommended way to build CUDA extensions with PyTorch
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
        
        cuda_archs = get_cuda_archs()
        nvcc_flags = [
            '-O3', 
            '--use_fast_math',
        ] + cuda_archs + [
            '--expt-relaxed-constexpr'  # For WMMA API
        ]
        
        ext_modules.append(
            CUDAExtension(
                "libortho._C_ops",
                [
                    "src/dual_gemm.cu",
                    "src/dual_gemm_tensor_core.cu",  # Tensor Core implementation
                    "torch_bind/bindings.cpp",
                ],
                include_dirs=[
                    "include",
                    pybind11.get_include(),
                ],
                extra_compile_args={
                    'cxx': ['-O3', '-std=c++17'],
                    'nvcc': nvcc_flags
                },
                libraries=['cublas'],
            )
        )
        build_ext_class = BuildExtension
    except ImportError:
        # Fallback to pybind11 if PyTorch is not available
        print("Warning: PyTorch not found, falling back to pybind11")
        from setuptools import Extension
        from pybind11.setup_helpers import build_ext as pybind11_build_ext
        
        cuda_archs = get_cuda_archs()
        nvcc_flags = [
            '-O3', 
            '--use_fast_math',
        ] + cuda_archs + [
            '--expt-relaxed-constexpr'
        ]
        
        # Note: Standard Extension doesn't handle .cu files well
        # This is a fallback that may not work
        ext_modules.append(
            Extension(
                "libortho._C_ops",
                [
                    "torch_bind/bindings.cpp",  # Only C++ files
                ],
                include_dirs=[
                    "include",
                    pybind11.get_include(),
                ],
                language='c++',
                extra_compile_args={
                    'cxx': ['-O3', '-std=c++17'],
                },
                define_macros=[('VERSION_INFO', '"dev"')],
            )
        )
        build_ext_class = pybind11_build_ext
else:
    build_ext_class = None

setup(
    name="libortho",
    version="0.1.0",
    description="Dual-Manifold LLM Runtime Library",
    # FIXED: Remove author to avoid pyproject.toml warning
    # author="libortho contributors",  # Moved to avoid conflict with pyproject.toml
    python_requires=">=3.8",
    packages=["libortho", "libortho.torch_bind", "libortho.tools", "libortho.experiments"],
    package_dir={
        "libortho": ".",
        "libortho.torch_bind": "torch_bind",
        "libortho.tools": "tools",
        "libortho.experiments": "experiments",
    },
    package_data={
        "libortho": ["include/*.h"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext_class} if build_ext_class else {},
    # FIXED: install_requires is defined in pyproject.toml, avoid duplication
    # install_requires is handled by pyproject.toml dependencies
    zip_safe=False,
)

