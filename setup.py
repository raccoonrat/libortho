"""
libortho - Setup Configuration
"""

from setuptools import setup, Extension
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
    # CUDA extension with Tensor Core support
    # FIXED: Use Extension instead of Pybind11Extension for better CUDA control
    # Pybind11Extension has issues with dictionary extra_compile_args in some versions
    from pybind11.setup_helpers import build_ext
    
    cuda_archs = get_cuda_archs()
    nvcc_flags = [
        '-O3', 
        '--use_fast_math',
    ] + cuda_archs + [
        '--expt-relaxed-constexpr'  # For WMMA API
    ]
    
    ext_modules.append(
        Extension(
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
            language='c++',
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': nvcc_flags
            },
            libraries=['cudart', 'cublas'],
            define_macros=[('VERSION_INFO', '"dev"')],
        )
    )

setup(
    name="libortho",
    version="0.1.0",
    description="Dual-Manifold LLM Runtime Library",
    author="libortho contributors",
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
    cmdclass={"build_ext": build_ext} if HAS_CUDA and ext_modules else {},
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "pybind11>=2.10.0",
    ],
    zip_safe=False,
)

