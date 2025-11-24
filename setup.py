"""
libortho - Setup Configuration
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
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

HAS_CUDA = check_cuda()

ext_modules = []

if HAS_CUDA:
    # CUDA extension with Tensor Core support
    ext_modules.append(
        Pybind11Extension(
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
            cxx_std=17,
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3', 
                    '--use_fast_math',
                    '-arch=sm_75',  # Turing (Tensor Core INT8)
                    '-arch=sm_80',  # Ampere
                    '-arch=sm_86',  # Ampere (consumer)
                    '-arch=sm_89',  # Ada Lovelace
                    '--expt-relaxed-constexpr'  # For WMMA API
                ]
            },
            libraries=['cudart', 'cublas'],
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
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
        "pybind11>=2.10.0",
    ],
    zip_safe=False,
)

