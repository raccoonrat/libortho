from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the project root directory (two levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(project_root, 'src')
include_dir = os.path.join(project_root, 'include')
current_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='libortho_ops',
    ext_modules=[
        CUDAExtension('libortho_ops', [
            os.path.join(current_dir, 'libortho_ops.cpp'),
            os.path.join(src_dir, 'dual_gemm.cu'),
            os.path.join(src_dir, 'dual_gemm_tensor_core.cu'), 
            os.path.join(src_dir, 'ortho.c')
        ],
        include_dirs=[include_dir],
        extra_compile_args={'cxx': ['-std=c++14'],
                            'nvcc': ['-gencode=arch=compute_80,code=sm_80', # Ampere
                                     '-gencode=arch=compute_75,code=sm_75', # Turing
                                     '-O3',
                                     '--expt-relaxed-constexpr']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)