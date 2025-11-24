# libortho Project Structure

```
libortho/
├── README.md                 # Project overview and quick start
├── LICENSE                   # MIT License
├── CONTRIBUTING.md           # Contribution guidelines
├── setup.py                 # Python package setup
├── pyproject.toml           # Modern Python project config
│
├── include/                  # C/CUDA headers
│   └── ortho.h              # Core data structures and API
│
├── src/                     # C/CUDA implementation
│   ├── ortho.c             # CPU fallback implementation
│   └── dual_gemm.cu        # CUDA kernel for dual-stream GEMM
│
├── torch_bind/             # PyTorch bindings
│   ├── __init__.py
│   ├── ortho_linear.py     # PyTorch module (OrthoLinear)
│   └── bindings.cpp        # PyBind11 C++ bindings
│
├── tools/                   # Python utilities
│   ├── __init__.py
│   └── sieve.py            # Hessian sieve for weight separation
│
├── experiments/             # Experimental scripts
│   ├── __init__.py
│   └── verify_core_logic.py # Core hypothesis verification
│
├── examples/                # Usage examples
│   └── basic_usage.py      # Basic usage demonstration
│
└── docs/                    # Documentation
    └── 1124-新的思路-3.md  # Original design document
```

## Key Components

### Core Library (`include/`, `src/`)
- **ortho.h**: Defines `orth_layer_t` structure and C API
- **ortho.c**: CPU implementation (fallback)
- **dual_gemm.cu**: CUDA kernel implementing dual-stream matrix multiplication

### Python Interface (`torch_bind/`)
- **OrthoLinear**: PyTorch module that replaces `nn.Linear`
- Supports privacy kill switch via `alpha` parameter
- Automatically handles base/ortho separation

### Tools (`tools/`)
- **hessian_sieve**: Separates weights into base and orthogonal components
- Uses Hessian-based geometric discriminator
- Supports both threshold and sparsity-target modes

### Experiments (`experiments/`)
- **verify_core_logic.py**: Minimal verification of dual-manifold hypothesis
- Tests privacy kill switch functionality
- Validates that privacy can be removed without destroying general capability

## Build Process

1. **Development setup**: `pip install -e .`
2. **CUDA support**: Automatically detected if `nvcc` is available
3. **Python-only mode**: Works without CUDA (uses CPU fallback)

## Design Philosophy

1. **Simplicity**: No more than 3 levels of indentation
2. **Physical separation**: Base and Ortho are stored separately
3. **No dynamic branching**: Kernel uses static indices
4. **Memory alignment**: 128-byte aligned for Tensor Core access
5. **Good taste**: Clean, minimal API

