# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd libortho

# Install in development mode
pip install -e .
```

## Verify Core Logic

First, verify that the core hypothesis works:

```bash
python experiments/verify_core_logic.py
```

Expected output:
```
--- [LibOrtho] Initializing Minimal Verification ---
Training Loss: 0.xxxxxx
Original Model -> Privacy Error: 0.xxxx (Should be low)
Original Model -> General Error: 0.xxxx (Should be low)
Sieve Complete. Ortho Sparsity: ~95%

--- Testing The Kill Switch ---
[Alpha=1.0] Privacy Error: 0.xxxx (Target: Low)
[Alpha=1.0] General Error: 0.xxxx (Target: Low)
[Alpha=0.0] Privacy Error: X.xxxx (Target: HIGH -> Forgot Privacy!)
[Alpha=0.0] General Error: 0.xxxx (Target: LOW -> Kept Logic!)
✅ SUCCESS: Privacy successfully forgotten (Exploded Error).
✅ SUCCESS: General logic preserved (Robust Base).
```

## Basic Usage

### 1. Separate Weights Using Hessian Sieve

```python
from tools.sieve import hessian_sieve, compute_hessian_diag_approx
import torch

# Your trained weight matrix
weight = torch.randn(4096, 4096)

# Generate input data for Hessian computation
inputs = torch.randn(1000, 4096)

# Compute Hessian diagonal
H_diag = compute_hessian_diag_approx(inputs)

# Separate weights
w_base, w_ortho = hessian_sieve(
    weight,
    H_diag,
    curvature_thresh=10.0  # or use sparsity_target=0.95
)
```

### 2. Use OrthoLinear Layer

```python
from torch_bind.ortho_linear import OrthoLinear

# Create layer
layer = OrthoLinear(in_features=4096, out_features=4096, q_bits=4)

# Load separated weights
layer.load_from_weights(w_base, w_ortho)

# Full intelligence mode
layer.set_alpha(1.0)
output = layer(input)

# Privacy-safe mode (only base)
layer.set_alpha(0.0)
output_safe = layer(input)
```

### 3. Run Examples

```bash
python examples/basic_usage.py
```

## Key Concepts

- **Base (W_base)**: Low-precision quantized weights (INT4), represents common knowledge
- **Ortho (W_ortho)**: High-precision sparse weights (FP16), represents privacy/specificity
- **Alpha (α)**: Privacy kill switch (1.0 = full, 0.0 = privacy-safe)

## Next Steps

1. Read `docs/1124-新的思路-3.md` for theoretical background
2. Check `PROJECT_STRUCTURE.md` for architecture details
3. See `CONTRIBUTING.md` for development guidelines

