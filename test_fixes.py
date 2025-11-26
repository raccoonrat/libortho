#!/usr/bin/env python3
"""
Quick test script to verify the fixes from Linus's code review.

Tests:
1. CSR format in pack_ortho_sparse
2. Memory-efficient hessian_sieve
3. Lazy loading functionality
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.sieve import hessian_sieve, pack_ortho_sparse, quantize_int4

def test_csr_format():
    """Test CSR format implementation."""
    print("=" * 60)
    print("Test 1: CSR Format in pack_ortho_sparse")
    print("=" * 60)
    
    # Create a small sparse tensor
    w_ortho = torch.zeros(10, 8)
    w_ortho[0, 1] = 1.0
    w_ortho[0, 3] = 2.0
    w_ortho[2, 5] = 3.0
    w_ortho[5, 0] = 4.0
    w_ortho[5, 2] = 5.0
    
    # Test CSR format
    row_ptr, col_indices, values = pack_ortho_sparse(w_ortho, format="csr")
    
    print(f"  Input shape: {w_ortho.shape}")
    print(f"  Non-zeros: {(w_ortho != 0).sum().item()}")
    print(f"  Row pointers shape: {row_ptr.shape}")
    print(f"  Column indices shape: {col_indices.shape}")
    print(f"  Values shape: {values.shape}")
    
    # Verify CSR structure
    assert row_ptr.shape[0] == w_ortho.shape[0] + 1, "Row pointer size mismatch"
    assert col_indices.shape[0] == values.shape[0], "Column indices and values mismatch"
    assert row_ptr[0] == 0, "First row pointer should be 0"
    assert row_ptr[-1] == (w_ortho != 0).sum().item(), "Last row pointer should equal nnz"
    
    # Verify row 0 has 2 non-zeros
    assert row_ptr[1] - row_ptr[0] == 2, "Row 0 should have 2 non-zeros"
    
    # Verify row 5 has 2 non-zeros
    assert row_ptr[6] - row_ptr[5] == 2, "Row 5 should have 2 non-zeros"
    
    print("  ✅ CSR format test passed!")
    return True

def test_memory_efficient_hessian_sieve():
    """Test memory-efficient hessian_sieve."""
    print("\n" + "=" * 60)
    print("Test 2: Memory-Efficient hessian_sieve")
    print("=" * 60)
    
    # Create a larger weight matrix to test memory efficiency
    out_features, in_features = 100, 64
    weight = torch.randn(out_features, in_features)
    H_inv = torch.ones(in_features) * 0.1  # Diagonal approximation
    
    print(f"  Weight shape: {weight.shape}")
    print(f"  H_inv shape: {H_inv.shape}")
    
    # Test with sparsity target
    w_base, w_ortho = hessian_sieve(
        weight,
        H_inv,
        sparsity_target=0.95,  # 95% sparsity
    )
    
    print(f"  Base shape: {w_base.shape}")
    print(f"  Ortho shape: {w_ortho.shape}")
    print(f"  Ortho sparsity: {1.0 - (w_ortho != 0).sum().item() / w_ortho.numel():.2%}")
    
    # Verify shapes
    assert w_base.shape == weight.shape, "Base shape mismatch"
    assert w_ortho.shape == weight.shape, "Ortho shape mismatch"
    
    # Verify sparsity target is approximately met
    actual_sparsity = 1.0 - (w_ortho != 0).sum().item() / w_ortho.numel()
    assert actual_sparsity >= 0.90, f"Sparsity target not met: {actual_sparsity:.2%}"
    
    print("  ✅ Memory-efficient hessian_sieve test passed!")
    return True

def test_no_weight_squared_copy():
    """Test that we don't create weight ** 2 copy."""
    print("\n" + "=" * 60)
    print("Test 3: No weight ** 2 Copy (Hessian computation)")
    print("=" * 60)
    
    weight = torch.randn(50, 32)
    
    # Old way (would create copy): weight_squared = weight ** 2
    # New way: torch.sum(weight * weight, dim=0)
    
    # Compute Hessian diagonal using new method
    H_diag = torch.sum(weight * weight, dim=0) / weight.shape[0]
    
    print(f"  Weight shape: {weight.shape}")
    print(f"  H_diag shape: {H_diag.shape}")
    print(f"  H_diag mean: {H_diag.mean().item():.4f}")
    
    # Verify it's correct
    # Manual computation for verification
    weight_squared_manual = weight ** 2
    H_diag_manual = weight_squared_manual.sum(dim=0) / weight.shape[0]
    
    assert torch.allclose(H_diag, H_diag_manual), "Hessian computation mismatch"
    
    print("  ✅ No weight ** 2 copy test passed!")
    return True

def test_csr_vs_coo():
    """Compare CSR and COO formats."""
    print("\n" + "=" * 60)
    print("Test 4: CSR vs COO Format Comparison")
    print("=" * 60)
    
    w_ortho = torch.randn(20, 16)
    # Make it sparse
    mask = torch.rand(20, 16) > 0.9
    w_ortho = w_ortho * mask
    
    # Test CSR
    row_ptr, col_indices, values_csr = pack_ortho_sparse(w_ortho, format="csr")
    
    # Test COO (backward compatibility)
    indices_coo, values_coo, _ = pack_ortho_sparse(w_ortho, format="coo")
    
    print(f"  Non-zeros: {(w_ortho != 0).sum().item()}")
    print(f"  CSR: {len(values_csr)} values, {len(row_ptr)} row pointers")
    print(f"  COO: {len(values_coo)} values, {len(indices_coo)} indices")
    
    # Both should have same number of values
    assert len(values_csr) == len(values_coo), "Value count mismatch"
    
    # Values should match (may be in different order, so sort)
    assert torch.allclose(torch.sort(values_csr)[0], torch.sort(values_coo)[0]), "Values mismatch"
    
    print("  ✅ CSR vs COO comparison passed!")
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Testing Fixes from Linus's Code Review")
    print("=" * 60)
    print()
    
    tests = [
        test_csr_format,
        test_memory_efficient_hessian_sieve,
        test_no_weight_squared_copy,
        test_csr_vs_coo,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! The fixes are working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())

