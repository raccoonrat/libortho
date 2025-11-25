# libortho: Design Rationale
## Why We Made These Decisions (Linus's Perspective)

---

## 1. Why Physical Separation Instead of Logical Separation?

### The Question

Why not just use a single weight matrix with flags indicating which elements are "privacy-sensitive"?

### The Answer

**Because flags are data, and data requires branching.**

If you have a flag for every weight, you're checking `if (is_privacy_sensitive)` in your inner loop. That's a branch prediction nightmare. Every cache miss on the flag array stalls the pipeline.

**Physical separation means:**
- Base stream: No flags, no checks, pure dense computation
- Ortho stream: Separate kernel launch, or separate warp allocation

When `alpha = 0.0`, the Ortho kernel doesn't even launch. Zero overhead.

### The "Good Taste" Principle

> "Sometimes you can look at a problem from a different angle and rewrite it so that the special case disappears and becomes the normal case."

Privacy isn't a "special case" that needs flags. It's a **normal component** that can be added or removed. By making it physically separate, we eliminate the need for flags entirely.

---

## 2. Why Hessian-Based Sieve Instead of Magnitude Thresholding?

### The Question

Why not just keep the top-k largest residuals as Ortho?

### The Answer

**Because magnitude doesn't distinguish privacy from noise.**

A large residual might be:
- Privacy-sensitive pattern (what we want)
- Quantization noise (what we don't want)
- Genius reasoning pattern (what we want, but different from privacy)

**Hessian captures task-specific importance:**

$$\text{Impact} = \frac{||\text{Residual}||^2}{\text{diag}(H^{-1})}$$

A small residual with high curvature (large Hessian inverse) can be more important than a large residual with low curvature.

### The Geometric Intuition

Hessian measures the "bending" of the loss landscape. High curvature means the weight is critical for the specific task. This is exactly what we need to identify privacy-sensitive patterns.

### The Pragmatic Choice

We use **diagonal approximation** of Hessian, not full Hessian. Why? Because computing full Hessian is O(N²) expensive. Diagonal approximation is O(N) and captures 90% of the information.

**Pragmatism over perfection.**

---

## 3. Why Not Try to Distinguish "Genius" from "Privacy"?

### The Question

Can't we use higher-order derivatives or other geometric properties to separate Type A (genius) from Type B (privacy)?

### The Answer

**No. And we shouldn't try.**

**Reason 1: We can't reliably distinguish them.**

Both have:
- Large normal component $\Delta w_{\perp}$
- High curvature (large Hessian)
- Task-specific importance

The difference (smooth transition vs. Dirac delta) is subtle and requires expensive computation.

**Reason 2: User control is better than algorithmic guessing.**

Users know their use case:
- Research lab: Keep genius, remove privacy → `alpha = 0.0` for privacy, but keep Ortho for genius
- Production deployment: Remove both → `alpha = 0.0`
- Privacy-preserving research: Keep both, apply DP to Ortho only

**Reason 3: Simplicity.**

One parameter (`alpha`) instead of complex logic. If you need more than 3 levels of indentation, you're screwed.

### The "Never Break Userspace" Principle

> "We don't break userspace!"

If we try to automatically distinguish genius from privacy and get it wrong, we break the user's workflow. Better to give them control.

---

## 4. Why Coordinate Stream Instead of CSR?

### The Question

CSR (Compressed Sparse Row) is the standard format for sparse matrices. Why use a custom "coordinate stream"?

### The Answer

**Because CSR row pointers are static, but our access pattern is dynamic.**

CSR is optimized for row-wise access:
```
for each row:
    for each non-zero in row:
        process
```

But our kernel might want:
- Column-wise access (for some optimizations)
- Block-wise access (for Tensor Core tiles)
- Warp-level partitioning (different warps handle different patterns)

**Coordinate stream (pre-sorted flat indices) gives us flexibility:**

```c
uint16_t *indices;  // Flat index, sorted by row then column
float *values;       // Corresponding values
```

We can:
- Iterate row-wise (early exit when row changes)
- Iterate column-wise (re-sort if needed)
- Partition by blocks (compute block boundaries on-the-fly)

**The "Good Taste" Principle:**

We don't optimize for one access pattern. We optimize for **flexibility**, then let the kernel choose the best pattern.

---

## 5. Why Kernel-Level Branching Instead of Element-Level?

### The Question

Why check `if (alpha > 0.0f)` at the kernel level? Why not check per-element in the inner loop?

### The Answer

**Because kernel-level branching is uniform for the launch.**

When you launch a CUDA kernel, all threads see the same `alpha` value. The branch predictor can handle this perfectly:
- First iteration: Predict "taken" (or "not taken")
- All subsequent iterations: Same prediction
- Zero misprediction penalty

**Element-level branching would be:**
```cuda
for each element:
    if (alpha > 0.0f && is_ortho_element):  // Different per element!
        acc += ...
```

This creates unpredictable branches. Branch predictor fails. Pipeline stalls.

**The "Simplicity" Principle:**

> "If you need more than 3 levels of indentation, you're screwed."

Kernel-level branching = 1 level of indentation.
Element-level branching = 3+ levels of indentation.

We chose the simpler path.

---

## 6. Why 128-Byte Alignment?

### The Question

Why not 64-byte (cache line) or 256-byte alignment?

### The Answer

**Because Tensor Cores require 128-byte alignment.**

Modern GPU Tensor Cores (NVIDIA Ampere, Ada) load data in 128-byte chunks. If your data isn't aligned, you get:
- Extra memory transactions (load 2 chunks instead of 1)
- Stalled pipeline (wait for second chunk)
- Wasted bandwidth

**128-byte alignment ensures:**
- Single memory transaction per Tensor Core load
- Optimal bandwidth utilization
- Zero wasted cycles

### The "Pragmatism" Principle

We don't optimize for theoretical cache line sizes. We optimize for **actual hardware behavior**.

---

## 7. Why Pre-Sorted Indices?

### The Question

Why sort indices offline? Can't we sort on-the-fly in the kernel?

### The Answer

**Because sorting is expensive, and we only need to do it once.**

Sorting N elements is O(N log N). In the kernel, that's:
- Extra computation cycles
- Extra shared memory usage
- Extra synchronization

**Pre-sorting offline (in Python, during weight separation) is:**
- Done once, not every inference
- Can use efficient CPU algorithms (quicksort, radix sort)
- Zero runtime overhead

### The "Good Taste" Principle

> "Eliminate the special case."

Sorting isn't a "special case" that needs to happen in the kernel. It's a **normal preprocessing step** that happens once.

---

## 8. Why Not Use C++ Templates?

### The Question

C++ templates would allow compile-time optimization for different bit widths, data types, etc. Why use plain C?

### The Answer

**Because templates are complexity, and complexity is the root of all evil.**

Templates create:
- Longer compile times
- Harder debugging (template instantiation errors)
- Code bloat (multiple instantiations)
- Harder to understand (template metaprogramming)

**Plain C gives us:**
- Fast compilation
- Clear code (no template magic)
- Easy debugging
- Simple mental model

### The "Simplicity" Principle

> "Functions must be short, do one thing, and do it well."

Templates try to do "everything" at compile time. We prefer to do "one thing" (dual-stream GEMM) and do it well.

**Exception**: If we need multiple bit widths (INT2, INT3, INT4), we can add runtime dispatch. But we don't need templates for that.

---

## 9. Why Python for Sieve, C/CUDA for Runtime?

### The Question

Why not implement the Hessian sieve in C++ for consistency?

### The Answer

**Because the sieve runs once, the runtime runs millions of times.**

**Sieve (Python):**
- Runs offline, during weight separation
- Needs flexibility (experiment with thresholds, formats)
- Needs libraries (PyTorch for Hessian computation)
- Performance doesn't matter (runs once)

**Runtime (C/CUDA):**
- Runs every inference
- Needs maximum performance
- Needs minimal dependencies
- Performance is critical

### The "Pragmatism" Principle

> "I'm a damn pragmatist."

Use the right tool for the job:
- Python for flexibility and experimentation
- C/CUDA for performance and production

---

## 10. Why the "Null Test" Requirement?

### The Question

Why require that disabled Ortho has zero overhead? Isn't 1-2% acceptable?

### The Answer

**No. Zero means zero.**

**Reason 1: If we allow 1%, we'll allow 2%, then 5%, then 10%.**

Slippery slope. Better to enforce zero from the start.

**Reason 2: Zero overhead is achievable.**

When `alpha = 0.0` or `ortho.count == 0`:
- Ortho kernel doesn't launch
- No memory allocated for Ortho
- Base kernel is identical to pure INT4 kernel

If we have overhead, it means we're doing something wrong (checking flags, allocating memory, etc.).

**Reason 3: It forces good design.**

The null test requirement forces us to:
- Physically separate Base and Ortho (not logical flags)
- Use kernel-level branching (not element-level)
- Avoid unnecessary allocations

**The "Good Taste" Principle:**

If the special case (Ortho disabled) has overhead, we haven't eliminated it properly. Make it the normal case.

---

## Summary: The Linus Philosophy

Every design decision follows these principles:

1. **Good Taste**: Eliminate special cases. Make them normal cases.
2. **Never Break Userspace**: Don't break existing code. Give users control.
3. **Pragmatism**: Solve real problems. Use the right tool for the job.
4. **Simplicity**: If you need >3 levels of indentation, fix your program.

**The result**: A system that is simple, fast, and actually works.

---

**End of Design Rationale**

