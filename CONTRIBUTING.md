# Contributing to libortho

## Design Principles

1. **No Dynamic Branching**: In kernel internals, don't branch based on input data content. Ortho indices must be statically compiled (CSR row pointers).

2. **Memory Alignment**: Ortho weights, though sparse, must be 128-byte aligned, otherwise Memory Controller will be slower than reading Dense matrices.

3. **The "Null" Test**: If `W_ortho` is all zeros, system performance must be **completely equivalent** to a standard INT4 model. If supporting sparse stream causes Base stream to slow down by 1%, it's a failure.

4. **Simplicity**: If you need more than 3 levels of indentation, you're screwed. Fix your algorithm.

5. **Good Taste**: We don't mix Base and Ortho storage. They are physically isolated.

## Code Style

- C/CUDA: Follow Linux kernel style guide (Linus's preference)
- Python: Follow PEP 8, use black for formatting
- Keep functions small and focused
- No unnecessary abstractions

## Testing

Before submitting PR:

1. Run `python experiments/verify_core_logic.py` - should show two ✅ SUCCESS
2. If CUDA available, test with actual GPU
3. Verify no performance regression in Base stream when Ortho is disabled

## Pull Request Checklist

- [ ] Code follows style guidelines
- [ ] No more than 3 levels of indentation
- [ ] Tests pass
- [ ] No performance regression
- [ ] Documentation updated if needed

