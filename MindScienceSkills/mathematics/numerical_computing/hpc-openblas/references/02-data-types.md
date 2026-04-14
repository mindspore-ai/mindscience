# BLAS Data Types

## Precision Prefixes

| Prefix | Type | Description |
|--------|------|-------------|
| s | float | Single precision (32-bit) |
| d | double | Double precision (64-bit) |
| c | float complex | Complex single precision |
| z | double complex | Complex double precision |

## Function Naming Convention

```
[Precision][Matrix Type][Operation]

Example: dgemm
- d: double precision
- ge: general matrix
- mm: matrix-matrix multiply
```

## Matrix Types

| Code | Type | Description |
|------|------|-------------|
| ge | General | No special structure |
| sy | Symmetric | A = A^T |
| he | Hermitian | A = A^H (complex) |
| tr | Triangular | Upper or lower triangular |
| po | Positive definite | Symmetric positive definite |
| ba | Banded | Banded matrix |
| sp | Packed | Symmetric packed storage |

## Complex Number Handling

### CBLAS Complex Types
```c
#include <cblas.h>

// Single precision complex
void cblas_cgemv(... const void *A, ... const void *x, ... void *y);

// Double precision complex
void cblas_zgemv(... const void *A, ... const void *x, ... void *y);
```

### Complex Number Layout
```c
// Complex numbers stored as [real, imaginary]
float complex_var[2] = {real_part, imag_part};

// Example: 1 + 2i
float z[2] = {1.0f, 2.0f};
```

## Performance Considerations

### Single vs Double Precision
- Single precision: 2x faster, 2x less memory
- Double precision: Required for accuracy-sensitive applications
- Mixed precision: Use single for speed, refine with double

### Complex vs Real
- Complex operations: ~4x more flops than real
- Consider real formulation if possible
- Example: 2n x 2n real system for n x n complex
