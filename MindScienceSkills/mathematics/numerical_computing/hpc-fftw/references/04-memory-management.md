# FFTW Memory Management

## SIMD-Aligned Allocation

### Why Alignment Matters
- SIMD instructions (SSE, AVX) require 16/32-byte alignment
- Misaligned memory causes segfaults or severe slowdown
- Standard malloc() does NOT guarantee alignment

### Allocation Functions
```c
// Complex arrays
fftw_complex *c_arr = fftw_alloc_complex(n);

// Real arrays
double *r_arr = fftw_alloc_real(n);

// Generic (returns void*)
void *arr = fftw_malloc(size_in_bytes);
```

### Deallocation
```c
// Always use fftw_free, NOT free()
fftw_free(c_arr);
fftw_free(r_arr);
```

## Memory Layout

### Complex Numbers
```c
typedef double fftw_complex[2];
// [0] = real part
// [1] = imaginary part
```

### Multi-dimensional Arrays
```c
// Row-major order (C convention)
double *arr = fftw_alloc_real(N0 * N1 * N2);
arr[i*N1*N2 + j*N2 + k] = value;
```

### In-place Transforms
```c
// For in-place R2C, allocate extra space
double *in = fftw_alloc_real(2 * (N/2 + 1));  // For 1D R2C
fftw_complex *out = (fftw_complex*)in;  // Same memory
```

## Memory Usage Estimation

```c
// Complex DFT: N complex numbers
size_t complex_dft_mem = N * sizeof(fftw_complex);

// Real DFT: N real input + N/2+1 complex output
size_t real_dft_mem = N * sizeof(double) + (N/2+1) * sizeof(fftw_complex);

// 3D DFT: N0 * N1 * N2 complex numbers
size_t dft_3d_mem = N0 * N1 * N2 * sizeof(fftw_complex);
```

## Large Arrays

For very large arrays (>2GB), ensure:
1. 64-bit FFTW build (--enable-64bit in configure)
2. Sufficient virtual memory
3. Use `ptrdiff_t` for array indices
