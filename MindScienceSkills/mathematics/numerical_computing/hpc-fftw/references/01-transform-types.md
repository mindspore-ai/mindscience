# FFTW Transform Types

## Complex DFT

### 1D Complex DFT
```c
fftw_plan fftw_plan_dft_1d(int n, fftw_complex *in, fftw_complex *out,
                            int sign, unsigned flags);
```
- `sign`: FFTW_FORWARD (-1) or FFTW_BACKWARD (+1)
- Output is NOT normalized; divide by n for inverse

### Multi-dimensional Complex DFT
```c
fftw_plan fftw_plan_dft_2d(int n0, int n1, ...);
fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, ...);
fftw_plan fftw_plan_dft(int rank, const int *n, ...);
```

## Real DFT (R2C / C2R)

### Real-to-Complex
```c
fftw_plan fftw_plan_dft_r2c_1d(int n, double *in, fftw_complex *out, ...);
fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double *in, fftw_complex *out, ...);
```
- Output size: n/2 + 1 complex elements (Hermitian symmetry)
- 2x faster than complex DFT for real input

### Complex-to-Real
```c
fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex *in, double *out, ...);
```
- Input must satisfy Hermitian symmetry

## Real Even/Odd Transforms (DCT/DST)

### Discrete Cosine Transform (DCT)
```c
fftw_plan fftw_plan_r2r_1d(int n, double *in, double *out,
                            fftw_r2r_kind kind, unsigned flags);
```
- FFTW_REDFT00: DCT-I
- FFTW_REDFT10: DCT-II (common DCT)
- FFTW_REDFT01: DCT-III
- FFTW_REDFT11: DCT-IV

### Discrete Sine Transform (DST)
- FFTW_RODFT00: DST-I
- FFTW_RODFT10: DST-II
- FFTW_RODFT01: DST-III
- FFTW_RODFT11: DST-IV

## Discrete Hartley Transform
```c
fftw_plan fftw_plan_r2r_1d(int n, double *in, double *out,
                            FFTW_DHT, unsigned flags);
```
