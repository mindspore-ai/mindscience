# OpenBLAS Error Recovery

## Common Errors

### Linker Errors

**Error: undefined reference to `cblas_dgemm`**
```bash
# Solution: Link with OpenBLAS
gcc -o myprogram myprogram.c -lopenblas

# Or specify library path
gcc -o myprogram myprogram.c -L/usr/local/lib -lopenblas
```

**Error: cannot find -lopenblas**
```bash
# Solution: Install OpenBLAS or set library path
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Or use full path
gcc -o myprogram myprogram.c /usr/local/lib/libopenblas.a -lpthread -lm
```

### Runtime Errors

**Error: Segmentation fault**
- Check array bounds
- Verify leading dimensions
- Ensure correct matrix orientation (row vs column major)

```c
// WRONG: Leading dimension too small
cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n-1, x, 1, 0.0, y, 1);

// CORRECT: Leading dimension >= n
cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, A, n, x, 1, 0.0, y, 1);
```

**Error: Wrong results**
- Check row-major vs column-major
- Verify transpose flags
- Check alpha/beta values

```c
// Row-major: A[i*lda + j]
// Column-major: A[i + j*lda]

// Row-major example
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);

// Column-major example
cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);
```

### Threading Issues

**Error: Poor performance with OpenMP**
```bash
# Solution: Avoid nested parallelism
export OPENBLAS_NUM_THREADS=1  # In OpenMP parallel regions
export OMP_NUM_THREADS=8        # For OpenMP parallel regions
```

**Error: Thread conflicts**
```c
// Solution: Use thread-local storage or single-threaded BLAS
#pragma omp parallel
{
    // Each thread uses single-threaded BLAS
    openblas_set_num_threads(1);
    cblas_dgemm(...);
}
```

### Performance Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Slow for small matrices | Threading overhead | Set OPENBLAS_NUM_THREADS=1 |
| Slow for large matrices | Single-threaded | Rebuild with USE_OPENMP=1 |
| Inconsistent performance | CPU frequency scaling | Disable turbo boost |
| Memory errors | Array out of bounds | Check dimensions and lda |

## Debugging Tips

1. **Check library version**
```c
printf("OpenBLAS version: %s\n", openblas_get_config());
printf("Core type: %s\n", openblas_get_corename());
```

2. **Verify threading**
```c
printf("Threads: %d\n", openblas_get_num_threads());
```

3. **Test with single thread**
```bash
export OPENBLAS_NUM_THREADS=1
./myprogram
```
