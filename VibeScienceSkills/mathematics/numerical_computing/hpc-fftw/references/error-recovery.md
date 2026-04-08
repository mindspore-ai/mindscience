# FFTW Error Recovery

## Common Errors

### Segmentation Fault

**Cause**: Unaligned memory access
```c
// WRONG
double *in = malloc(N * sizeof(double));

// CORRECT
double *in = fftw_alloc_real(N);
```

### Incorrect Results

**Cause 1**: Missing normalization
```c
// FFTW does NOT normalize inverse FFT
// For inverse, divide by N
for (int i = 0; i < N; i++) {
    out[i][0] /= N;
    out[i][1] /= N;
}
```

**Cause 2**: Wrong sign
```c
// Forward: FFTW_FORWARD (-1)
// Inverse: FFTW_BACKWARD (+1)
```

### MPI Deadlock

**Cause**: Incorrect array distribution
```c
// Always use fftw_mpi_local_size_* functions
ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                                &local_n0, &local_0_start);
```

### Thread Safety Issues

**Cause**: Shared plans across threads
```c
// WRONG: Shared plan
fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
#pragma omp parallel
{
    fftw_execute(plan);  // Race condition!
}

// CORRECT: One plan per thread
#pragma omp parallel
{
    fftw_plan plan = fftw_plan_dft_1d(N, local_in, local_out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}
```

## Debugging Tips

1. **Enable debug mode**: Compile with `-DFFTW_DEBUG`
2. **Check alignment**: Use `fftw_alignment_of(ptr)`
3. **Validate plans**: Use `fftw_print_plan(plan)`
4. **Memory check**: Run with Valgrind

## Performance Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Slow execution | ESTIMATE mode | Use MEASURE/PATIENT |
| Slow first run | Plan measurement | Save wisdom file |
| Poor scaling | Memory bandwidth | Reduce threads, use MPI |
| Cache misses | Non-contiguous access | Reorder loops |
