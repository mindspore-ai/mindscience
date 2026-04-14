# OpenBLAS Threading

## Threading Models

### OpenMP (Recommended)
```bash
# Build with OpenMP
make USE_OPENMP=1

# Set thread count
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
```

### pthreads
```bash
# Build with pthreads
make USE_THREAD=1

# Set thread count
export OPENBLAS_NUM_THREADS=8
```

### Single-threaded
```bash
# Build single-threaded
make USE_THREAD=0

# Or force single-threaded at runtime
export OPENBLAS_NUM_THREADS=1
```

## Thread Control

### Environment Variables
```bash
# Number of threads
export OPENBLAS_NUM_THREADS=8

# Thread affinity (bind to cores)
export OPENBLAS_MAIN_FREE=1  # Don't bind main thread

# Disable threading
export OPENBLAS_NUM_THREADS=1
```

### Runtime Control
```c
#include <openblas/cblas.h>

// Set number of threads
openblas_set_num_threads(8);

// Get current thread count
int num_threads = openblas_get_num_threads();

// Get number of parallel threads
int parallel_threads = openblas_get_num_procs();
```

## Thread Safety

### Thread-Safe Operations
- All BLAS operations are thread-safe
- Each thread can call BLAS independently

### Avoiding Conflicts
```c
// WRONG: Nested parallelism
#pragma omp parallel
{
    openblas_set_num_threads(4);  // Race condition!
    cblas_dgemm(...);
}

// CORRECT: Set threads before parallel region
openblas_set_num_threads(1);  // Single-threaded BLAS
#pragma omp parallel
{
    cblas_dgemm(...);  // Each thread uses single-threaded BLAS
}

// CORRECT: Use OpenMP parallel BLAS
export OPENBLAS_NUM_THREADS=4
cblas_dgemm(...);  // BLAS uses 4 threads
```

## Performance Tuning

### Optimal Thread Count
- Match physical cores (not hyperthreads)
- Consider memory bandwidth limits
- Test different thread counts

### Thread Affinity
```bash
# Intel MPI
export I_MPI_PIN_DOMAIN=omp

# OpenMP
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

### Memory Bandwidth
- Level 3 BLAS: Compute-bound, scales well
- Level 1/2 BLAS: Memory-bound, limited scaling
- Consider single-threaded for small matrices
