# FFTW Parallelization

## OpenMP (Multi-threaded)

### Initialization
```c
#include <fftw3.h>
#include <omp.h>

int main() {
    // Initialize threads
    fftw_init_threads();
    
    // Set number of threads
    fftw_plan_with_nthreads(omp_get_max_threads());
    
    // Create plan (will use multiple threads)
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
    
    // Execute (parallelized)
    fftw_execute(plan);
    
    // Cleanup
    fftw_cleanup_threads();
    return 0;
}
```

### Compilation
```bash
gcc -fopenmp -lfftw3 -lfftw3_omp myprogram.c
```

## MPI (Distributed Memory)

### Initialization
```c
#include <fftw3-mpi.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    fftw_mpi_init();
    
    // Get local array size
    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD,
                                         &local_n0, &local_0_start);
    
    // Allocate local portion
    fftw_complex *local_data = fftw_alloc_complex(alloc_local);
    
    // Create MPI plan
    fftw_plan plan = fftw_mpi_plan_dft_2d(N0, N1, local_data, local_data,
                                           MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    
    // Execute
    fftw_execute(plan);
    
    fftw_destroy_plan(plan);
    fftw_free(local_data);
    fftw_mpi_cleanup();
    MPI_Finalize();
    return 0;
}
```

### Compilation
```bash
mpicc -lfftw3 -lfftw3_mpi myprogram.c
```

## Hybrid OpenMP + MPI

```c
// Initialize both
fftw_init_threads();
MPI_Init(&argc, &argv);
fftw_mpi_init();

// Set threads per MPI rank
fftw_plan_with_nthreads(omp_get_max_threads());

// Create plan
fftw_plan plan = fftw_mpi_plan_dft_3d(...);
```

## Performance Tips

1. **Thread count**: Match physical cores, not hyperthreads
2. **Problem size**: Powers of 2 are fastest
3. **Memory locality**: Use first-touch initialization
4. **Wisdom**: Save/load plans for repeated sizes
