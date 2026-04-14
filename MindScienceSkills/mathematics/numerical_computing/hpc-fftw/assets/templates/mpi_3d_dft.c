#include <fftw3-mpi.h>
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    fftw_mpi_init();
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    ptrdiff_t N0 = 128, N1 = 128, N2 = 128;
    ptrdiff_t alloc_local, local_n0, local_0_start;
    
    // Get local data size
    alloc_local = fftw_mpi_local_size_3d(N0, N1, N2, MPI_COMM_WORLD,
                                          &local_n0, &local_0_start);
    
    // Allocate
    fftw_complex *data = fftw_alloc_complex(alloc_local);
    
    // Create plan
    fftw_plan plan = fftw_mpi_plan_dft_3d(N0, N1, N2, data, data,
                                           MPI_COMM_WORLD, FFTW_FORWARD, FFTW_MEASURE);
    
    // Initialize local data
    for (ptrdiff_t i = 0; i < local_n0 * N1 * N2; i++) {
        data[i][0] = 0.0;
        data[i][1] = 0.0;
    }
    
    // Execute
    fftw_execute(plan);
    
    if (rank == 0) {
        printf("MPI 3D DFT completed.\n");
    }
    
    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(data);
    MPI_Finalize();
    
    return 0;
}
