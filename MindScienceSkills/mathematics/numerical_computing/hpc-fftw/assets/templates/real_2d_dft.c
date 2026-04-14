#include <fftw3.h>
#include <stdio.h>

int main() {
    int N0 = 512, N1 = 512;
    
    // Real input, complex output for 2D R2C DFT
    double *in = fftw_alloc_real(N0 * N1);
    fftw_complex *out = fftw_alloc_complex(N0 * (N1/2 + 1));
    
    // Create plan for real-to-complex 2D DFT
    fftw_plan plan = fftw_plan_dft_r2c_2d(N0, N1, in, out, FFTW_MEASURE);
    
    // Initialize input
    for (int i = 0; i < N0 * N1; i++) {
        in[i] = (double)i / (N0 * N1);
    }
    
    // Execute
    fftw_execute(plan);
    
    printf("2D R2C DFT completed.\n");
    printf("Output size: %d x %d complex elements\n", N0, N1/2 + 1);
    
    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    
    return 0;
}
