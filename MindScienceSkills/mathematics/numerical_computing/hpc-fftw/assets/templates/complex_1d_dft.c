#include <fftw3.h>
#include <stdio.h>
#include <math.h>

int main() {
    int N = 1024;
    
    // Allocate aligned memory
    fftw_complex *in = fftw_alloc_complex(N);
    fftw_complex *out = fftw_alloc_complex(N);
    
    // Create plan
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);
    
    // Initialize input (example: sine wave)
    for (int i = 0; i < N; i++) {
        in[i][0] = sin(2.0 * M_PI * 10.0 * i / N);  // Real part
        in[i][1] = 0.0;  // Imaginary part
    }
    
    // Execute FFT
    fftw_execute(plan);
    
    // Print first few results
    printf("FFT Results (first 10 bins):\n");
    for (int i = 0; i < 10; i++) {
        printf("Bin %d: %.6f + %.6fi\n", i, out[i][0], out[i][1]);
    }
    
    // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    
    return 0;
}
