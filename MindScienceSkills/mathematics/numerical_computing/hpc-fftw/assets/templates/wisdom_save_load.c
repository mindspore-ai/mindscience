#include <fftw3.h>
#include <stdio.h>

void save_wisdom(const char *filename) {
    FILE *f = fopen(filename, "w");
    if (f) {
        fftw_export_wisdom_to_file(f);
        fclose(f);
        printf("Wisdom saved to %s\n", filename);
    }
}

void load_wisdom(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (f) {
        fftw_import_wisdom_from_file(f);
        fclose(f);
        printf("Wisdom loaded from %s\n", filename);
    }
}

int main() {
    int N = 4096;
    
    // Load existing wisdom (if available)
    load_wisdom("fftw_wisdom.dat");
    
    // Allocate and create plan
    fftw_complex *in = fftw_alloc_complex(N);
    fftw_complex *out = fftw_alloc_complex(N);
    
    // Use PATIENT mode - slow first time, fast after wisdom loaded
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_PATIENT);
    
    // Save wisdom for future runs
    save_wisdom("fftw_wisdom.dat");
    
    // Execute and cleanup
    fftw_execute(plan);
    
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    
    return 0;
}
