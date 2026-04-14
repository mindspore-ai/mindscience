#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <time.h>

int main() {
    int m = 500, n = 400, k = 300;
    double alpha = 1.0, beta = 0.0;
    
    // Allocate matrices A (m x k), B (k x n), C (m x n)
    double *A = (double*)malloc(m * k * sizeof(double));
    double *B = (double*)malloc(k * n * sizeof(double));
    double *C = (double*)malloc(m * n * sizeof(double));
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < m * k; i++) A[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < k * n; i++) B[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < m * n; i++) C[i] = 0.0;
    
    // Level 3 BLAS operations
    
    // 1. DGEMM: C = alpha*A*B + beta*C
    clock_t start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
    clock_t end = clock();
    
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    double gflops = 2.0 * m * n * k / time_spent / 1e9;
    
    printf("DGEMM completed in %.3f seconds\n", time_spent);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("C[0] = %.6f, C[m*n-1] = %.6f\n", C[0], C[m*n-1]);
    
    // 2. DGEMM with transpose: C = alpha*A^T*B + beta*C
    double *C2 = (double*)malloc(m * n * sizeof(double));
    for (int i = 0; i < m * n; i++) C2[i] = 0.0;
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C2, n);
    printf("DGEMM (A^T * B) completed, C2[0] = %.6f\n", C2[0]);
    
    // 3. DSYMM: C = alpha*A*B + beta*C (symmetric A)
    int n_sym = 200;
    double *A_sym = (double*)malloc(n_sym * n_sym * sizeof(double));
    double *B_sym = (double*)malloc(n_sym * n_sym * sizeof(double));
    double *C_sym = (double*)malloc(n_sym * n_sym * sizeof(double));
    
    for (int i = 0; i < n_sym * n_sym; i++) {
        A_sym[i] = (i % n_sym == i / n_sym) ? 2.0 : 0.1;  // Symmetric
        B_sym[i] = (double)rand() / RAND_MAX;
        C_sym[i] = 0.0;
    }
    
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower,
                n_sym, n_sym, alpha, A_sym, n_sym, B_sym, n_sym, beta, C_sym, n_sym);
    printf("DSYMM completed, C_sym[0] = %.6f\n", C_sym[0]);
    
    // 4. DTRMM: B = alpha*op(A)*B (triangular A)
    // Create lower triangular matrix
    for (int i = 0; i < n_sym; i++) {
        for (int j = 0; j < n_sym; j++) {
            if (j > i) A_sym[i * n_sym + j] = 0.0;  // Lower triangular
        }
    }
    
    cblas_dtrmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                n_sym, n_sym, alpha, A_sym, n_sym, B_sym, n_sym);
    printf("DTRMM completed\n");
    
    free(A);
    free(B);
    free(C);
    free(C2);
    free(A_sym);
    free(B_sym);
    free(C_sym);
    return 0;
}
