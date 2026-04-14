#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

int main() {
    int m = 100, n = 50;
    double alpha = 1.0, beta = 0.0;
    
    // Allocate matrix A (m x n) and vectors x (n), y (m)
    double *A = (double*)malloc(m * n * sizeof(double));
    double *x = (double*)malloc(n * sizeof(double));
    double *y = (double*)malloc(m * sizeof(double));
    
    // Initialize
    for (int i = 0; i < m * n; i++) A[i] = (double)i / (m * n);
    for (int i = 0; i < n; i++) x[i] = 1.0;
    for (int i = 0; i < m; i++) y[i] = 0.0;
    
    // Level 2 BLAS operations
    
    // 1. GEMV: y = alpha*A*x + beta*y
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, n, x, 1, beta, y, 1);
    printf("After DGEMV, y[0] = %.6f, y[m-1] = %.6f\n", y[0], y[m-1]);
    
    // 2. GEMV with transpose: y = alpha*A^T*x + beta*y
    double *x2 = (double*)malloc(m * sizeof(double));
    double *y2 = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < m; i++) x2[i] = 1.0;
    for (int i = 0; i < n; i++) y2[i] = 0.0;
    
    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, alpha, A, n, x2, 1, beta, y2, 1);
    printf("After DGEMV (transposed), y2[0] = %.6f\n", y2[0]);
    
    // 3. GER: A = alpha*x*y^T + A (rank-1 update)
    cblas_dger(CblasRowMajor, m, n, alpha, y, 1, x, 1, A, n);
    printf("After DGER, A[0] = %.6f\n", A[0]);
    
    // 4. SYMV: y = alpha*A*x + beta*y (symmetric matrix)
    // For symmetric matrix, only store lower/upper triangle
    int n_sym = 100;
    double *A_sym = (double*)malloc(n_sym * n_sym * sizeof(double));
    double *x_sym = (double*)malloc(n_sym * sizeof(double));
    double *y_sym = (double*)malloc(n_sym * sizeof(double));
    
    for (int i = 0; i < n_sym; i++) {
        x_sym[i] = 1.0;
        y_sym[i] = 0.0;
        for (int j = 0; j < n_sym; j++) {
            A_sym[i * n_sym + j] = (i == j) ? 2.0 : 0.1;  // Diagonal dominant
        }
    }
    
    cblas_dsymv(CblasRowMajor, CblasLower, n_sym, alpha, A_sym, n_sym, x_sym, 1, beta, y_sym, 1);
    printf("After DSYMV, y_sym[0] = %.6f\n", y_sym[0]);
    
    free(A);
    free(x);
    free(y);
    free(x2);
    free(y2);
    free(A_sym);
    free(x_sym);
    free(y_sym);
    return 0;
}
