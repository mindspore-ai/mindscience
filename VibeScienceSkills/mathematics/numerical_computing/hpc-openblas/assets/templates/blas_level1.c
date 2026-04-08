#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

int main() {
    int n = 1000;
    
    // Allocate vectors
    double *x = (double*)malloc(n * sizeof(double));
    double *y = (double*)malloc(n * sizeof(double));
    
    // Initialize
    for (int i = 0; i < n; i++) {
        x[i] = (double)i / n;
        y[i] = (double)(n - i) / n;
    }
    
    // Level 1 BLAS operations
    
    // 1. Dot product: result = x^T * y
    double dot = cblas_ddot(n, x, 1, y, 1);
    printf("Dot product: %.6f\n", dot);
    
    // 2. AXPY: y = alpha * x + y
    double alpha = 2.0;
    cblas_daxpy(n, alpha, x, 1, y, 1);
    printf("After DAXPY, y[0] = %.6f\n", y[0]);
    
    // 3. Norm: nrm2 = ||x||_2
    double nrm2 = cblas_dnrm2(n, x, 1);
    printf("L2 norm of x: %.6f\n", nrm2);
    
    // 4. Scale: x = alpha * x
    cblas_dscal(n, 0.5, x, 1);
    printf("After DSCAL, x[0] = %.6f\n", x[0]);
    
    // 5. Copy: y = x
    cblas_dcopy(n, x, 1, y, 1);
    printf("After DCOPY, y[0] = %.6f\n", y[0]);
    
    // 6. Swap: x <-> y
    cblas_dswap(n, x, 1, y, 1);
    printf("After DSWAP, x[0] = %.6f, y[0] = %.6f\n", x[0], y[0]);
    
    // 7. Index of max absolute value
    int imax = cblas_idamax(n, x, 1);
    printf("Index of max |x[i]|: %d\n", imax);
    
    // 8. ASUM: sum of absolute values
    double asum = cblas_dasum(n, x, 1);
    printf("Sum of |x[i]|: %.6f\n", asum);
    
    free(x);
    free(y);
    return 0;
}
