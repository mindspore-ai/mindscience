#include <stdio.h>
#include <stdlib.h>
#include <lapacke.h>

int main() {
    // Example 1: Solve linear system Ax = b using DGESV
    int n = 5;
    int nrhs = 1;
    int lda = n;
    int ldb = n;
    int info;
    
    // Matrix A (column-major for LAPACK)
    double A[] = {
        2.0,  0.0,  0.0,  0.0,  0.0,
        1.0,  2.0,  0.0,  0.0,  0.0,
        0.0,  1.0,  2.0,  0.0,  0.0,
        0.0,  0.0,  1.0,  2.0,  0.0,
        0.0,  0.0,  0.0,  1.0,  2.0
    };
    
    double b[] = {2.0, 4.0, 6.0, 8.0, 10.0};
    int ipiv[5];
    
    printf("Solving Ax = b using DGESV...\n");
    info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, A, lda, ipiv, b, ldb);
    
    if (info == 0) {
        printf("Solution x:\n");
        for (int i = 0; i < n; i++) {
            printf("  x[%d] = %.6f\n", i, b[i]);
        }
    } else {
        printf("Error: info = %d\n", info);
    }
    
    // Example 2: Compute eigenvalues using DSYEV
    printf("\nComputing eigenvalues using DSYEV...\n");
    
    double A_eig[] = {
        4.0,  1.0,  1.0,
        1.0,  3.0,  2.0,
        1.0,  2.0,  3.0
    };
    int n_eig = 3;
    double w[3];
    
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', n_eig, A_eig, n_eig, w);
    
    if (info == 0) {
        printf("Eigenvalues:\n");
        for (int i = 0; i < n_eig; i++) {
            printf("  lambda[%d] = %.6f\n", i, w[i]);
        }
    } else {
        printf("Error: info = %d\n", info);
    }
    
    // Example 3: Matrix inversion using DGETRI + DGETRF
    printf("\nMatrix inversion using DGETRF + DGETRI...\n");
    
    double A_inv[] = {
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0
    };
    int n_inv = 3;
    int ipiv_inv[3];
    
    // LU factorization
    info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n_inv, n_inv, A_inv, n_inv, ipiv_inv);
    
    if (info == 0) {
        // Inversion
        info = LAPACKE_dgetri(LAPACK_COL_MAJOR, n_inv, A_inv, n_inv, ipiv_inv);
        
        if (info == 0) {
            printf("Inverse matrix:\n");
            for (int i = 0; i < n_inv; i++) {
                printf("  [");
                for (int j = 0; j < n_inv; j++) {
                    printf(" %.4f", A_inv[i + j * n_inv]);
                }
                printf(" ]\n");
            }
        }
    }
    
    // Example 4: SVD using DGESDD
    printf("\nSVD using DGESDD...\n");
    
    double A_svd[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };
    int m_svd = 3, n_svd = 2;
    double s[2], u[9], vt[4];
    int ldu = m_svd, ldvt = n_svd;
    
    info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', m_svd, n_svd, A_svd, n_svd, s, u, ldu, vt, ldvt);
    
    if (info == 0) {
        printf("Singular values:\n");
        for (int i = 0; i < n_svd; i++) {
            printf("  s[%d] = %.6f\n", i, s[i]);
        }
    }
    
    return 0;
}
