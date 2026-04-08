# BLAS Levels

## Level 1: Vector Operations

Operations on O(n) data, O(n) work.

| Function | Operation | Description |
|----------|-----------|-------------|
| dswap | x <-> y | Swap vectors |
| dscal | x = alpha * x | Scale vector |
| dcopy | y = x | Copy vector |
| daxpy | y = alpha * x + y | AXPY operation |
| ddot | dot = x^T * y | Dot product |
| dnrm2 | nrm = ||x||_2 | Euclidean norm |
| dasum | sum = sum(|x_i|) | Sum of absolute values |
| idamax | imax = argmax(|x_i|) | Index of max absolute value |

### Example
```c
// Dot product
double result = cblas_ddot(n, x, 1, y, 1);

// AXPY: y = alpha * x + y
cblas_daxpy(n, alpha, x, 1, y, 1);

// Norm
double norm = cblas_dnrm2(n, x, 1);
```

## Level 2: Matrix-Vector Operations

Operations on O(n^2) data, O(n^2) work.

| Function | Operation | Description |
|----------|-----------|-------------|
| dgemv | y = alpha*A*x + beta*y | General matrix-vector |
| dsymv | y = alpha*A*x + beta*y | Symmetric matrix-vector |
| dtrmv | x = A*x | Triangular matrix-vector |
| dger | A = alpha*x*y^T + A | Rank-1 update |
| dsyr | A = alpha*x*x^T + A | Symmetric rank-1 update |
| dsyr2 | A = alpha*x*y^T + alpha*y*x^T + A | Symmetric rank-2 update |

### Example
```c
// General matrix-vector multiply
cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, lda, x, 1, beta, y, 1);

// Symmetric matrix-vector multiply
cblas_dsymv(CblasRowMajor, CblasLower, n, alpha, A, lda, x, 1, beta, y, 1);
```

## Level 3: Matrix-Matrix Operations

Operations on O(n^2) data, O(n^3) work. Highest performance potential.

| Function | Operation | Description |
|----------|-----------|-------------|
| dgemm | C = alpha*A*B + beta*C | General matrix-matrix |
| dsymm | C = alpha*A*B + beta*C | Symmetric matrix-matrix |
| dsyrk | C = alpha*A*A^T + beta*C | Symmetric rank-k update |
| dsyr2k | C = alpha*A*B^T + alpha*B*A^T + beta*C | Symmetric rank-2k update |
| dtrmm | B = alpha*A*B | Triangular matrix-matrix |
| dtrsm | B = alpha*A^{-1}*B | Triangular solve |

### Example
```c
// General matrix-matrix multiply
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

// Symmetric rank-k update
cblas_dsyrk(CblasRowMajor, CblasLower, CblasNoTrans,
            n, k, alpha, A, lda, beta, C, ldc);
```
