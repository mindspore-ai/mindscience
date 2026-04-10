---
name: hpc-openblas
description: OpenBLAS optimized BLAS/LAPACK library for HPC. High-performance linear algebra operations used as backend for scientific computing. Use when users need:(1) Matrix-matrix multiplication (GEMM), (2) Linear system solving, (3) Eigenvalue computation, (4) Optimizing scientific code performance, (5) Building GROMACS, LAMMPS, TensorFlow, PyTorch, or SciPy backends.
---

# HPC-OpenBLAS Skill

OpenBLAS is an optimized BLAS library providing high-performance linear algebra operations for CPUs. It serves as the computational backend for GROMACS, LAMMPS, NWChem, TensorFlow, PyTorch, and SciPy.

## Quick Start

### Typical Workflow
1. Install or load OpenBLAS on the target system — see [references/04-compilation.md](references/04-compilation.md)
2. Choose BLAS level based on your operation — see [references/01-blas-levels.md](references/01-blas-levels.md)
3. Select data types (float, double, complex) — see [references/02-data-types.md](references/02-data-types.md)
4. Configure threading model — see [references/03-threading.md](references/03-threading.md)
5. Copy and modify a template from `assets/templates/`
6. Compile with appropriate flags — see [references/04-compilation.md](references/04-compilation.md)
7. Run and diagnose performance or errors — see [references/error-recovery.md](references/error-recovery.md)

## Skill Map

```
User Requirements
├─ Linear Algebra Operations
│  ├─ Vector ops (dot, axpy, norm) → 01-blas-levels.md (Level 1)
│  ├─ Matrix-vector ops (gemv, symv) → 01-blas-levels.md (Level 2)
│  └─ Matrix-matrix ops (gemm, symm) → 01-blas-levels.md (Level 3)
├─ Data Type Selection
│  ├─ Precision prefix (s/d/c/z) → 02-data-types.md
│  ├─ Matrix type (ge/sy/he/tr) → 02-data-types.md
│  └─ Complex number handling → 02-data-types.md
├─ Threading Configuration
│  ├─ OpenMP model → 03-threading.md
│  ├─ pthreads model → 03-threading.md
│  └─ Nested parallelism avoidance → 03-threading.md
├─ Compilation & Linking
│  ├─ Build from source → 04-compilation.md
│  ├─ Static/dynamic linking → 04-compilation.md
│  └─ CMake/pkg-config integration → 04-compilation.md
└─ Error Recovery
   ├─ Linker errors → error-recovery.md
   ├─ Segfaults / wrong results → error-recovery.md
   └─ Threading conflicts → error-recovery.md
```

## Reference Documents

| Document | Content |
|----------|---------|
| [references/01-blas-levels.md](references/01-blas-levels.md) | BLAS Level 1 (vector), Level 2 (matrix-vector), Level 3 (matrix-matrix) operations and CBLAS function signatures |
| [references/02-data-types.md](references/02-data-types.md) | Precision prefixes (s/d/c/z), matrix type codes (ge/sy/he/tr), complex number layout |
| [references/03-threading.md](references/03-threading.md) | OpenMP/pthreads/single-threaded models, OPENBLAS_NUM_THREADS, thread safety and nested parallelism |
| [references/04-compilation.md](references/04-compilation.md) | Build from source, TARGET architecture, static/dynamic linking, CMake, pkg-config |
| [references/error-recovery.md](references/error-recovery.md) | Linker errors, segmentation faults, wrong results, threading conflicts, performance diagnosis |

## BLAS Level Selection

| Level | Complexity | Best For | Reference |
|-------|------------|----------|-----------|
| Level 1 | O(n) | Vector operations (dot, axpy, norm) | [01-blas-levels.md](references/01-blas-levels.md) |
| Level 2 | O(n²) | Matrix-vector operations (gemv, symv) | [01-blas-levels.md](references/01-blas-levels.md) |
| Level 3 | O(n³) | Matrix-matrix operations (best scaling) | [01-blas-levels.md](references/01-blas-levels.md) |

## Data Type Selection

| Prefix | Type | Use Case | Reference |
|--------|------|----------|-----------|
| `s` | float | 2× speed, lower accuracy | [02-data-types.md](references/02-data-types.md) |
| `d` | double | Default for scientific computing | [02-data-types.md](references/02-data-types.md) |
| `c` | float complex | Signal processing | [02-data-types.md](references/02-data-types.md) |
| `z` | double complex | High-precision complex | [02-data-types.md](references/02-data-types.md) |

## Threading Configuration

| Model | Best For | Reference |
|-------|----------|-----------|
| OpenMP (`USE_OPENMP=1`) | Most HPC applications | [03-threading.md](references/03-threading.md) |
| pthreads | Fine-grained control | [03-threading.md](references/03-threading.md) |
| Single-threaded | Embedded, debugging | [03-threading.md](references/03-threading.md) |

## Template Files

Template files in `assets/templates/` are ready-to-use starting scaffolds that can be copied and modified:

| Template | Type | Use Case | Reference |
|----------|------|---------|-----------|
| [assets/templates/blas_level1.c](assets/templates/blas_level1.c) | C source | Vector ops (dot, axpy, norm) | [01-blas-levels.md](references/01-blas-levels.md), [02-data-types.md](references/02-data-types.md) |
| [assets/templates/blas_level2.c](assets/templates/blas_level2.c) | C source | Matrix-vector ops (gemv, symv) | [01-blas-levels.md](references/01-blas-levels.md), [02-data-types.md](references/02-data-types.md) |
| [assets/templates/blas_level3.c](assets/templates/blas_level3.c) | C source | Matrix-matrix ops (gemm, symm) | [01-blas-levels.md](references/01-blas-levels.md), [02-data-types.md](references/02-data-types.md) |
| [assets/templates/lapack_example.c](assets/templates/lapack_example.c) | C source | LAPACK routines (solve, eigenvalues) | [01-blas-levels.md](references/01-blas-levels.md), [04-compilation.md](references/04-compilation.md) |
| [assets/templates/openblas_slurm.sh](assets/templates/openblas_slurm.sh) | Batch script | SLURM submission for OpenBLAS jobs | [04-compilation.md](references/04-compilation.md), [03-threading.md](references/03-threading.md) |

## Guardrails

### Threading Rules
- Set `OPENBLAS_NUM_THREADS` **before** calling BLAS routines — see [03-threading.md](references/03-threading.md)
- Never nest OpenMP parallelism with OpenBLAS threading — causes race conditions
- Set `OPENBLAS_NUM_THREADS=1` when your code uses OpenMP — see [03-threading.md](references/03-threading.md)

### Memory Layout
- Match `CblasRowMajor`/`CblasColMajor` to your data layout — see [error-recovery.md](references/error-recovery.md)
- Leading dimension (`lda`) must be >= actual dimension
- Use 64-byte alignment for AVX-512

### Performance
- Small matrices (n < 100): use single thread — see [03-threading.md](references/03-threading.md)
- Level 3 BLAS scales well with threads — see [01-blas-levels.md](references/01-blas-levels.md)
- Level 1/2 BLAS are memory-bound — see [03-threading.md](references/03-threading.md)

## Error Recovery

Consult [references/error-recovery.md](references/error-recovery.md) for structured diagnosis of:

- **Linker errors** — undefined reference to `cblas_dgemm`, cannot find `-lopenblas`
- **Segmentation faults** — array bounds, leading dimensions, matrix orientation
- **Wrong results** — row-major vs column-major, transpose flags, alpha/beta values
- **Threading conflicts** — nested parallelism, poor OpenMP performance
- **Performance issues** — single-threaded builds, CPU frequency scaling
