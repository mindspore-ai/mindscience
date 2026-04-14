---
name: hpc-fftw
description: FFTW (Fastest Fourier Transform in the West) — high-performance C library for discrete Fourier transforms in one or more dimensions, supporting complex, real, and separable transforms (DCT/DST). Use for signal processing, spectral analysis, PDE solvers, and large-scale parallel computing on HPC clusters.
---

# HPC FFTW

FFTW is a C library for computing discrete Fourier transforms (DFTs) in one or more dimensions, of arbitrary input size, and of both real and complex data. It is the de-facto standard for high-performance FFT on HPC systems.

## Quick Start

| Step | Task | Reference |
|------|------|-----------|
| 1 | Choose transform type (Complex/Real/DCT/DST) | [01-transform-types](references/01-transform-types.md) |
| 2 | Select planning mode (ESTIMATE/MEASURE/PATIENT) | [02-planning-modes](references/02-planning-modes.md) |
| 3 | Configure parallelization (OpenMP/MPI/Hybrid) | [03-parallelization](references/03-parallelization.md) |
| 4 | Handle memory allocation (SIMD alignment) | [04-memory-management](references/04-memory-management.md) |
| - | Diagnose and fix runtime errors | [error-recovery](references/error-recovery.md) |

## Skill Map

```
FFTW Skill
├─ Transform Types          → references/01-transform-types.md
├─ Planning Modes           → references/02-planning-modes.md
├─ Parallelization          → references/03-parallelization.md
├─ Memory Management        → references/04-memory-management.md
└─ Error Recovery           → references/error-recovery.md
```

## Key Decision Points

| Question | Options | Recommendation |
|----------|---------|----------------|
| Transform type? | Complex DFT / Real DFT (R2C/C2R) / DCT / DST | Use Real DFT for real input (2x faster, half memory) |
| Planning mode? | ESTIMATE / MEASURE / PATIENT / EXHAUSTIVE | MEASURE for default; PATIENT for production |
| Parallelization? | OpenMP / MPI / Hybrid | OpenMP for shared memory; MPI for distributed |
| Precision? | double / float / long double | Default double; float for memory-constrained |
| Reuse plans? | Wisdom save/load | Enable for repeated transform sizes in production |

## References

Detailed guidance for each topic area:

- [Transform Types](references/01-transform-types.md) — Complex DFT, Real DFT (R2C/C2R), DCT/DST, Discrete Hartley Transform; covers 1D/2D/3D/multi-dimensional variants and normalization
- [Planning Modes](references/02-planning-modes.md) — ESTIMATE/MEASURE/PATIENT/EXHAUSTIVE flags, wisdom file save/load, system-wide wisdom configuration
- [Parallelization](references/03-parallelization.md) — OpenMP multi-threading, MPI distributed memory, hybrid OpenMP+MPI; includes initialization sequences and compilation flags
- [Memory Management](references/04-memory-management.md) — SIMD-aligned allocation (`fftw_malloc`, `fftw_alloc_complex`, `fftw_alloc_real`), in-place transform sizing, large-array considerations (64-bit build, `ptrdiff_t`)
- [Error Recovery](references/error-recovery.md) — Segfaults from unaligned memory, incorrect normalization, MPI deadlock, thread safety races, debugging tips and performance symptom table

## Guardrails

### Memory Alignment
- **Always** use `fftw_malloc` / `fftw_alloc_complex` / `fftw_alloc_real` for SIMD-aligned allocation
- Free with `fftw_free`, never standard `free()`
- Misaligned arrays cause segfaults or severe performance degradation

### Plan Creation
- Create plans **before** filling input arrays when using MEASURE or PATIENT modes (they overwrite arrays)
- Destroy plans after execution to avoid memory leaks
- Use wisdom files to save optimal plans for repeated runs

### Multi-threading
- Call `fftw_init_threads()` before any FFTW functions
- Call `fftw_plan_with_nthreads(n)` before creating plans
- Plans are **not** thread-safe; use a separate plan per thread

### MPI
- Call `fftw_mpi_init()` at program start
- Use `fftw_mpi_local_size_*` functions to determine local array distribution
- Array distribution differs from serial FFTW — never assume same layout

## Assets

**When to include**: When the skill needs files that will be used in the final output.

**Use cases**: Templates, boilerplate code, batch scripts that get copied or modified.

| File | Purpose |
|------|---------|
| `assets/templates/complex_1d_dft.c` | Basic 1D complex DFT with sine-wave input, aligned allocation, MEASURE planning |
| `assets/templates/real_2d_dft.c` | 2D real-to-complex (R2C) DFT showing half-sized complex output |
| `assets/templates/mpi_3d_dft.c` | MPI-distributed 3D DFT with `fftw_mpi_local_size_3d` and `fftw_mpi_plan_dft_3d` |
| `assets/templates/wisdom_save_load.c` | Save/load wisdom files to reuse expensive plan measurements across runs |
| `assets/templates/fftw_slurm.sh` | SLURM submission script for MPI FFTW (2 nodes, 64 ranks, 64 GB memory) |

## Outputs

Always report:
- Transform type, dimensions, and precision
- Planning mode and whether wisdom was used
- Parallelization strategy (OpenMP thread count, MPI rank count)
- Execution time and speedup vs. ESTIMATE baseline
- Memory usage and alignment verification (`fftw_alignment_of`)

## Reference Summary

All references are used in this skill:

| Document | Topic |
|----------|-------|
| [01-transform-types](references/01-transform-types.md) | Complex DFT, Real DFT (R2C/C2R), DCT/DST, multi-dimensional variants |
| [02-planning-modes](references/02-planning-modes.md) | ESTIMATE/MEASURE/PATIENT/EXHAUSTIVE, wisdom save/load |
| [03-parallelization](references/03-parallelization.md) | OpenMP, MPI, hybrid parallelization |
| [04-memory-management](references/04-memory-management.md) | SIMD-aligned allocation, large-array support |
| [error-recovery](references/error-recovery.md) | Segfaults, normalization, MPI deadlock, thread safety |

## Installation

```bash
./configure --enable-mpi --enable-openmp --enable-sse2 --enable-avx
make -j8
sudo make install
```

For 64-bit large-array support: add `--enable-64bit`.
