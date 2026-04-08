---
name: hpc-fenics
description: Build, review, and debug FEniCS or DOLFINx PDE scripts for finite-element workflows. Use when translating PDEs into UFL, selecting function spaces, applying boundary conditions, choosing between classic FEniCS and DOLFINx, or fixing FEM runtime errors.
---

# HPC FEniCS

FEniCS is a finite element framework for solving partial differential equations (PDEs) in science and engineering. It spans structural mechanics, heat transfer, fluid dynamics, and coupled multi-physics problems.

Treat FEniCS as a family with two main stacks: **classic FEniCS** (legacy) and **DOLFINx** (modern, MPI-first).

## Quick Start

| Step | Task | Reference |
|------|------|-----------|
| 1 | Choose stack (FEniCS vs DOLFINx) | [01-stack-selection](references/01-stack-selection.md) |
| 2 | Translate PDE into UFL weak form | [02-weak-form-construction](references/02-weak-form-construction.md) |
| 3 | Select function space (scalar/vector/mixed) | [03-function-spaces](references/03-function-spaces.md) |
| 4 | Apply boundary conditions | [04-boundary-conditions](references/04-boundary-conditions.md) |
| 5 | Configure linear solver | [05-linear-solvers](references/05-linear-solvers.md) |
| 6 | Handle nonlinear PDEs | [06-nonlinear-problems](references/06-nonlinear-problems.md) |
| 7 | Handle transient/time-dependent PDEs | [07-transient-problems](references/07-transient-problems.md) |
| 8 | Handle mixed formulations (Stokes, Darcy) | [08-mixed-problems](references/08-mixed-problems.md) |
| 9 | Execute on HPC cluster with MPI | [09-cluster-execution](references/09-cluster-execution.md) |
| - | Diagnose and fix runtime errors | [error-recovery](references/error-recovery.md) |
| - | PETSc solver option patterns | [petsc-solver-playbook](references/petsc-solver-playbook.md) |
| - | Parallel safety and MPI ownership rules | [parallel-and-mpi-caveats](references/parallel-and-mpi-caveats.md) |

## Skill Map

```
                        PDE TO CODE WORKFLOW
     ┌──────────────────────────────────────────────────────┐
     │  Stack-Selection (01)                                 │
     │  ──────────────────────────────────────────────────  │
     │  Weak-Form-Construction (02) │ Function-Spaces (03)  │
     │  ───────────────────────────│─────────────────────  │
     │  Boundary-Conditions (04)   │ Linear-Solvers (05)   │
     │  ────────────────────────────│─────────────────────  │
     │  Nonlinear-Problems (06)     │ Transient-Problems (07)│
     │  ───────────────────────────│─────────────────────  │
     │  Mixed-Problems (08)        │ Cluster-Execution (09) │
     │  ──────────────────────────────────────────────────  │
     │  Error-Recovery │ PETSc-Solver-Playbook │ MPI-Caveats │
     └──────────────────────────────────────────────────────┘
```

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Classic FEniCS or DOLFINx? | `01-stack-selection` | DOLFINx for new projects and MPI; classic for legacy code |
| Scalar or vector unknown? | `03-function-spaces` | Match element family and order to unknown type |
| Linear or nonlinear PDE? | `02` / `06` | Linear: `TrialFunction`; Nonlinear: `Function` with residual |
| Steady or transient? | `07-transient-problems` | Time-stepping loop vs single solve |
| Mixed formulation? | `08-mixed-problems` | Block systems with field-split preconditioners |
| Running on a cluster? | `09-cluster-execution` | MPI `srun`/`sbatch` workflow and mesh partitioning |

## Guardrails

- Do not mix classic FEniCS imports with DOLFINx APIs in one script.
- Do not use `TrialFunction` for a nonlinear unknown — use `Function`.
- Do not guess a boundary condition if the PDE is under-constrained; state what is missing.
- Do not ignore shape and rank mismatches in UFL expressions — inspect `value_shape`.
- Do not assume boundary marking strategies that work in serial are automatically parallel-safe.

## Outputs

Always report:

- chosen stack and version family
- PDE form and function space selection
- boundary conditions applied (type and location)
- expected output files and format
- the exact failure class when repairing a script

## Assets

**When to include**: When the skill needs files that will be used in the final output.

**Use cases**: Templates, boilerplate code, batch scripts that get copied or modified.

| File | Use Case | References |
|------|----------|------------|
| `assets/templates/poisson_dolfinx.py` | Steady Poisson — simplest DOLFINx linear case | `01`, `02`, `03`, `04`, `05` |
| `assets/templates/transient_diffusion_dolfinx.py` | Time-dependent diffusion with Backward Euler | `01`, `07` |
| `assets/templates/fenics-dolfinx-slurm.sh` | SLURM batch submission for MPI runs | `09` |

## Reference Summary

All references are used in this skill:

| Document | Topic |
|----------|-------|
| [01-stack-selection](references/01-stack-selection.md) | FEniCS vs DOLFINx API, migration path, version detection |
| [02-weak-form-construction](references/02-weak-form-construction.md) | UFL operators, integration measures, common PDE weak forms |
| [03-function-spaces](references/03-function-spaces.md) | Lagrange, DG, RT, Nédélec; scalar/vector/mixed spaces |
| [04-boundary-conditions](references/04-boundary-conditions.md) | Dirichlet, Neumann, Robin, periodic; multi-BC handling |
| [05-linear-solvers](references/05-linear-solvers.md) | PETSc KSP/preconditioner selection by problem scale and type |
| [06-nonlinear-problems](references/06-nonlinear-problems.md) | Newton solve, line search, continuation, SNES options |
| [07-transient-problems](references/07-transient-problems.md) | Backward Euler, Crank-Nicolson, BDF2, adaptive timestep |
| [08-mixed-problems](references/08-mixed-problems.md) | Block systems, Schur complement, Stokes/Darcy/Navier-Stokes |
| [09-cluster-execution](references/09-cluster-execution.md) | SLURM/PBS script patterns, mesh partitioning, MPI-IO output |
| [error-recovery](references/error-recovery.md) | Structured diagnosis: import, form, solver, BC, mesh, parallel |
| [petsc-solver-playbook](references/petsc-solver-playbook.md) | Robust PETSc option sets for linear and nonlinear problems |
| [parallel-and-mpi-caveats](references/parallel-and-mpi-caveats.md) | Parallel entity ownership, boundary marking safety, options consistency |
