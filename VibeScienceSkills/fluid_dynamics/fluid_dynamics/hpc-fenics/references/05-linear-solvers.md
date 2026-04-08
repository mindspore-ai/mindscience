# Linear Solvers and PETSc Configuration

## DOLFINx LinearProblem

For linear PDEs in DOLFINx, use `LinearProblem`:

```python
from dolfinx import fem, petsc
from petsc4py import PETSc

problem = fem.petsc.LinearProblem(
    a,                    # bilinear form
    L,                    # linear form
    bcs=[bc1, bc2],      # boundary conditions
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }
)
u = problem.solve()
```

## Classic FEniCS solve()

```python
solve(a == L, u, bcs,
      solver_parameters={"linear_solver": "mumps"})
```

## Solver Selection Strategy

```
Problem Scale & Type
│
├─ DOF < 100,000
│   └─ Use direct solver (LU/MUMPS) — reliable, no convergence tuning
│
├─ DOF 100K - 1M
│   └─ Direct if memory allows, otherwise iterative with AMG
│
└─ DOF > 1M
    └─ Iterative solver with AMG or Gamgee preconditioner
```

## PETSc KSP Types

| KSP | Full Name | Best For |
|-----|-----------|----------|
| `preonly` | No Krylov iteration | Use with direct solver (LU, Cholesky) |
| `gmres` | Generalized MINRES | General nonsymmetric |
| `fgmres` | Flexible GMRES | Nonsymmetric with variable preconditioner |
| `bicgstab` | BiCGStab | General nonsymmetric, less memory |
| `cg` | Conjugate Gradient | Symmetric positive definite |
| `minres` | MINRES | Symmetric indefinite |

## PETSc Preconditioners

| PC | Full Name | Best For |
|----|-----------|----------|
| `lu` | Direct LU | Small-medium problems, debugging |
| `mumps` | MUMPS | Large sparse direct solve |
| `ilu` | ILU (incomplete LU) | General iterative, moderate problems |
| `bjacobi` | Block Jacobi | Parallel, scalable |
| `gamgee` | Gamgee ( AMG) | Large sparse problems, scalable |
| `hypre` | Hypre (BoomerAMG) | Large parallel AMG |
| `fieldsplit` | Field split | Block systems (Stokes, poroelasticity) |

## Direct Solver (Debugging Baseline)

```python
petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}
```

Use this first when developing or debugging — it eliminates solver
convergence as a variable. Once the formulation is correct, switch to
iterative for scale.

## Iterative Solver for Large Problems

### For Symmetric Positive Definite (Heat, Elasticity)

```python
petsc_options = {
    "ksp_type": "cg",
    "pc_type": "gamgee",       # algebraic multigrid
    "pc_gamgee_technology": "p4est",
    "ksp_rtol": 1e-10,
    "ksp_atol": 1e-12,
}
```

### For Nonsymmetric (Advection-Diffusion, Navier-Stokes)

```python
petsc_options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "upper",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "bjacobi"
    },
    "fieldsplit_1": {
        "ksp_type": "cg",
        "pc_type": "ilu"
    }
}
```

## Schur Complement (Block Systems)

For saddle-point problems (Stokes, Darcy, Navier-Stokes):

```python
# Block system: [[A, B^T], [B, 0]] [u, p]^T = [f, g]^T
petsc_options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_schur_fact_type": "full",
    # Velocity block (field 0)
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "gamgee",
    # Pressure block (field 1) — use approximate inverse
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "jacobi"
}
```

## Setting PETSc Options at Runtime

### Via petsc4py

```python
from petsc4py import PETSc

opts = PETSc.Options()
opts["ksp_type"] = "cg"
opts["pc_type"] = "gamgee"
```

### Via Command Line (for debugging)

```python
# Set via options prefix for the problem
problem = fem.petsc.LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_view": None}  # prints solver config
)
```

## Convergence Monitoring

```python
# Add to petsc_options for convergence info
"ksp_converged_reason": None    # print convergence reason
"ksp_monitor_true_residual": None  # print residual history
```

## Solver Tolerance Guidelines

| Problem | KSP RTOL | KSP ATOL | Notes |
|---------|----------|----------|-------|
| Debugging | 1e-10 | 1e-14 | Very tight |
| Standard | 1e-6 | 1e-10 | Balance accuracy/cost |
| Engineering | 1e-4 | 1e-8 | Production runs |

Let the solver converge to its natural tolerance. Tightening tolerances
beyond what's needed wastes compute with no accuracy benefit.

## Performance Tuning

### For Large Scale (>1M DOF)

```python
petsc_options = {
    "ksp_type": "fgmres",
    "ksp_gmres_restart": 100,
    "pc_type": "gamgee",
    "pc_gamgee_technology": "p4est",
    "gamgee_refine": 2,          # AMG levels
    "gamgee_coarse": 1,          # Coarse solver
    "ksp_max_it": 500,
}
```

### For Anisotropic Problems (stretched meshes)

```python
petsc_options = {
    "ksp_type": "cg",
    "pc_type": "gamgee",
    "pc_gamgee_anisotropic": None,  # Enable
}
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Slow convergence | Weak preconditioner | Switch to AMG or ILU |
| Divergence | Preconditioner failing | Use direct solver temporarily |
| Divergence at high frequency | High-frequency modes not damped | Add Jacobi smoothing |
| "Diverged due to nan" | Preconditioner breakdown | Check form assembly |
| Convergence stalls | Locking (LBB stability) | Use pressure-stabilized formulation |
