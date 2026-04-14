# Error Recovery

## Import and Environment Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'dolfinx'` | Wrong Python environment | Load correct conda/env module |
| `ModuleNotFoundError: No module named 'fenics'` | Classic FEniCS not installed | Install or switch to DOLFINx |
| `ImportError: cannot import name '...' from 'dolfinx'` | API changed between versions | Check DOLFINx version |
| `petsc4py` import fails | PETSc not built | Rebuild PETSc with Python bindings |

## Form Assembly Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ArityMismatch` | Wrong form arity | Using `TrialFunction` for nonlinear unknown |
| `ShapeMismatch` | Space/rank mismatch | Check function space of unknown vs form |
| `InvalidRank` | Scalar vs vector confusion | Ensure `grad(u)` for scalar, `div(u)` for vector |
| `DomainMismatch` | Integration measure error | Use `dx` for volume, `ds` for boundary |
| `QuadratureError` | Quadrature degree too low | Increase cell quadrature rule |

**ArityMismatch fix:**

```python
# WRONG
u = TrialFunction(V)
F = u**2 * v * dx  # F is nonlinear but u is TrialFunction

# CORRECT
u = Function(V)
F = u**2 * v * dx
```

## Solver Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Diverged: 'diverged due to nan'` | Bad Jacobian or preconditioner | Check form assembly |
| `Diverged: 'diverged due to...' | Preconditioner failing | Switch to direct LU |
| `Max iterations reached` | Tolerance too tight or bad PC | Loosen tolerance, try different PC |
| `Singular matrix` | Missing BCs or pure Neumann | Add Dirichlet or point constraint |
| `Zero pivot` | Rigid body mode present | Add BC to constrain rigid modes |

**Troubleshooting solver divergence:**

1. Switch to direct solver (LU/MUMPS) — if it converges, the form is correct
2. If still diverges, check the form itself
3. If direct works, tune iterative solver

```python
# Use direct solver for debugging
problem = fem.petsc.LinearProblem(
    a, L, bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
```

## Nonlinear Solver Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Newton diverged` | Bad initial guess | Provide better initial guess |
| `Divided by zero in Jacobian` | u=0 at some points causing division | Avoid 0 denominator in forms |
| `Line search failed` | Step too large | Enable line search |
| `Not converged` | Tolerance too tight | Loosen `rtol`, increase max_it` |

## Boundary Condition Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `DirichletBC: cannot locate dofs` | Wrong entity dimension | Use correct tdim for DOF location |
| `Shape mismatch in BC application` | BC value has wrong shape | Ensure BC function matches space |
| `BC not applied` | BC list not passed to solve | Add `bcs=[bc]` to solve call |
| Under-constrained PDE | Not enough BCs | Add BCs to fully constrain the problem |

**Pure Neumann problem fix:**

```python
# Method 1: Add point constraint
u_D = Function(V)
u_D.x.array[0] = 0.0  # Constrain one DOF
bc = DirichletBC(V, u_D, [])

# Method 2: Use Lagrange multiplier
# Add constraint equation: mean(u) = 0
```

## Mesh and DOF Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Mesh topology not created` | Missing `create_connectivity` | Call `mesh.topology.create_connectivity()` |
| `DOF not found on boundary` | Entity dimension wrong | Use `tdim - 1` for facets |
| `Boundary marker empty` | Lambda function always returns False | Debug boundary detection function |

## Parallel Execution Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `MPI error: process exited` | Rank failed | Check logs for errors |
| `Segmentation fault` | Memory access error | Check array indexing |
| `Deadlock` | Allreduce called on subset | Use `mesh.comm` for collective calls |
| `Mesh not partitioned` | All ranks need mesh access | Partition mesh before scatter |

## Transient Simulation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `NaN in solution` | Timestep too large | Reduce dt |
| `Solution grows unboundedly` | Physics instability | Check PDE physics |
| `IndexError in state update` | Forgot to copy solution | Use `u_n.x.array[:] = u.x.array[:]` |

## Recovery Workflow

```
1. Error occurs
   ↓
2. Read full traceback (last 50 lines)
   ↓
3. Classify: Import / Form / Solver / BC / Mesh / Parallel
   ↓
4. For form errors: check TrialFunction vs Function, ranks, spaces
   ↓
5. For solver errors: switch to direct LU for debugging
   ↓
6. For BC errors: verify entity dimension and space match
   ↓
7. For parallel errors: run on 1 rank first
   ↓
8. Isolate with minimal reproduction
   ↓
9. Apply fix
```
