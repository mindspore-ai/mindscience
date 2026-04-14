# Mixed and Coupled Formulations

## Mixed Problems Require Mixed Spaces

When a PDE has multiple unknowns of different types (e.g., velocity + pressure
in Stokes), use a mixed function space:

```
Stokes:  (u, p) where u is vector, p is scalar
Darcy:   (u, p) where u is vector (flux), p is scalar (pressure)
Navier-Stokes: (u, p) same as Stokes + nonlinear advection
```

## Mixed Space Construction

### DOLFINx Taylor-Hood Elements

```python
from dolfinx import fem
from basix.ufl import mixed_element

# Velocity (P2) and pressure (P1)
P2 = fem.functionspace(mesh, ("Lagrange", 2))
P1 = fem.functionspace(mesh, ("Lagrange", 1))

# Or using mixed element directly
element = mixed_element([P2, P1])
W = fem.functionspace(mesh, element)

# Split for use
u = fem.Function(W.sub(0))  # velocity
p = fem.Function(W.sub(1))  # pressure
```

### Classic FEniCS Mixed Space

```python
P2 = VectorElement("Lagrange", triangle, 2)
P1 = FiniteElement("Lagrange", triangle, 1)
ME = MixedElement([P2, P1])
W = FunctionSpace(mesh, ME)

u, p = TrialFunctions(W)
v, q = TestFunctions(W)
```

## Stokes Flow

PDE:
- `-ν ∇²u + ∇p = f`
- `∇·u = 0`

Weak form (with u=0 on boundary):

```python
# Classic FEniCS
a = (nu * inner(grad(u), grad(v)) - div(v) * p + q * div(u)) * dx
L = dot(f, v) * dx

solve(a == L, w, bcs)
```

### Stabilization (LBB Condition)

Standard Taylor-Hood (P2-P1) is LBB-stable. If using equal-order elements,
add stabilization:

```python
h = CellSize(mesh)
alpha = 0.1

# PSPG (pressure-stabilized Petrov-Galerkin) term
a += alpha * h**2 * dot(grad(p), grad(q)) * dx
```

## Raviart-Thomas for Darcy/Mixed Poisson

For `∇·u = f` with `u = -k∇p`:

```python
from basix.ufl import wrapped_element

RT = fem.functionspace(mesh, ("RT", 1))  # Raviart-Thomas
P = fem.functionspace(mesh, ("Lagrange", 1))  # Lagrange for pressure

W = fem.functionspace(mesh, ("RT", 1) * ("Lagrange", 1))

sigma, u = TrialFunctions(W)
tau, v = TestFunctions(W)

k = Constant(mesh, 1.0)
a = (1/k) * dot(sigma, tau) * dx + div(sigma) * v * dx + div(tau) * u * dx
L = -f * v * dx
```

## Block Systems

Mixed problems produce block matrices:

```
┌───────┬───────┐
│   A   │  Bᵀ  │  [A][u] + [Bᵀ][p] = [f]
├───────┼───────┤  [B][u] + [0] [p] = [g]
│   B   │   0   │
└───────┴───────┘
```

## Solving Block Systems with PETSc

### Field Split Preconditioner

```python
from petsc4py import PETSc

# Create the block matrix
# A, B are assembled separately
# Use FieldSplit to block

K = PETSc.Mat().createNest([[A, B_T], [B, P]])
K.setUp()

# Set up field split
FSP = K.getFieldSplitSubSchur()
FSP.setType(PETSc.FieldSplit.Type.SCHUR)
FSP.setSchurFactType(PETSc.FieldSplit.SchurFactType.UPPER)
```

### Block Diagonal Preconditioner

For SPD blocks, use block Jacobi or block Gauss-Seidel.

## Robust Solver Patterns

### Stokes (from DOLFINx demos)

```python
# Direct solver for debugging
problem = fem.petsc.LinearProblem(
    a, L, bcs=bcs,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
    }
)
```

### Stokes Iterative (for scale)

```python
# Use Schur complement
problem = fem.petsc.LinearProblem(
    a, L, bcs=bcs,
    petsc_options={
        "ksp_type": "fgmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_schur_fact_type": "full",
        "fieldsplit_0_ksp_type": "cg",
        "fieldsplit_0_pc_type": "gamgee",
        "fieldsplit_1_pc_type": "none",  # Approximate inverse for pressure
        "fieldsplit_1_ksp_type": "cg",
    }
)
```

## Nitsche's Method for Interface Conditions

For weak imposition of continuity across interfaces:

```python
# On interface Γ: u_1 = u_2, ∂u_1/∂n_1 = -∂u_2/∂n_2
h = CellDiameter(mesh)
gamma = 10  # penalty parameter

# Nitsche contribution
F += - dot(average(grad(u)), n) * jump(v) * dS \
     - dot(average(grad(v)), n) * jump(u) * dS \
     + (gamma / h) * jump(u) * jump(v) * dS
```

## Output for Mixed Problems

```python
# Extract components and save separately
u_sub = u.sub(0)
p_sub = u.sub(1)

u_file = io.VTXWriter(mesh.comm, "velocity.bp", [u_sub])
p_file = io.VTXWriter(mesh.comm, "pressure.bp", [p_sub])

u_file.write(t)
p_file.write(t)
```

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Wrong subspace | Shape mismatch in form | Check value_shape of each subspace |
| LBB instability | Pressure oscillations | Use Taylor-Hood or add stabilization |
| Locking | Too stiff, wrong solution | Use reduced integration or mixed interpolation |
| Interface oscillation | Discontinuous solution | Check weak form of interface condition |
