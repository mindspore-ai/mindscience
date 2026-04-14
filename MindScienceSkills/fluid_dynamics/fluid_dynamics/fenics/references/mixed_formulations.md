# Mixed Formulations in FEniCS

Saddle-point problems and mixed finite element methods.

## Mixed Poisson Equation

Introduce flux variable $\sigma = -\nabla u$:

$$\sigma + \nabla u = 0$$

$$\nabla \cdot \sigma = f$$

### H(div) Conforming Elements

**Raviart-Thomas (RT1):**
```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Create mesh
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

# Mixed function space (RT1-P0)
RT1 = dfx.fem.functionspace(mesh, ("RT", 1))
P0 = dfx.fem.functionspace(mesh, ("DG", 0))
V = dfx.fem.functionspace(mesh, [RT1, P0])

# Trial and test functions
(sigma, u) = ufl.TrialFunctions(V)
(tau, v) = ufl.TestFunctions(V)

# Source term
f = dfx.fem.Constant(mesh, 1.0)

# Variational form
a = (ufl.inner(sigma, tau) * ufl.dx
      - ufl.div(tau) * u * ufl.dx
      - ufl.div(sigma) * v * ufl.dx)
L = f * v * ufl.dx

# Boundary conditions
# u = 0 on boundary
def boundary(x):
    return np.logical_or(
        np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
        np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)
    )

u0 = dfx.fem.Function(P0)
u0.x.array[:] = 0.0
bc = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(P0, boundary))

# Solve with block preconditioner
w_h = dfx.fem.Function(V)

options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "jacobi",
    "ksp_rtol": 1e-10
}

problem = dfx.fem.petsc.LinearProblem(a, L, [bc], w_h, petsc_options=options)
dfx.nls.petsc.solve(problem)

w_sol, u_sol = w_h.sub(0), w_h.sub(1)
```

**Brezzi-Douglas-Marini (BDM1):**
```python
# Higher-order H(div) elements
BDM1 = dfx.fem.functionspace(mesh, ("BDM", 1))
P0 = dfx.fem.functionspace(mesh, ("DG", 0))
V = dfx.fem.functionspace(mesh, [BDM1, P0])
```

## Darcy Flow

Flow through porous media:

$$\mathbf{u} = -\frac{K}{\mu} \nabla p$$

$$\nabla \cdot \mathbf{u} = 0$$

### Mixed Formulation

```python
# Parameters
K = dfx.fem.Constant(mesh, 1.0)  # Permeability
mu = dfx.fem.Constant(mesh, 1.0)  # Viscosity

# Mixed function space (RT1-P0)
RT1 = dfx.fem.functionspace(mesh, ("RT", 1))
P0 = dfx.fem.functionspace(mesh, ("DG", 0))
V = dfx.fem.functionspace(mesh, [RT1, P0])

(u, p) = ufl.TrialFunctions(V)
(v, q) = ufl.TestFunctions(V)

# Variational form
a = (mu / K * ufl.inner(u, v) * ufl.dx
      - ufl.div(v) * p * ufl.dx
      - ufl.div(u) * q * ufl.dx)
L = 0 * q * ufl.dx

# Boundary conditions
# Pressure on left and right
def left_boundary(x):
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    return np.isclose(x[0], 1.0)

p_left = dfx.fem.Function(P0)
p_left.x.array[:] = 1.0
bc_left = dfx.fem.dirichletbc(p_left, dfx.fem.locate_dofs_geometrical(P0, left_boundary))

p_right = dfx.fem.Function(P0)
p_right.x.array[:] = 0.0
bc_right = dfx.fem.dirichletbc(p_right, dfx.fem.locate_dofs_geometrical(P0, right_boundary))

bcs = [bc_left, bc_right]

# Solve
w_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, w_h, petsc_options=options)
dfx.nls.petsc.solve(problem)
```

## Stokes Flow

Mixed velocity-pressure formulation:

$$-\nabla^2 \mathbf{u} + \nabla p = \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

### Taylor-Hood Elements

```python
# See references/fluid_flow.md for complete implementation
# P2-P1 element pair
```

### MINI Elements

Alternative stable element pair:

```python
# P1bubble-P1
P1bubble = dfx.fem.functionspace(mesh, ("Bubble", 1))
P1 = dfx.fem.functionspace(mesh, ("Lagrange", 1))
V = dfx.fem.functionspace(mesh, [P1bubble, P1])

# Rest of formulation similar to Taylor-Hood
```

## Block Preconditioners

### Schur Complement

For saddle-point problems of form:

$$\begin{bmatrix} A & B^T \\ B & 0 \end{bmatrix} \begin{bmatrix} u \\ p \end{bmatrix} = \begin{bmatrix} f \\ g \end{bmatrix}$$

Schur complement: $S = -B A^{-1} B^T$

### PETSc Configuration

```python
options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    
    # Velocity block (field 0)
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "hypre",
    
    # Pressure block (field 1)
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "jacobi",
    
    # Schur complement
    "pc_fieldsplit_schur_fact_type": "full",
    "ksp_rtol": 1e-10
}
```

### Approximate Schur Complement

```python
# Use mass matrix for Schur complement
options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "diag",
    "ksp_rtol": 1e-10
}
```

## Inf-Sup Condition

Mixed problems must satisfy inf-sup condition:

$$\inf_{q \neq 0} \sup_{v \neq 0} \frac{b(v, q)}{||v||_V ||q||_Q} \geq \beta > 0$$

### Stable Element Pairs

**2D:**
- Taylor-Hood (P2-P1)
- MINI (P1bubble-P1)
- RT1-P0

**3D:**
- Taylor-Hood (P2-P1)
- MINI (P1bubble-P1)
- RT1-P0

### Checking Stability

**Pressure modes:**
```python
# Check for spurious pressure modes
# Should be zero for stable elements
pressure_mean = dfx.fem.assemble_scalar(p_sol * ufl.dx)
print(f"Pressure mean: {pressure_mean:.6e}")
```

## Post-Processing

### Recovering Primal Variable

For mixed Poisson, recover $u$ from $\sigma$:

```python
# Solve: -∇²u = f with u from mixed solution
# Or use post-processing techniques
```

### Divergence Check

```python
# Verify divergence-free condition for Stokes
div_u = ufl.div(u_sol)
div_norm = dfx.fem.assemble_scalar(div_u**2 * ufl.dx)
print(f"Divergence norm: {np.sqrt(div_norm):.6e}")
```

### Flux Conservation

```python
# Check flux conservation across boundaries
# Should be zero for exact solution
```

## Common Issues

### Spurious Pressure Modes

**Issue:** Checkerboard pressure pattern

**Solutions:**
- Use stable element pair
- Check inf-sup condition
- Verify boundary conditions

### Poor Solver Performance

**Issue:** Slow convergence of block preconditioner

**Solutions:**
- Tune Schur complement approximation
- Try different field splits
- Use appropriate preconditioners for each block

### Divergence Errors

**Issue:** Divergence not zero

**Solutions:**
- Check mixed formulation
- Verify element choice
- Examine boundary conditions

## Advanced Topics

### Three-Field Formulations

```python
# Introduce additional field
# e.g., elasticity with pressure
V = dfx.fem.functionspace(mesh, [V_u, V_p, V_lambda])
```

### Augmented Lagrangian

```python
# Alternative to mixed formulation
# Uses penalty and Lagrange multipliers
```

### Static Condensation

```python
# Eliminate interior degrees of freedom
# See FEniCS static condensation demo
```

### Discontinuous Galerkin

```python
# DG methods for mixed problems
# See references/advanced_elements.md
```
