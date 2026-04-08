---
name: fenics
description: Finite element computing platform for solving PDEs with FEniCSx. Use when solving partial differential equations with: (1) Poisson/Helmholtz equations, (2) Stokes/Navier-Stokes fluid flow, (3) Linear/nonlinear elasticity, (4) Cahn-Hilliard phase field, (5) Mixed formulations, (6) Complex geometries with Gmsh, (7) High-performance parallel computing, (8) Custom finite elements, or (9) Multi-physics problems requiring FEM
license: LGPL-3.0
metadata:
    skill-author: K-Dense Inc.
---

# FEniCS

Finite element computing platform for automated solution of differential equations.

## Overview

FEniCSx is a modern open-source computing platform for solving partial differential equations (PDEs) using the finite element method (FEM). It enables users to quickly translate scientific models into efficient finite element code through high-level Python and C++ interfaces.

## Quick Start

**Installation:**
```bash
conda install -c conda-forge fenics-dolfinx
# or
pip install fenics-dolfinx
```

**Basic Poisson equation:**
```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Create mesh and function space
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

# Define variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dfx.fem.Constant(mesh, 1.0)

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Solve
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, [], u_h)
dfx.nls.petsc.solve(problem)
```

## Core Workflow

### 1. Mesh Creation

**Built-in meshes:**
```python
import dolfinx as dfx
from mpi4py import MPI

# 1D interval
mesh = dfx.mesh.create_interval(MPI.COMM_WORLD, 100)

# 2D rectangle
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
mesh = dfx.mesh.create_rectangle(MPI.COMM_WORLD, [0, 0], [1, 1], [32, 32])

# 3D box
mesh = dfx.mesh.create_unit_cube(MPI.COMM_WORLD, 16, 16, 16)
mesh = dfx.mesh.create_box(MPI.COMM_WORLD, [0, 0, 0], [1, 1, 1], [16, 16, 16])
```

**Loading from file:**
```python
# XDMF format (recommended)
with dfx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
```

**Complex geometries with Gmsh:**
```python
# See references/meshing.md for detailed Gmsh integration
mesh = dfx.io.gmsh_to_dolfinx("geometry.msh", MPI.COMM_WORLD)
```

### 2. Function Spaces and Elements

**Lagrange elements (continuous):**
```python
# P1 (linear)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

# P2 (quadratic)
V = dfx.fem.functionspace(mesh, ("Lagrange", 2))

# P3 (cubic)
V = dfx.fem.functionspace(mesh, ("Lagrange", 3))
```

**Vector function spaces:**
```python
# Vector P2 (2D)
V = dfx.fem.functionspace(mesh, ("Lagrange", 2, (2,)))

# Vector P2 (3D)
V = dfx.fem.functionspace(mesh, ("Lagrange", 2, (3,)))
```

**Mixed function spaces:**
```python
# Taylor-Hood for Stokes (P2-P1)
P2 = dfx.fem.functionspace(mesh, ("Lagrange", 2))
P1 = dfx.fem.functionspace(mesh, ("Lagrange", 1))
V = dfx.fem.functionspace(mesh, [P2, P1])
```

**Discontinuous Galerkin:**
```python
# DG elements
V = dfx.fem.functionspace(mesh, ("DG", 0))
V = dfx.fem.functionspace(mesh, ("DG", 1))
```

### 3. Defining Functions

**Trial and test functions:**
```python
import ufl

u = ufl.TrialFunction(V)  # Unknown solution
v = ufl.TestFunction(V)   # Test function
```

**Known functions:**
```python
# Constant
f = dfx.fem.Constant(mesh, 1.0)

# Function (for initial conditions, boundary values)
u_h = dfx.fem.Function(V)

# Set values
u_h.x.array[:] = 1.0

# Vector function
u_vec = dfx.fem.Function(V_vec)
u_vec.x.array[:] = [1.0, 0.0]
```

### 4. Variational Forms

**Weak form definition:**
```python
# Poisson: -∇²u = f
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
```

**Helmholtz equation:**
```python
# -∇²u + k²u = f
k = dfx.fem.Constant(mesh, 10.0)
a = (ufl.dot(ufl.grad(u), ufl.grad(v)) + k**2 * u * v) * ufl.dx
L = f * v * ufl.dx
```

**Time-dependent problems:**
```python
# Heat equation: ∂u/∂t - α∇²u = f
alpha = dfx.fem.Constant(mesh, 0.1)
u_n = dfx.fem.Function(V)  # Previous time step

# BDF1 (backward Euler)
a = (u * v + alpha * dt * ufl.dot(ufl.grad(u), ufl.grad(v))) * ufl.dx
L = (u_n * v + dt * f * v) * ufl.dx
```

### 5. Boundary Conditions

**Dirichlet (fixed value):**
```python
# Define boundary
def boundary(x):
    return np.isclose(x[0], 0.0)  # x = 0

# Create boundary condition
u0 = dfx.fem.Function(V)
u0.x.array[:] = 0.0
bc = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(V, boundary))
```

**Multiple boundary conditions:**
```python
def left_boundary(x):
    return np.isclose(x[0], 0.0)

def right_boundary(x):
    return np.isclose(x[0], 1.0)

u_left = dfx.fem.Function(V)
u_left.x.array[:] = 0.0
bc_left = dfx.fem.dirichletbc(u_left, dfx.fem.locate_dofs_geometrical(V, left_boundary))

u_right = dfx.fem.Function(V)
u_right.x.array[:] = 1.0
bc_right = dfx.fem.dirichletbc(u_right, dfx.fem.locate_dofs_geometrical(V, right_boundary))

bcs = [bc_left, bc_right]
```

**Time-dependent boundary conditions:**
```python
# Update boundary condition at each time step
u_left.x.array[:] = np.sin(t)
```

### 6. Solving

**Linear problems:**
```python
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h)
dfx.nls.petsc.solve(problem)
```

**Nonlinear problems:**
```python
# Newton solver
problem = dfx.nls.petsc.NewtonProblem(a, L, bcs, u_h)
solver = dfx.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
solver.solve(problem)
```

**Custom solver parameters:**
```python
# PETSc options
options = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": 1e-10,
    "ksp_atol": 1e-14
}

problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h, petsc_options=options)
dfx.nls.petsc.solve(problem)
```

## Common Problem Types

### Poisson Equation

**Standard Poisson:**
```python
# -∇²u = f in Ω, u = 0 on ∂Ω
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = dfx.fem.Constant(mesh, 1.0)

a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, [], u_h)
dfx.nls.petsc.solve(problem)
```

**Neumann boundary conditions:**
```python
# -∇²u = f in Ω, ∂u/∂n = g on ∂Ω
g = dfx.fem.Constant(mesh, 1.0)

# Add Neumann term to right-hand side
L = (f * v + g * v) * ufl.ds  # ds = boundary measure
```

### Stokes Flow

**Taylor-Hood elements:**
```python
# See references/fluid_flow.md for complete implementation
# -∇²u + ∇p = 0, ∇·u = 0

P2 = dfx.fem.functionspace(mesh, ("Lagrange", 2))
P1 = dfx.fem.functionspace(mesh, ("Lagrange", 1))
V = dfx.fem.functionspace(mesh, [P2, P1])

(u, p) = ufl.TrialFunctions(V)
(v, q) = ufl.TestFunctions(V)

f = dfx.fem.Constant(mesh, (0.0, 0.0))

a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v) * p + ufl.div(u) * q) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h)
dfx.nls.petsc.solve(problem)
```

### Navier-Stokes

**Transient Navier-Stokes:**
```python
# See references/fluid_flow.md for complete implementation
# ∂u/∂t + (u·∇)u - ν∇²u + ∇p = 0, ∇·u = 0

nu = dfx.fem.Constant(mesh, 0.01)
u_n = dfx.fem.Function(P2_vec)  # Previous time step

# Semi-discrete form (BDF1)
a = (ufl.inner(u, v) * ufl.dx
      + dt * ufl.inner(ufl.grad(u), ufl.grad(u_n)) * v * ufl.dx
      + dt * nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
      - dt * ufl.div(v) * p * ufl.dx
      + dt * ufl.div(u) * q * ufl.dx)
L = ufl.inner(u_n, v) * ufl.dx
```

### Linear Elasticity

**Plane strain:**
```python
# See references/elasticity.md for complete implementation
# ∇·σ = f, σ = λ tr(ε) I + 2μ ε, ε = (∇u + ∇uᵀ)/2

E = dfx.fem.Constant(mesh, 1.0e5)  # Young's modulus
nu_mat = dfx.fem.Constant(mesh, 0.3)  # Poisson's ratio
mu = E / (2 * (1 + nu_mat))
lmbda = E * nu_mat / ((1 + nu_mat) * (1 - 2 * nu_mat))

def epsilon(u):
    return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

def sigma(u):
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
```

### Cahn-Hilliard

**Phase field model:**
```python
# See references/phase_field.md for complete implementation
# ∂c/∂t = ∇·(M∇μ), μ = f'(c) - κ∇²c

M = dfx.fem.Constant(mesh, 1.0)
kappa = dfx.fem.Constant(mesh, 0.01)

V = dfx.fem.functionspace(mesh, ("Lagrange", 1))
c = ufl.TrialFunction(V)
d = ufl.TestFunction(V)
mu = ufl.TrialFunction(V)
e = ufl.TestFunction(V)

# Free energy derivative: f'(c) = c³ - c
f_prime = c**3 - c

# Coupled equations
a1 = (c * d + dt * M * ufl.dot(ufl.grad(mu), ufl.grad(d))) * ufl.dx
L1 = (c_n * d) * ufl.dx

a2 = (mu * e + kappa * ufl.dot(ufl.grad(c), ufl.grad(e)) - f_prime * e) * ufl.dx
L2 = 0 * e * ufl.dx
```

## Solver Configuration

### PETSc Solvers

**Basic configuration:**
```python
options = {
    "ksp_type": "cg",        # Conjugate gradient
    "pc_type": "hypre",      # Hypre preconditioner
    "ksp_rtol": 1e-10,
    "ksp_max_it": 1000
}

problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h, petsc_options=options)
dfx.nls.petsc.solve(problem)
```

**Multigrid:**
```python
options = {
    "ksp_type": "richardson",
    "pc_type": "mg",
    "pc_mg_type": "multiplicative",
    "ksp_rtol": 1e-10
}
```

**Block preconditioner for mixed problems:**
```python
options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "hypre"
}
```

### Newton Solver

**Nonlinear problems:**
```python
problem = dfx.nls.petsc.NewtonProblem(a, L, bcs, u_h)
solver = dfx.nls.petsc.NewtonSolver(MPI.COMM_WORLD)

# Set parameters
solver.set_param("rtol", 1e-8)
solver.set_param("atol", 1e-10)
solver.set_param("max_it", 50)

# Solve
solver.solve(problem)

# Convergence info
print(f"Newton iterations: {solver.iteration}")
print(f"Final residual: {solver.residual}")
```

## Visualization

**Using PyVista:**
```python
import pyvista as pv

# Create VTK grid
grid = pv.UnstructuredGrid(*dfx.io.vtk_mesh(mesh))

# Add solution
grid.point_data["u"] = u_h.x.array.real

# Plot
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="u", show_edges=True)
plotter.show()
```

**Saving to file:**
```python
# XDMF format (recommended)
with dfx.io.XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h)

# VTK format
with dfx.io.VTKFile(MPI.COMM_WORLD, "solution.pvd", "w") as vtk:
    vtk.write_function(u_h, 0.0)
```

## Parallel Execution

**MPI parallelization:**
```bash
# Run with 4 processors
mpirun -np 4 python script.py
```

**Check parallel configuration:**
```python
from mpi4py import MPI
print(f"Rank: {MPI.COMM_WORLD.rank}, Size: {MPI.COMM_WORLD.size}")
```

**Parallel mesh:**
```python
# Mesh is automatically partitioned
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 100, 100)
print(f"Local cells: {mesh.topology.index_map(mesh.topology.dim, 0).size_local}")
```

## Best Practices

### 1. Element Selection
- **P1**: Fast, lower accuracy, good for simple problems
- **P2**: Good balance of speed and accuracy
- **P3+**: High accuracy, more expensive
- **Taylor-Hood (P2-P1)**: Inf-sup stable for Stokes
- **Mixed elements**: Use appropriate pairs for saddle-point problems

### 2. Solver Selection
- **Direct (LU)**: Small problems (< 100k DOFs)
- **CG + Hypre/AMG**: Large elliptic problems
- **GMRES + ILU**: Non-symmetric systems
- **FGMRES + Block preconditioner**: Mixed formulations

### 3. Boundary Conditions
- Use geometric location for simple boundaries
- Use mesh tags for complex geometries
- Update time-dependent BCs in time loop
- Be careful with corner points for multiple BCs

### 4. Performance
- Use appropriate mesh resolution
- Choose efficient solvers/preconditioners
- Enable parallel execution for large problems
- Consider matrix-free methods for very large problems

### 5. Numerical Stability
- Ensure inf-sup condition for mixed problems
- Use appropriate time stepping for transient problems
- Check convergence of Newton solver
- Monitor residuals during solving

## Resources

### Scripts

**`scripts/template_poisson.py`**
Basic Poisson equation solver template.

**`scripts/template_stokes.py`**
Stokes flow solver with Taylor-Hood elements.

**`scripts/template_navier_stokes.py`**
Transient Navier-Stokes solver template.

**`scripts/template_elasticity.py`**
Linear elasticity solver template.

### References

- **`references/phase_field.md`** - Cahn-Hilliard and Allen-Cahn equations
- **`references/fluid_flow.md`** - Stokes and Navier-Stokes implementations
- **`references/elasticity.md`** - Linear and nonlinear elasticity
- **`references/mixed_formulations.md`** - Mixed Poisson and saddle-point problems
- **`references/meshing.md`** - Complex mesh generation with Gmsh
- **`references/solvers.md`** - Solver configuration and parallel computing
- **`references/advanced_elements.md`** - Custom finite elements and DG methods

## Common Issues

**Divergence in Newton solver:**
- Check initial guess
- Reduce time step
- Use line search
- Examine problem formulation

**Poor solver performance:**
- Try different preconditioner
- Check mesh quality
- Use appropriate solver for problem type
- Enable parallel execution

**Boundary condition conflicts:**
- Check corner point handling
- Use proper mesh tags
- Verify geometric location functions

**Memory issues:**
- Reduce mesh resolution
- Use iterative solvers
- Enable matrix-free methods
- Check for memory leaks in custom code

## Additional Resources

- Official documentation: https://docs.fenicsproject.org/
- DOLFINx demos: https://docs.fenicsproject.org/dolfinx/v0.10.0.post1/python/demos.html
- FEniCSx tutorial: https://jorgensd.github.io/dolfinx-tutorial/
- GitHub repository: https://github.com/FEniCS/dolfinx
- Community forum: https://fenicsproject.discourse.group/
