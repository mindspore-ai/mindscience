---
name: fipy
description: Finite volume PDE solver for diffusion, convection, phase field, and coupled equations. Use when solving partial differential equations with: (1) Diffusion processes, (2) Convection-diffusion problems, (3) Phase field models (Allen-Cahn, Cahn-Hilliard), (4) Level set methods, (5) Coupled PDE systems, (6) Moving boundary problems, (7) Fluid flow (Stokes, Navier-Stokes), or (8) Multi-physics simulations requiring adaptive meshing or parallel solving
license: https://github.com/usnistgov/fipy/blob/main/LICENSE
metadata:
    skill-author: K-Dense Inc.
---

# FiPy

Finite volume PDE solver using Python for solving coupled sets of partial differential equations.

## Overview

FiPy is an object-oriented, partial differential equation solver based on the finite volume method. It provides extensible tools for solving arbitrary combinations of coupled elliptic, hyperbolic, and parabolic PDEs, with built-in support for diffusion, convection, and source terms.

## Quick Start

**Installation:**
```bash
conda install -c conda-forge fipy
# or
pip install fipy
```

**Basic diffusion equation:**
```python
from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm

# Create mesh and variable
mesh = Grid1D(nx=50, dx=0.01)
phi = CellVariable(mesh=mesh, value=0., name='phi')

# Set initial and boundary conditions
phi.setValue(1., where=mesh.x < 0.1)
phi.constrain(0., where=mesh.facesLeft)
phi.constrain(1., where=mesh.facesRight)

# Create and solve equation
eq = TransientTerm() == DiffusionTerm(coeff=1.0)
for step in range(100):
    eq.solve(var=phi, dt=0.001)
```

## Core Workflow

### 1. Mesh Creation

Choose appropriate mesh type for your problem dimensionality:

**1D:**
```python
from fipy import Grid1D
mesh = Grid1D(nx=100, dx=0.01)
```

**2D:**
```python
from fipy import Grid2D
mesh = Grid2D(nx=100, ny=100, dx=0.01, dy=0.01)
```

**3D:**
```python
from fipy import Grid3D
mesh = Grid3D(nx=50, ny=50, nz=50, dx=0.01, dy=0.01, dz=0.01)
```

**Non-uniform grids:**
```python
mesh = Grid1D(dx=[0.01]*50 + [0.02]*50)
```

For complex geometries, use Gmsh:
```python
from fipy import Gmsh2D
mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
''')
```

### 2. Variable Definition

Create CellVariable for cell-centered values:
```python
from fipy import CellVariable
phi = CellVariable(mesh=mesh, value=0., name='phi')
```

Create FaceVariable for face-centered values:
```python
from fipy import FaceVariable
flux = FaceVariable(mesh=mesh, value=0., rank=1)
```

Set initial conditions:
```python
phi.setValue(1., where=mesh.x < 0.5)
phi.setValue(2., where=(mesh.x > 0.5) & (mesh.y > 0.5))
```

### 3. Boundary Conditions

**Dirichlet (fixed value):**
```python
phi.constrain(0., where=mesh.facesLeft)
phi.constrain(1., where=mesh.facesRight)
```

**Neumann (fixed gradient):**
```python
phi.faceGrad.constrain(0., where=mesh.facesTop)
```

**Robin (mixed):**
```python
# Heat transfer: -k*n·grad(T) = h*(T - T_inf)
# See references/boundary_conditions.md for implementation
```

**Spatially varying:**
```python
X, Y = mesh.faceCenters
phi.constrain(X * Y, where=mesh.exteriorFaces)
```

### 4. Equation Construction

**Diffusion:**
```python
from fipy import DiffusionTerm
eq = TransientTerm() == DiffusionTerm(coeff=D)
```

**Convection-diffusion:**
```python
from fipy import ConvectionTerm, DiffusionTerm
velocity = FaceVariable(mesh=mesh, value=(1., 0.), rank=1)
eq = (TransientTerm() + ConvectionTerm(coeff=velocity)
      == DiffusionTerm(coeff=D))
```

**With source term:**
```python
from fipy import ExplicitSourceTerm
source = CellVariable(mesh=mesh, value=1.0)
eq = TransientTerm() == DiffusionTerm() + ExplicitSourceTerm(source)
```

**Steady-state:**
```python
eq = DiffusionTerm() == ExplicitSourceTerm(source)
eq.solve(var=phi)
```

### 5. Solving

**Time stepping:**
```python
for step in range(1000):
    eq.solve(var=phi, dt=0.001)
```

**With convergence criteria:**
```python
eq.solve(var=phi, dt=0.001, solver=LinearLUSolver())
```

**Parallel solving:**
```bash
mpirun -np 4 python script.py --petsc
```

## Common Problem Types

### Diffusion Problems

**Simple diffusion:**
```python
eq = TransientTerm() == DiffusionTerm(coeff=1.0)
```

**Anisotropic diffusion:**
```python
D = FaceVariable(mesh=mesh, value=((1., 0.), (0., 2.)), rank=2)
eq = TransientTerm() == DiffusionTerm(coeff=D)
```

**Variable diffusion coefficient:**
```python
D = CellVariable(mesh=mesh, value=1.0)
eq = TransientTerm() == DiffusionTerm(coeff=D)
```

### Phase Field Models

**Allen-Cahn (non-conserved order parameter):**
```python
from fipy import ImplicitSourceTerm
epsilon = 0.01
M = 1.0
eq = (TransientTerm()
      == DiffusionTerm(coeff=epsilon**2 * M)
      - ImplicitSourceTerm(M * phi * (1 - phi) * (1 - 2 * phi)))
```

**Cahn-Hilliard (conserved order parameter):**
```python
# See references/phase_field.md for complete implementation
# Requires coupled equations
```

### Coupled Equations

**Two coupled variables:**
```python
from fipy import ImplicitSourceTerm

phi = CellVariable(mesh=mesh, value=0., name='phi')
psi = CellVariable(mesh=mesh, value=0., name='psi')

eq1 = (TransientTerm(var=phi)
       == DiffusionTerm(coeff=1.0, var=phi)
       + ImplicitSourceTerm(psi, var=phi))

eq2 = (TransientTerm(var=psi)
       == DiffusionTerm(coeff=0.5, var=psi)
       + ImplicitSourceTerm(phi, var=psi))

coupled_eq = eq1 & eq2
coupled_eq.solve(dt=0.001)
```

### Fluid Flow

**Stokes flow:**
```python
# See references/fluid_flow.md for implementation
# Requires coupled velocity-pressure equations
```

## Solver Selection

FiPy supports multiple solver suites:

**Default (SciPy):**
```python
eq.solve(var=phi, dt=0.001)
```

**PETSc (parallel):**
```bash
python script.py --petsc
# or
FIPY_SOLVERS=petsc python script.py
```

**Trilinos (parallel):**
```bash
python script.py --trilinos
```

**PyAMGX (GPU):**
```bash
python script.py --pyamgx
```

For detailed solver configuration, see `references/solvers.md`.

## Visualization

**Using Matplotlib:**
```python
import matplotlib.pyplot as plt
plt.contourf(mesh.cellCenters[0], mesh.cellCenters[1], phi)
plt.colorbar()
plt.show()
```

**Using FiPy viewers:**
```python
from fipy import MatplotlibViewer
viewer = MatplotlibViewer(vars=phi)
for step in range(100):
    eq.solve(var=phi, dt=0.001)
    viewer.plot()
```

## Best Practices

### 1. Mesh Selection
- Use Grid meshes for rectangular domains (faster)
- Use Gmsh for complex geometries
- Ensure mesh is sufficiently resolved for your problem
- Avoid highly non-orthogonal meshes (accuracy issues)

### 2. Time Step Selection
- Use explicit stability criterion: dt < dx² / (2*D) for diffusion
- Use adaptive stepping with steppyngstounes for stiff problems
- Smaller dt improves stability but increases computation time

### 3. Boundary Conditions
- Apply constraints before solving
- Use `constrain()` for Dirichlet conditions
- Use `faceGrad.constrain()` for Neumann conditions
- Default is zero flux (no constraint needed)

### 4. Performance Optimization
- Use `--inline` flag for C-optimized operations
- Set `OMP_NUM_THREADS=1` for parallel MPI runs
- Use PETSc or Trilinos for large problems
- Consider mesh partitioning for parallel efficiency

### 5. Numerical Stability
- Check CFL condition for convection-dominated problems
- Use upwind schemes for high Peclet numbers
- Ensure proper scaling of variables
- Monitor residuals during solving

## Resources

### Scripts

**`scripts/template_diffusion.py`**
Basic diffusion template with mesh creation, boundary conditions, and time stepping.

**`scripts/template_phase_field.py`**
Allen-Cahn phase field model template.

**`scripts/template_coupled.py`**
Coupled equation solver template.

### References

- **`references/phase_field.md`** - Phase field methods (Allen-Cahn, Cahn-Hilliard)
- **`references/level_set.md`** - Level set methods for moving boundaries
- **`references/fluid_flow.md`** - Stokes and Navier-Stokes implementations
- **`references/boundary_conditions.md`** - Advanced boundary conditions (Robin, mixed)
- **`references/solvers.md`** - Solver configuration and parallel execution
- **`references/meshing.md`** - Complex mesh generation with Gmsh
- **`references/advanced_terms.md`** - Higher-order diffusion, nonlinear terms

## Common Issues

**Convergence failure:**
- Reduce time step
- Check boundary condition consistency
- Try different solver
- Examine mesh quality

**Parallel scaling issues:**
- Set `OMP_NUM_THREADS=1`
- Check mesh partitioning
- Verify MPI configuration
- Monitor load balance

**Memory issues:**
- Reduce mesh resolution
- Use iterative solvers
- Enable memory-efficient solvers

## Additional Resources

- Official documentation: https://www.ctcms.nist.gov/fipy/
- GitHub repository: https://github.com/usnistgov/fipy
- Examples: https://www.ctcms.nist.gov/fipy/examples.html
- Discussion forum: https://github.com/usnistgov/fipy/discussions
