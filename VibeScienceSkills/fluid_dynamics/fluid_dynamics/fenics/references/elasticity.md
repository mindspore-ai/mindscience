# Elasticity in FEniCS

Linear and nonlinear elasticity using finite element method.

## Linear Elasticity

Hooke's law for small deformations:

$$\nabla \cdot \sigma = \mathbf{f}$$

$$\sigma = \lambda \text{tr}(\varepsilon) \mathbf{I} + 2\mu \varepsilon$$

$$\varepsilon = \frac{1}{2}(\nabla \mathbf{u} + \nabla \mathbf{u}^T)$$

### Plane Strain

```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Create mesh
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

# Vector function space (P1)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))

# Trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Material parameters
E = dfx.fem.Constant(mesh, 1.0e5)  # Young's modulus
nu_mat = dfx.fem.Constant(mesh, 0.3)  # Poisson's ratio

# Lame parameters
mu = E / (2 * (1 + nu_mat))
lmbda = E * nu_mat / ((1 + nu_mat) * (1 - 2 * nu_mat))

# Strain and stress
def epsilon(u):
    return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

def sigma(u):
    return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

# Source term (body force)
f = dfx.fem.Constant(mesh, (0.0, -9.81))  # Gravity

# Variational form
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Boundary conditions
def clamped_boundary(x):
    return np.isclose(x[0], 0.0)

u0 = dfx.fem.Function(V)
u0.x.array[:] = [0.0, 0.0]
bc = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(V, clamped_boundary))

# Solve
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, [bc], u_h)
dfx.nls.petsc.solve(problem)
```

### Plane Stress

```python
# Plane stress: different Lame parameters
mu = E / (2 * (1 + nu_mat))
lmbda = E * nu_mat / ((1 + nu_mat) * (1 - nu_mat))

# Rest of formulation same as plane strain
```

### 3D Elasticity

```python
# 3D mesh
mesh = dfx.mesh.create_unit_cube(MPI.COMM_WORLD, 16, 16, 16)

# 3D vector function space
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))

# Rest of formulation similar to 2D
```

## Nonlinear Elasticity

Finite deformations using Neo-Hookean material:

$$\mathbf{F} = \mathbf{I} + \nabla \mathbf{u}$$

$$\mathbf{C} = \mathbf{F}^T \mathbf{F}$$

$$\psi = \frac{\mu}{2}(\text{tr}(\mathbf{C}) - 3) - \mu \ln J + \frac{\lambda}{2}(\ln J)^2$$

where $J = \det(\mathbf{F})$

### Neo-Hookean Material

```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Create mesh
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))

# Functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
u_h = dfx.fem.Function(V)

# Material parameters
mu = dfx.fem.Constant(mesh, 10.0)
lmbda = dfx.fem.Constant(mesh, 100.0)

# Deformation gradient
F = ufl.Identity(len(u)) + ufl.grad(u)

# Right Cauchy-Green tensor
C = F.T * F

# Jacobian
J = ufl.det(F)

# Strain energy density (Neo-Hookean)
psi = (mu / 2) * (ufl.tr(C) - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2

# First Piola-Kirchhoff stress
P = ufl.diff(psi, F)

# Variational form
a = ufl.inner(P, ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Boundary conditions
def clamped_boundary(x):
    return np.isclose(x[0], 0.0)

u0 = dfx.fem.Function(V)
u0.x.array[:] = [0.0, 0.0]
bc = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(V, clamped_boundary))

# Solve with Newton
problem = dfx.nls.petsc.NewtonProblem(a, L, [bc], u_h)
solver = dfx.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
solver.solve(problem)
```

### Hyperelastic Materials

**Saint Venant-Kirchhoff:**
```python
# Linear strain energy
psi = (lmbda / 2) * ufl.tr(epsilon(u))**2 + mu * ufl.tr(epsilon(u) * epsilon(u))
```

**Mooney-Rivlin (incompressible rubber):**
```python
# Requires mixed formulation with pressure
# See mixed formulations section
```

## Boundary Conditions

### Clamped (Dirichlet)

```python
def clamped_boundary(x):
    return np.isclose(x[0], 0.0)

u0 = dfx.fem.Function(V)
u0.x.array[:] = [0.0, 0.0]
bc = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(V, clamped_boundary))
```

### Traction (Neumann)

```python
# Apply traction force on boundary
traction = dfx.fem.Constant(mesh, (1.0, 0.0))

# Add to right-hand side
L += ufl.inner(traction, v) * ufl.ds
```

### Point Load

```python
# Apply point load at specific location
# Requires careful handling (singularity)
# Use point source or distributed load over small area
```

### Symmetry

```python
# Symmetry on left boundary (x = 0)
def symmetry_boundary(x):
    return np.isclose(x[0], 0.0)

# u_x = 0, ∂u_y/∂x = 0
u_sym = dfx.fem.Function(V)
u_sym.x.array[:] = [0.0, 0.0]  # Fix x, allow y
bc_sym = dfx.fem.dirichletbc(u_sym, dfx.fem.locate_dofs_geometrical(V, symmetry_boundary))
```

## Post-Processing

### Displacement

```python
# Already computed as u_h
# Extract components
u_x = u_h.sub(0)
u_y = u_h.sub(1)
```

### Strain

```python
# Compute strain tensor
def compute_strain(u):
    return 0.5 * (ufl.grad(u) + ufl.grad(u).T)

# Project to function space
V_tensor = dfx.fem.functionspace(mesh, ("Lagrange", 1, (2, 2)))
strain = dfx.fem.Function(V_tensor)

# This requires careful handling of tensor projection
# See FEniCS documentation for tensor projection
```

### Stress

```python
# Compute stress tensor
def compute_stress(u):
    eps = 0.5 * (ufl.grad(u) + ufl.grad(u).T)
    return lmbda * ufl.tr(eps) * ufl.Identity(len(u)) + 2 * mu * eps

# Von Mises stress
def von_mises(stress):
    s11, s12 = stress[0, 0], stress[0, 1]
    s21, s22 = stress[1, 0], stress[1, 1]
    return ufl.sqrt(s11**2 - s11*s22 + s22**2 + 3*(s12**2 + s21**2))
```

### Energy

```python
# Compute strain energy
strain_energy = dfx.fem.assemble_scalar(
    0.5 * ufl.inner(sigma(u_h), epsilon(u_h)) * ufl.dx
)
print(f"Strain energy: {strain_energy:.6f}")
```

## Common Problems

### Cantilever Beam

```python
# Clamped at x = 0, load at x = L
def clamped(x):
    return np.isclose(x[0], 0.0)

def loaded(x):
    return np.isclose(x[0], 1.0)

# Apply load on right boundary
traction = dfx.fem.Constant(mesh, (0.0, -100.0))
L += ufl.inner(traction, v) * ufl.ds
```

### Plate with Hole

```python
# Requires complex meshing with Gmsh
# Circular hole in center of plate
# See references/meshing.md
```

### Block under Compression

```python
# Compressive load on top, fixed bottom
def bottom(x):
    return np.isclose(x[1], 0.0)

def top(x):
    return np.isclose(x[1], 1.0)

# Compressive traction
traction = dfx.fem.Constant(mesh, (0.0, -1000.0))
L += ufl.inner(traction, v) * ufl.ds
```

### Contact Problems

```python
# Requires variational inequalities
# Penalty method or Lagrange multipliers
# See FEniCS contact mechanics demos
```

## Numerical Considerations

### Locking Phenomenon

**Volumetric locking** for nearly incompressible materials:

**Solutions:**
- Use mixed formulation (u-p)
- Use higher-order elements (P2)
- Use reduced integration

**Shear locking** for thin structures:

**Solutions:**
- Use higher-order elements
- Use appropriate element types
- Consider shell/beam elements

### Mesh Quality

**Requirements:**
- Well-shaped elements (aspect ratio < 10)
- Sufficient resolution for stress concentrations
- Refined mesh near geometric features

### Solver Selection

**Linear elasticity:**
- CG + AMG for large problems
- Direct solvers for small problems

**Nonlinear elasticity:**
- Newton solver with line search
- Good initial guess important
- May require load stepping

## Applications

### Structural Mechanics

- Beam bending
- Plate vibration
- Shell analysis
- Frame structures

### Geomechanics

- Soil consolidation
- Rock mechanics
- Tunnel stability
- Slope stability

### Biomechanics

- Tissue deformation
- Bone mechanics
- Cardiovascular flow
- Cell mechanics

### Materials Science

- Crystal deformation
- Phase transformations
- Defect mechanics
- Thin film stress

## Troubleshooting

### Non-convergence

**Issue:** Newton solver fails to converge

**Solutions:**
- Improve initial guess
- Use line search
- Reduce load increments
- Check material parameters

### Stress Singularities

**Issue:** Infinite stress at corners/points

**Solutions:**
- Use mesh refinement near singularities
- Use graded meshes
- Apply distributed loads instead of point loads

### Poor Accuracy

**Issue:** Results don't match analytical solutions

**Solutions:**
- Increase mesh resolution
- Use higher-order elements
- Check boundary conditions
- Verify material parameters

### Memory Issues

**Issue:** Out of memory for large problems

**Solutions:**
- Use iterative solvers
- Reduce mesh resolution
- Use matrix-free methods
- Enable parallel execution

## Advanced Topics

### Mixed Formulation

For nearly incompressible materials:

```python
# Displacement-pressure formulation
V_u = dfx.fem.functionspace(mesh, ("Lagrange", 2, (2,)))
V_p = dfx.fem.functionspace(mesh, ("Lagrange", 1))
V = dfx.fem.functionspace(mesh, [V_u, V_p])

# (u, p) = TrialFunctions(V)
# (v, q) = TestFunctions(V)

# Mixed variational form
# See mixed formulations reference
```

### Plasticity

```python
# Requires history variables and return mapping
# See FEniCS plasticity demos
```

### Dynamic Problems

```python
# Add inertia term: ρ∂²u/∂t²
# Requires time-stepping scheme
# Newmark-beta or central difference
```

### Parallel Computation

```python
# Mesh automatically partitioned
# Use MPI for parallel execution
mpirun -np 4 python script.py
```
