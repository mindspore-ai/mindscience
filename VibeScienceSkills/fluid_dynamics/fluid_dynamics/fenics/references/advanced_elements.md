# Advanced Finite Elements

Custom finite elements and discontinuous Galerkin methods in FEniCS.

## Discontinuous Galerkin (DG)

### DG for Poisson

```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Create mesh
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

# DG function space
V = dfx.fem.functionspace(mesh, ("DG", 1))

# Trial and test functions
u = = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Parameters
alpha = 1.0  # Penalty parameter
f = dfx.fem.Constant(mesh, 1.0)

# Interior and exterior facets
n = dfx.fem.FacetNormal(mesh)
h = dfx.fem.CellDiameter(mesh)

# Jump and average operators
def jump(u):
    return u('+') - u('-')

def avg(u):
    return 0.5 * (u('+') + u('-'))

# DG variational form
a = (ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
      - ufl.dot(avg(ufl.grad(u)), jump(v) * n * ufl.dS
      - ufl.dot(jump(u) * n, avg(ufl.grad(v)) * ufl.dS
      + alpha / h * ufl.dot(jump(u), jump(v)) * ufl.dS)
L = f * v * ufl.dx

# Boundary conditions
# Weakly enforced through DG formulation

# Solve
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, [], u_h)
dfx.nls.petsc.solve(problem)
```

### HDG (Hybridizable DG)

```python
# See FEniCS HDG demo
# Hybridizable formulation with static condensation
```

## Custom Elements

### Using Basix

```python
import dolfinx as dfx
import basix

# Define custom element
# See FEniCS custom elements demo
```

### TNT Elements

```python
# Tensor-product elements
# See FEniCS TNT elements demo
```

## Element Families

### Lagrange Variants

```python
# Standard Lagrange
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

# Serendipity P1
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, variant="serendipity"))

# Bubble elements
V = dfx.fem.functionspace(mesh, ("Bubble", 1))
```

### Vector Elements

```python
# 2D vector
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))

# 3D vector
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (3,)))
```

### Tensor Elements

```python
# 2D tensor (2x2)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (2, 2)))

# 3D tensor (3x3)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1, (3, 3)))
```

## H(div) Elements

### Raviart-Thomas (RT)

```python
# RT1 (lowest order)
V = dfx.fem.functionspace(mesh, ("RT", 1))

# RT2 (higher order)
V = dfx.fem.functionspace(mesh, ("RT", 2))
```

### BDM (Brezzi-Douglas-Marini)

```python
# BDM1
V = dfx.fem.functionspace(mesh, ("BDM", 1))

# BDM2
V = dfx.fem.functionspace(mesh, ("BDM", 2))
```

## Nédélec Elements

```python
# Nédélec first kind
V = dfx.fem.functionspace(mesh, ("N1curl", 1))

# Nédélec second kind
V = dfx.fem.functionspace(mesh, ("N2curl", 1))
```

## Element Properties

### Continuity

**Continuous (H1):**
```python
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))
```

**Discontinuous (L2):**
```python
V = dfx.fem.functionspace(mesh, ("DG", 0))
```

**H(div) conforming:**
```python
V = dfx.fem.functionspace(mesh, ("RT", 1))
```

**H(curl) conforming:**
```python
V = dfx.fem.functionspace(mesh, ("N1curl", 1))
```

### Order

```python
# P0 (constant)
V = dfx.fem.functionspace(mesh, ("Lagrange", 0))

# P1 (linear)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

# P2 (quadratic)
V = dfx.fem.functionspace(mesh, ("Lagrange", 2))

# P3 (cubic)
V = dfx.fem.functionspace(mesh, ("Lagrange", 3))
```

## Applications

### Mixed Problems

**Darcy flow:**
```python
# RT1-P0 elements
RT1 = dfx.fem.functionspace(mesh, ("RT", 1))
P0 = dfx.fem.functionspace(mesh, ("DG", 0))
V = dfx.fem.functionspace(mesh, [RT1, P0])
```

**Stokes flow:**
```python
# Taylor-Hood (P2-P1)
P2 = dfx.fem.functionspace(mesh, ("Lagrange", 2))
P1 = dfx.fem.functionspace(mesh, ("Lagrange", 1))
V = dfx.fem.functionspace(mesh, [P2, P1])
```

### Electromagnetics

**Maxwell's equations:**
```python
# Nédélec elements for electric field
V_E = dfx.fem.functionspace(mesh, ("N1curl", 1))

# Lagrange for magnetic field
V_H = dfx.fem.functionspace(mesh, ("Lagrange", 1))
```

### Transport Problems

**DG for convection:**
```python
# DG for upwind schemes
V = dfx.fem.functionspace(mesh, ("DG", 1))
```

## Advanced Topics

### Static Condensation

```python
# Eliminate interior DOFs
# See FEniCS static condensation demo
```

### hp-Adaptivity

```python
# Adapt mesh and element order
# Requires advanced implementation
```

### Spectral Elements

```python
# High-order elements on structured meshes
# See FEniCS spectral elements demo
```

## Common Issues

### Element Compatibility

**Issue:** Elements not compatible with problem type

**Solutions:**
- Check inf-sup condition for mixed problems
- Use appropriate element pairs
- Verify element continuity requirements

### Poor Convergence

**Issue:** Slow convergence with high-order elements

**Solutions:**
- Use appropriate preconditioners
- Check mesh quality
- Consider lower-order elements

### Implementation Errors

**Issue:** Custom element not working

**Solutions:**
- Check element definition
- Verify DOF mapping
- Consult Basix documentation

## Resources

- Basix documentation: https://docs.fenicsproject.org/basix/
- FEniCS custom elements demo: https://docs.fenicsproject.org/dolfinx/v0.10.0.post1/python/demos/demo_tnt-elements.html
- FEniCS HDG demo: https://docs.fenicsproject.org/dolfinx/v0.10.0.post1/python/demos/demo_hdg.html
