# Boundary Conditions

## BC Types in FEniCS

| BC Type | Application | Notes |
|---------|-------------|-------|
| Dirichlet | u = g on ∂Ω | Strong enforcement |
| Neumann | ∂u/∂n = g on ∂Ω | Natural BC in weak form |
| Robin | αu + β∂u/∂n = g | Mixed BC |
| Periodic | u(x) = u(x+L) | Cyclic geometry |

## Dirichlet BCs in DOLFINx

```python
from dolfinx import fem
import numpy as np

V = fem.functionspace(mesh, ("Lagrange", 1))

# Create a function for the boundary value
u_D = fem.Function(V)
u_D.x.array[:] = 1.0  # Set all to 1.0

# Locate boundary DOFs
import basix
tdim = mesh.topology.dim
mesh.topology.create_entities(tdim - 1)

def boundary_marker(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)

facets = mesh.locate_entities_boundary(mesh, tdim - 1, boundary_marker)
dofs = fem.locate_dofs_topological(V, tdim - 1, facets)

# Apply DirichletBC
bc = fem.dirichletbc(u_D, dofs)
```

## Dirichlet BCs in Classic FEniCS

```python
V = FunctionSpace(mesh, "P", 1)

# Constant boundary value
u_D = Constant(domain, 0.0)
bc = DirichletBC(V, u_D, "on_boundary")

# Function-based boundary
def boundary(x):
    return x[0] < 1e-10

bc = DirichletBC(V, u_D, boundary)
```

## Spatially Varying BCs

### DOLFINx

```python
u_D = fem.Function(V)

# Set values based on coordinates
@fem.function.spatial_entity(mesh, 0)  # Node-wise
def u_D_expr(x):
    return x[1]**2  # Parabolic inlet profile

# Or directly set DOF values after locating
dofs = fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
u_D.x.array[dofs] = evaluate_at_coords(coords[dofs])
```

### Classic FEniCS

```python
class InletProfile(Expression):
    def __init__(self, mesh, **kwargs):
        self.mesh = mesh
        super().__init__(**kwargs)

    def eval(self, x, values):
        r = sqrt(x[1]**2 + x[2]**2)
        values[0] = 1.0 - r / 0.5  # parabolic

u_inlet = InletProfile(mesh, degree=2)
bc = DirichletBC(V, u_inlet, inlet_facet_ids)
```

## Neumann BCs (Natural)

Neumann BCs appear as **surface integrals** in the weak form:

Weak form of `-∇²u = f` with `∂u/∂n = g` on ∂Ω:

```
∫Ω ∇u·∇v dx = ∫Ω f v dx + ∫∂Ω g v ds
```

```python
# The linear form L includes the Neumann term
g = Constant(mesh, 0.0)  # or spatially varying
L = f * v * dx + g * v * ds
```

## Robin BCs

For `αu + β∂u/∂n = g`:

```python
alpha = Constant(mesh, 1.0)
beta = Constant(mesh, 0.1)
g_robin = Constant(mesh, 0.0)

# In weak form:
a = dot(grad(u), grad(v)) * dx + (alpha/beta) * u * v * dx
L = (g_robin/beta) * v * ds
```

## Periodic BCs

### DOLFINx Periodicity

```python
from dolfinx import DirichletBC

# Create a periodic boundary map
def periodic_boundary(x):
    return np.isclose(x[0], 1.0)

def periodic_relation(x):
    out = x.copy()
    out[:, 0] = 0.0  # Map x=1 to x=0
    return out

V = fem.functionspace(mesh, ("Lagrange", 1))
bc = DirichletBC(V, u0, [], periodic_relation=periodic_relation)
```

### Classic FEniCS Periodicity

```python
class PeriodicBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0)

    def map(self, x, y):
        y[0] = x[0] - 1.0

pb = PeriodicBoundary()
V = FunctionSpace(mesh, "P", 1, constrained_domain=pb)
```

## Multiple Boundary Conditions

For different BCs on different parts of the boundary:

```python
# Locate facets on left boundary
left_facets = mesh.locate_entities_boundary(mesh, tdim - 1,
    lambda x: np.isclose(x[0], 0.0))
left_dofs = fem.locate_dofs_topological(V, tdim - 1, left_facets)
bc1 = fem.dirichletbc(u_left, left_dofs, V)

# Locate facets on right boundary
right_facets = mesh.locate_entities_boundary(mesh, tdim - 1,
    lambda x: np.isclose(x[0], 1.0))
right_dofs = fem.locate_dofs_topological(V, tdim - 1, right_facets)
bc2 = fem.dirichletbc(u_right, right_dofs, V)

# Collect as list
bcs = [bc1, bc2]
```

## Applying BCs to the Linear System

### DOLFINx with LinearProblem

```python
problem = fem.petsc.LinearProblem(a, L, bcs=[bc1, bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u = problem.solve()
```

### Classic FEniCS

```python
solve(a == L, u, bcs)
```

### Manual Assembly + BC Application

```python
# Assemble matrix and vector
A = fem.assemble_matrix(a, bcs=bcs)
b = fem.assemble_vector(L)

# Apply BCs manually (for more control)
fem.apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.set_bc(b, bcs)
```

## BC Enforcement Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Elimination (default) | Modify matrix rows/cols | Standard Dirichlet |
| Lifting | Add contributions to RHS | Weak enforcement |
| Penalty | Add large term to diagonal | XFEM, when DOFs unknown |

FEniCS uses elimination by default for DirichletBC.

## Under-Constrained PDEs

If the PDE needs boundary conditions but you cannot specify them:

```
Solver will converge to a solution + constant
The constant is determined by the null space
```

For the Poisson equation with no BCs:
- Solution is unique only up to an additive constant
- Use `FunctionSpace(mesh, "P", 1, constraints=None)` and add a Lagrange multiplier

For Darcy flow or other saddle-point problems, verify that:
- The kernel (null space) is accounted for
- Pressure is grounded at one point
