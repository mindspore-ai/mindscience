# Function Spaces

## Space Selection by Unknown Type

| Unknown | Function Space | UFL Element |
|---------|---------------|-------------|
| Scalar (temperature, pressure) | `FunctionSpace` | `"Lagrange", k` |
| Vector (displacement, velocity) | `VectorFunctionSpace` | `VectorElement("Lagrange", k)` |
| Tensor (stress, strain) | `TensorFunctionSpace` | `TensorElement("Lagrange", k)` |
| Mixed (Stokes, Navier-Stokes) | `FunctionSpace` with mixed element | `MixedElement([...])` |

## DOLFINx Space Creation

### Scalar Space

```python
V = fem.functionspace(mesh, ("Lagrange", 1))  # P1 (linear)
V = fem.functionspace(mesh, ("Lagrange", 2))  # P2 (quadratic)
V = fem.functionspace(mesh, ("DG", 0))        # Discontinuous Galerkin P0
```

### Vector Space

```python
V = fem.functionspace(mesh, ("Lagrange", 1))  # Returns VectorFunctionSpace for vector unknowns
# For explicit VectorElement:
V = fem.functionspace(mesh, (fem.ElementMetaObject("Lagrange", 1), 2))  # 2D vector
```

### Classic FEniCS Space Creation

```python
V = FunctionSpace(mesh, "P", 1)      # Scalar P1
W = VectorFunctionSpace(mesh, "P", 1)  # Vector P1
P = FunctionSpace(mesh, "P", 1)        # Scalar P1 for pressure (Stokes)
```

## Element Families

### Lagrange (Continuous Galerkin)

Standard continuous finite elements. Use for:
- Most elliptic problems
- Structural mechanics
- Heat transfer

| Order | DOF per element | Shape |
|-------|-----------------|-------|
| P0/DG0 | 1 | Constant |
| P1 | 3 (2D) / 4 (3D) | Linear |
| P2 | 6 (2D) / 10 (3D) | Quadratic |
| P3+ | Higher order | Cubic etc. |

### Discontinuous Galerkin (DG)

Discontinuous across element boundaries. Use for:
- Hyperbolic problems (advection, transport)
- Conservation laws
- High-order methods

### Real vs Virtual Spaces

For Raviart-Thomas (RT) or Nédélec elements used in mixed Poisson:

```python
# DOLFINx
RT = fem.functionspace(mesh, ("RT", 1))      # Raviart-Thomas (H(div))
N = fem.functionspace(mesh, ("N1curl", 1))    # Nédélec (H(curl))
```

## Mixed Function Spaces

For coupled problems where different fields need different spaces:

### DOLFINx Mixed Space

```python
P2 = fem.functionspace(mesh, ("Lagrange", 2))  # Velocity
P1 = fem.functionspace(mesh, ("Lagrange", 1))  # Pressure

W = fem.functionspace(mesh, ("Lagrange", 2))  # For vector (velocity)
W = fem.functionspace(mesh, ("Lagrange", 1))  # For scalar (pressure)
# Then use Function(W) and extract subspaces
```

### Classic FEniCS Mixed Space

```python
P2 = VectorElement("Lagrange", triangle, 2)
P1 = FiniteElement("Lagrange", triangle, 1)
ME = MixedElement([P2, P1])
W = FunctionSpace(mesh, ME)

u, p = TrialFunctions(W)  # Split the mixed function
v, q = TestFunctions(W)
```

## Subspace Extraction

### DOLFINx

```python
W = fem.functionspace(mesh, ("Lagrange", 2))
u = fem.Function(W)  # Vector function

# Extract scalar components for a mixed problem
u_sub = fem.Function(W.sub(0))  # First component
u_1 = u.sub(1)                   # Second component (lazy)
```

### Classic FEniCS

```python
u = Function(W)  # Mixed function
u0, u1 = u.split()  # Split into components
```

## Function Space for Boundary Conditions

For DirichletBC, the function space must match the trial space:

```python
V = fem.functionspace(mesh, ("Lagrange", 1))
u0 = fem.Function(V)
u_D = fem.Function(V)

# For vector problems
W = fem.functionspace(mesh, ("Lagrange", 1))  # Vector
bc = fem.dirichletbc(u_D, dofs, W)  # Apply to all DOFs
```

## DOF Location and Boundary Detection

### DOLFINX Boundary Finding

```python
import basix

# Find boundary facets
tdim = mesh.topology.dim
mesh.topology.create_entities(tdim - 1)
facet_markers = mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: ...)

# Locate DOFs on boundary facets
dofs = fem.locate_dofs_topological(W, tdim - 1, facet_markers)

# Apply BC
bc = fem.dirichletbc(value, dofs, W)
```

### Classic FEniCS Boundary Finding

```python
# Automatic boundary detection using a function
def boundary(x):
    return np.isclose(x[0], 0.0) or np.isclose(x[1], 0.0)

bc = DirichletBC(V, u_D, boundary)
```

## Space Compatibility Rules

| Form Type | Requires | Notes |
|-----------|----------|-------|
| `dot(grad(u), grad(v)) * dx` | u: scalar, v: scalar | Laplace operator |
| `inner(sigma(u), epsilon(v)) * dx` | u, v: vectors | Elasticity |
| `dot(grad(u), v) * dx` | u: scalar, v: vector | Advection |
| `div(u) * q * dx` | u: vector, q: scalar | Continuity |

## Rank and Shape Inspection

When debugging form errors, inspect the unknown:

```python
print(f"u rank: {u.function_space.value_shape}")
# () = scalar
# (2,) = 2D vector
# (3,) = 3D vector
# (2, 2) = 2D tensor
```
