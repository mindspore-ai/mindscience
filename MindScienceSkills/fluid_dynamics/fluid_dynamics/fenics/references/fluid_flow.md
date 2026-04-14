# Fluid Flow in FEniCS

Complete implementations of fluid flow equations using finite element method.

## Stokes Flow

Low Reynolds number flow (creeping flow):

$$-\nabla^2 \mathbf{u} + \nabla p = \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

### Taylor-Hood Elements

Inf-sup stable element pair (P2-P1):

```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Create mesh
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

# Taylor-Hood function space (P2-P1)
P2 = dfx.fem.functionspace(mesh, ("Lagrange", 2))
P1 = dfx.fem.functionspace(mesh, ("Lagrange", 1))
V = dfx.fem.functionspace(mesh, [P2, P1])

# Trial and test functions
(u, p) = ufl.TrialFunctions(V)
(v, q) = ufl.TestFunctions(V)

# Source term
f = dfx.fem.Constant(mesh, (0.0, 0.0))

# Variational form
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - ufl.div(v) * p + ufl.div(u) * q) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

# Boundary conditions
def walls(x):
    return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

def lid(x):
    return np.isclose(x[1], 1.0)

# No-slip on walls
u0 = dfx.fem.Function(P2)
u0.x.array[:] = 0.0
bc_walls = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(P2, walls))

# Moving lid
u_lid = dfx.fem.Function(P2)
u_lid.x.array[:] = 1.0
bc_lid = dfx.fem.dirichletbc(u_lid, dfx.fem.locate_dofs_geometrical(P2, lid))

bcs = [bc_walls, bc_lid]

# Solve
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h)
dfx.nls.petsc.solve(problem)

# Extract velocity and pressure
u_sol, p_sol = u_h.sub(0), u_h.sub(1)
```

### Block Preconditioner

Efficient solver for mixed problems:

```python
# PETSc options for block preconditioner
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

problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h, petsc_options=options)
dfx.nls.petsc.solve(problem)
```

## Navier-Stokes Flow

Incompressible flow at finite Reynolds number:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \nabla^2 \mathbf{u} + \nabla p = \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

### Semi-Discrete Formulation

Time-stepping with BDF1 (backward Euler):

```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Parameters
nu = 0.01              # Kinematic viscosity
T = 1.0                # Final time
dt = 0.01               # Time step
num_steps = int(T / dt)

# Create mesh and function space
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
P2 = dfx.fem.functionspace(mesh, ("Lagrange", 2))
P1 = dfx.fem.functionspace(mesh, ("Lagrange", 1))
V = dfx.fem.functionspace(mesh, [P2, P1])

# Trial and test functions
(u, p) = ufl.TrialFunctions(V)
(v, q) = ufl.TestFunctions(V)

# Previous time step
u_n = dfx.fem.Function(P2)

# Source term
f = dfx.fem.Constant(mesh, (0.0, 0.0))

# Variational form (BDF1)
a = (ufl.inner(u, v) * ufl.dx
      + dt * ufl.inner(ufl.grad(u), ufl.grad(u_n)) * v * ufl.dx
      + dt * nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
      - dt * ufl.div(v) * p * ufl.dx
      + dt * ufl.div(u) * q * ufl.dx)
L = ufl.inner(u_n, v) * ufl.dx + dt * ufl.inner(f, v) * ufl.dx

# Boundary conditions
def walls(x):
    return np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))

def lid(x):
    return np.isclose(x[1], 1.0)

u0 = dfx.fem.Function(P2)
u0.x.array[:] = 0.0
bc_walls = dfx.fem.dirichletbc(u0, dfx.fem.locate_dofs_geometrical(P2, walls))

u_lid = dfx.fem.Function(P2)
u_lid.x.array[:] = 1.0
bc_lid = dfx.fem.dirichletbc(u_lid, dfx.fem.locate_dofs_geometrical(P2, lid))

bcs = [bc_walls, bc_lid]

# Solve
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h, petsc_options=options)

# Time stepping
for step in range(num_steps):
    t = step * dt
    
    # Update boundary condition if time-dependent
    # u_lid.x.array[:] = np.sin(t)
    
    # Solve
    dfx.nls.petsc.solve(problem)
    
    # Update previous time step
    u_n.x.array[:] = u_h.sub(0).x.array[:]
    
    print(f"Step {step}/{num_steps}, t = {t:.3f}")
```

### Chorin's Projection Method

Splitting method for incompressible flow:

```python
# Step 1: Solve for intermediate velocity (without pressure)
# Step 2: Solve pressure Poisson equation
# Step 3: Correct velocity

# This is more complex but can be more efficient
# See FEniCS demos for complete implementation
```

### Oseen Equation

Linearization about a base flow:

$$-\nu \nabla^2 \mathbf{u} + \mathbf{u}_0 \cdot \nabla \mathbf{u} + \mathbf{u} \cdot \nabla \mathbf{u}_0 + \nabla p = \mathbf{f}$$

```python
# Base flow (e.g., from Stokes solution)
u0 = dfx.fem.Function(P2)

# Oseen linearization
a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v))
      + ufl.inner(ufl.grad(u), u0) * v
      + ufl.inner(ufl.grad(u0), u) * v
      - ufl.div(v) * p
      + ufl.div(u) * q) * ufl.dx
L = ufl.inner(f, v) * ufl.dx
```

## Darcy Flow

Flow through porous media:

$$\mathbf{u} = -\frac{K}{\mu} \nabla p$$

$$\nabla \cdot \mathbf{u} = 0$$

### Mixed Formulation

```python
# Parameters
K = 1.0          # Permeability
mu = 1.0          # Viscosity

# Mixed function space (RT1-P0)
RT1 = dfx.fem.functionspace(mesh, ("RT", 1))
P0 = dfx.fem.functionspace(mesh, ("DG", 0))
V = dfx.fem.functionspace(mesh, [RT1, P0])

(sigma, u) = ufl.TrialFunctions(V)
(tau, v) = ufl.TestFunctions(V)

f = dfx.fem.Constant(mesh, 0.0)

# Variational form
a = (ufl.inner(sigma, tau) * ufl.dx
      - ufl.div(tau) * u * ufl.dx
      - ufl.div(sigma) * v * ufl.dx)
L = f * v * ufl.dx

# Boundary conditions
# Pressure on left and right
p_left = dfx.fem.Function(P0)
p_left.x.array[:] = 1.0
bc_left = dfx.fem.dirichletbc(p_left, dfx.fem.locate_dofs_geometrical(P0, left_boundary))

p_right = dfx.fem.Function(P0)
p_right.x.array[:] = 0.0
bc_right = dfx.fem.dirichletbc(p_right, dfx.fem.locate_dofs_geometrical(P0, right_boundary))

bcs = [bc_left, bc_right]

# Solve
u_h = dfx.fem.Function(V)
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h)
dfx.nls.petsc.solve(problem)
```

## Boundary Conditions

### Dirichlet (Velocity)

```python
# Fixed velocity
u_fixed = dfx.fem.Function(P2)
u_fixed.x.array[:] = [1.0, 0.0]  # u = (1, 0)
bc = dfx.fem.dirichletbc(u_fixed, dfx.fem.locate_dofs_geometrical(P2, boundary))
```

### Neumann (Stress)

```python
# Natural boundary condition (traction)
# Add to right-hand side:
L += ufl.inner(traction, v) * ufl.ds
```

### Outflow Condition

```python
# Do-nothing boundary (natural)
# No explicit condition needed
```

### Symmetry Condition

```python
# Symmetry on bottom wall (y = 0)
def symmetry_boundary(x):
    return np.isclose(x[1], 0.0)

# u_y = 0, ∂u_x/∂y = 0
u_sym = dfx.fem.Function(P2)
u_sym.x.array[:] = 0.0
bc_sym = dfx.fem.dirichletbc(u_sym, dfx.fem.locate_dofs_geometrical(P2, symmetry_boundary))
```

## Post-Processing

### Velocity Magnitude

```python
# Compute velocity magnitude
u_sol = u_h.sub(0)
velocity_mag = dfx.fem.Function(P1)
velocity_mag.x.array[:] = np.sqrt(u_sol.x.array[0::2]**2 + u_sol.x.array[1::2]**2)
```

### Vorticity

```python
# Compute vorticity: ω = ∂v/∂x - ∂u/∂y
# Requires gradient computation
grad_u = ufl.grad(u_sol)
vorticity = grad_u[1, 0] - grad_u[0, 1]

# Project to function space
W = dfx.fem.functionspace(mesh, ("Lagrange", 1))
omega = dfx.fem.Function(W)
problem = dfx.fem.petsc.LinearProblem(
    ufl.inner(ufl.TrialFunction(W), ufl.TestFunction(W)) * ufl.dx,
    vorticity * ufl.TestFunction(W) * ufl.dx,
    [], omega
)
dfx.nls.petsc.solve(problem)
```

### Pressure Drop

```python
# Compute pressure drop across domain
p_sol = u_h.sub(1)
p_min = p_sol.x.array.min()
p_max = p_sol.x.array.max()
pressure_drop = p_max - p_min
print(f"Pressure drop: {pressure_drop:.6f}")
```

### Flow Rate

```python
# Compute flow rate through boundary
u_sol = u_h.sub(0)
nQ = ufl.inner(u_sol, dfx.fem.FacetNormal(mesh)) * ufl.ds

# Integrate
Q = dfx.fem.assemble_scalar(nQ)
print(f"Flow rate: {Q:.6f}")
```

## Stability Considerations

### CFL Condition

For explicit time stepping:

$$\Delta t < \frac{\Delta x}{|\mathbf{u}|_{\max}}$$

```python
# Estimate maximum velocity
u_max = np.max(np.abs(u_sol.x.array))
dx = mesh.geometry.x[:, 0].max() / 32  # Approximate cell size
dt_cfl = 0.5 * dx / u_max
print(f"CFL-limited dt: {dt_cfl:.6f}")
```

### Inf-Sup Condition

For mixed formulations, use stable element pairs:

- **Taylor-Hood (P2-P1)**: Stable for Stokes
- **MINI (P1bubble-P1)**: Stable alternative
- **RT1-P0**: Stable for Darcy flow

### Reynolds Number Effects

- **Low Re (< 1)**: Stokes flow (neglect convection)
- **Moderate Re (1-100)**: Steady Navier-Stokes
- **High Re (> 100)**: Unsteady, turbulence models needed

## Common Applications

### Lid-Driven Cavity

Classic benchmark problem:

```python
# See Taylor-Hood example above
# Moving lid at y = 1, stationary walls
```

### Channel Flow (Poiseuille)

```python
# Parabolic velocity profile
# u = (u_max/4h²) * (h² - y²)
```

### Flow Past Obstacle

```python
# Requires complex meshing with Gmsh
# See references/meshing.md
```

### Backward-Facing Step

```python
# Flow expansion, recirculation region
# Test for separation and reattachment
```

## Troubleshooting

### Pressure Oscillations

**Issue:** Pressure checkerboard pattern

**Solutions:**
- Use stable element pair (Taylor-Hood)
- Check inf-sup condition
- Use appropriate preconditioner

### Velocity Divergence

**Issue:** Non-zero divergence

**Solutions:**
- Check continuity equation formulation
- Verify boundary conditions
- Ensure proper element pairing

### Poor Convergence

**Issue:** Slow solver convergence

**Solutions:**
- Use block preconditioner
- Try different solver/preconditioner
- Check mesh quality
- Reduce time step for transient problems

### Mass Conservation Errors

**Issue:** Mass not conserved

**Solutions:**
- Verify divergence-free condition
- Check boundary conditions
- Use appropriate time stepping
- Monitor flow rate through boundaries
