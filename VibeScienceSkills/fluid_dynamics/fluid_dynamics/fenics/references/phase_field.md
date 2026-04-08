# Phase Field Methods in FEniCS

Implementation of phase field models using finite element method.

## Allen-Cahn Equation

Non-conserved order parameter evolution:

$$\frac{\partial \phi}{\partial t} = -L \frac{\delta F}{\delta \phi}$$

where $F[\phi] = \int_\Omega \left( \frac{\epsilon^2}{2} |\nabla \phi|^2 + f(\phi) \right) dx$

### Double-Well Potential

$$f(\phi) = A \phi^2 (1 - \phi)^2$$

Derivative: $f'(\phi) = 2A\phi(1 - \phi)(1 - 2\phi)$

### Implementation

```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Parameters
L = 1.0              # Mobility
epsilon = 0.01        # Interface width
A = 1.0               # Energy barrier height
T = 1.0                # Final time
dt = 0.001             # Time step
num_steps = int(T / dt)

# Create mesh
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

# Functions
phi = ufl.TrialFunction(V)
d = ufl.TestFunction(V)
phi_n = dfx.fem.Function(V)  # Previous time step
phi_h = dfx.fem.Function(V)  # Current solution

# Initial condition: random perturbation
np.random.seed(42)
phi_n.x.array[:] = 0.5 + 0.1 * (np.random.random(len(phi_n.x.array)) - 0.5)

# Variational form (BDF1)
# ∂φ/∂t = L[ε²∇²φ - f'(φ)]
f_prime = 2 * A * phi * (1 - phi) * (1 - 2 * phi)

a = (phi * d + dt * L * epsilon**2 * ufl.dot(ufl.grad(phi), ufl.grad(d))
      + dt * L * f_prime * d) * ufl.dx
L_form = phi_n * d * ufl.dx

# Solve
problem = dfx.fem.petsc.LinearProblem(a, L_form, [], phi_h)

for step in range(num_steps):
    t = step * dt
    dfx.nls.petsc.solve(problem)
    
    # Update previous time step
    phi_n.x.array[:] = phi_h.x.array[:]
    
    if step % 100 == 0:
        print(f"Step {step}/{num_steps}, t = {t:.3f}")
        print(f"φ range: [{phi_h.x.array.min():.3f}, {phi_h.x.array.max():.3f}]")
```

## Cahn-Hilliard Equation

Conserved order parameter (spinodal decomposition):

$$\frac{\partial c}{\partial t} = \nabla \cdot \left( M \nabla \mu \right)$$

$$\mu = \frac{\delta F}{\delta c} = f'(c) - \kappa \nabla^2 c$$

### Double-Well Potential

$$f(c) = A c^2 (1 - c)^2$$

Derivative: $f'(c) = 2A c (1 - c) (1 - 2c)$

### Implementation (Mixed Formulation)

```python
import dolfinx as dfx
from mpi4py import MPI
import ufl
import numpy as np

# Parameters
M = 1.0              # Mobility
kappa = 0.01          # Gradient energy coefficient
A = 1.0               # Interaction parameter
T = 1.0                # Final time
dt = 0.0001            # Time step
num_steps = int(T / dt)

# Create mesh
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))

# Mixed function space for c and μ
W = dfx.fem.functionspace(mesh, [V, V])

# Trial and test functions
(c, mu) = ufl.TrialFunctions(W)
(d, e) = ufl.TestFunctions(W)

# Previous time step
c_n = dfx.fem.Function(V)
w_h = dfx.fem.Function(W)

# Initial condition: random perturbation
np.random.seed(42)
c_n.x.array[:] = 0.5 + 0.1 * (np.random.random(len(c_n.x.array)) - 0.5)

# Free energy derivative
f_prime = 2 * A * c * (1 - c) * (1 - 2 * c)

# Coupled variational form
# Equation 1: ∂c/∂t = ∇·(M∇μ)
a1 = (c * d + dt * M * ufl.dot(ufl.grad(mu), ufl.grad(d))) * ufl.dx
L1 = c_n * d * ufl.dx

# Equation 2: μ = f'(c) - κ∇²c
a2 = (mu * e + kappa * ufl.dot(ufl.grad.grad(c), ufl.grad(e))
      - f_prime * e) * ufl.dx
L2 = 0 * e * ufl.dx

# Combine
a = a1 + a2
L_form = L1 + L2

# Solve with block preconditioner
options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
"
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "jacobi",
    "ksp_rtol": 1e-10
}

problem = dfx.fem.petsc.LinearProblem(a, L_form, [], w_h, petsc_options=options)

for step in range(num_steps):
    t = step * dt
    dfx.nls.petsc.solve(problem)
    
    # Extract c and update previous time step
    c_h = w_h.sub(0)
    c_n.x.array[:] = c_h.x.array[:]
    
    if step % 100 == 0:
        print(f"Step {step}/{num_steps}, t = {t:.3f}")
        print(f"c range: [{c_h.x.array.min():.3f}, {c_h.x.array.max():.3f}]")
```

## Multi-Component Phase Field

Multiple order parameters with constraint $\sum_i \phi_i = 1$:

```python
# N order parameters
N = 3

# Create function space
V = dfx.fem.functionspace(mesh, ("Lagrange", 1))
W = dfx.fem.functionspace(mesh, [V] * N)

# Trial and test functions
phis = ufl.TrialFunctions(W)
ds = ufl.TestFunctions(W)

# Lagrange multiplier for constraint
V_lagrange = dfx.fem.functionspace(mesh, ("Lagrange", 1))
lambda_ = ufl.TrialFunction(V_lagrange)
nu = ufl.TestFunction(V_lagrange)

# Coupled equations
equations = []
for i in range(N):
    # Allen-Cahn for each component
    phi_i = phis[i]
    d_i = ds[i]
    
    f_prime_i = 2 * A * phi_i * (1 - phi_i) * (1 - 2 * phi_i)
    
    # Add coupling terms
    coupling = 0
    for j in range(N):
        if i != j:
            coupling += phi_j**2
    
    a_i = (phi_i * d_i + dt * L * epsilon**2 * ufl.dot(ufl.grad(phi_i), ufl.grad(d_i))
           + dt * L * (f_prime_i + coupling) * d_i) * ufl.dx
    L_i = phi_n_i * d_i * ufl.dx
    
    equations.append(a_i)
    equations.append(L_i)

# Add constraint equation: Σφᵢ = 1
constraint_eq = (sum(phis) * nu - 1.0 * nu) * ufl.dx

# Combine all equations
a = sum(equations[:-1:2]) + constraint_eq
L_form = sum(equations[1::2])
```

## Anisotropy

For dendritic growth with anisotropic interface energy:

```python
# Anisotropic mobility
def anisotropic_mobility(grad_phi):
    theta = ufl.atan_2(grad_phi[1], grad_phi[0])
    epsilon_bar = 0.01
    delta = 0.04
    return epsilon_bar * (1 + delta * ufl.cos(4 * theta))

# Use in variational form
mobility = anisotropic_mobility(ufl.grad(phi))
a = (phi * d + dt * L * mobility * ufl.dot(ufl.grad(phi), ufl.grad(d))) * ufl.dx
```

## Boundary Conditions

### Periodic Boundaries

```python
# FEniCS handles periodic boundaries through mesh
# Specify periodic boundary conditions in mesh generation
```

### No-Flux Boundaries

```python
# Natural boundary condition (Neumann = 0)
# No explicit condition needed
```

### Fixed Value Boundaries

```python
def boundary(x):
    return np.isclose(x[0], 0.0)

phi0 = dfx.fem.Function(V)
phi0.x.array[:] = 0.0
bc = dfx.fem.dirichletbc(phi0, dfx.fem.locate_dofs_geometrical(V, boundary))
```

## Numerical Considerations

### Interface Resolution

Grid spacing should resolve interface:

$$\Delta x < \frac{\epsilon}{2}$$

Typical: $\epsilon = 4-6$ grid points

### Time Step Stability

**Explicit:**
$$\Delta t < \frac{\Delta x^2}{4 L \epsilon^2}$$

**Implicit (BDF1):**
Larger time steps possible

### Initial Conditions

**Random perturbations:**
```python
phi_n.x.array[:] = 0.5 + 0.1 * (np.random.random(len(phi_n.x.array)) - 0.5)
```

**Smooth interface:**
```python
# Initialize with tanh profile
X, Y = mesh.geometry.x[:, 0], mesh.geometry.x[:, 1]
r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
phi_n.x.array[:] = 0.5 * (1 - np.tanh((r - 0.3) / epsilon))
```

## Post-Processing

### Interface Detection

```python
# Find interface cells (φ ≈ 0.5)
interface_cells = np.abs(phi_h.x.array - 0.5) < 0.1
print(f"Interface cells: {np.sum(interface_cells)}")
```

### Phase Fraction

```python
# Compute volume fraction of each phase
phase1_fraction = np.mean(phi_h.x.array > 0.5)
phase2_fraction = 1.0 - phase1_fraction
print(f"Phase 1: {phase1_fraction:.3f}, Phase 2: {phase2_fraction:.3f}")
```

### Interface Area

```python
# Compute interface area (gradient magnitude)
grad_phi = ufl.grad(phi_h)
interface_area = dfx.fem.assemble_scalar(ufl.sqrt(ufl.inner(grad_phi, grad_phi)) * ufl.dx)
print(f"Interface area: {interface_area:.6f}")
```

### Free Energy

```python
# Compute total free energy
F = dfx.fem.assemble_scalar(
    (0.5 * epsilon**2 * ufl.inner(ufl.grad(phi_h), ufl.grad(phi_h))
     + A * phi_h**2 * (1 - phi_h)**2) * ufl.dx
)
print(f"Free energy: {F:.6f}")
```

## Applications

### Grain Growth

Multiple order parameters with constraint:

```python
# See multi-component example above
# Each order parameter represents a grain orientation
```

### Spinodal Decomposition

Cahn-Hilliard equation:

```python
# See Cahn-Hilliard example above
# Phase separation in binary alloys
```

### Dendritic Growth

Anisotropic Allen-Cahn:

```python
# See anisotropy example above
# Coupled with heat equation
```

### Solidification

Coupled phase field and heat:

```python
# Temperature field
T = ufl.TrialFunction(V_T)
s = ufl.TestFunction(V_T)

# Coupled system:
# 1) Phase field evolution
# 2) Heat equation with latent heat

# Latent heat release
L = dfx.fem.Constant(mesh, 1.0)
latent_heat = L * (phi - phi_n) / dt

# Heat equation with latent heat
a_T = (T * s + alpha * dt * ufl.dot(ufl.grad(T), ufl.grad(s))) * ufl.dx
L_T = (T_n * s + dt * latent_heat * s) * ufl.dx
```

## Common Issues

### Mass Conservation (Cahn-Hilliard)

**Issue:** Total mass not conserved

**Solutions:**
- Use mixed formulation
- Ensure proper time stepping
- Check numerical integration accuracy

### Interface Smearing

**Issue:** Interface too diffuse

**Solutions:**
- Reduce epsilon
- Increase mesh resolution
- Check time step

### Unphysical Oscillations

**Issue:** Spurious oscillations near interface

**Solutions:**
- Reduce time step
- Use stable time stepping
- Check mesh quality

### Slow Coarsening

**Issue:** Phase separation too slow

**Solutions:**
- Increase mobility
- Check energy parameters
- Verify initial conditions

## Advanced Topics

### Adaptive Mesh Refinement

Refine mesh near interface:

```python
# Requires mesh adaptation capabilities
# See FEniCS documentation for details
```

### Parallel Computation

```python
# Mesh automatically partitioned
# Use MPI for parallel execution
mpirun -np 4 python script.py
```

### Higher-Order Elements

```python
# Use P2 or P3 for better accuracy
V = dfx.fem.functionspace(mesh, ("Lagrange", 2))
```

### Vector-Valued Phase Fields

```python
# For vector order parameters (e.g., liquid crystals)
V_vec = dfx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
```
