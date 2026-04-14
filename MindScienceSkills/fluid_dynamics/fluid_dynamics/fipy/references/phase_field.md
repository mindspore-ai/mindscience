# Phase Field Methods

Phase field methods describe moving interfaces without explicit tracking using order parameters.

## Allen-Cahn Equation

Non-conserved order parameter evolution:

$$\frac{\partial \phi}{\partial t} = -L \frac{\delta F}{\delta \phi}$$

where $F$ is the free energy functional.

**Implementation:**
```python
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, ImplicitSourceTerm
import numpy as np

# Parameters
L = 1.0          # Mobility
epsilon = 0.01    # Interface width
A = 1.0           # Energy barrier height

# Setup
mesh = Grid2D(nx=100, ny=100, dx=0.01, dy=0.01)
phi = CellVariable(mesh=mesh, name='order parameter')

# Initial condition: random perturbation around 0.5
phi.setValue(0.5 + 0.1 * (np.random.random(mesh.numberOfCells) - 0.5))

# Equation: ∂φ/∂t = L[ε²∇²φ - f'(φ)]
# where f'(φ) = A * φ * (1 - φ) * (1 - 2φ)
eq = (TransientTerm()
      == DiffusionTerm(coeff=L * epsilon**2)
      - ImplicitSourceTerm(L * A * phi * (1 - phi) * (1 - 2 * phi)))

# Time stepping
for step in range(1000):
    eq.solve(var=phi, dt=0.0001)
```

**Key points:**
- Order parameter $\phi$ typically varies between 0 and 1
- Interface region has width proportional to $\epsilon$
- Energy functional drives phase separation
- Suitable for non-conserved order parameters (grain growth, solidification)

## Cahn-Hilliard Equation

Conserved order parameter evolution (spinodal decomposition):

$$\frac{\partial c}{\partial t} = \nabla \cdot \left( M \nabla \mu \right)$$

$$\mu = \frac{\delta F}{\delta c} = f'(c) - \kappa \nabla^2 c$$

**Implementation (coupled equations):**
```python
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm

# Parameters
M = 1.0          # Mobility
kappa = 0.01     # Gradient energy coefficient
A = 1.0           # Interaction parameter

# Setup
mesh = Grid2D(nx=100, ny=100, dx=0.01, dy=0.01)
c = CellVariable(mesh=mesh, name='concentration', value=0.5)
mu = CellVariable(mesh=mesh, name='chemical potential')

# Initial condition: random perturbation
c.setValue(0.5 + 0.1 * (np.random.random(mesh.numberOfCells) - 0.5))

# Coupled equations:
# 1) ∂c/∂t = ∇·(M∇μ)
# 2) μ = f'(c) - κ∇²c
#    where f'(c) = A * (2c - 1) * (2c - 1) * (2c - 1)

eq1 = TransientTerm(var=c) == DiffusionTerm(coeff=M, var=mu)
eq2 = (DiffusionTerm(coeff=kappa, var=c) + ImplicitSourceTerm(var=c)
       == mu - A * (2*c - 1)**3)

coupled_eq = eq1 & eq2

for step in range(1000):
    coupled_eq.solve(dt=0.0001)
```

**Key points:**
- Conserves total concentration/integrated order parameter
- Describes spinodal decomposition and phase separation
- Requires coupled equations for c and μ
- Suitable for binary alloys, polymer blends

## Multi-Component Systems

For multiple order parameters (e.g., multi-phase, polycrystalline):

```python
# N order parameters with constraint Σφᵢ = 1
phi = [CellVariable(mesh=mesh, name=f'phi{i}') for i in range(N)]

# Initialize with random values and normalize
for i in range(N):
    phi[i].setValue(np.random.random(mesh.numberOfCells))
sum_phi = sum(phi)
for i in range(N):
    phi[i].setValue(phi[i] / sum_phi)

# Coupled equations for each component
equations = []
for i in range(N):
    # Allen-Cahn with coupling terms
    eq_i = (TransientTerm(var=phi[i])
            == DiffusionTerm(coeff=L * epsilon**2, var=phi[i])
            - ImplicitSourceTerm(L * f_prime(phi[i]), var=phi[i]))
    
    # Add coupling terms
    for j in range(N):
        if i != j:
            eq_i += ImplicitSourceTerm(L * coupling(phi[i], phi[j]), var=phi[i])
    
    equations.append(eq_i)

coupled_eq = equations[0] & equations[1] & ... & equations[N-1]
```

## Anisotropy

For anisotropic interface energy (dendritic growth):

```python
# Anisotropic mobility
def anisotropic_mobility(gradient_angle):
    theta = np.arctan2(gradient_angle[1], gradient_angle[0])
    epsilon_bar = 0.01
    delta = 0.04
    return epsilon_bar * (1 + delta * np.cos(4 * theta))

# Use in diffusion term
epsilon = FaceVariable(mesh=mesh, value=epsilon_bar)
eq = TransientTerm() == DiffusionTerm(coeff=L * epsilon**2)
```

## Boundary Conditions

**Periodic boundaries (default for phase field):**
```python
# No explicit constraints needed for periodic
```

**No-flux boundaries:**
```python
phi.faceGrad.constrain(0., where=mesh.exteriorFaces)
```

**Fixed value at boundaries:**
```python
phi.constrain(0., where=mesh.facesLeft)
phi.constrain(1., where=mesh.facesRight)
```

## Numerical Considerations

**Interface resolution:**
- Grid spacing should resolve interface: dx < ε/2
- Typical: ε = 4-6 grid points

**Time step stability:**
- Explicit: dt < dx² / (4Lε²)
- Implicit: larger steps possible

**Initial conditions:**
- Random perturbations for spinodal decomposition
- Smooth interface profiles for nucleation problems

## Applications

- **Grain growth** - Allen-Cahn with multiple order parameters
- **Solidification** - Phase field with temperature coupling
- **Spinodal decomposition** - Cahn-Hilliard
- **Dendritic growth** - Anisotropic phase field
- **Microstructure evolution** - Multi-component phase field
