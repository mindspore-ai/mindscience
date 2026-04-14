# Fluid Flow

Implementation of fluid flow equations in FiPy.

## Stokes Flow

For low Reynolds number flow (creeping flow):

$$\nabla \cdot \mathbf{u} = 0$$

$$-\nabla p + \mu \nabla^2 \mathbf{u} = \mathbf{f}$$

**Implementation (lid-driven cavity):**
```python
from fipy import CellVariable, Grid2D, DiffusionTerm, ImplicitSourceTerm
from fipy.tools import numerix as nx

# Parameters
mu = 1.0          # Viscosity
L = 1.0            # Domain Cavity size
N = 50             # Grid resolution

# Setup
mesh = Grid2D(nx=N, ny=N, dx=L/N, dy=L/N)

# Variables
u = CellVariable(mesh=mesh, name='u velocity', rank=1)
p = CellVariable(mesh=mesh, name='pressure')
pCorrection = CellVariable(mesh=mesh, name='pressure correction')

# Boundary conditions
# Lid moving at top
u.constrain((1., 0.), where=mesh.facesTop)
# No-slip on other walls
u.constrain((0., 0.), where=mesh.facesBottom)
u.constrain((0., 0.), where=mesh.facesLeft)
u.constrain((0., 0.), where=mesh.facesRight)

# Pressure reference (arbitrary constant)
p.constrain(0., where=mesh.facesLeft)

# Coupled Stokes equations
# Using SIMPLE algorithm approach
from fipy import FaceVariable

# Face-centered velocity
uFace = FaceVariable(mesh=mesh, value=(0., 0.), rank=1)

# Pressure gradient
pressureGrad = p.faceGrad

# Momentum equation (simplified)
# For full implementation, see examples.flow.stokesCavity
```

**Full implementation details:**
See `examples.flow.stokesCavity` in FiPy examples for complete Stokes flow solver with SIMPLE algorithm.

## Navier-Stokes Flow

For higher Reynolds number flows:

$$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}$$

$$\nabla \cdot \mathbf{u} = 0$$

**Implementation approach:**
```python
from fipy import TransientTerm, ConvectionTerm, DiffusionTerm

# Parameters
rho = 1.0          # Density
nu = 0.1           # Kinematic viscosity

# Velocity and pressure
u = CellVariable(mesh=mesh, name='velocity', rank=1)
p = CellVariable(mesh=mesh, name='pressure')

# Momentum equation
momentum_eq = (TransientTerm()
              + ConvectionTerm(coeff=u)
              == DiffusionTerm(coeff=nu)
              - p.grad)

# Continuity equation (incompressibility)
# Requires pressure-velocity coupling
```

**Key considerations:**
- Requires pressure-velocity coupling (SIMPLE, PISO, or projection methods)
- Time step limited by CFL condition: dt < dx / |u|
- For steady state, use iterative methods

## Darcy Flow

Flow through porous media:

$$\mathbf{u} = -\frac{K}{\mu} \nabla p$$

$$\nabla \cdot \mathbf{u} = 0$$

**Implementation:**
```python
# Parameters
K = 1.0           # Permeability
mu = 1.0           # Viscosity

# Pressure
p = CellVariable(mesh=mesh, name='pressure')

# Darcy's law
velocity = -(K/mu) * p.grad

# Continuity equation
eq = (-(K/mu) * p.grad).divergence == 0

# Boundary conditions
p.constrain(1.0, where=mesh.facesLeft)    # High pressure
p.constrain(0.0, where=mesh.facesRight)   # Low pressure

eq.solve(var=p)
```

## Brinkman Flow

Combines Darcy and Stokes flow:

$$\frac{\mu}{K} \mathbf{u} = -\nabla p + \mu \nabla^2 \mathbf{u}$$

```python
# Parameters
mu = 1.0           # Viscosity
K = 0.1            # Permeability

# Variables
u = CellVariable(mesh=mesh, name='velocity', rank=1)
p = CellVariable(mesh=mesh, name='pressure')

# Brinkman equation
eq_momentum = (mu/K * u
               == -p.grad
               + mu * u.divergence)

# Continuity
eq_continuity = u.divergence == 0

# Solve coupled system
```

## Boundary Conditions for Flow

**No-slip (velocity = 0):**
```python
u.constrain((0., 0.), where=mesh.facesBottom)
```

**Moving wall:**
```python
u.constrain((1., 0.), where=mesh.facesTop)
```

**Inlet (prescribed velocity):**
```python
u.constrain((1., 0.), where=mesh.facesLeft)
```

**Outlet (zero pressure gradient):**
```python
p.faceGrad.constrain(0., where=mesh.facesRight)
```

**Symmetry:**
```python
u[0].faceGrad.constrain(0., where=mesh.facesBottom)  # ∂u/∂y = 0
u[1].constrain(0., where=mesh.facesBottom)              # v = 0
```

## Solver Strategies

**SIMPLE Algorithm (Semi-Implicit Method for Pressure-Linked Equations):**
1. Solve momentum equation with guessed pressure
2. Solve pressure correction equation
3. Correct velocity and pressure
4. Repeat until convergence

**PISO Algorithm (Pressure Implicit with Splitting of Operators):**
- Similar to SIMPLE but with multiple pressure corrections
- Better for transient flows

**Projection Method:**
- Solve intermediate velocity
- Project onto divergence-free field
- Update pressure

## Common Applications

**Lid-driven cavity:**
- Classic benchmark problem
- Tests pressure-velocity coupling
- Available in `examples.flow.stokesCavity`

**Channel flow:**
- Poiseuille flow for analytical validation
- Useful for testing boundary conditions

**Flow past obstacle:**
- Requires complex meshing (Gmsh)
- Tests mesh handling and boundary conditions

**Porous media flow:**
- Darcy or Brinkman equations
- Applications in groundwater, oil reservoirs

## Performance Tips

**For large problems:**
- Use PETSc or Trilinos solvers
- Enable parallel execution
- Use appropriate preconditioners

**For steady-state:**
- Use implicit solvers
- Consider continuation methods for high Reynolds

**For transient:**
- Use adaptive time stepping
- Monitor CFL condition
- Consider operator splitting

## Validation

**Compare with analytical solutions:**
- Poiseuille flow: parabolic velocity profile
- Couette flow: linear velocity profile
- Stokes flow: exact solutions for simple geometries

**Check mass conservation:**
```python
# Verify divergence-free condition
divergence = u.divergence
print(f"Max divergence: {abs(divergence).max()}")
```

**Check boundary conditions:**
```python
# Verify no-slip at walls
print(f"Velocity at bottom: {u[mesh.facesBottom]}")
```
