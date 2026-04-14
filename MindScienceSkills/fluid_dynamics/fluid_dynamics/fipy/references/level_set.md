# Level Set Methods

Level set methods track moving interfaces implicitly using signed distance functions.

## Basic Level Set

The level set function φ(x,t) represents the interface as φ = 0 contour:

$$\frac{\partial \phi}{\partial t} + \mathbf{v} \cdot \nablaustphi = 0$$

**Implementation:**
```python
from fipy import CellVariable, Grid2D, TransientTerm, ConvectionTerm
import numpy as np

# Setup
mesh = Grid2D(nx=100, ny=100, dx=0.01, dy=0.01)
phi = CellVariable(mesh=mesh, name='level set')

# Initialize: circle of radius 0.3 centered at (0.5, 0.5)
X, Y = mesh.cellCenters
r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
phi.setValue(r - 0.3)

# Velocity field (expanding circle)
velocity = FaceVariable(mesh=mesh, value=(1., 1.), rank=1)

# Level set equation
eq = TransientTerm() + ConvectionTerm(coeff=velocity) == 0

# Time stepping
for step in range(100):
    eq.solve(var=phi, dt=0.001)
```

## Distance Function

Initialize level set as signed distance to interface:

```python
# For circle
X, Y = mesh.cellCenters
center_x, center_y = 0.5, 0.5
radius = 0.3

# Signed distance function
phi.setValue(np.sqrt((X - center_x)**2 + (Y - center_y)**2) - radius)

# For rectangle
x_min, x_max = 0.3, 0.7
y_min, y_max = 0.3, 0.7

phi.setValue(np.maximum.reduce([
    x_min - X,
    X - x_max,
    y_min - Y,
    Y - y_max
]))
```

## Reinitialization

Maintain φ as signed distance function:

$$\frac{\partial \phi}{\partial t} + \text{sign}(\phi_0)(|\nabla\phi| - 1) = 0$$

**Implementation:**
```python
def reinitialize(phi, iterations=5):
    """Reinitialize level set to signed distance function."""
    phi0 = phi.copy()
    
    for _ in range(iterations):
        # Compute sign function
        sign_phi = phi0 / np.sqrt(phi0**2 + 1e-6)
        
        # Compute gradient magnitude
        grad_mag = np.sqrt(phi.faceGrad[0]**2 + phi.faceGrad[1]**2)
        
        # Reinitialization equation
        eq = TransientTerm() == sign_phi * (grad_mag - 1)
        eq.solve(var=phi, dt=0.1)
```

## Velocity Extension

Extend velocity from interface to entire domain:

```python
def extend_velocity(phi, velocity):
    """Extend velocity field from interface to whole domain."""
    # Solve Laplace equation with boundary conditions at interface
    # to smoothly extend velocity
    
    vel_extended = velocity.copy()
    
    # Find interface cells
    interface = abs(phi) < dx
    
    # Extend using diffusion
    for i in range(velocity.rank):
        eq = DiffusionTerm() == 0
        vel_extended[i].constrain(velocity[i], where=interface)
        eq.solve(var=vel_extended[i])
    
    return vel_extended
```

## Curvature Calculation

Compute interface curvature from level set:

$$\kappa = \nabla \cdot \left( \frac{\nabla \phi}{|\nabla \phi|} \right)$$

```python
def compute_curvature(phi):
    """Compute curvature of level set interface."""
    grad_phi = phi.faceGrad
    grad_mag = np.sqrt(grad_phi[0]**2 + grad_phi[1]**2 + 1e-10)
    
    # Normal vector
    normal = grad_phi / grad_mag
    
    # Curvature = div(normal)
    curvature = normal.divergence
    
    return curvature
```

## Mean Curvature Flow

Interface motion by mean curvature:

$$\frac{\partial \phi}{\partial t} = \kappa |\nabla \phi|$$

```python
def mean_curvature_flow(phi, iterations=100, dt=0.001):
    """Evolve interface by mean curvature."""
    for step in range(iterations):
        # Compute curvature
        curvature = compute_curvature(phi)
        
        # Compute gradient magnitude
        grad_phi = phi.faceGrad
        grad_mag = np.sqrt(grad_phi[0]**2 + grad_phi[1]**2 + 1e-10)
        
        # Evolve
        eq = TransientTerm() == curvature * grad_mag
        eq.solve(var=phi, dt=dt)
```

## Applications

### Electrodeposition

```python
# See examples.levelSet.electroChem for complete implementation
# Level set tracks electrode-electrolyte interface
# Velocity determined by electric field
```

### Bubble Dynamics

```python
# Level set represents bubble interface
# Velocity includes surface tension and pressure
velocity = pressure_grad + surface_tension * curvature
```

### Crystal Growth

```python
# Level set for solid-liquid interface
# Velocity includes thermal undercooling and anisotropy
velocity = undercooling * mobility * (1 + anisotropy)
```

## Numerical Considerations

**Time step:**
- CFL condition: dt < dx / |v|
- For reinitialization: dt < dx

**Spatial resolution:**
- Interface should be resolved by several grid points
- Typical: 5-10 grid points across interface

**Stability:**
- Use upwind schemes for convection
- Reinitialize periodically to maintain signed distance
- Handle narrow band level sets for efficiency

## Narrow Band Level Set

For efficiency, only compute near interface:

```python
# Define narrow band
band_width = 5 * dx
narrow_band = abs(phi) < band_width

# Only solve in narrow band
eq = TransientTerm() + ConvectionTerm(coeff=velocity) == 0
eq.solve(var=phi, dt=0.001, where=narrow_band)
```

## Coupling with Physics

**Heat equation with moving boundary:**
```python
# Temperature field
T = CellVariable(mesh=mesh, name='temperature')

# Heat equation
heat_eq = TransientTerm() == DiffusionTerm(coeff=alpha)

# Level set evolution
velocity = normal * thermal_velocity
level_set_eq = TransientTerm() + ConvectionTerm(coeff=velocity) == 0

# Solve coupled
for step in range(1000):
    heat_eq.solve(var=T, dt=dt)
    level_set_eq.solve(var=phi, dt=dt)
```

## Visualization

```python
import matplotlib.pyplot as plt

# Plot level set contours
plt.contour(mesh.cellCenters[0], mesh.cellCenters[1], 
            phi, levels=[0], colors='red')
plt.contourf(mesh.cellCenters[0], mesh.cellCenters[1], 
             phi, levels=20, cmap='RdBu')
plt.colorbar()
plt.show()
```

## Common Issues

**Loss of signed distance property:**
- Reinitialize periodically
- Use proper time step

**Interface breakup/merger:**
- Handled automatically by level set method
- No explicit topology tracking needed

**Mass conservation:**
- Level set not inherently mass-conserving
- Use volume correction methods if needed

**Numerical dissipation:**
- Use higher-order schemes
- Ensure adequate resolution
