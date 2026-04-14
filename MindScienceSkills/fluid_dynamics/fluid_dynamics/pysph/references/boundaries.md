# Boundary Conditions in PySPH

PySPH provides multiple boundary condition types for different scenarios.

## Solid Boundaries

### Generalized Wall Boundary Conditions

**Reference**: Adami et al. 2012, JCP

**Use case**: No-slip or free-slip walls, moving boundaries

**Characteristics**:
- Exact enforcement of wall conditions
- Support for moving walls
- Can include wall rotation
- Pressure boundary conditions included

**Implementation**:
```python
from pysph.sph.bc.inlet_outlet_manager import InletOutletManager

# Create wall boundary
wall = InletOutletManager(
    fname_wall=None,  # No wall file needed
    num_layers=2,     # Number of boundary layers
    rho0=1000.0,      # Reference density
    p0=0.0,           # Reference pressure
    u0=0.0, v0=0.0,   # Reference velocity
)
```

**Moving wall example**:
```python
# Moving wall with velocity
wall = InletOutletManager(
    num_layers=2,
    rho0=1000.0,
    p0=0.0,
    u0=1.0, v0=0.0    # Wall moving in x-direction
)
```

**Rotating wall example**:
```python
# Rotating wall
wall = InletOutletManager(
    num_layers=2,
    rho0=1000.0,
    p0=0.0,
    angular_velocity=1.0    # Rotation speed
)
```

## Inlet/Outlet Boundaries

### Inlet Boundary

**Use case**: Flow entering domain, prescribed inflow

**Characteristics**:
- Prescribed velocity or flow rate
- Can be time-dependent
- Supports various inflow profiles

**Implementation**:
```python
from pysph.sph.bc.inlet_outlet_manager import InletOutletManager

# Constant velocity inlet
inlet = InletOutletManager(
    fname_inlet='inlet_profile.h5',
    num_layers=2,
    rho0=1000.0,
    p0=0.0,
    u0=1.0, v0=0.0    # Inflow velocity
)
```

**Time-dependent inlet**:
```python
def inlet_velocity(t, dt):
    # Varying inflow velocity
    u0 = 1.0 + 0.5 * np.sin(2 * np.pi * t)
    return u0, 0.0

inlet = InletOutletManager(
    fname_inlet='inlet_profile.h5',
    num_layers=2,
    rho0=1000.0,
    p0=0.0,
    u0=0.0, v0=0.0
)

# Update inlet velocity during simulation
def update_inlet(dst, t, dt):
    u0, v0 = inlet_velocity(t, dt)
    # Update inlet particles
```

### Outlet Boundary

**Use case**: Flow leaving domain, open boundaries

**Characteristics**:
- Multiple outlet types available
- Zero-gradient or extrapolation
- Non-reflecting conditions

**Outlet types**:

1. **Do-nothing outlet**: Simple extrapolation
```python
outlet = InletOutletManager(
    outlet_type='do-nothing',
    num_layers=2
)
```

2. **Mirror outlet**: Reflective boundary
```python
outlet = InletOutletManager(
    outlet_type='mirror',
    num_layers=2
)
```

3. **Method of Characteristics**: Non-reflecting outlet
```python
outlet = InletOutletManager(
    outlet_type='moc',
    num_layers=2
)
```

## Periodic Boundaries

**Use case**: Infinite domains, repeated patterns

**Characteristics**:
- Particles wrap around boundaries
- Automatic ghost particle creation
- Efficient for large domains

**Implementation**:
```python
from pysph.base.nnps import DomainManager

# 2D periodic domain
domain = DomainManager(
    xmin=0.0, xmax=1.0,
    ymin=0.0, ymax=1.0,
    periodic_in_x=True,
    periodic_in_y=True
)

# Create NNPS with periodic domain
nps = LinkedListNNPS(
    dim=2,
    particles=[particles],
    radius_scale=3.0,
    domain=domain
)
```

**3D periodic example**:
```python
domain = DomainManager(
    xmin=0.0, xmax=1.0,
    ymin=0.0, ymax=1.0,
    zmin=0.0, zmax=1.0,
    periodic_in_x=True,
    periodic_in_y=True,
    periodic_in_z=True
)
```

## Free-Surface Boundaries

**Use case**: Open to atmosphere, free-surface flows

**Characteristics**:
- Zero pressure at free surface
- Automatic detection
- Particles can separate from domain

**Implementation**:
```python
# Free-surface particles have tag=0
# Pressure equation automatically sets p=0 for free surface

# In WCSPH formulation
class PressureEquation(Equation):
    def post_loop(self, d_idx, d_rho, d_p, d_h):
        # Equation of state
        d_p[d_idx] = c0 * (d_rho[d_idx]/rho0)**gamma

# Free-surface detection
# Particles with insufficient neighbors are marked as free surface
```

## Solid-Fluid Interaction

**Use case**: Fluid flowing around solid objects

**Characteristics**:
- Solid particles interact with fluid
- No-slip or free-slip conditions
- Moving solid objects

**Implementation**:
```python
# Create solid particle array
solid = get_particle_array(name='solid', x=sx, y=sy, m=sm, rho=rho_s)

# Create fluid particle array
fluid = get_particle_array(name='fluid', x=fx, y=fy, m=fm, rho=rho_f)

# Equations with solid-fluid interaction
equations = [
    Group(
        equations=[
            DensityEquation(dest='fluid', sources=['fluid', 'solid']),
            DensityEquation(dest='solid', sources=['fluid', 'solid'])
        ]
    ),
    Group(
        equations=[
            PressureEquation(dest='fluid', sources=['fluid']),
            AccelerationEquation(dest='fluid', sources=['fluid', 'solid'])
        ]
    )
]
```

**Moving solid example**:
```python
# Update solid particle positions
def move_solid(dst, t, dt):
    # Move solid particles
    velocity = 0.5
    dst.x += velocity * dt

equations = [
    Group(
        equations=[...],
        post=move_solid  # Update solid position after each step
    )
]
```

## Boundary Condition Selection Guide

| Boundary Type | Use Case | Recommended Method |
|--------------|-----------|-------------------|
| Stationary wall | No-slip wall | Generalized wall BC |
| Moving wall | Moving boundary | Generalized wall BC with velocity |
| Rotating wall | Rotating object | Generalized wall BC with angular velocity |
| Constant inflow | Prescribed inlet | Inlet with constant velocity |
| Time-varying inflow | Unsteady inlet | Inlet with time-dependent velocity |
| Open outlet | Flow leaving domain | Do-nothing or MOC outlet |
| Reflective outlet | Closed domain | Mirror outlet |
| Infinite domain | Periodic pattern | Periodic boundaries |
| Free surface | Open to atmosphere | WCSPH with p=0 detection |
| Solid object | Fluid-structure interaction | Solid particle array |

## Boundary Layer Particles

Multiple layers of boundary particles for accurate BCs:

```python
# Create 2 layers of boundary particles
wall = InletOutletManager(
    num_layers=2,  # Number of layers
    rho0=1000.0,
    p0=0.0
)
```

**Benefits**:
- More accurate gradient computation
- Better pressure boundary conditions
- Reduced boundary artifacts

**Trade-offs**:
- Increased particle count
- Higher computational cost
- More memory usage

## Boundary Condition Implementation

### Creating Boundary Particles

```python
from pysph.base.utils import get_boundary_particles

# Create wall boundary particles
wall_particles = get_boundary_particles(
    fluid_particles,
    x_min=0.0, x_max=1.0,
    y_min=0.0, y_max=0.1,
    create_boundary_at='y_min'
)
```

### Setting Boundary Properties

```python
# Set boundary particle properties
wall_particles.rho[:] = 1000.0  # Reference density
wall_particles.p[:] = 0.0      # Reference pressure
wall_particles.u[:] = 0.0       # No-slip velocity
wall_particles.v[:] = 0.0
```

### Free-Slip Conditions

```python
# Free-slip wall (zero normal velocity)
# Tangential velocity is free

class FreeSlipEquation(Equation):
    def loop(self, d_idx, d_au, d_u, d_v, s_m, s_u, s_v, s_idx, DWIJ):
        # Zero normal component, preserve tangential
        # Implementation depends on wall orientation
        pass
```

## Common Issues and Solutions

### Boundary Reflections

**Problem**: Unwanted reflections from boundaries

**Solutions**:
- Use non-reflecting outlet (MOC)
- Increase boundary layer count
- Use proper boundary conditions
- Check particle distribution near boundary

### Pressure Boundary Artifacts

**Problem**: Incorrect pressure near boundaries

**Solutions**:
- Increase boundary layers
- Use generalized wall BCs
- Ensure proper density computation
- Check kernel support near boundary

### Mass Loss/Gain

**Problem**: Total non-conservation at boundaries

**Solutions**:
- Verify boundary particle mass
- Check inlet/outlet mass balance
- Ensure proper time integration
- Validate boundary conditions

### Free-Surface Instability

**Problem**: Particles separate incorrectly

**Solutions**:
- Adjust free-surface detection threshold
- Use tensile instability correction
- Apply particle shifting
- Check neighbor count criterion

## Advanced Topics

- **Moving boundaries**: Time-dependent boundary conditions
- **Deforming boundaries**: Boundary motion and deformation
- **Multi-phase boundaries**: Interface between different fluids
- **Adaptive boundaries**: Dynamic boundary refinement
- **Boundary layer optimization**: Optimal layer count and distribution
