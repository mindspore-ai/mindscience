# Boundary Conditions

OpenFOAM provides comprehensive boundary condition support for various applications.

## Wall Boundaries

### Fixed Wall (No-Slip)

```python
# Fixed wall, no-slip
U = {'type': 'fixedValue', 'value': (0, 0, 0)}
```

### Moving Wall

```python
# Moving wall with velocity
U = {'type': 'fixedValue', 'value': (1, 0, 0)}
```

### Zero Gradient Wall

```python
# Zero gradient wall
U = {'type': 'zeroGradient'}
```

## Inlet/Outlet Boundaries

### Fixed Value Inlet

```python
# Fixed velocity inlet
U = {'type': 'fixedValue', 'value': (1, 0, 0)}
```

### Pressure Inlet

```python
# Pressure inlet
p_inlet = 101325  # Pa
```

### Outlet Boundary

```python
# Zero gradient outlet
U = {'type': 'zeroGradient'}
```

### Pressure Outlet

```python
# Pressure outlet
p_outlet = 101325  # Pa
```

## Cyclic Boundaries

### Cyclic AMI

```python
# Cyclic boundary
boundaryDict = {
    'type': 'cyclicAMI',
    'p': 101325  # Pa
}
```

### Cyclic AMI with Fan

```python
# Cyclic boundary with fan
boundaryDict = {
    'type': 'cyclicAMI',
    'fan': {
        'p': 101325,
        'origin': (0, 0, 0),
        'dir': (1, 0, 0),
        'distance': 0.1
    }
}
```

## Symmetry Plane

### Symmetry Plane

```python
# Symmetry plane
boundaryDict = {
    'type': 'symmetryPlane',
    'basePoint': (0, 0, 0),
    'normal': (1, 0, 0)
}
```

### Wedge

```python
# Wedge boundary
boundaryDict = {
    'type': 'wedge',
    'axis': 'y',
    'angle': 45  # degrees
}
```

## OpenFOAM Boundaries

### External Wall

```python
# External wall boundary
boundaryDict = {
    'type': 'externalWall',
    'patches': [
        {
            'name': 'myWall',
            'type': 'wall',
            'faces': 'f(xMin yMin yMin xMax)',
            'sampleMode': 'nearestPatch'
        }
    ]
}
```

### Patch Boundary

```python
# Patch boundary
boundaryDict = {
    'type': 'patch',
    'pName': 'myPatch',
    'sampleMode': 'nearestPatch'
}
```

### Mapped Wall

```python
# Mapped wall boundary
boundaryDict = {
    'type': 'mappedWall',
    'sampleMode': 'nearestCell'
}
```

## Boundary Condition Selection Guide

| Application | Recommended BC | Reason |
|--------------|----------------|---------|
| Incompressible flow | No-slip wall | Standard viscous flow |
| Free-surface flow | Zero gradient | Free surface condition |
| Inlet flow | Fixed value inlet | Prescribed inflow |
| Outlet flow | Zero gradient outlet | Open boundary |
| Rotating machinery | Cyclic AMI | Periodic motion |
| Symmetry plane | Symmetry plane | Mirror symmetry |
| External flow | External wall | Multi-region coupling |

## Boundary Condition Implementation

### in 0 File

```python
# 0 file boundary conditions
boundaryField {
    type            fixedValue;
    value           uniform (0 0 0);
}

inlet {
    type            fixedValue;
    value           uniform (1 0 0);
}

outlet {
    type            zeroGradient;
}
```

### Control Dictionary

```python
# Control dictionary
application {
    type            compressible;
    
    startFrom       latestTime;
    
    P {
        type            codedFixedValue;
        value           1e5;
    }
    
    U {
        type            fixedValue;
        value           uniform (0 0 0);
    }
}
```

### Dynamic Boundary Conditions

```python
# Time-dependent boundary
import math

def update_boundary(t):
    # Oscillating inlet velocity
    U_inlet = 1.0 + 0.5 * math.sin(2 * math.pi * t)
    return U_inlet
```

## Boundary Condition Best Practices

1. **Start simple**: Use fixed walls for initial testing
2. **Validate physics**: Ensure BCs match physical setup
3. **Check compatibility**: Verify BCs with solver choice
4. **Test convergence**: BCs can affect solver convergence
5. **Document choices**: Keep track of BC decisions
6. **Use appropriate types**: Match BC type to physics

## Common Issues and Solutions

### Boundary Artifacts

**Problem**: Unphysical behavior near boundaries

**Solutions**:
- Refine mesh near boundaries
- Use boundary layers
- Check boundary condition type
- Verify boundary values

### Mass Conservation Issues

**Problem**: Mass not conserved at boundaries

**Solutions**:
- Check inlet/outlet mass balance
- Verify boundary conditions
- Check time step size
- Monitor mass conservation

### Pressure Oscillations

**Problem**: Pressure field oscillations near boundaries

**Solutions**:
- Use appropriate pressure boundary conditions
- Improve mesh quality near boundaries
- Check solver settings
- Reduce time step size

### Convergence Failures

**Problem**: Solver fails to converge

**Solutions**:
- Check boundary condition compatibility
- Improve initial guess
- Relax boundary conditions gradually
- Try different solver

## Advanced Topics

- **Moving boundaries**: Time-dependent boundary conditions
- **Deforming boundaries**: Boundary motion and deformation
- **Multi-region boundaries**: Complex boundary setups
- **Boundary layer optimization**: Optimal layer thickness
- **Conjugate heat transfer**: Coupled thermal BCs
