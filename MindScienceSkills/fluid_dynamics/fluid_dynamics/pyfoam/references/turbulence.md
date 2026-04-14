# Turbulence Modeling

OpenFOAM provides comprehensive turbulence modeling capabilities.

## RAS Models

### Standard k-epsilon

**Description**: Standard k-epsilon model

**Configuration**:
```python
applicationDict = {
    'RAS': {
        'type': 'RASAS',
        'k': 0.41,
        'epsilon': 0.09,
        'Cmu': 0.09
    }
}
```

**Use case**: Standard incompressible flows, moderate turbulence

### Realizable k-epsilon

**Description**: Realizable k-epsilon model

**Configuration**:
```python
applicationDict = {
    'RAS': {
        'type': 'realizableKE',
        'k': 0.41,
        'epsilon': 0.09,
        'Cmu': 0.09
    }
}
```

**Use case**: High Reynolds number flows, better near-wall treatment

### k-omega SST

**Description**: Menter shear stress transport model

**Configuration**:
```python
applicationDict = {
    'RAS': {
        'type': 'kOmegaSST',
        'k': 0.41,
        'omega': 0.3,
        'Cmu': 0.09
    }
}
```

**Use case**: Separated flows, adverse pressure gradients, better boundary layer treatment

### Spalart Allmaras

**Description**: Spalart Allmaras model

**Configuration**:
```python
applicationDict = {
    'RAS': {
        'type': 'SpalartAllmaras',
        'k': 0.42,
        'epsilon': 0.09,
        'Cmu': 0.09
    }
}
```

**Use case**: Rotating flows, high accuracy requirements

## LES Modeling

### Smagorinsky

**Description**: Smagorinsky subgrid-scale model

**Configuration**:
```python
applicationDict = {
    'LES': True,
    'delta': 1.0,
    'pRef': 2.0
}
```

**Use case**: High Reynolds number flows, detailed turbulence structures

### Dynamic LES

**Description**: Dynamic k-equation model

**Configuration**:
```python
applicationDict = {
    'LES': True,
    'dynamicKEq': True,
    'pRef': 2.0
}
```

**Use case**: Transient flows, time-varying turbulence

### Wall-Modeled LES

**Description**: Wall-modeled LES for near-wall treatment

**Configuration**:
```python
applicationDict = {
    'LES': True,
    'wallModeled': True,
    'pRef': 2.0
}
```

**Use case**: Wall-bounded flows, accurate near-wall turbulence

## Detached Eddy Simulation (DES)

### Standard DES

**Description**: Standard detached eddy simulation

**Configuration**:
```python
applicationDict = {
    'LES': True,
    'DES': True,
    'delta': 1.0
}
```

**Use case**: Transitional flows, accurate turbulence prediction

### Delayed DES

**Description**: Delayed detached eddy simulation

**Configuration**:
```python
applicationDict = {
    'LES': True,
    'DES': True,
    'DESdelta': 0.65
}
```

**Use case**: Improved stability, reduced computational cost

## Turbulence Model Selection Guide

| Flow Type | Reynolds Number | Recommended Model | Reason |
|-----------|--------------|------------------|---------|
| Laminar | Re < 2300 | Laminar | Low turbulence |
| Transitional | 2300 < Re < 10000 | k-epsilon SST | Moderate turbulence |
| Turbulent | Re > 10000 | k-omega SST or Spalart | High turbulence |
| Wall-bounded | Any | Wall-modeled LES | Accurate near-wall |
| Rotating | Any | Spalart Allmaras | Rotating flows |

## Turbulence Model Performance

| Model | Computational Cost | Accuracy | Use Case |
|-------|-------------------|---------|---------|
| Laminar | Low | Good for low Re | Laminar flows |
| k-epsilon | Medium | Better accuracy | Moderate Re |
| k-omega SST | Medium-High | Good accuracy | High Re |
| Spalart Allmaras | High | Best accuracy | High Re, rotating |
| LES | High | Detailed turbulence | Very high Re |
| DES | Very High | Accurate transition | Transitional flows |

## Turbulence Damping

### Near-Wall Damping

```python
applicationDict = {
    'RAS': {
        'type': 'kOmegaSST',
        'k': 0.41,
        'omega': 0.3,
        'Cmu': 0.09,
        'damping': {
            'type': 'nearWallDamping',
            'scale': 0.5
        }
    }
}
```

### Low-Re Number Damping

```python
applicationDict = {
    'RAS': {
        'type': 'kOmegaSST',
        'k': 0.41,
        'omega': 0.3,
        'Cmu': 0.09,
        'damping': {
            'type': 'vanDriest',
            'scale': 0.01
        }
    }
}
```

## Turbulence Wall Functions

### Wall Functions

```python
applicationDict = {
    'RAS': {
        'type': 'kOmegaSST',
        'k': 0.41,
        'omega': 0.3,
        'Cmu': 0.09,
        'wallFunction': 'wallFunction'
    }
}
```

### Wall Function Specification

```python
# In 0 file
wallFunctionCoeffs {
    type: cubic;
    sampleValues (0 1 2 3 4 5);
    value uniform 1;
}

# In control dictionary
wallFunction {
    type: cubic;
    sampleValues (0 1 2 3 4 5);
    value uniform 1;
}
```

## Common Issues and Solutions

### Divergence with Turbulence

**Problem**: Simulation diverges with turbulence model

**Solutions**:
- Reduce time step size
- Improve mesh quality
- Check boundary conditions
- Try more stable turbulence model
- Use appropriate damping

### High Residuals

**Problem**: High residuals with turbulence

**Solutions**:
- Improve initial conditions
- Check mesh quality
- Verify turbulence model parameters
- Ensure proper near-wall treatment
- Reduce time step size

### Poor Turbulence Prediction

**Problem**: Turbulence model doesn't capture flow physics

**Solutions**:
- Try different turbulence model
- Adjust model parameters
- Improve mesh resolution
- Check Reynolds number
- Verify flow regime

## Advanced Topics

- **Zonal DES**: Zonal detached eddy simulation
- **Hybrid RAS-LES**: Combined RAS-LES model
- **Anisotropic models**: Direction-dependent turbulence
- **Compressibility corrections**: Turbulence-aware compressibility
- **Heat transfer turbulence**: Buoyancy-affected turbulence
- **Multiphase turbulence**: Multi-phase turbulence modeling
