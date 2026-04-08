# SPH Formulations

PySPH provides multiple SPH formulations for different physics problems.

## Weakly Compressible SPH (WCSPH)

**Reference**: Gesteira et al. 2010, Journal of Hydraulic Research

**Use case**: Free-surface flows, dam break, wave-structure interaction

**Characteristics**:
- Explicit time integration
- Equation of state for pressure
- Suitable for low Mach number flows
- Robust for free-surface problems

**Example usage**:
```python
from pysph.sph.wc.basic import get_density_equations, get_pressure_equations

equations = [
    get_density_equations(dest='fluid', sources=['fluid']),
    get_pressure_equations(dest='fluid', sources=['fluid'])
]
```

## Transport Velocity Formulation (TVF)

**Reference**: Adami et al. 2013, JCP

**Use case**: Incompressible flows, lid-driven cavity, internal flows

**Characteristics**:
- Exact enforcement of incompressibility
- Transport velocity eliminates tensile instability
- Higher accuracy for velocity field
- More computationally expensive

**Example usage**:
```python
from pysph.sph.tvf.transport_velocity import get_equations

equations = get_equations(dest='fluid', sources=['fluid', 'boundary'])
```

## Entropically Damped Artificial Compressibility (EDAC)

**Reference**: Ramachandran et al. 2019, Computers and Fluids

**Use case**: Incompressible flows with faster convergence

**Characteristics**:
- Artificial compressibility with entropy damping
- Faster convergence than ISPH
- Good balance between accuracy and cost
- Robust for complex geometries

**Example usage**:
```python
from pysph.sph.edac.basic import get_equations

equations = get_equations(dest='fluid', sources=['fluid'])
```

## Incompressible SPH (ISPH)

**Reference**: Cummins et al. 1999, JCP

**Use case**: Strictly incompressible flows

**Characteristics**:
- Pressure Poisson equation solved each timestep
- Exact incompressibility
- Highest accuracy for incompressible flows
- Most computationally expensive

**Example usage**:
```python
from pysph.sph.isph.basic import get_equations

equations = get_equations(dest='fluid', sources=['fluid'])
```

## Elastic SPH

**Reference**: Gray et al. 2001, CMAME

**Use case**: Solid mechanics, elasticity, collision dynamics

**Characteristics**:
- Stress-strain relationships
- Material models (elastic, plastic)
- Solid-solid and solid-fluid interactions

**Example usage**:
```python
from pysph.sph.elastic.basic import get_equations

equations = get_equations(dest='solid', sources=['solid'])
```

## Formulation Selection Guide

| Problem Type | Recommended Formulation | Reason |
|--------------|----------------------|---------|
| Free-surface flows (dam break, sloshing) | WCSPH | Robust, efficient |
| Incompressible internal flows | TVF or EDAC | Balance of accuracy and cost |
| High-accuracy incompressible flows | ISPH | Exact incompressibility |
| Solid mechanics | Elastic SPH | Stress-strain formulation |
| Wave-structure interaction | WCSPH | Free-surface capability |
| Multi-phase flows | TVF | Transport velocity stability |

## Performance Considerations

- **WCSPH**: Fastest, suitable for most free-surface problems
- **TVF**: Moderate cost, better velocity accuracy
- **EDAC**: Faster convergence than ISPH, good for incompressible flows
- **ISPH**: Slowest due to pressure solve, highest accuracy
- **Elastic**: Cost depends on material model and time step

## Accuracy Considerations

- **WCSPH**: Good for free-surface, compressibility errors acceptable
- **TVF**: Excellent velocity field accuracy, eliminates tensile instability
- **EDAC**: Good balance, artificial compressibility controlled
- **ISPH**: Best incompressibility enforcement
- **Elastic**: Depends on material model and time integration

## Stability Considerations

- **WCSPH**: Requires CFL condition, may need artificial viscosity
- **TVF**: More stable due to transport velocity
- **EDAC**: Entropy damping improves stability
- **ISPH**: Most stable for incompressible flows
- **Elastic**: Depends on material model, may need time step control

## Common Issues and Solutions

### Tensile Instability

**Problem**: Particles clump together under tension

**Solutions**:
- Use TVF formulation
- Apply tensile instability correction
- Use particle shifting algorithms

### Pressure Noise

**Problem**: Oscillations in pressure field

**Solutions**:
- Use higher-order kernels
- Apply density correction
- Use EDAC or ISPH formulations

### Free-Surface Distortion

**Problem**: Incorrect free-surface boundary

**Solutions**:
- Use WCSPH formulation
- Ensure proper boundary conditions
- Check particle distribution near free surface
