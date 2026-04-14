# Integrators in PySPH

Integrators advance particle states in time using various schemes.

## Euler Integrator

First-order explicit time integration.

**Characteristics**:
- First-order accuracy
- Simple implementation
- Good for testing
- Requires small time steps

**Implementation**:
```python
from pysph.sph.integrator import EulerIntegrator

integrator = EulerIntegrator()
```

**Usage**:
```python
app = Application(
    particle_arrays=[particles],
    nnps_factory=lambda: nps,
    integrator=integrator,
    equations=equations
)
```

## Verlet Integrator

Second-order explicit time integration.

**Characteristics**:
- Second-order accuracy
- Better stability than Euler
- Standard for many SPH applications
- Moderate computational cost

**Implementation**:
```python
from pysph.sph.integrator import VerletIntegrator

integrator = VerletIntegrator()
```

## Predictor-Corrector Integrator

Second-order predictor-corrector scheme.

**Characteristics**:
- Second-order accuracy
- Good stability properties
- Suitable for compressible flows
- Two stages per timestep

**Implementation**:
```python
from pysph.sph.integrator import PECIntegrator

integrator = PECIntegrator()
```

## RK2 Integrator

Second-order Runge-Kutta scheme.

**Characteristics**:
- Second-order accuracy
- Better accuracy than Verlet
- More computational cost
- Two function evaluations per timestep

**Implementation**:
```python
from pysph.sph.integrator import RK2Integrator

integrator = RK2Integrator()
```

## RK4 Integrator

Fourth-order Runge-Kutta scheme.

**Characteristics**:
- Fourth-order accuracy
- Highest accuracy among standard integrators
- Most computationally expensive
- Four function evaluations per timestep

**Implementation**:
```python
from pysph.sph.integrator import RK4Integrator

integrator = RK4Integrator()
```

## Custom Integrators

Define custom multi-stage integrators.

**Structure**:
```python
from pysph.sph.integrator import Integrator, IntegratorStep

class MyIntegrator(Integrator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def one_timestep(self, t, dt):
        # Custom time stepping logic
        self.compute_accelerations()
        self.stage1()
        self.do_post_stage(dt, 1)

        self.compute_accelerations(update_nnps=False)
        self.stage2()
        self.update_domain()
        self.do_post_stage(dt, 2)
```

**Multi-stage example**:
```python
class MyIntegrator(Integrator):
    def one_timestep(self, t, dt):
        # Stage 1
        self.compute_accelerations(0)
        self.stage1()
        self.do_post_stage(dt, 1)

        # Stage 2
        self.compute_accelerations(1, update_nnps=False)
        self.stage2()
        self.update_domain()
        self.do_post_stage(dt, 2)

        # Stage 3
        self.compute_accelerations(2, update_nnps=False)
        self.stage3()
        self.do_post_stage(dt, 3)
```

## Integrator Selection Guide

| Requirement | Recommended Integrator | Reason |
|-------------|----------------------|---------|
| Quick testing | Euler | Simple, fast |
| Standard applications | Verlet | Good balance of accuracy and cost |
| Compressible flows | PEC | Good stability |
| High accuracy | RK2 or RK4 | Higher-order accuracy |
| Custom time stepping | Custom integrator | Specialized requirements |

## Time Step Control

### CFL Condition

Automatic time step based on Courant-Friedrichs-Lewy condition:

```python
app = Application(
    particle_arrays=[particles],
    nnps_factory=lambda: nps,
    integrator=integrator,
    equations=equations,
    cfl=0.3  # CFL number (default 0.3)
)
```

**Time step calculation**:
```
dt = cfl * min(dt_velocity, dt_force, dt_viscosity)
```

### Adaptive Time Step

Per-particle adaptive time stepping:

```python
# Add dt_adapt property to particle array
particles.add_property('dt_adapt', data=0.01)

# Equations compute dt_adapt per particle
# Minimum value is used as timestep
```

### Fixed Time Step

Constant time step:

```python
app = Application(
    particle_arrays=[particles],
    nnps_factory=lambda: nps,
    integrator=integrator,
    equations=equations,
    tf=1.0,  # Final time
    dt=0.01   # Fixed time step
)
```

## Multi-Stage Equations

Different equations for different integrator stages.

**Implementation**:
```python
from pysph.sph.equation import MultiStageEquations

def create_equations():
    # Stage 1 equations
    eqs_stage1 = [
        DensityEquation(dest='fluid', sources=['fluid']),
        PressureEquation(dest='fluid', sources=['fluid'])
    ]

    # Stage 2 equations
    eqs_stage2 = [
        DensityEquation(dest='fluid', sources=['fluid']),
        'Different pressure equation'
    ]

    return MultiStageEquations([eqs_stage1, eqs_stage2])
```

**Usage with custom integrator**:
```python
class MyIntegrator(Integrator):
    def one_timestep(self, t, dt):
        # Stage 1 equations
        self.compute_accelerations(0)
        self.stage1()
        self.do_post_stage(dt, 1)

        # Stage 2 equations
        self.compute_accelerations(1, update_nnps=False)
        self.stage2()
        self.update_domain()
        self.do_post_stage(dt, 2)
```

## Acceleration Computation

### Standard Acceleration

Compute accelerations before integration stages:

```python
class MyIntegrator(Integrator):
    def one_timestep(self, t, dt):
        # Compute accelerations
        self.compute_accelerations()

        # Integration stages
        self.stage1()
        self.do_post_stage(dt, 1)
```

### Multi-Stage Accelerations

Different accelerations for different stages:

```python
class MyIntegrator(Integrator):
    def one_timestep(self, t, dt):
        # Stage 1 accelerations
        self.compute_accelerations(0)
        self.stage1()
        self.do_post_stage(dt, 1)

        # Stage 2 accelerations
        self.compute_accelerations(1, update_nnps=False)
        self.stage2()
        self.update_domain()
        self.do_post_stage(dt, 2)
```

### No NNPS Update

Skip neighbor search for intermediate stages:

```python
class MyIntegrator(Integrator):
    def one_timestep(self, t, dt):
        # Stage 1 with NNPS update
        self.compute_accelerations()
        self.stage1()
        self.do_post_stage(dt, 1)

        # Stage 2 without NNPS update
        self.compute_accelerations(update_nnps=False)
        self.stage2()
        self.do_post_stage(dt, 2)
```

## Domain Update

Update particle positions and neighbor search:

```python
class MyIntegrator(Integrator):
    def one_timestep(self, t, dt):
        # Integration stages
        self.compute_accelerations()
        self.stage1()
        self.do_post_stage(dt, 1)

        # Update domain (positions and NNPS)
        self.update_domain()

        # Continue integration
        self.compute_accelerations(update_nnps=False)
        self.stage2()
        self.do_post_stage(dt, 2)
```

**When to call update_domain()**:
- After particles have moved significantly
- Before next stage requiring updated positions
- When periodic boundaries need update
- When ghost particles need refresh

## Post-Stage Operations

Operations after each integration stage:

```python
class MyIntegrator(Integrator):
    def one_timestep(self, t, dt):
        self.compute_accelerations()
        self.stage1()
        self.do_post_stage(dt, 1)  # Post-stage 1

        self.update_domain()
        self.stage2()
        self.do_post_stage(dt, 2)  # Post-stage 2
```

**Typical post-stage operations**:
- Boundary condition updates
- Particle shifting
- Density filtering
- Property updates
- Output generation

## Performance Considerations

### Accuracy vs Cost

| Integrator | Accuracy | Cost | Stages |
|-----------|----------|------|---------|
| Euler | O(dt) | Low | 1 |
| Verlet | O(dt²) | Medium | 1 |
| PEC | O(dt²) | Medium | 2 |
| RK2 | O(dt²) | Medium-High | 2 |
| RK4 | O(dt⁴) | High | 4 |

### Stability Considerations

- **Euler**: Most restrictive, requires small dt
- **Verlet**: Good stability for many problems
- **PEC**: Good for compressible flows
- **RK2**: Better stability than Verlet
- **RK4**: Best stability properties

### Computational Cost

- **Function evaluations**: More stages = more cost
- **NNPS updates**: Expensive operation
- **Domain updates**: Required after particle motion
- **Boundary conditions**: May add overhead

## Common Issues and Solutions

### Instability

**Problem**: Solution blows up

**Solutions**:
- Reduce time step
- Use more stable integrator
- Check CFL condition
- Verify equation implementation

### Inaccuracy

**Problem**: Results not accurate enough

**Solutions**:
- Use higher-order integrator
- Reduce time step
- Check equation accuracy
- Verify boundary conditions

### Energy Drift

**Problem**: Energy not conserved

**Solutions**:
- Use symplectic integrator if available
- Check time integration accuracy
- Verify conservation equations
- Reduce numerical dissipation

## Advanced Topics

- **Symplectic integrators**: Energy-conserving schemes
- **Adaptive time stepping**: Variable time steps
- **Sub-cycling**: Different time steps for different equations
- **Implicit integration**: For stiff problems
- **Multi-rate integration**: Different time scales

## Best Practices

1. **Start with simple integrator**: Test with Euler or Verlet first
2. **Verify stability**: Check solution stability before increasing accuracy
3. **Monitor energy**: Track conservation properties
4. **Profile performance**: Measure computational cost
5. **Choose appropriate order**: Balance accuracy and cost
6. **Test convergence**: Verify convergence with decreasing dt
