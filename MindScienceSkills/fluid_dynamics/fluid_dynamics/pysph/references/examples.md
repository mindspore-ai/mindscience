# PySPH Examples

PySPH includes numerous examples demonstrating various capabilities.

## Built-in Examples

### Elliptical Drop

**Description**: 2D elliptical drop deformation

**Run**:
```bash
pysph run elliptical_drop
```

**View**:
```bash
pysph view elliptical_drop_output/
```

**Characteristics**:
- Quick test (~20 seconds)
- 2D simulation
- Elastic solid mechanics
- Demonstrates particle deformation

**Use case**: Testing installation, basic SPH validation

### Dam Break 2D

**Description**: 2D dam break problem (SPHERIC Test 2)

**Run**:
```bash
pysph run dam_break_2d
```

**View live**:
```bash
pysph view
```

**Characteristics**:
- Free-surface flow
- WCSPH formulation
- ~30 minutes runtime
- SPHERIC benchmark test

**Use case**: Free-surface flows, wave breaking, validation

### Dam Break 3D

**Description**: 3D dam break problem

**Run**:
```bash
pysph run dam_break_3d
```

**Characteristics**:
- 3D free-surface flow
- Large particle count
- SPHERIC benchmark test
- Requires significant computational resources

**Use case**: 3D free-surface flows, large-scale validation

### Lid-Driven Cavity

**Description**: 2D lid-driven cavity flow

**Run**:
```bash
pysph run cavity
```

**Characteristics**:
- Incompressible flow
- Transport velocity formulation
- Internal flow
- Generates streamlines

**Use case**: Incompressible flows, internal flows, validation

### Solid Mechanics Examples

**Run**:
```bash
pysph run solid_mech.rings
```

**Characteristics**:
- Elastic solid mechanics
- Collision dynamics
- Solid-solid interaction
- Demonstrates stress-strain relationships

**Use case**: Solid mechanics, collision, elasticity

## Example Categories

### Free-Surface Flows

**Examples**:
- `elliptical_drop`: Deforming elastic drop
- `dam_break_2d`: 2D dam break
- `dam_break_3d`: 3D dam break
- `sloshing`: Wave sloshing

**Common features**:
- WCSPH formulation
- Free-surface boundary conditions
- Equation of state for pressure
- Gravity effects

### Incompressible Flows

**Examples**:
- `cavity`: Lid-driven cavity
- `channel_flow`: Channel flow
- `backward_facing_step`: Backward facing step

**Common features**:
- TVF or EDAC formulation
- Incompressibility enforcement
- Prescribed boundary conditions
- Internal flow patterns

### Solid Mechanics

**Examples**:
- `solid_mech.rings`: Ring collision
- `solid_mech.impact`: Impact problems
- `solid_mech.vibration`: Vibration analysis

**Common features**:
- Elastic SPH formulation
- Stress-strain relationships
- Material models
- Solid-solid interaction

### Multi-Phase Flows

**Examples**:
- `bubble_rise`: Bubble rising in liquid
- `droplet_impact`: Droplet impact
- `wave_breaking`: Wave breaking

**Common features**:
- Multiple particle types
- Interface tracking
- Surface tension effects
- Density ratios

## Running Examples

### Basic Execution

```bash
# List available examples
pysph run

# Run specific example
pysph run dam_break_2d
```

### With Backend Selection

```bash
# CPU with OpenMP
pysph run dam_break_2d --backend=cython --threads=8

# GPU execution
pysph run dam_break_2d --backend=opencl

# MPI parallel
mpirun -np 4 pysph run dam_break_2d --backend=mpich
```

### With Time Control

```bash
# Specify final time
pysph run dam_break_2d --tf=2.0

# Specify time step
pysph run dam_break_2d --dt=0.001

# Specify both
pysph run dam_break_2d --tf=2.0 --dt=0.001
```

### With Output Control

```bash
# Specify output directory
pysph run dam_break_2d --output-dir=my_output

# Output frequency
pysph run dam_break_2d --output-freq=100
```

## Viewing Results

### Interactive Viewer

```bash
# View most recent output
pysph view

# View specific output
pysph view dam_break_2d_output/

# Live viewer (while simulation running)
pysph view --live
```

### Post-Processing

```python
# Load particle data
from pysph.tools.loader import load

particles = load('dam_break_2d_output/')

# Access properties
x = particles.x
y = particles.y
rho = particles.rho
p = particles.p
```

### Visualization

```python
import matplotlib.pyplot as plt

# Plot particle positions
plt.scatter(particles.x, particles.y, c=particles.rho)
plt.colorbar()
plt.show()

# Plot velocity field
plt.quiver(particles.x, particles.y, particles.u, particles.v)
plt.show()
```

## Example Analysis

### Dam Break 2D

**Physical setup**:
- Water column: 2.0m × 1.0m
- Domain: 4.0m × 3.2m
- Particle spacing: 0.012m
- Total particles: ~65,000

**Numerical parameters**:
- Formulation: WCSPH
- Kernel: Cubic spline
- Time step: Adaptive
- CFL number: 0.3

**Expected behavior**:
- Initial collapse of water column
- Wave propagation
- Free-surface evolution
- Impact on opposite wall

**Validation**:
- Compare with experimental data
- Check mass conservation
- Verify energy dissipation
- Compare with other SPH codes

### Lid-Driven Cavity

**Physical setup**:
- Cavity: 1.0m × 1.0m
- Lid velocity: 1.0 m/s
- Reynolds number: 100-1000
- Fluid: Water or air

**Numerical parameters**:
- Formulation: TVF
- Kernel: Cubic spline
- Time step: Adaptive
- CFL number: 0.3

**Expected behavior**:
- Vortex formation
- Steady-state circulation
- Velocity profile development
- Pressure distribution

**Validation**:
- Compare with Ghia et al. (1982) benchmark
- Check velocity profiles
- Verify circulation strength
- Compare with grid-based methods

## Custom Examples

### Creating Custom Example

**Structure**:
```python
# example/my_problem.py
from pysph.application import Application
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Group
from pysph.sph.integrator import VerletIntegrator

def create_particles():
    # Create particle distribution
    import numpy as np
    x, y = np.meshgrid(...)
    particles = get_particle_array(name='fluid', x=x.ravel(), y=y.ravel())
    return particles

def create_equations():
    # Define your equations
    from pysph.sph.wc.basic import get_density_equations
    from pysph.sph.wc.basic import get_pressure_equations
    from pysph.sph.wc.basic import get_equations as get_momentum_eqs

    equations = [
        Group(equations=get_density_equations(dest='fluid', sources=['fluid'])),
        Group(equations=get_pressure_equations(dest='fluid', sources=['fluid'])),
        Group(equations=get_momentum_eqs(dest='fluid', sources=['fluid']))
    ]
    return equations

if __name__ == '__main__':
    # Create particles
    particles = create_particles()

    # Create equations
    equations = create_equations()

    # Create integrator
    integrator = VerletIntegrator()

    # Setup application
    app = Application(
        particle_arrays=[particles],
        nnps_factory=lambda: nps,
        integrator=integrator,
        equations=equations,
        tf=1.0,
        dt=0.01
    )

    # Run simulation
    app.run()

    # Save output
    particles.save('my_output')
```

**Run custom example**:
```bash
pysph run my_problem
```

## Performance Benchmarks

### Standard Benchmarks

**SPHERIC Tests**:
- Test 2: 2D dam break
- Test 3: 2D oscillating drop
- Test 5: 3D dam break
- Test 7: 3D lock exchange

**Run benchmarks**:
```bash
pysph run dam_break_2d
pysph run oscillating_drop
pysph run dam_break_3d
```

### Performance Comparison

Compare different formulations:

```bash
# WCSPH
pysph run dam_break_2d --scheme=wcsph

# TVF
pysph run cavity --scheme=tvf

# EDAC
pysph run cavity --scheme=edac
```

Compare different backends:

```bash
# Cython
pysph run dam_break_2d --backend=cython

# OpenCL
pysph run dam_break_2d --backend=opencl

# MPI
mpirun -np 4 pysph run dam_break_2d --backend=mpich
```

## Learning from Examples

### Study Equation Structure

Examine equation implementations:

```python
# Look at source code
import pysph.sph.wc.basic as wc

# Read density equation
import inspect
print(inspect.getsourcefile(wc.get_density_equations))
```

### Study Formulation Differences

Compare different formulations:

```python
# WCSPH formulation
from pysph.sph.wc.basic import get_density_equations as wcsph_density

# TVF formulation
from pysph.sph.tvf.basic import get_density_equations as tvf_density

# Compare implementations
```

### Study Boundary Conditions

Examine boundary condition implementations:

```python
from pysph.sph.bc.inlet_outlet_manager import InletOutletManager

# Create and examine boundary
wall = InletOutletManager(...)
```

## Common Issues and Solutions

### Installation Issues

**Problem**: Cannot run examples

**Solutions**:
- Verify installation: `pysph run elliptical_drop`
- Check dependencies: `pip install -r requirements.txt`
- Build from source: `python setup.py install`

### Visualization Issues

**Problem**: Cannot view results

**Solutions**:
- Install Mayavi: `pip install mayavi`
- Check output directory: `ls output_dir/`
- Use alternative visualization: Python/matplotlib

### Performance Issues

**Problem**: Example runs too slowly

**Solutions**:
- Use appropriate backend: `--backend=opencl`
- Reduce particle count: Modify example parameters
- Increase parallelism: `--threads=8`
- Check hardware requirements

### Accuracy Issues

**Problem**: Results look incorrect

**Solutions**:
- Verify formulation: Check scheme selection
- Check time step: Reduce `--dt`
- Verify boundary conditions: Check inlet/outlet
- Compare with benchmarks: Use SPHERIC tests

## Advanced Topics

- **Parameter studies**: Vary physical parameters
- **Convergence studies**: Grid/time step convergence
- **Validation studies**: Compare with experiments
- **Optimization studies**: Performance tuning
- **Extension studies**: Add new physics

## Best Practices

1. **Start with simple examples**: Begin with `elliptical_drop`
2. **Understand the physics**: Know what example demonstrates
3. **Study the code**: Examine equation implementations
4. **Modify parameters**: Experiment with changes
5. **Validate results**: Check conservation and accuracy
6. **Build on examples**: Use examples as templates for custom problems
