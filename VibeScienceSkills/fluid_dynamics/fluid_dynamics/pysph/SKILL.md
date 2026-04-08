---
name: pysph
description: Python-based framework for Smoothed Particle Hydrodynamics (SPH) simulations. Use when working with particle-based fluid dynamics, free-surface flows, elastic solid mechanics, or when implementing custom SPH formulations. Supports high-performance execution via Cython, OpenCL, OpenMP, and MPI.
---

# PySPH

PySPH is a Python framework for Smoothed Particle Hydrodynamics with high-performance backends.

## Quick Start

Run built-in examples:

```bash
pysph run elliptical_drop      # Quick test (~20s)
pysph run dam_break_2d        # 2D dam break (~30 min)
pysph run dam_break_3d        # 3D dam break
pysph run cavity              # Lid-driven cavity
```

View results:

```bash
pysph view elliptical_drop_output/    # View saved output
pysph view                     # Live viewer
```

## Core Concepts

### Particle Arrays

Store particle properties with typed arrays:

```python
from pysph.base.utils import get_particle_array
import numpy as np

x, y = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
particles = get_particle_array(name='fluid', x=x.ravel(), y=y.ravel())
```

Default properties: `x, y, z` (position), `u, v, w` (velocity), `h, m, rho` (smoothing, mass, density), `p` (pressure), `au, av, aw` (acceleration)

### Nearest Neighbor Search

Find neighbors within interaction radius:

```python
from pysph.base.nnps import LinkedListNNPS, DomainManager

domain = DomainManager(0., 1., 0., 1., periodic_in_x=True)
nps = LinkedListNNPS(dim=2, particles=[particles], radius_scale=3, domain=domain)
```

### Equations

Define SPH equations in Python:

```python
from pysph.sph.equation import Equation, Group

class DensityEquation(Equation):
    def __init__(self, dest, sources):
        super().__init__(dest, sources)

    def loop(self, d_idx, d_rho, s_m, s_idx, WIJ):
        d_rho[d_idx] += s_m[s_idx] * WIJ

equations = [
    Group(equations=[DensityEquation(dest='fluid', sources=['fluid'])])
]
```

## Basic Simulation Workflow

```python
from pysph.base.nnps import LinkedListNNPS
from pysph.sph.integrator import EulerIntegrator
from pysph.application import Application

# Create particles
particles = get_particle_array(name='fluid', x=x, y=y, h=h, m=m, rho=0.0)

# Setup neighbor search
nps = LinkedListNNPS(dim=2, particles=[particles], radius_scale=2.0)

# Define equations
equations = [Group(equations=[DensityEquation(dest='fluid', sources=['fluid'])])]

# Create integrator
integrator = EulerIntegrator()

# Setup application
app = Application(
    particle_arrays=[particles],
    nnps_factory=lambda: nps,
    integrator=integrator,
    equations=equations
)

# Run simulation
app.run(t_end=1.0, dt=0.01)
```

## SPH Formulations

PySPH includes multiple formulations:

- **WCSPH**: Weakly Compressible SPH for free-surface flows
- **TVF**: Transport Velocity Formulation for incompressible fluids
- **EDAC**: Entropically Damped Artificial Compressibility
- **ISPH**: Incompressible SPH with pressure projection
- **Elastic SPH**: For solid mechanics and elasticity

See [formulations.md](references/formulations.md) for detailed comparison and usage.

## Performance Backends

Select backend at runtime:

```bash
pysph run dam_break_2d --backend=cython    # CPU with OpenMP
pysph run dam_break_2d --backend=opencl     # GPU
pysph run dam_break_2d --backend=mpich      # MPI parallel
```

## Resources

- **SPH formulations**: See [formulations.md](references/formulations.md)
- **Equation writing**: See [equations.md](references/equations.md)
- **Parallel execution**: See [parallel.md](references/parallel.md)
- **Boundary conditions**: See [boundaries.md](references/boundaries.md)
- **Custom integrators**: See [integrators.md](references/integrators.md)
- **Examples**: See [examples.md](references/examples.md)
