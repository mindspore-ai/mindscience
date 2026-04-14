---
name: phiflow
description: Differentiable PDE solving framework for fluid simulations and physics-based machine learning. Use when working with fluid dynamics, incompressible flow, advection-diffusion problems, or when differentiable physics simulations are needed. Supports NumPy, TensorFlow, PyTorch, and JAX backends.
---

# PhiFlow

PhiFlow is a differentiable PDE solving framework for fluid simulations and physics-based machine learning.

## Backend Selection

Choose backend before importing:

```python
from phi.flow import *              # NumPy backend (non-differentiable)
from phi.torch.flow import *       # PyTorch backend (differentiable)
from phi.tf.flow import *           # TensorFlow backend (differentiable)
from phi.jax.flow import *          # JAX backend (differentiable)
```

## Grid Types

**CenteredGrid** - Scalar fields sampled at cell centers:

```python
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=32, y=40, bounds=Box(x=32, y=40))
```

**StaggeredGrid** - Velocity fields sampled at face centers:

```python
velocity = StaggeredGrid(0, extrapolation.ZERO, x=32, y=40, bounds=Box(x=32, y=40))
```

## Fluid Simulation Steps

Standard incompressible flow simulation:

```python
# 1. Advect velocity
velocity = advect.semi_lagrangian(velocity, velocity, dt=1)

# 2. Advect scalar field
smoke = advect.mac_cormack(smoke, velocity, dt=1)

# 3. Add forces (buoyancy, external)
velocity += smoke * (0, 0.5) @ velocity

# 4. Make incompressible
velocity, pressure = fluid.make_incompressible(velocity, (), Solve(rank_deficiency=0))
```

## Obstacles

Define obstacles with geometry and optional motion:

```python
obstacle = Obstacle(Sphere(center=(16, 20), radius=5), velocity=(0.5, 0))
velocity, pressure = fluid.make_incompressible(velocity, obstacle)
```

## Differentiability

For gradient-based optimization, use differentiable backend:

```python
from phi.torch.flow import *

# Define simulation function
def simulate(smoke, velocity):
    for _ in range(20):
        smoke = advect.mac_cormack(smoke, velocity, dt=1)
        velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
        velocity, _ = fluid.make_incompressible(velocity, ())
    return smoke

# Compute gradient
grad_fn = field.gradient(simulate, wrt='velocity')
velocity_grad = grad_fn(smoke, velocity)
```

## Visualization

```python
vis.plot(smoke)
vis.plot(velocity)
vis.plot(field.stack(trajectory, batch('time')), animate='time')
```

## Resources

- **Advanced advection schemes**: See references/advection.md
- **Higher-order methods**: See references/higher_order.md
- **Batch execution**: See references/batching.md
- **Obstacle handling**: See references/obstacles.md
