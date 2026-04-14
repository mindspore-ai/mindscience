# Batch Execution

PhiFlow supports data-parallel execution via batch dimensions.

## Creating Batched Fields

```python
# Batch over inflow locations
INFLOW_LOCATION = tensor(
    [(4, 5), (8, 5), (12, 5), (16, 5)],
    batch('inflow_loc'),
    channel(vector='x,y')
)

INFLOW = 0.6 * CenteredGrid(
    Sphere(center=INFLOW_LOCATION, radius=3),
    extrapolation.BOUNDARY, x=32, y=40
)
```

## Batch Operations

All operations automatically parallelize over batch dimensions:

```python
# This runs 4 simulations simultaneously
for i in range(20):
    smoke = advect.mac_cormack(smoke, velocity, dt=1) + INFLOW
    velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
    velocity, _ = fluid.make_incompressible(velocity, ())
```

## Accessing Batch Results

```python
# Access single simulation
smoke.inflow_loc[0]  # First simulation
smoke.inflow_loc[-1]  # Last simulation

# Stack results over time
trajectory = field.stack([smoke1, smoke2, smoke3], batch('time'))
vis.plot(trajectory, animate='time')
```

## Use Cases

- Parameter sweeps (different inflow positions, velocities)
- Optimization over multiple initial conditions
- Ensemble simulations
- Training neural networks with multiple samples

## Performance Notes

Batch execution is highly efficient - simulations run in parallel on GPU without Python overhead.
