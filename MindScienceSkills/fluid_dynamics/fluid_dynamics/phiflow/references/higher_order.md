# Higher-Order Methods

PhiFlow supports higher-order spatial schemes for improved accuracy.

## Pressure Solver Order

Control spatial order for pressure gradient computation:

```python
# 2nd order (default, supports obstacles)
velocity, pressure = fluid.make_incompressible(velocity, order=2)

# 4th order (no obstacles)
velocity, pressure = fluid.make_incompressible(velocity, order=4)
```

## RK4 Time Integration

4th-order Runge-Kutta for time advancement:

```python
def momentum_equation(v):
    return advect.semi_lagrangian(v, v, dt=1) + forces

velocity, pressure = fluid.incompressible_rk4(
    momentum_equation, velocity, pressure, dt=1,
    pressure_order=4, pressure_solve=Solve('CG')
)
```

**Benefits**: Better temporal accuracy, reduced numerical dissipation

**Limitations**: Higher computational cost, obstacles only supported with 2nd order

## Accuracy vs Performance Trade-off

- **2nd order**: Fast, robust, supports all features
- **4th order**: Higher accuracy, no obstacles
- **RK4**: Best temporal accuracy, higher cost

Use higher-order methods when:
- Precise gradient information needed (optimization)
- Long simulation times
- Benchmarking numerical accuracy
