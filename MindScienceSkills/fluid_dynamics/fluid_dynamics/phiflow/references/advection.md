# Advection Schemes

PhiFlow provides multiple advection algorithms with different trade-offs:

## Semi-Lagrangian Advection

Stable and efficient for velocity fields:

```python
velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
```

**Use when**: Advecting velocity, stability is critical

## MacCormack Advection

Second-order accurate, good for scalar fields:

```python
smoke = advect.mac_cormack(smoke, velocity, dt=1)
```

**Use when**: High accuracy needed for smoke/density fields

## Finite Difference Advection

Upwind scheme for simple cases:

```python
field = advect.upwind(field, velocity, dt=1)
```

**Use when**: Simple advection, lower computational cost

## Selection Guide

| Field Type | Recommended Scheme | Reason |
|------------|-------------------|---------|
| Velocity | semi_lagrangian | Stable, efficient |
| Smoke/Density | mac_cormack | Higher accuracy |
| Temperature | semi_lagrangian | Stable, sufficient accuracy |
| Concentration | mac_cormack | Preserves sharp gradients |
