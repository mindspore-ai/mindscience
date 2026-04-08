# Boundary Conditions in gprMax

gprMax supports various boundary conditions for electromagnetic simulations.

## PML Boundaries

### Basic PML

```python
# PML absorbing boundaries
#pml: 1.0 1.0 1.0
```

### PML Configuration

```python
# PML with profile
#pml: 1.0 1.0 1.0 pml_profile 0.05 0.05
```

### PML Placement

```python
# PML on all sides
#pml: 1.0 1.0 1.0

# PML on specific sides
#pml_x: 1.0 0.0 0.0  # x direction
#pml_y: 0.0 1.0 0.0  # y direction
#pml_z: 0.0 0.0 1.0  # z direction
```

## Other Boundaries

### Perfect Electric Conductor

```python
# Perfect electric conductor
#boundary: 6.0 1.0 pec
```

### Perfect Magnetic Conductor

```python
# Perfect magnetic conductor
#boundary: 6.0 1.0 pmc
```

### Periodic Boundaries

```python
# Bloch-periodic boundaries
#k_point: 0.1 0.0 0.0  # Bloch wavevector
```

## Boundary Configuration

### Boundary Selection

```python
# PML boundaries (recommended)
#pml: 1.0 1.0 1.0

# Mixed boundaries
#pml: 1.0 0.0 0.0
#boundary: 6.0 1.0 pec
```

### Boundary Parameters

```python
# PML thickness
#pml: 1.0 1.0 1.0 1.5  # 1.5 wavelengths

# PML profile
#pml: 1.0 1.0 1.0 pml_profile 0.05 0.05
```

## Best Practices

### PML Thickness

1. **Minimum thickness**: 1-2 wavelengths
2. **Recommended thickness**: 2-3 wavelengths
3. **High accuracy**: 3+ wavelengths

### Boundary Placement

1. **Enclose domain**: PML should enclose computational domain
2. **Object padding**: Keep objects away from PML
3. **Source placement**: Keep sources away from PML

### Boundary Selection

1. **Absorbing problems**: Use PML boundaries
2. **Periodic structures**: Use periodic boundaries
3. **Resonantors**: Use PEC boundaries
4. **Waveguides**: Use PML boundaries

## Troubleshooting

### Simulation Divergence

**Problem**: Simulation diverges or produces NaN/Inf values

**Solutions**:
1. Reduce time step (decrease Courant number)
2. Check material stability
3. Verify boundary conditions
4. Check source configuration

### Poor Absorption

**Problem**: Reflections from boundaries

**Solutions**:
1. Increase PML thickness
2. Use PML profile
3. Check PML placement
4. Verify PML parameters

### Periodic Issues

**Problem**: Incorrect periodic behavior

**Solutions**:
1. Verify k_point vector
2. Check structure periodicity
3. Verify frequency range
4. Check material parameters

### Conductor Issues

**Problem**: Unexpected reflections

**Solutions**:
1. Verify conductor type
2. Check conductor placement
3. Check material interfaces
4. Check source configuration