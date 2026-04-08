# Materials in NGsolve

NGsolve supports various material types for electromagnetic and mechanical simulations.

## Basic Materials

### Constant Material

```python
import ngsolve as ns

# Constant material
E = ns.Constant('E')
E.set_value(1.0)
```

### Function Material

```python
# Function material
E = ns.Function('E')
E.set_value('1.0 + x**2')
```

## Electromagnetic Materials

### Dielectric Material

```python
# Dielectric material
E = ns.Constant('E')
E.set_value(1.0)
```

### Magnetic Material

```python
# Magnetic material
mu = ns.Constant('mu')
mu.set_value(1.0)
```

### Anisotropic Material

```python
# Anisotropic material
E = ns.Function('E')
E.set_value([[1.0, 0.5], [0.5, 2.0]])
```

## Frequency-Dependent Materials

### Complex Permittivity

```python
# Complex permittivity
E = ns.Function('E')
E.set_value(1.0 + 0.1j)
```

### Conductive Material

```python
# Conductive material
E = ns.Constant('E')
E.set_value(1.0)
E.set_conductivity(0.01)
```

## Material Properties

### Material Assignment

```python
# Assign material to function space
V = ns.FunctionSpace('V')
V.set_material(E)
```

### Material Visualization

```python
# Get material distribution
E_array = E.vector().get_array()

# Plot material
import matplotlib.pyplot as plt
plt.imshow(E_array.reshape(100, 100), cmap='viridis')
plt.colorbar(label='Permittivity')
plt.title('Material Distribution')
plt.show()
```

## Best Practices

### Material Selection

1. **Use appropriate models**: Choose correct material model for application
2. **Physical properties**: Use realistic material parameters
3. **Frequency range**: Ensure material valid over frequency range

### Material Stability

1. **Check for stability**: Verify material doesn't cause numerical issues
2. **Use proper scaling**: Ensure material values are properly scaled
3. **Consider convergence**: Material properties affect convergence

### Material Modeling

1. **Spatial variation**: Model spatial material variations
2. **Frequency dependence**: Model frequency-dependent materials
3. **Anisotropy**: Model anisotropic materials

## Troubleshooting

### Material Instability

**Problem**: Simulation diverges with material

**Solutions**:
1. Check material parameters
2. Verify material stability
3. Check mesh quality
4. Check solver configuration

### Unexpected Behavior

**Problem**: Material doesn't behave as expected

**Solutions**:
1. Verify material definition
2. Check function space
3. Check solver type
4. Check boundary conditions

### Convergence Issues

**Problem**: Results don't converge with material

**Solutions**:
1. Increase mesh resolution
2. Improve solver tolerance
3. Check material parameters
4. Verify solver settings

### Performance Issues

**Problem**: Simulation is slow with material

**Solutions**:
1. Use appropriate solver
2. Use preconditioners
3. Optimize mesh
4. Use parallel computing