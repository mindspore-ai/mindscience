# Python scripting in NGsolve

NGsolve provides flexible Python scripting for complex simulation workflows.

## Python Integration

### Basic Python Usage

```python
import ngsolve as ns

# Create mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 1.0, 1.0)

# Create function space
V = ns.FunctionSpace('V')
V.set_parameter_order('u', 'u')

# Create material
u = ns.Constant('u')
u.set_value(1.0)
V.set_material(u)

# Create source
f = ns.Constant('f')
f.set_value(1.0)
V.set_source(f)

# Create problem
problem = ns.Poisson('u', mesh)

# Create solver
solver = ns.PoissonSolver('u', mesh, problem)

# Solve
solver.solve()
```

### Parameter Sweeps

```python
# Parameter sweep over frequencies
import numpy as np

frequencies = np.linspace(1.0, 10.0, 10)

for freq in frequencies:
    f.set_value(freq)
    solver.solve()
```

### Complex Geometries

```python
# Create complex geometry programmatically
import numpy as np

# Create parameterized geometry
for x_pos in np.linspace(-0.5, 0.5, 10):
    for y_pos in np.linspace(-0.5, 0.5, 10):
        mesh.add_point((x_pos, y_pos))
```

## Advanced Features

### Material Definitions

```python
# Dispersive materials
from ngsolve.materials import Drude, Lorentzian

# Drude material (metal-like)
drude_material = Drude(epsilon_inf=1.0, 
                           plasma=1.0, 
                           gamma=0.05, 
                           sigma=0.1)

# Lorentzian dispersive material
lorentzian_material = Lorentzian(epsilon_r=4.0, 
                                         epsilon_inf=10.0, 
                                         plasma=1e12, 
                                         gamma=1e11, 
                                         sigma=0.1)
```

### Custom Sources

```python
# Custom source function
def my_source(t, position):
    # User-defined source function
    return np.sin(2 * np.pi * 1.5e9 * t)

# Add custom source
V.set_source(my_source)
```

### Output Control

```python
# Control output frequency
V.set_output_frequency(0.1)  # Output every 0.1 time units

# Output specific field components
V.set_output_components(['u'])  # Output u-component only
```

## Simulation Control

### Time Stepping

```python
# Run for specific time
solver.solve(until=200)

# Run until condition
solver.solve(until=200, 
            stop_when_fields_decayed=True, 
            decay_by=1e-3)

# Run multiple times
solver.solve(until=200, n=10)  # Run 10 times
```

### Restart Simulation

```python
# Restart simulation
solver.restart()

# Reset simulation
solver.reset()
```

## Post-Processing

### Field Analysis

```python
# Get field data
u = V.get_subfunction('u')
u_array = u.vector().get_array()

# Calculate field statistics
print(f"Max u field: {np.max(np.abs(u_array))}")
print(f"Min u field: {np.min(np.abs(u_array))}")
```

### Parameter Extraction

```python
# Get solver statistics
stats = solver.get_solver_stats()
print(f"Iterations: {stats['iterations']}")
print(f"Residual: {stats['residual']}")
```

## Best Practices

### Python Scripting

1. **Use descriptive variable names**: Clear and meaningful names
2. **Comment complex logic**: Document non-obvious operations
3. **Use functions**: Break complex code into functions
4. **Error handling**: Add try-except blocks for robustness

### Parameter Management

1. **Use physical units**: Work in meters, millimeters, etc.
2. **Document assumptions**: Record material assumptions
3. **Validate inputs**: Check parameter validity
4. **Consistent units**: Use consistent unit systems

### Simulation Design

1. **Resolution requirements**: Follow λ_min/10 rule
2. **Time window sizing**: Include full source duration
3. **PML thickness**: Use 1-2 wavelengths
4. **Source placement**: Keep sources away from boundaries

### Performance Optimization

1. **Profile first**: Profile before optimizing
2. **Measure before optimizing**: Establish baseline
3. **Optimize systematically**: Change one parameter at a time
4. **Verify results**: Check that optimization helps

## Troubleshooting

### Python Errors

**Problem**: `ImportError: No module named 'ngsolve'`

**Solution**: Install ngsolve
```bash
pip install ngsolve
```

### Simulation Divergence

**Problem**: Simulation diverges or produces NaN/Inf values

**Solutions**:
1. Reduce time step (decrease Courant number)
2. Check material stability
3. Verify boundary conditions
4. Check source configuration

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce mesh resolution
2. Reduce output frequency
3. Use parallel computing
4. Optimize Python code

### Performance Issues

**Problem**: Simulation is too slow

**solutions**:
1. Use parallel computing
2. Enable OpenMP/MPI support
3. Reduce output frequency
4. Optimize Python code