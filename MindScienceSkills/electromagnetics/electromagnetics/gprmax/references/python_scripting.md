# Python Scripting in gprMax

gprMax provides Python scripting capabilities for complex simulation workflows.

## Python Integration

### Basic Python Usage

```python
# Import gprMax module
import gprMax

# Define simulation parameters
domain = [0.240, 0.210, 0.002]
dx_dy_dz = [0.002, 0.002, 0 ddz]

# Create materials
half_space = gprMax.Medium(epsilon_r=6.0, sigma=0.0)
pec_material = gprMax.Medium(epsilon_r=10.0, sigma=0.0)

# Create geometry
gprMax.geometry([
    gprMax.Box(size=domain, material=half_space),
    gprMax.Cylinder(size=[0.120, 0.080, 0.120], 
                  center=[0.0, 0.0, 0.0], 
                  material=pec_material)
])

# Create source
waveform = gprMax.Waveform(type='Ricker', 
                              f=1.5e9, 
                              amplitude=1.0)
gprMax.source([
    gprMax.HertzianDipole(center=[0.0, 0.0, 0.100], 
                               z=0.170, 
                               waveform=waveform)
])

# Run simulation
gprMax.run(until=200)
```

### Parameter Sweeps

```python
# Parameter sweep over frequencies
import numpy as np

frequencies = np.linspace(1.0, 2.0, 10)

for freq in frequencies:
    gprMax.Waveform(type='Ricker', f=freq, amplitude=1.0)
    gprMax.source([gprMax.H.ertzianDipole(waveform=waveform)])
    gprMax.run(until=200)
```

### Complex Geometries

```python
# Create complex geometry programmatically
import numpy as np

# Create parameterized geometry
for x_pos in np.linspace(-0.05, 0.05, 10):
    for y_pos in np.linspace(-0.05, 0.05, 10):
        gprMax.geometry([
            gprMax.Cylinder(size=[0.02, 0.02, 0.02], 
                          center=[x_pos, y_pos, 0.0], 
                          material=pec_material)
        ])
```

## Advanced Features

### Material Definitions

```python
# Dispersive materials
from gprMax.materials import Drude, Lorentzian

# Drude material (metal-like)
drude_material = Drude(epsilon_inf=1.0, 
                         plasma=1.0, 
                         gamma=0.05, 
                         sigma=0.1)

# Lorentzian dispersive material
lorentzian_material = Lorentzian(epsilon_r=4.0, 
                                     epsilon_inf=10.0, 
                                     plasma=0.1, 
                                     gamma=0.05, 
                                     sigma=0.1)
```

### Custom Sources

```python
# Custom source function
def my_source(t, position):
    # User-defined source function
    return np.sin(2 * np.pi * 1.5e9 * t)

# Add custom source
gprMax.source([
    gprMax.Source(my_source, center=[0.0, 0.0, 0.100])
])
```

### Output Control

```python
# Control output frequency
gprMax.output_freq(0.1)  # Output every 0.1 time units

# Output specific field components
gprMax.output_efield(0.1)  # Output Ez field
gprMax.output_hfield(0.1)  # Output H field

# Output at specific times
gprMax.output_at_time(100)  # Output at time=100
```

## Simulation Control

### Time Stepping

```python
# Run for specific time
gprMax.run(until=200)

# Run until condition
gprMax.run(until=200, 
            stop_when_fields_decayed=True, 
            decay_by=1e-3)

# Run multiple times
gprMax.run(until=200, n=10)  # Run 10 times
```

### Restart Simulation

```python
# Restart simulation
gprMax.restart()

# Reset simulation
gprMax.reset()
```

## Post-Processing

### Field Analysis

```python
# Load output data
import gprMax.io

# Load field data
ez_data = gprMax.io.load_output('simulation.out', 'Ez')

# Load geometry
geometry = gprMax.io.load_geometry('simulation.out')

# Analyze fields
print(f"Max Ez field: {np.max(np.abs(ez_data))}")
print(f"Min Ez field: {np.min(np.abs(ez_data))}")
```

### Parameter Extraction

```python
# Extract simulation parameters
import gprMax.parameters

# Get time step
dt = gprMax.parameters.get_dt()

# Get Courant number
courant = gprMax.parameters.get_courant()

# Get cell size
cell_size = gprMax.parameters.get_cell_size()
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

**Problem**: `ImportError: No module named 'gprMax'`

**Solution**: Install gprMax
```bash
conda activate gprmax
```

### Simulation Divergence

**Problem**: Simulation diverges or produces NaN values

**Solutions**:
1. Reduce time step (decrease Courant number)
2. Check material stability
3. Verify boundary conditions
4. Check source configuration

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce domain size
2. Reduce output frequency
'3. Use GPU acceleration
4. Reduce number of output components

### Performance Issues

**Problem**: Simulation is too slow

**solutions**:
1. Use GPU acceleration
2. Enable OpenMP parallelization
3. Reduce output frequency
4. Optimize Python code