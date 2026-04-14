# Output and Visualization in gprMax

gprMax provides various output options for simulation results and visualization.

## Output File Types

### Output Files (.out)

Primary output files containing simulation results:
- **Field data**: Electric and magnetic field components
- **Currents**: Surface and volume currents
- **Parameters**: Material and source parameters
- **Time data**: Simulation time information

### HDF5 Files (.h5)

HDF5 format files for field data:
- **Field snapshots**: 3D field arrays
- **Geometry data**: Material distribution
- **Source data**: Source time evolution
- **Parameter data**: Material parameters

### Geometry Files (.vtk, .stl)

3D geometry visualization files:
- **VTK format**: 3D geometry meshes
- **STL format**: Stereo lithography files

## Output Commands

### Field Output

```python
# Electric field output
#output_efield: 0 0 0.0  # Output Ez field
#output_hfield: 0 0 0.0  # Output Hx, Hy, Hz fields

# Magnetic field output
#output_bfield: 0 0 0.0  # Output Bx, By, Bz fields
#output_dfield: 0 0 0 0  # Output Dx, Dy, Dz fields
```

### Current Output

```python
# Surface current output
#output_current: 0 0 0.0 0  # Surface currents

# Volume current output
#output_current_vol: 0 0 0 0.0 0  # Volume currents
```

### Parameter Output

```python
# Material parameter output
#output_material: 0 0 0.0 0 0 0 0 0  # Material parameters

# Source parameter output
#output_source: 0 0 0.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  fanc
```

## Visualization Tools

### Built-in Plotting Tools

```bash
# Plot A-scan results
python -m tools.plot_Ascan simulation.out

# Plot B-scan results
python -m tools.plot_Bscan simulation.out

# Convert to PNG
python -m tools.convert_png2h5 simulation.out
```

### Python Visualization

```python
import gprMax
import numpy as np
import matplotlib.pyplot as plt

# Load output data
data = gprMax.io.load_output('simulation.out')

# Plot electric field
plt.figure(figsize=(10, 8))
plt.imshow(data['ez'][-1], cmap='RdBu', aspect='auto')
plt.colorbar(label='Ez field')
plt.title('Electric Field')
plt.show()
```

### 3D Visualization

```python
# Create 3D field animation
from gprMax.viz import plot_3d_fields

# Plot Ez field evolution
plot_3d_fields('simulation.out', component='ez', 
                  output_dir='animation/')
```

## Output Configuration

### Output Frequency

```python
# Output frequency control
#output_freq: 0.0 0.0 0.0  # Output frequency
#output_freq_step: 0.0 0.0.0 0  # Output frequency step
```

### Output Components

```python
# Select field components to output
#output_efield: 0 0 0.0 0  # Output electric fields
#output_hfield: 0 0 0.0 0  # Output magnetic fields
#output_bfield: 0 0 0 0.0  # Output B and D fields
#output_current: 0 0 0.0 0 0 0  # Output currents
```

### Output Volumes

```python
# Output specific volumes
#output_volume: 0.0 0 0 0 0.0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0  fanc
```

## Output Best Practices

### Output Frequency Selection

1. **Match source bandwidth**: Output frequency should capture source spectrum
2. **Consider time requirements**: Higher frequency needs more time steps
3. **Balance resolution**: Higher frequency needs higher spatial resolution

### Output Component Selection

1. **Output needed components only**: Reduce output file size
2. **Electric vs magnetic**: Choose based on application
3. **Consider polarization**: Match source polarization

### Output Volume Selection

1. **Output regions of interest**: Focus on important spatial regions
2. **Reduce output size**: Minimize output file size
3. **Consider symmetry**: Exploit symmetry in output

### Output File Management

1. **Organize output files**: Use descriptive names
2. **Archive results**: Keep important results
3. **Clean up temporary files**: Remove unnecessary output files

## Output Troubleshooting

### No Output Files Generated

**Problem**: Simulation completes but no output files

**Solutions**:
1. Check output commands in input file
2. Verify output frequency is appropriate
3. Check simulation time window

### Incorrect Output Data

**Problem**: Output data appears incorrect

**Solutions**:
1. Verify output commands
2. Check material definitions
3. Check source configuration
4. Check boundary conditions

### Large Output Files

**Problem**: Output files are very large

**Solutions**:
1. Reduce output frequency
2. Reduce output components
3. Reduce output volumes
4. Reduce output time window

### Visualization Issues

**Problem**: Plots don't display correctly

**Solutions**:
1. Check output data format
2. Verify visualization tools are installed
3. Check plotting parameters
4. Try different visualization methods