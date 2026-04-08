# Input File Commands in gprMax

gprMax uses text-based input files (.in) to define simulations.

## Essential Commands

### Domain Definition

```python
# 2D domain (z = 0)
#domain: 0.240 0.210 0.002

# 3D domain
#domain: 0.240 0.210 0.002 0.002
```

### Spatial Resolution

```python
# Spatial resolution (meters)
#dx_dy_dz: 0.002 0.002 0.002

# Resolution rule: dx <= lambda_min / 10
# where lambda_min is minimum wavelength in simulation
```

### Material Definitions

```python
# Built-in materials
#material: 6 0 1 0 free_space  # Air (free space)
#material: 6 0 1 0 pec  # Perfect electric conductor
#material: 6 0 1 0 half_space epsilon_r=6.0 sigma=0.0

# User-defined materials
#material: 6 0 1 0 my_material epsilon_r=10.0 sigma=0.01
```

### Source Commands

```python
# Hertzian dipole source
#waveform: ricker 1.5e9 my_ricker
#hertzian_dipole: z 0.100 0.170 0 my_ricker

# Gaussian pulse source
#waveform: gaussian 1.5e9 0.2e9
#source: 0 0 0.100 0.170 0 gaussian 1.5e9 0.2e9

# Plane wave source
#waveform: plane_wave
#source: 0 0 0.100 0.170 0 plane_wave
```

### Object Construction

```python
# Box (parallelepiped)
#box: 0 0 0 0.240 0.170 0.002 half_space

# Cylinder
#cylinder: 0.120 0.080 0.120 0.080 0.002 pec

# Sphere
#sphere: 0.0 0 0.0 0.0 0.050 pec
```

### Output Commands

```python
# Electric field output
#output_efield: 0 0 0 0.0 0.0

# Magnetic field output
#output_hfield: 0 0 0 0.0 0.0

# Current output
#output_current: 0 0 0.0.0 0.0

# Geometry view
#geometry_view: 0 0 0.0.240 0.210 0.002
```

## General Commands

### Comments

```python
# This is a comment
# Comments start with #
```

### Variables

```python
# Variable definitions
#my_variable: 10.0
```

### Python Code

```python
# Python code in input file
import gprMax
import numpy as np

# Your Python code here
```

## Advanced Commands

### PML Configuration

```python
# PML absorbing boundaries
#pml: 1.0 1.0 1.0 0.0

# PML with profile
#pml: 1.0 1.0 1.0 0.0 pml_profile 0.05 0.05 0.05
```

### Boundary Conditions

```python
# Periodic boundary conditions
#k_point: 0.1 0.0 0.0  # Bloch wavevector

# Perfect electric conductor
#boundary: 6 0 1 0 pec
```

### Time Stepping

```python
# Courant number (CFL condition)
#courant: 0.5

# Time window
#time_window: 3e-9
```

## Best Practices

### File Organization

1. **Use descriptive names**: `cylinder_Ascan_2D.in`
2. **Include comments**: Document simulation purpose
3. **Logical ordering**: Domain → Materials → Objects → Sources → Output

### Parameter Selection

1. **Resolution**: Follow λ_min/10 rule
2. **Time window**: Include full source pulse duration
3. **PML thickness**: 1-2 wavelengths
4. **Domain size**: Include PML and padding

### Material Modeling

1. **Use appropriate models**: Choose correct material for application
2. **Frequency range**: Ensure material valid over frequency range
3. **Stability**: Check for numerical stability issues

### Source Configuration

1. **Source placement**: Keep sources away from boundaries
2. **Polarization**: Specify correct field polarization
3. **Frequency**: Use appropriate frequency for application

### Output Configuration

1. **Field components**: Output needed components only
2. **Output frequency**: Balance resolution and storage
3. **Geometry views**: Useful for verification

## Troubleshooting

### Simulation Divergence

**Problem**: Simulation diverges or produces NaN/Inf values

**Solutions**:
1. Reduce time step (decrease Courant number)
2. Check material stability
3. Verify boundary conditions
4. Check source configuration

### Poor Convergence

**Problem**: Results don't converge with increasing resolution

**Solutions**:
1. Increase spatial resolution
2. Use subpixel averaging
3. Check PML thickness
4. Verify material parameters

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce domain size
2. Reduce output frequency
3. Use GPU if available
4. Use MPI for distributed memory

### Performance Issues

**Problem**: Simulation too slow

**Solutions**:
1. Use GPU acceleration
2. Enable OpenMP parallelization
3. Reduce output frequency
4. Optimize Python code

### Output Issues

**Problem**: Output files not generated or incorrect

**Solutions**:
1. Check output commands are correct
2. Verify output file paths
3. Check file permissions
4. Check disk space