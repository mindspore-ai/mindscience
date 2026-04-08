---
name: gprmax
description: Open-source FDTD electromagnetic wave propagation simulator for Ground Penetrating Radar (GPR). Use for: (1) 2D/3D electromagnetic simulations using Finite-Difference Time-Domain (FDTD) method, (2) GPR antenna modeling and ground penetration analysis, (3) Material modeling including dielectric, magnetic, and dispersive media, (4) Source excitation including Hertzian dipoles and Gaussian pulses, (5) PML absorbing boundary conditions, (6) Field output and visualization, (7) Parallel computing with OpenMP/MPI/GPU support, (8) Python scripting for complex simulation workflows.
license: MIT License
metadata:
    skill-author: Mindspore Science Team
---

# gprMax: Ground Penetrating Radar Simulator

gprMax is an open-source software package for electromagnetic wave propagation simulation using the Finite-Difference Time-Domain (FDTD) method. It was designed for Ground Penetrating Radar (GPR) modeling but can be used for a wide range of electromagnetic applications.

## Quick Start

Basic 2D simulation workflow:

```bash
# Run basic simulation
python -m gprMax user_models/cylinder_Ascan_2D.in

# Plot results
python -m tools.plot_Ascan user_models/cylinder_Ascan_2D.out
```

## Core Concepts

### FDTD Method

gprMax solves Maxwell's equations in 3D using the FDTD method:
- **Spatial discretization**: Yee lattice with electric and magnetic fields offset
- **Temporal discretization**: Leapfrog time stepping based on CFL condition
- **Boundary conditions**: PML absorbing layers, perfect conductors, periodic boundaries
- **Material modeling**: Frequency-dependent and nonlinear materials

### Coordinate System

gprMax uses a Cartesian coordinate system:
- **2D simulations**: x, y coordinates (z = 0)
- **3D simulations**: x, y, z coordinates
- **Units**: User-defined length units (meters, millimeters, etc.)

### Spatial Resolution

Resolution is critical for accuracy:
- **Rule of thumb**: Δx ≤ λ_min/10 (at least 10 cells per minimum wavelength)
- **Higher resolution**: Δx ≤ λ_min/20 for better accuracy
- **Courant number**: CFL = c·Δt/Δx ≤ 1/√3 (typically 0.5)

### Time Stepping

- **Automatic**: Time step calculated automatically from CFL condition
- **Manual override**: Can specify custom time step if needed
- **Time window**: Duration of simulation based on source and propagation

## Common Workflows

### Workflow 1: Basic 2D Simulation

Simple 2D electromagnetic simulation:

```python
# Input file commands
#domain: 0.240 0.210 0.002 0.002 0.002 2D
#dx_dy_dz: 0.002 0.002 0.002
#material: 6 0 1 0 free_space
#source: 0 0.0 0 0.100 0.170 0 ricker 1.5e9 my_ricker
#waveform: ricker 1.5e9 my_ricker
#geometry_view: 0 0 0 0.240 0.210 0.002
```

### Workflow 2: GPR A-Scan

Ground penetrating radar simulation:

```python
# Input file for GPR A-scan
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#material: 6 0 1 0 half_space
#source: 0 0 0 0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker
#src_steps: 0.002 0
#rx_steps: 0.002 0
#box: 0 0 0 0.240 0.170 0.002
```

### Workflow 3: B-Scan

Multiple trace simulation:

```python
# Input file for B-scan
#domain: 0.240 0.210 0.002
#dx_dy_dz: 0.002 0.002 0.002
#material: 6 0 1 0 half_space
#source: 0 0 0 0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker
#src_steps: 0.002 0
#rx_steps: 0.002 0
#box: 0 0 0 0.240 0.170 0.002
```

### Workflow 4: Material 3D Simulation

3D simulation with complex materials:

```python
# Input file with material definitions
#domain: 0.240 0.210 0.002 0.002 0.002 3D
#dx_dy_dz: 0.002 0.002 0.002
#material: 6 0 1 0 half_space
#material: 6 0 1 0 pec 10.0
#source: 0 0 0 0 0.100 0.170 0 gaussian 1.5e9 0.2e9
```

### Workflow 5: Python Scripting

Complex simulation with Python scripting:

```python
# Python code in input file
import gprMax

# Define parameters
domain = [0.240, 0.210, 0.002]
dx_dy_dz = [0.002, 0.002, 0.002]

# Create materials
half_space = gprMax.Medium(epsilon_r=6.0, sigma=0.0)
pec_material = gprMax.Medium(epsilon_r=10.0, sigma=0.0)

# Add geometry
gprMax.geometry([gprMax.Box(size=[0.240, 0.210, 0.002], 
                         material=half_space)])

# Add source
gprMax.source([gprMax.HertzianDipole(center=[0, 0, 0.100], 
                                     z=0.170, 
                                     f=1.5e9, 
                                     polarization='z')])

# Run simulation
gprMax.run(until=200)
```

## Running gprMax

### Basic Execution

```bash
# Activate gprMax environment
conda activate gprmax

# Run simulation
python -m gprMax input_file.in

# Plot results
python -m tools.plot_Ascan input_file.out
```

### Command Line Options

```bash
# Number of runs
python -m gprMax input_file.in -n 60

# GPU acceleration
python -m gprMax input_file.in -gpu 0

# Restart from specific run
python -m gprMax input_file.in -n 15 -restart 45

# MPI parallelization
python -m gprMax input_file.in -mpi 61

# Benchmarking mode
python -m gprMax input_file.in -benchmark

# Geometry only (no simulation)
python -m gprMax input_file.in --geometry-only
```

### Output Files

gprMax generates several output files:
- **`.out` files: Simulation results (fields, currents, etc.)
- **`.h5` files: HDF5 format field data
- **Geometry files**: 3D geometry views (VTK format)
- **Python processed files**: Files after Python code execution

## Materials

### Built-in Materials

```python
# Free space (air)
#material: 6 0 1 0 free_space

# Perfect electric conductor
#material: 6 0 1 0 pec

# Dielectric half-space
#material: 6 0 1 0 half_space epsilon_r=6.0 sigma=0.0

# Magnetic material
#material: 6 0 1 0 half_space epsilon_r=1.0 mu_r=2.0 sigma=0.0
```

### Material Properties

See [materials.md](references/materials.md) for detailed material definitions including:
- Frequency-dependent materials
- Dispersive materials (Drude, Lorentzian)
- Nonlinear materials (Kerr, Pockels)
- Magnetic materials
- Conductivity

### Python Material Creation

```python
# Create custom materials
import gprMax

# Simple dielectric
dielectric = gprMax.Medium(epsilon_r=4.0, sigma=0.0)

# Dispersive material
from gprMax.materials import Drude
dispersive = Drude(epsilon_inf=4.0, plasma=0.1, gamma=0.05)

# Magnetic material
magnetic = gprMax.Medium(epsilon_r=1.0, mu_r=2.0, sigma=0.0)

# Nonlinear material
kerr = gprMax.Medium(epsilon_r=2.0, chi3=0.1)
```

## Sources

### Source Types

See [sources.md](references/sources.md) for detailed source configurations:
- **Hertzian dipole**: Point source with Ricker waveform
- **Gaussian pulse**: Time-domain Gaussian pulse
- **Plane wave**: Extended plane wave source
- **Custom sources**: User-defined source functions

### Source Configuration

```python
# Hertzian dipole
#source: 0 0 0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker

# Gaussian pulse
#source: 0 0 0 0.100 0.170 0 gaussian 1.5e9 0.2e9

# Plane wave
#source: 0 0 0 0.100 0.170 0 plane_wave
```

### Source Parameters

- **Center**: Source position [x, y, z]
- **Frequency**: Center frequency (Hz)
- **Polarization**: Field polarization (x, y, z)
- **Amplitude**: Source amplitude
- **Width**: Temporal width (for pulses)

## Boundary Conditions

### PML Boundaries

See [boundaries.md](references/boundaries.md) for boundary condition details:
- **PML thickness**: Absorbing layer thickness
- **PML profile**: Absorbing profile parameters
- **Placement**: Around computational domain

### Other Boundaries

```python
# PML boundaries
#pml: 1.0 1.0 1.0

# Perfect electric conductor
#boundary: 6 0 1 0 pec

# Periodic boundaries
#k_point: 0.1 0 0 0  # Bloch wavevector
```

## Output and Visualization

### Output Options

See [output.md](references/output.md) for output configuration:
- **Field components**: Electric and magnetic fields
- **Currents**: Surface and volume currents
- **Geometry views**: 3D geometry visualization
- **HDF5 output**: Field data in HDF5 format

### Visualization Tools

```python
# Plot A-scan
python -m tools.plot_Ascan simulation.out

# Plot B-scan
python -m tools.plot_Bscan simulation.out

# Convert to PNG
python -m tools.convert_png2h5 simulation.out
```

## Parallel Computing

### OpenMP Parallelization

```bash
# Use OpenMP for CPU parallelization
# Automatically detected and used by gprMax
```

### MPI Parallelization

```bash
# MPI task farm
python -m gprMax input_file.in -mpi 61

# Without spawn mechanism
python -m gprMax input_file.in -mpi 61 --mpi-no-spawn
```

### GPU Acceleration

```bash
# Use NVIDIA GPU
python -m gprMax input_file.in -gpu 0

# Use specific GPU
python -m gprMax input_file.in -gpu 0 1
```

See [parallel.md](references/parallel.md) for detailed parallel computing information.

## Python Scripting

### Basic Python Scripting

See [python_scripting.md](references/python_scripting.md) for Python scripting capabilities:
- **Parameter sweeps**: Automated parameter variations
- **Complex geometries**: Programmatic geometry creation
- **Custom analysis**: User-defined analysis functions

### Scripting Examples

```python
# Parameter sweep in input file
import gprMax
import numpy as np

# Sweep parameter
for freq in np.linspace(1.0, 2.0, 10):
    gprMax.source([gprMax.HertzianDipole(f=freq, ...)])
    gprMax.run(until=200)
    gprMax.output(f'results_freq_{freq:.2f}.out')
```

## Best Practices

### Resolution Selection

1. **Minimum resolution**: 10 cells per minimum wavelength
2. **High accuracy**: 20+ cells per minimum wavelength
3. **Material regions**: Higher resolution in high-index materials
4. **Source regions**: Higher resolution near sources

### Domain Sizing

1. **Include PML**: Add PML thickness to domain size
2. **Source distance**: Keep sources at least 10 cells from boundaries
3. **Object padding**: Add padding around objects of interest
4. **Propagation distance**: Ensure sufficient distance for wave propagation

### Time Window

1. **Source duration**: Include full source pulse duration
2. **Propagation time**: Add time for wave propagation
3. **Decay time**: Include time for field decay
4. **Safety margin**: Add 10-20% extra time

### Material Modeling

1. **Use appropriate models**: Choose correct material model for application
2. **Frequency range**: Ensure material valid over frequency range
3. **Stability**: Check for numerical stability issues
4. **Discretization**: Consider subpixel averaging

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
1. Increase resolution
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
2. Enable OpenMP/MPI parallelization
3. Reduce output frequency
4. Optimize Python code

## Resources

### References

- [input_file.md](references/input_file.md) - Input file commands and syntax
- [materials.md](references/materials.md) - Material definitions and properties
- [sources.md](references/sources.md) - Source types and configurations
- [boundaries.md](references/boundaries.md) - Boundary conditions
- [output.md](references/output.md) - Output options and visualization
- [parallel.md](references/parallel.md) - Parallel computing
- [python_scripting.md](references/python_scripting.md) - Python scripting
- [examples.md](references/examples.md) - Common simulation examples

### Scripts

- [basic_simulation.py](scripts/basic_simulation.py) - Basic simulation setup
- [gpr_simulation.py](scripts/gpr_simulation.py) - GPR simulation tools
- [visualization.py](scripts/visualization.py) - Field visualization
- [parameter_sweep.py](scripts/parameter_sweep.py) - Parameter sweep automation

### External Resources

- Official documentation: https://docs.gprmax.com/
- GitHub repository: https://github.com/gprMax/gprMax
- User guide: https://docs.gprmax.com/en/latest/
- Examples: https://docs.gprmax.com/en/latest/examples_simple_2D.html