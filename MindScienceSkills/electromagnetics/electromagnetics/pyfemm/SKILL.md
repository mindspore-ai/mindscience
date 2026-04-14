---
name: pyfemm
description: Python interface for FEMM (Finite Element Method Magnetics) providing electromagnetic and thermal analysis capabilities
license: GPL License
---

# PyFEMM

## Overview
PyFEMM is a Python interface for FEMM (Finite Element Method Magnetics), a suite of programs for solving low-frequency electromagnetic problems on 2D planar and axisymmetric domains. It provides Python scripting capabilities for electromagnetic and thermal analysis.

## When to Use This Skill
Use this skill when you need to:
- Solve low-frequency electromagnetic problems using FEM
- Perform magnetic field analysis and design
- Analyze electric field distributions
- Perform thermal analysis coupled with electromagnetic fields
- Design and optimize electromagnetic devices
- Automate FEMM simulations
- Perform parametric studies and optimizations
- Analyze motors, transformers, and inductors

## Core Capabilities
- **Magnetic Analysis**: 2D planar and axisymmetric magnetic field analysis
- **Electric Analysis**: Electric field and current flow analysis
- **Thermal Analysis**: Heat conduction and thermal analysis
- **Circuit Coupling**: Coupled electromagnetic-circuit analysis
- **Geometry Creation**: Create and modify 2D geometries
- **Mesh Generation**: Automatic mesh generation and refinement
- **Post-Processing**: Extract field values and calculate derived quantities
- **Parametric Studies**: Perform parametric sweeps and optimizations

## Installation
```bash
pip install pyfemm
```

Requirements:
- FEMM software installation (Windows)
- COM interface access

## Usage Examples

### Basic Magnetic Analysis
```python
import pyfemm

# Open FEMM
femm = pyfemm.femm()
femm.openfemm()

# Create new magnetics problem
femm.newdocument(0)  # 0 = magnetics problem

# Define problem type
femm.mi_probdef(0, "meters", "planar", 1e-8, 0, 30)

# Draw geometry
femm.mi_drawrectangle(0, 0, 0.1, 0.05)

# Define materials
femm.mi_getmaterial("Air")
femm.mi_selectgroup(1)
femm.mi_setblockprop("Air", 0, 0, 0, 0, 0, 0)

# Define boundary conditions
femmiesmi_addboundprop("BC0", 0, 0, 0, 0, 0, 0, 0, 0)
femm.mi_selectsegment(0, 0, 0.1, 0)
femm.mi_setsegmentprop("BC0", 0, 1)

# Generate mesh
femm.mi_createmesh()

# Solve problem
femm.mi_analyze()

# Load solution
femm.mi_loadsolution()

# Get field values
bx, by = femm.mi_getb(0.025, 0.025)
print(f"Bx: {bx} T, By: {by} T")
```

### Electric Analysis
```python
import pyfemm

# Open FEMM
femm = pyfemm.femm()
femm.openfemm()

# Create new electrostatics problem
femm.newdocument(1)  # 1 = electrostatics problem

# Define problem type
femm.ei_probdef(0, "meters", "planar", 1e-8, 0, 30)

# Draw geometry
femm.ei_drawrectangle(0, 0, 0.1, 0.05)

# Define materials
femm.ei_getmaterial("Air")
femm.ei_selectgroup(1)
femm.ei_setblockprop("Air", 0, 0, 0)

# Define boundary conditions
femm.ei_addboundprop("BC0", 0, 0, 0, 0, 0, 0, 0)
femm.ei_selectsegment(0, 0, 0.1, 0)
femm.ei_setsegmentprop("BC0", 0, 1)

# Generate mesh
femm.ei_createmesh()

# Solve problem
femm.ei_analyze()

# Load solution
femm.ei_loadsolution()

# Get field values
ex, ey = femm.ei_gete(0.025, 0.025)
print(f"Ex: {ex} V/m, Ey: {ey} V/m")
```

### Thermal Analysis
```python
import pyfemm

# Open FEMM
femm = pyfemm.femm()
femm.openfemm()

# Create new heat flow problem
femm.newdocument(2)  # 2 = heat flow problem

# Define problem type
femm.ht_probdef(0, "meters", "planar", 1e-8, 0, 30)

# Draw geometry
femm.ht_drawrectangle(0, 0, 0.1, 0.05)

# Define materials
femm.ht_getmaterial("Air")
femm.ht_selectgroup(1)
femm.ht_setblockprop("Air", 0, 0, 0, 0, 0)

# Define boundary conditions
femm.ht_addboundprop("BC0", 0, 0, 0, 0, 0, 0, 0)
femm.ht_selectsegment(0, 0, 0.1, 0)
femm.ht_setsegmentprop("BC0", 0, 1)

# Generate mesh
femm.ht_createmesh()

# Solve problem
femm.ht_analyze()

# Load solution
femm.ht_loadsolution()

# Get temperature
temp = femm.ht_gettemperature(0.025, 0.025)
print(f"Temperature: {temp} K")
```

### Parametric Study
```python
import pyfemm
import numpy as np

# Open FEMM
femm = pyfemm.femm()
femm.openfemm()

# Create new magnetics problem
femm.newdocument(0)
femm.mi_probdef(0, "meters", "planar", 1e-8, 0, 30)

# Define geometry
femm.mi_drawrectangle(0, 0, 0.1, 0.05)

# Define materials
femm.mi_getmaterial("Air")
femm.mi_selectgroup(1)
femm.mi_setblockprop("Air", 0, 0, 0, 0, 0, 0)

# Parametric sweep
widths = np.linspace(0.05, 0.15, 11)
results = []

for width in widths:
    # Modify geometry
    femm.mi_clearselected()
    femm.mi_selectsegment(0, 0, 0.1, 0)
    femm.mi_deleteselectedsegments()
    femm.mi_drawrectangle(0, 0, width, 0.05)
    
    # Generate mesh and solve
    femm.mi_createmesh()
    femm.mi_analyze()
    femm.mi_loadsolution()
    
    # Get field
    bx, by = femm.mi_getb(width/2, 0.025)
    results.append((width, bx, by))

# Print results
for width, bx, by in results:
    print(f"Width: {width:.3f} m, Bx: {bx:.3e} T, By: {by:.3e} T")
```

### Circuit Coupling
```python
import pyfemm

# Open FEMM
femm = pyfemm.femm()
femm.openfemm()

# Create new magnetics problem
femm.newdocument(0)
femm.mi_probdef(0, "meters", "planar", 1e-8, 0, 30)

# Draw coil geometry
femm.mi_drawrectangle(0, 0, 0.1, 0.05)

# Define coil material
femm.mi_getmaterial("Copper")
femm.mi_selectgroup(1)
femm.mi_setblockprop("Copper", 100, 0, 0, 0, 0, 0)

# Add circuit
femm.mi_addcircprop("Circuit1", 1, 0, 0, 0, 0, 0)
femm.mi_selectgroup(1)
femm.mi_setblockprop("Copper", 100, 0, 0, 0, 0, "Circuit1")

# Define circuit properties
femm.mi_circuitprop("Circuit1", 1, 0, 0, 0, 0, 0)

# Generate mesh and solve
femm.mi_createmesh()
femm.mi_analyze()
femm.mi_loadsolution()

# Get circuit properties
current = femm.mi_getcircuitcurrent("Circuit1")
voltage = femm.mi_getcircuitvoltage("Circuit1")
print(f"Current: {current} A, Voltage: {voltage} V")
```

## Applications
- **Motor Design**: Design and optimize electric motors
- **Transformer Design**: Design and optimize transformers
- **Inductor Design**: Design and optimize inductors
- **Actuator Design**: Design electromagnetic actuators
- **Sensor Design**: Design magnetic sensors
- **EM Shielding**: Analyze electromagnetic shielding
- **Thermal Analysis**: Analyze thermal effects
- **Circuit Design**: Coupled electromagnetic-circuit analysis

## Advanced Features
- **Complex Geometries**: Create complex 2D geometries
- **Material Libraries**: Access extensive material libraries
- **Mesh Control**: Control mesh density and refinement
- **Field Calculations**: Calculate derived field quantities
- **Export Results**: Export results for further analysis
- **Scripting**: Full scripting capability for automation
- **Integration**: Integrate with other Python tools

## Best Practices
- **Geometry Cleanup**: Clean up geometry to avoid issues
- **Mesh Quality**: Ensure adequate mesh quality
- **Material Properties**: Use correct material properties
- **Boundary Conditions**: Define appropriate boundary conditions
- **Convergence**: Monitor convergence for accurate results
- **Validation**: Validate results against analytical solutions

## References
- FEMM: http://www.femm.info/
- Documentation: http://www.femm.info/wiki/
- PyFEMM: https://pypi.org/project/pyfemm/
- Requirements: FEMM software (Windows), COM interface
