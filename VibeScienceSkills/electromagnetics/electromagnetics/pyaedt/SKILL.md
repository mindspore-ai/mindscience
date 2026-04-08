---
name: pyaedt
description: Official Ansys Electronics Desktop (AEDT) Python API providing comprehensive control of Ansys electromagnetic simulation tools
license: Commercial License
---

# PyAEDT

## Overview
PyAEDT is the official Python API for Ansys Electronics Desktop (AEDDT). It provides comprehensive control of Ansys electromagnetic simulation tools including HFSS, Maxwell, Icepak, Q3D, and other Ansys electromagnetic solvers, enabling automation of electromagnetic simulations.

## When to Use This Skill
Use this skill when you need to:
- Automate Ansys HFSS electromagnetic simulations
- Control Ansys Maxwell electromagnetic field simulations
- Automate Ansys Icepak thermal simulations
- Work with Ansys Q3D extractor simulations
- Batch process electromagnetic simulations
- Integrate Ansys simulations into Python workflows
- Perform parametric sweeps and optimizations
- Automate electromagnetic design and analysis

## Core Capabilities
- **HFSS Control**: Automate 3D full-wave electromagnetic simulations
- **Maxwell Control**: Control low-frequency electromagnetic simulations
- **Icepak Control**: Automate thermal and fluid flow simulations
- **Q3D Control**: Automate quasi-static electromagnetic simulations
- **Parametric Analysis**: Perform parametric sweeps and optimizations
- **Geometry Creation**: Create and modify 3D geometry
- **Mesh Control**: Control mesh generation and refinement
- **Post-Processing**: Extract and analyze simulation results

## Installation
```bash
pip install pyaedt
```

Requirements:
- Ansys Electronics Desktop (AEDT) installation
- Valid Ansys license

## Usage Examples

### Basic HFSS Simulation
```python
import pyaedt

# Launch or connect to AEDT
hfss = pyaedt.Hfss()

# Create variable
hfss["freq"] = "10GHz"
hfss["length"] = "10mm"

# Create box
box = hfss.modeler.create_box(
    position=[0, 0, 0],
    size_list=["10mm", "10mm", "10mm"],
    name="waveguide"
)

# Assign material
box.material_name = "copper"

# Create port
port = hfss.wave_port(
    face_id=box.faces[0],
    name="Port1",
    integration_line_offset=0.5
)

# Setup solution setup
setup = hfss.create_setup(name="Setup1")
setup.props["Frequency"] = "10GHz"

# Add frequency sweep
hfss.create_frequency_sweep(
    setup_name="Setup1",
    sweep_type="Interpolating",
    start_freq="1GHz",
    stop_freq="20GHz",
    num_of_freq_points=100
)

# Analyze
hfss.analyze_all()

# Get S-parameters
s_params = hfss.get_s_parameters()
print(f"S11: {s_params[0, 0, :]}")
```

### Maxwell Simulation
```python
import pyaedt

# Launch Maxwell
maxwell = pyaedt.Maxwell3d()

# Create geometry
coil = maxwell.modeler.create_coil(
    name="coil",
    radius="5mm",
    height="10mm",
    number_of_turns=100
)

# Assign material
coil.material_name = "copper"

# Create excitation
excitation = maxwell.assign_current(
    obj_name=coil.name,
    amplitude="1A",
    name="Current1"
)

# Setup solution
setup = maxwell.create_setup(name="Setup1")
setup.props["TimeStop"] = "1ms"
setup.props["TimeStep"] = "1us"

# Analyze
maxwell.analyze_all()

# Get results
flux = maxwell.get_flux_density()
print(f"Max flux density: {max(flux)}")
```

### Parametric Sweep
```python
import pyaedt

# Launch HFSS
hfss = pyaedt.Hfss()

# Create parameter
hfss["width"] = "5mm"

# Create geometry with parameter
box = hfss.modeler.create_box(
    position=[0, 0, 0],
    size_list=["width", "10mm", "10mm"],
    name="waveguide"
)

# Setup solution
setup = hfss.create_setup(name="Setup1")
setup.props["Frequency"] = "10GHz"

# Create parametric sweep
sweep = hfss.parametrics.add(
    setup_name="Setup1",
    variable_name="width",
    start="5mm",
    stop="10mm",
    step="1mm"
)

# Analyze
hfss.analyze_all()

# Get parametric results
results = hfss.get_parametric_results("width", "S11")
print(f"Parametric results: {results}")
```

### Geometry Manipulation
```python
import pyaedt

# Launch HFSS
hfss = pyaedt.Hfss()

# Create cylinder
cyl = hfss.modeler.create_cylinder(
    position=[0, 0, 0],
    radius="5mm",
    height="20mm",
    axis="Z",
    name="cylinder"
)

# Create sphere
sphere = hfss.modeler.create_sphere(
    position=["10mm", 0, 0],
    radius="5mm",
    name="sphere"
)

# Boolean operation
union = hfss.modeler.unite([cyl.name, sphere.name],
                           name="combined")

# Create air region
air_region = hfss.modeler.create_region(
    padding_percentage=50,
    name="Region"
)

# Assign air material
air_region.material_name = "air"
```

### Post-Processing
```python
import pyaedt

# Launch HFSS
hfss = pyaedra.Hfss()

# Analyze simulation
hfss.analyze_all()

# Get S-parameters
s11 = hfss.get_s_parameter(port1=1, port2=1)
s21 = hfss.get_s_parameter(port1=2, port2=1)

# Get field data
fields = hfss.get_fields_data(
    setup_name="Setup1",
    sweep_name="LastAdaptive",
    quantity="Mag_E",
    filter_objects=["waveguide"]
)

# Export fields
hfss.export_fields(
    setup_name="Setup1",
    sweep_name="LastAdaptive",
    file_name="fields.fld",
    quantity="Mag_E"
)

# Create report
report = hfss.post.create_report(
    name="S-Parameters",
    setup_name="Setup1",
    sweep_name="LastAdaptive",
    expressions=["S11", "S21"],
    plot_type="Rectangular"
)
```

## Applications
- **RF/Microwave Design**: Design and optimize RF components
- **Antenna Design**: Simulate and analyze antenna performance
- **EMC/EMI Analysis**: Electromagnetic compatibility analysis
- **Signal Integrity**: Analyze signal integrity in high-speed designs
- **Power Integrity**: Analyze power distribution networks
- **Thermal Analysis**: Thermal and fluid flow simulations
- **Motor Design**: Design and optimize electric motors
- **Transformer Design**: Design and optimize transformers

## Advanced Features
- **Script Recording**: Record GUI actions as Python scripts
- **Custom Materials**: Define custom material properties
- **Mesh Operations**: Advanced mesh control and refinement
- **Optimization**: Built-in optimization algorithms
- **Sensitivity Analysis**: Perform sensitivity analysis
- **Distributed Computing**: Use distributed computing for large simulations
- **Batch Processing**: Process multiple simulations in batch

## Best Practices
- **Resource Management**: Monitor and manage memory usage
- **Geometry Cleanup**: Clean up geometry to avoid issues
- **Mesh Quality**: Ensure adequate mesh quality for accuracy
- **Convergence**: Monitor convergence for accurate results
- **Validation**: Validate results against analytical solutions
- **Documentation**: Document simulation setup and parameters

## References
- Documentation: https://aedt.docs.pyansys.com/
- PyAEDT GitHub: https://github.com/ansys/pyaedt
- Ansys: https://www.ansys.com/
- Requirements: Ansys Electronics Desktop, valid license
