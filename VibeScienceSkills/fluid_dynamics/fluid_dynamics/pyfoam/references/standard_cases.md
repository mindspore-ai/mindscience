# Standard Cases

OpenFOAM provides numerous standard cases for testing, validation, and learning.

## Tutorial Cases

### Cavity Case

**Description**: Lid-driven cavity flow (Re = 100)

**Run**:
```bash
cd $FOAM_TUTORIALS
python cavity.py
```

**Characteristics**:
- 2D incompressible flow
- Re = 100 (laminar)
- Lid velocity: 1 m/s
- Steady-state solution
- Standard benchmark case

**Physics**:
- Incompressible Navier-Stokes equations
- SIMPLE solver
- Upwind differencing scheme
- No-slip walls

**Mesh**:
- Structured hexahedral mesh
- Boundary layer refinement
- Non-orthogonal near lid
- Uniform cell size

**Solver**:
- simpleFoam (SIMPLE algorithm)
- Steady-state solver
- Pressure-velocity coupling
- Max Courant number: 0.5

**Boundary Conditions**:
- Fixed walls (no-slip)
- Moving lid (U = (1, 0))
- Zero-gradient walls

**Initial Conditions**:
- U = (0, 0) everywhere
- P = 0 (gauge pressure)
- T = 300 K (isothermal)

**Output**:
- Velocity field U(x,y,t)
- Pressure field p
- Convergence history
- Residual fields

**Typical Results**:
- Lid-driven circulation
- Corner vortices
- Parabolic velocity profile
- Steady-state convergence in ~500 iterations

### Dam Break Case

**Description**: 2D dam break problem

**Run**:
```bash
cd $FOAM_RUN
blockMeshDict damBreak
```

**Characteristics**:
- Free-surface flow
- Gravity-driven collapse
- Wave propagation
- SPHERIC benchmark test (Test 2)

**Physics**:
- Weakly compressible SPH formulation
- Gravity effects
- Free-surface boundary conditions
- Surface tension effects

**Mesh**:
- Structured hexahedral mesh
- Boundary layer refinement
- Uniform cell size
- Higher resolution near free surface

**Solver**:
- interFoam (transient solver)
- PISO algorithm
- Adaptive time stepping
- Max Courant number: 0.5

**Boundary Conditions**:
- Fixed walls (no-slip)
- Zero-gradient walls
- Free-surface at top
- Atmospheric pressure at sides

**Initial Conditions**:
- Water column: 2m × 1m
- U = (0, 0) everywhere
- P = 101325 Pa (atmospheric)
- T = 300 K

**Output**:
- Volume fraction field alpha
- Velocity field U
- Free-surface detection
- Wave propagation

**Typical Results**:
- Column collapse
- Wave front propagation
- Free-surface evolution
- Impact on opposite wall

### Dam Break 3D

**Description**: 3D dam break problem

**Run**:
```bash
cd $FOAM_RUN
blockMeshDict damBreak3D
```

**Characteristics**:
- 3D free-surface flow
- Large particle count
- SPHERIC benchmark test (Test 5)
- Requires significant computational resources

**Physics**:
- Weakly compress SPH formulation
- Gravity effects
- Free-surface boundary conditions
- Surface tension effects

**Mesh**:
- 3D structured mesh
- High resolution near walls
- Boundary layer refinement
- Uniform cell size

**Solver**:
- interFoam (transient solver)
- PISO algorithm
- Adaptive time stepping
- Parallel execution recommended

**Boundary Conditions**:
- Fixed walls (no-slip)
- Zero-gradient walls
- Free-surface at top
- Atmospheric pressure at sides

**Initial Conditions**:
- Water column: 2m × 1m × 1m
- U = (0, 0, 0) everywhere
- P = 101325 Pa (atmospheric)
- T = 300 K

**Output**:
- Volume fraction field alpha
- Velocity field U
- Free-surface detection
- Wave propagation

**Typical Results**:
- 3D collapse behavior
- Wave propagation in 3D
- Free-surface evolution
- Complex wave interactions

### Hot Room Case

**Description**: Buoyant convection in enclosure

**Run**:
```bash
cd $FOAM_TUTORIALS
buoyantHotRoom
```

**Characteristics**:
- Natural convection
- Heat transfer
- Buoyancy force
- Turbulence modeling

**Physics**:
- Boussinesq equation
- Turbulence modeling
- Heat transfer equation
- Ideal gas law for buoyancy

**Mesh**:
- Structured mesh
- Near-wall refinement
- Non-orthogonal cells
- Boundary layer for heat transfer

**Solver**:
- buoyantBoussinesqSimpleFoam
- Transient solver
- Turbulence model integration

**Boundary Conditions**:
- Fixed walls (no-slip)
- Zero-gradient walls
- Temperature boundary conditions

**Initial Conditions**:
- T = 300 K (room temperature)
- P = 101325 Pa
- U = (0, 0) (initially still)

**Output**:
- Temperature field T
- Velocity field U
- Buoyancy force field
- Heat flux

**Typical Results**:
- Buoyant convection patterns
- Temperature stratification
- Steady-state temperature field
- Heat transfer rates

### MotorBike Case

**Description**: Flow around rotating motorcycle

**Run**:
```bash
cd $FOAM_TUTORIALS
motorBike
```

**Characteristics**:
- Moving boundary (rotating wheel)
- Overset mesh
- Multiphase flow
- Reference frame motion

**Physics**:
- Incompressible Navier-Stokes
- Moving wall boundary conditions
- Reference frame transformation
- Overset mesh motion

**Mesh**:
- Overset structured mesh
- Moving boundary refinement
- Rotating reference frame
- Local refinement near wheel

**Solver**:
- SRFSimpleFoam (SRFS formulation)
- Transient solver
- Moving mesh capabilities
- Overset mesh motion

**Boundary Conditions**:
- Rotating wall (angular velocity)
- Fixed walls
- Overset boundary conditions

**Initial Conditions**:
- U = (0, 0) everywhere
- P = 101325 Pa
- T = 300 K

**Output**:
- Velocity field U
- Rotating frame position
- Forces on rotating wheel

**Typical Results**:
- Flow around rotating geometry
- Wake patterns
- Force distribution
- Reference frame tracking

## SPHERIC Benchmarks

OpenFOAM includes SPHERIC benchmark tests for validation:

### Test 2: 2D Oscillating Drop

**Description**: 2D oscillating drop under gravity

**Run**:
```bash
cd $FOAM_RUN
oscillatingDrop2D
```

**Characteristics**:
- Surface tension modeling
- Gravity effects
- Free-surface boundary conditions
- SPHERIC validation test

**Physics**:
- Weakly compressible SPH
- Surface tension force
- Gravity force
- Free-surface detection

**Mesh**:
- High-resolution structured mesh
- Boundary layer refinement
- Free-surface particle refinement

**Solver**:
- interFoam solver
- Adaptive time stepping
- High accuracy settings

**Boundary Conditions**:
- Free-surface boundaries
- Zero-gradient walls
- Atmospheric pressure

**Initial Conditions**:
- Drop radius: 0.25 m
- Drop position: (1, 1) m
- U = (0, 0) initially
- P = 101325 Pa

**Output**:
- Volume fraction field
- Velocity field
- Free-surface detection
- Surface tension force

**Validation**:
- Compare with experimental data
- Check mass conservation
- Verify surface tension modeling

### Test 3: 2D Dam Break

**Description**: Standard 2D dam break

**Run**:
```bash
cd $FOAM_RUN
damBreak2D
```

**Characteristics**:
- Free-surface flow
- Gravity-driven collapse
- Wave propagation
- SPHERSPH benchmark test

**Physics**:
- Weakly compressible SPH
- Gravity effects
- Free-surface boundary conditions
- Momentum conservation

**Mesh**:
- Structured hexahedral mesh
- Boundary layer refinement
- Free-surface refinement

**Solver**:
- interFoam solver
- PISO algorithm
- Adaptive time stepping

**Boundary Conditions**:
- Free-surface boundaries
- Zero-gradient walls
- Atmospheric pressure

**Initial Conditions**:
- Water column: 2m × 1m
- U = (0, 0) initially
- P = 101325 Pa

**Output**:
- Volume fraction field
- Velocity field
- Free-surface detection
- Wave propagation

**Validation**:
- Compare with experimental data
- Check mass conservation
- Verify wave propagation speed

### Test 5: 3D Dam Break

**Description**: 3D dam break

**Run**:
```bash
cd $FOAM_RUN
damBreak3D
```

**Characteristics**:
- 3D free-surface flow
- Large particle count
- SPHERIC benchmark test
- Requires significant computational resources

**Physics**:
- Weakly compressible SPH
- Gravity effects
- Free-surface boundary conditions
- Momentum conservation

**Mesh**:
- 3D structured mesh
- High resolution near walls
- Boundary layer refinement
- Free-surface refinement

**Solver**:
- interFoam solver
- PISO algorithm
- Adaptive time stepping
- Parallel execution recommended

**Boundary Conditions**:
- Free-surface boundaries
- Zero-gradient walls
- Atmospheric pressure

**Initial Conditions**:
- Water column: 2m × 1m × 1m
- U = (0, 0, 0) initially
- P = 101325 Pa

**Output**:
- Volume fraction field
- Velocity field
- Free-surface detection
- Wave propagation

**Validation**:
- Compare with experimental data
- Check mass conservation
- Verify 3D behavior

### Test 7: 3D Lock Exchange

**Description**: 3D lock exchange

**Run**:
```bash
cd $FOAM_RUN
lockExchange3D
```

**Characteristics**:
- 3D free-surface flow
- Lock exchange mechanism
- Particle tracking
- SPHERIC benchmark test

**Physics**:
- Weakly compressible SPH
- Lock exchange modeling
- Particle tracking
- Momentum conservation

**Mesh**:
- 3D structured mesh
- Lock exchange regions
- Boundary layer refinement

**Solver**:
- interFoam solver
- PISO algorithm
- Lock exchange model

**Boundary Conditions**:
- Lock exchange boundaries
- Free-surface boundaries
- Zero-gradient walls

**Initial Conditions**:
- Two chambers with lock
- U = (0, 0) initially
- P = 101325 Pa
- T = 300 K

**Output**:
- Volume fraction field
- Velocity field
- Lock exchange tracking
- Particle positions

**Validation**:
- Compare with experimental data
- Check mass conservation
- Verify lock exchange mechanism

## Case Selection Guide

| Problem Type | Recommended Case | Reason |
|--------------|------------------|---------|
| Incompressible flow | Cavity | Standard benchmark |
| Free-surface flow | Dam break | SPHERIC test |
| Heat transfer | Hot room | Buoyant convection |
| Moving boundaries | MotorBike | Overset mesh |
| Multiphase flow | MultiPhaseMixing | Phase interactions |

## Case Modification

### Modifying Standard Cases

```bash
# Copy standard case
cp -r $FOAM_TUTORIALS/cavity .
cd cavity

# Edit control dictionary
# Modify parameters in controlDict

# Run modified case
blockMeshDict cavity
```

### Creating Custom Cases

```python
# Create case structure
import os

case_dir = 'my_case'
os.makedirs(case_dir)

# Create necessary directories
os.makedirs(f'{case_dir}/0')
os.makedirs(f'{case_dir}/constant')
os.makedirs(f'{case_dir}/system')

# Create files
# 0/ mesh generation
# constant/ transport properties
# system/controlDict
```

## Running Cases

### Basic Execution

```bash
# Standard execution
cd $FOAM_RUN
blockMeshDict cavity

# With custom case
cd $FOAM_RUN
blockMeshDict ../my_case
```

### Parallel Execution

```bash
# Parallel with 4 cores
mpirun -np 4 blockMeshDict cavity -parallel

# With specific method
mpirun -np 4 blockMeshDict cavity -parallel -decompose metis
```

### Solver Selection

```bash
# Use specific solver
# Edit controlDict to change solver
application {
    solver pimpleFoam;
}
```

## Case Analysis

### Convergence Checking

```python
# Monitor convergence
import re

# Read log file
with open('log.simpleFoam') as f:
    for line in f:
        if 'Initialisation' in line:
            print(line)
        if 'solution diverges' in line.lower():
            print("Divergence detected!")
```

### Residual Analysis

```python
# Calculate residuals
import numpy as np

# Read field data
# Calculate L2 norm of residual
residual_norm = np.linalg.norm(residual_field)
```

### Validation Metrics

**Common metrics**:
- Convergence rate
- Residual norms
- Mass conservation
- Energy conservation
- Momentum conservation

**Validation tools**:
- paraFoam for post-processing
- sample for data analysis
- PyFoamPlotter for visualization

## Performance Optimization

### Mesh Optimization

**Guidelines**:
- Use appropriate cell size for flow features
- Refine boundary layers for accuracy
- Use non-orthogonal meshes for complex geometries
- Consider mesh quality metrics

### Solver Optimization

**Guidelines**:
- Choose appropriate solver for problem type
- Adjust solver tolerances for accuracy
- Use appropriate differencing schemes
- Consider preconditioning

### Time Stepping

**Guidelines**:
- Use adaptive time stepping for stability
- Adjust Courant number for accuracy
- Consider max Courant number for stability
- Use appropriate time step size

## Common Issues and Solutions

### Convergence Problems

**Problem**: Solution diverges or fails to converge

**Solutions**:
- Reduce time step size
- Improve mesh quality
- Check boundary conditions
- Try different solver
- Check initial conditions

### Stability Issues

**Problem**: Simulation becomes unstable

**Solutions**:
- Reduce Courant number
- Improve mesh quality
- Check boundary conditions
- Use appropriate differencing scheme
- Verify physical properties

### Accuracy Issues

**Problem**: Results differ from expected values

**Solutions**:
- Refine mesh
- Increase solver accuracy
- Check boundary conditions
- Verify physical properties
- Compare with analytical solutions

### Performance Issues

**Problem**: Simulation runs too slowly

**Solutions**:
- Use parallel execution
- Optimize mesh
- Use appropriate solver
- Reduce output frequency
- Consider GPU acceleration

## Best Practices

1. **Start with standard cases**: Use tutorial cases for learning
2. **Understand physics**: Know what each case demonstrates
3. **Study case structure**: Examine standard case organization
4. **Modify incrementally**: Start small, validate, then expand
5. **Use appropriate solver**: Match solver to problem type
6. **Monitor convergence**: Check solution quality during simulation
7. **Validate results**: Check against benchmarks or analytical solutions
