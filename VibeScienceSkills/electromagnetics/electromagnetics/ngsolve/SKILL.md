---
name: ngsolve
description: High-performance multiphysics finite element software for electromagnetic, solid mechanics, and fluid dynamics. Use for: (1) Electromagnetic wave propagation and antenna modeling, (2) RF/microwave circuit simulation, (3) Scattering and diffraction problems, (4) Material modeling and electromagnetic properties, (5) Coupled electromagnetic-structural problems, (6) Waveguide and resonator analysis, (7) Finite element analysis of complex geometries, (8) Time-domain and frequency-domain simulations, (9) Multiphysics coupling and fluid-structure interaction, (10) Python scripting for flexible simulation workflows, (11) High-order accurate numerical methods with adaptive meshing, (12) Parallel computing with OpenMP/MPI support.
---

# NGsolve: High-Performance Multiphysics Finite Element Software

NGsolve is a high-performance open-source multiphysics finite element software package. It provides a flexible Python interface for solving partial differential equations with applications in solid mechanics, fluid dynamics, and electromagnetics.

## Quick Start

Basic electromagnetic simulation workflow:

```python
import ngsolve as ns

# Create simple mesh
mesh = ns.Mesh()
mesh.add_rect(1.0, 1.0, 0.0, 0.5, 0.5, 0.5)

# Create function space
V = ns.FunctionSpace('V')
V.set_parameter_order('E', 'E', 'H')

# Define material
E = ns.Constant('E')
E.set_value(1.0)
V.set_material(E)

# Define problem
problem = ns.Poisson('V', mesh)
problem.set_formulation('sigma(u)*grad(u)**2', 
                              degree=ns.grad(grad)**2)

# Create solver
solver = ns.PoissonSolver('V', mesh, problem)

# Solve
solver.solve()
```

## Core Concepts

### Problem Types

NGsolve supports various problem types:
- **Poisson**: Electrostatic problems
- **Elasticity**: Elastic deformation
- **Incompressible**: Large deformation
- **Stokes**: Fluid flow
- **Navier-Stokes**: Navier-Stokes equations
- **Heat**: Heat conduction
- **Wave**: Wave equations
- **Electromagnetics**: Time-harmonic electromagnetic fields

### Mesh Types

```python
# Create different mesh types
mesh = ns.Mesh()

# Structured mesh
mesh.add_rect(0, 0, 0, 1, 1, 1)

# Unstructured mesh
mesh.add_point((0.5, 0.5), (0.5, 0.5), (0.5, 0.5))

# Curvilinear elements
mesh.add_circle((0, 0), 0.5, 1, 0.5, 1)
```

### Function Spaces

```python
# Scalar field
V = ns.FunctionSpace('V')

# Vector field (2D/3D)
V = ns.VectorFunctionSpace('V', 2)

# Mixed fields
V = ns.MixedFunctionSpace('V', 
                                   {'E': ns.VectorValue(2), 
                                    'H': ns.VectorValue(2)})
```

### Materials

```python
# Constant material
E = ns.Constant('E')
E.set_value(1.0)

# Function material
E = ns.Function('E')
E.set_value('1.0 + x**2')

# Anisotropic material
E = ns.Function('E')
E.set_value([[1.0, 0.5], [0.5, 2.0]])
```

## Common Workflows

### Workflow 1: Electromagnetic Wave Propagation

Simulate electromagnetic wave propagation:

```python
import ngsolve as ns

# Create mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 0, 2, 1, 0)

# Function space
V = ns.FunctionSpace('V', 2)
V.set_parameter_order('E', 'E', 'H')

# Material
E = ns.Constant('E')
E.set_value(1.0)
V.set_material(E)

# Source
f = ns.Constant('f')
f.set_value(1.0)
V.set_source(f)

# Problem
problem = ns.WaveEquation('V', mesh, 
                               degree=ns.grad(V)**2,
                               frequency=2.0*ns.pi)

# Solver
solver = ns.WaveSolver('V', mesh, problem)
solver.solve()
```

### Workflow 2: Antenna Modeling

Model antenna radiation pattern:

```python
import ngsolve as ns

# Create 3D mesh
mesh = ns.Mesh()
mesh.add_sphere((0, 0, 0), 0.5)

# Function space
V = ns.FunctionSpace('V', 3)
V.set_parameter_order('E', 'E', 'H')

# Material
E = ns.Constant('E')
E.set_value(1.0)
V.set_material(E)

# Source
f = ns.Constant('f')
f.set_value('sin(2*ns.pi*1e9)')
V.set_source(f)

# Problem
problem = ns.HelmholtzEquation('V', mesh, 
                                     degree=ns.grad(V)**2,
                                     frequency=1e9)

# Solver
solver = ns.HelmholtzSolver('V', mesh, problem)
solver.solve()
```

### Workflow 3: Microwave Resonator

Simulate microwave resonator:

```python
import ngsolve as ns

# Create cavity mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 0, 1, 1, 1)

# Function space
V = ns.FunctionSpace('V', 3)
V.set_parameter_order('E', 'E', 'H')

# Perfect conductor walls
E = ns.Constant('E')
E.set_value(1e10)
V.set_material(E)

# Source
f = ns.Constant('f')
f.set_value(1.0)
V.set_source(f)

# Problem
problem = ns.HelmholtzEquation('V', mesh, 
                                     degree=ns.grad(V)**2,
                                     frequency=2.45e9)

# Solver
solver = ns.HelmholtzSolver('V', mesh, problem)
solver.solve()
```

### Workflow 4: Scattering Problem

Simulate electromagnetic scattering:

```python
import ngsolve as ns

# Create domain mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 0, 2, 2, 2)

# Function space
V = ns.FunctionSpace('V', 3)
V.set_parameter_order('E', 'E', 'H')

# Background material
E_bg = ns.Constant('E')
E_bg.set_value(1.0)
V_bg.set_material(E_bg)

# Scatterer material
E_sc = ns.Constant('E')
E_sc.set_value(10.0)
V_sc.set_material(E_sc)

# Source
f = ns.Constant('f')
f.set_value(1.0)
V.set_source(f)

# Problem
problem = ns.HelmholtzEquation('V', mesh, 
                                     degree=ns.grad(V)**2,
                                     frequency=1e9)

# Solver
solver = ns.HelmholtzSolver('V', mesh, problem)
solver.solve()
```

### Workflow 5: Material Modeling

Model electromagnetic materials:

```python
import ngsolve as ns

# Create mesh
mesh = ns.Mesh()
mesh.add_rect(0, 0, 0, 1, 1, 1)

# Function space
V = ns.FunctionSpace('V', 3)
V.set_parameter_order('E', 'E', 'H')

# Spatially varying material
E = ns.Function('E')
E.set_value('1.0 + 0.5*x')

# Problem
problem = ns.HelmholtzEquation('V', mesh, 
                                     degree=ns.grad(V)**2,
                                     frequency=1e9)

# Solver
solver = ns.HelmholtzSolver('V', mesh, problem)
solver.solve()
```

## Solver Configuration

### Solver Types

```python
# Poisson solver
solver = ns.PoissonSolver('V', mesh, problem)

# Helmholtz solver
solver = ns.HelmholtzSolver('V', mesh, problem)

# Wave solver
solver = ns.WaveSolver('V', mesh, problem)

# Navier-Stokes solver
solver = ns.NavierStokesSolver('V', mesh, problem)
```

### Solver Parameters

```python
# Create solver with parameters
solver = ns.PoissonSolver('V', mesh, problem,
                                linear_solver='direct',
                                preconditioner='ilu',
                                absolute_tolerance=1e-6,
                                relative_tolerance=1e-6,
                                max_iterations=1000)
```

### Solver Execution

```python
# Solve problem
solver.solve()

# Solve with monitoring
class MyMonitor(ns.ProgressMonitor):
    def __init__(self):
        self.step = 0
    
    def __call__(self, mesh):
        self.step += 1
        if self.step % 10 == 0:
            print(f"Step {self.step}")

monitor = MyMonitor()
solver.solve(monitor=monitor)
```

## Output and Visualization

### Output Fields

```python
# Get electric field
E = V.get_subfunction('E')
E_array = E.vector().get_array()

# Get magnetic field
H = V.get.get_subfunction('H')
H_array = H.vector().get_array()
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np

# Get field data
E = V.get_subfunction('E')
E_array = E.vector().get_array()

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(E_array.reshape(100, 100), 
              cmap='RdBu', origin='lower')
plt.colorbar(im, label='E field')
plt.title('Electric Field')
plt.show()
```

## Best Practices

### Mesh Quality

1. **Element quality**: Use appropriate element types
2. **Mesh resolution**: Balance accuracy and computational cost
3. **Boundary layers**: Include PML or absorbing boundaries
4. **Source placement**: Place sources away from boundaries

### Solver Selection

1. **Problem type**: Choose appropriate solver for problem type
2. **Preconditioner**: Use appropriate preconditioner
3. **Tolerance**: Set appropriate convergence tolerance
4. **Iterations**: Set maximum iterations

### Material Modeling

1. **Physical properties**: Use realistic material parameters
2. **Spatial variation**: Model spatial material variations
3. **Frequency dependence**: Model frequency-dependent materials
4. **Anisotropy**: Model anisotropic materials

## Troubleshooting

### Convergence Issues

**Problem**: Solver doesn't converge

**Solutions**:
1. Increase mesh resolution
2. Improve preconditioner
3. Adjust tolerance
4. Check material properties

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce mesh resolution
2. Use coarser solver
3. Reduce output frequency
4. Use parallel computing

### Performance Issues

**Problem**: Simulation is too slow

**Solutions**:
1. Use parallel computing
2. Use appropriate preconditioner
3. Optimize mesh
4. Use compiled solver

## Resources

### References

- [meshing.md](references/meshing.md) - Mesh generation and quality
- [materials.md](references/materials.md) - Material definitions and properties
- [solvers.md](references/solvers.md) - Solver types and configuration
- [electromagnetics.md](references/electromagnetics.md) - Electromagnetic applications
- [visualization.md](references/visualization.md) - Output and visualization

### Scripts

- [antenna_modeling.py](scripts/antenna_modeling.py) - Antenna modeling tools
- [post_processing.py](scripts/post_processing.py) - Field analysis tools

### External Resources

- Official documentation: https://ngsolve.org/
- GitHub repository: https://github.com/NGSolve/ngsolve
- PyPI package: https://pypi.org/project/ngsolve/