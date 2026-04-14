---
name: pyfoam
description: Computational fluid dynamics framework using finite volume method. Use when solving CFD problems, heat transfer, multiphase flows, or complex fluid simulations. Supports C++ with Python bindings, parallel execution, and extensive solver library.
---

# PyFoam (OpenFOAM)

PyFoam is a free, open-source CFD software for computational fluid dynamics.

## Quick Start

### Basic Simulation

```bash
# Navigate to case directory
cd $FOAM_RUNS

# Run simulation
blockMesh
simpleFoam
paraFoam
```

### Python Interface

```python
import PyFoam

# Create mesh
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
args = ParsedParameterFile()
args.case = 'cavity'
args.parse()

# Execute solver

```

## Core Concepts

### Mesh

**Characteristics**:
- Unstructured mesh support
- Multiple mesh types (hex, poly, tet)
- Dynamic mesh refinement
- Boundary layer specification

**Example**:
```python
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict

# Create mesh
mesh = ParsedBlockMeshDict(args.case)
```

### Fields

**Characteristics**:
- VolScalarField: Scalar quantities (pressure, temperature)
- VolVectorField: Vector quantities (velocity)
- SurfaceScalarField: Boundary fields
- Point fields: Lagrangian particle tracking

**Example**:
```python
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict

# Create pressure field
p = VolScalarField(
    mesh=mesh,
    value="uniform 0",
    name="p"
)

# Create velocity field
U = VolVectorField(
    mesh=mesh,
    value="uniform (0 0 0)",
    name="U"
)
```

### Boundary Conditions

**Characteristics**:
- Dirichlet: Fixed value
- Neumann: Gradient zero
- Robin: Mixed condition
- Cyclic: Periodic boundaries

**Example**:
```python
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict

# Fixed wall
U_wall = DirichletField(
    mesh=mesh,
    value="uniform (0 0 0)",
    patch="walls"
)

# Inlet
U_inlet = DirichletField(
    mesh=mesh,
    value="uniform (1 0 0)",
    patch="inlet"
)

# Outlet
p_outlet = DirichletField(
    mesh=mesh,
    value="uniform 0",
    patch="outlet"
)
```

## Solvers

### Incompressible Solvers

**Characteristics**:
- icoFoam: Standard incompressible flow
- simpleFoam: Transient solver
- pimpleFoam: PISO solver
- pUFOAM: PISO with unstructured mesh support

**Example**:
```python
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict

# Setup solver
args = ParsedParameterFile()
args.case = 'cavity'
args.parse()

# Run solver
```

### Heat Transfer Solvers

**Characteristics**:
- laplacianFoam: Steady-state conduction
-chtMultiRegionFoam: Conjugate heat transfer
- solidEquilibriumFoam: Solid heat conduction

**Example**:
```python
# Heat conduction
args = ParsedParameterFile()
args.case = 'heatConduction'
args.parse()
```

### Multiphase Solvers

**Characteristics**:
- reactingFoam: Chemical reactions
- compressibleInterFoam: Compressible multiphase flows
- twoPhaseEulerFoam: Euler-Euler multiphase

**Example**:
```python
# Chemical reactions
args = ParsedParameterFile()
args.case = 'reactingPipe'
args.parse()
```

## Simulation Workflow

### Standard Workflow

```python
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from PyFoam.RunDictionary.ParsedBlockMeshDict import ParsedBlockMeshDict

# Parse arguments
args = ParsedParameterFile()
args.case = 'cavity'
args.parse()

# Create mesh
mesh = ParsedBlockMeshDict(args.case)

# Create fields
p = VolScalarField(mesh=mesh, value="uniform 0", name="p")
U = VolVectorField(mesh=mesh, value="uniform (0 0 0)", name="U")

# Create boundary conditions
U_wall = DirichletField(mesh=mesh, value="uniform (0 0 0)", patch="walls")
U_lid = DirichletField(mesh=mesh, value="uniform (1 0 0)", patch="movingWall")

# Setup solver
args.p = p
args.U = U
args.writeControl = True

# Run simulation
```

### Post-Processing

```python
# ParaView for visualization
paraFoam -case cavity

# Sample utility
sample -case cavity -latestTime
```

## Resources

- **Solvers**: See [solvers.md](references/solvers.md)
- **Mesh generation**: See [mesh.md](references/mesh.md)
- **Boundary conditions**: See [boundaries.md](references/boundaries.md)
- **Tutorials**: See [tutorials.md](references/tutorials.md)
- **Parallel execution**: See [parallel.md](references/parallel.md)
- **Advanced features**: See [advanced.md](references/advanced.md)
