# Stack Selection: FEniCS vs DOLFINx

## Two Stacks, One Family

```
FEniCS Family
├── Classic FEniCS (legacy)
│   ├── API: from fenics import * or from dolfin import *
│   ├── Mesh: Mesh, FunctionSpace, etc.
│   ├── Status: Maintenance mode
│   └── Use: Existing code, older tutorials
│
└── DOLFINx (modern)
    ├── API: from dolfinx import fem, io, mesh
    ├── MPI-first design
    ├── Uses: Basix (element basis), petsc4py, mpi4py
    └── Status: Active development
```

## When to Use Classic FEniCS

Use classic FEniCS when:
- You have existing code that cannot be migrated easily
- You are following older tutorials/books
- The environment only has classic FEniCS installed
- You need APIs like `FunctionSpace`, `DirichletBC`, `solve(a == L, ...)` directly

**Import pattern:**

```python
from fenics import *
# or
from dolfin import *
```

## When to Use DOLFINx

Use DOLFINx when:
- Starting a new project
- You need MPI parallel scaling
- You need current element support (high-order, etc.)
- You need better performance on modern hardware
- You want active ecosystem support

**Import pattern:**

```python
from mpi4py import MPI
from dolfinx import fem, io, mesh
from petsc4py import PETSc
import basix
```

## Switching Mid-Script is Forbidden

```python
# WRONG — mixing stacks
from fenics import *
from dolfinx import fem   # Cannot mix!

# CORRECT — choose one
from fenics import *      # classic FEniCS only

# OR
from dolfinx import ...   # DOLFINx only
```

## Environment Notes

DOLFINx documentation targets Linux-first workflows. On Windows, use WSL:

```bash
# Install DOLFINx via conda or spack in WSL
wsl -d Ubuntu
conda install -c conda-forge fenics-dolfinx
```

## Version Detection

To check which version is available:

```python
import sys
try:
    import dolfinx
    print(f"DOLFINx version: {dolfinx.__version__}")
except ImportError:
    pass

try:
    import fenics
    print(f"Classic FEniCS detected")
except ImportError:
    pass
```

## DOLFINx Components

| Package | Purpose |
|---------|---------|
| `dolfinx` | Core functionality |
| `basix` | Finite element basis functions |
| `petsc4py` | PETSc Python bindings |
| `mpi4py` | MPI Python bindings |
| `pyvista` | Visualization |

## Feature Comparison

| Feature | Classic FEniCS | DOLFINx |
|---------|---------------|---------|
| MPI support | Limited | Full |
| Performance | Good | Better |
| Element support | Standard | Broader (basix) |
| Active development | No | Yes |
| Python 3.11+ | Varies | Yes |
| Windows support | Native | WSL only |

## Migration from Classic to DOLFINx

| Classic FEniCS | DOLFINx Equivalent |
|---------------|-------------------|
| `FunctionSpace(mesh, "P", 1)` | `fem.functionspace(mesh, ("Lagrange", 1))` |
| `TrialFunction(V)` | Same for linear |
| `Function(V)` | Same |
| `DirichletBC(V, value, boundary)` | `fem.dirichletbc(value, dofs, V)` |
| `solve(a == L, u, bcs)` | `fem.petsc.LinearProblem(a, L, bcs, ...)` |
| `File("output.pvd")` | `io.VTXWriter(mesh.comm, "output.bp", [u])` |

## Parallel Considerations

DOLFINx is designed for MPI-first execution:

```python
from mpi4py import MPI

mesh = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
print(f"Rank {mesh.comm.rank} of {mesh.comm.size}")
```

Classic FEniCS serial execution is the default; MPI requires special handling.

## Choosing for Cluster Execution

| Cluster Environment | Recommendation |
|-------------------|----------------|
| Shared memory (single node) | Either works |
| Multi-node MPI | DOLFINx required |
| GPU acceleration | DOLFINx (via PETSc) |
| Large scale (>1M DOF) | DOLFINx required |
