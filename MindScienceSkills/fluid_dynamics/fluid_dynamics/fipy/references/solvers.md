# Solvers

Guide to solver configuration and parallel execution in FiPy.

## Available Solver Suites

### SciPy (Default)

Basic solver suite, always available:

```python
eq.solve(var=phi, dt=0.001)
```

**Characteristics:**
- Serial only
- Good for small to medium problems
- Direct and iterative solvers available
- No external dependencies

### PETSc

Powerful parallel solver suite:

```bash
python script.py --petsc
# or
Fpython script.py --trilinos
```

**Characteristics:**
- Parallel with MPI
- Wide range of solvers and preconditioners
- Excellent for large problems
- Requires PETSc installation

**Installation:**
```bash
conda install -c conda-forge petsc4py petsc
```

**Configuration:**
```python
from fipy.solvers import PETScSolver

solver = PETScSolver(
    solver='gmres',
    preconditioner='ilu',
    tolerance=1e-10,
    maxIterations=1000
)

eq.solve(var=phi, dt=0.001, solver=solver)
```

**PETSc options:**
```bash
# Set solver
PETSC_OPTIONS="-ksp_type gmres -pc_type ilu" python script.py --petsc

# Monitor convergence
PETSC_OPTIONS="-ksp_monitor" python script.py --petsc

# Multigrid
PETSC_OPTIONS="-pc_type mg -ksp_type richardson" python script.py --petsc
```

### Trilinos

Alternative parallel solver suite:

```bash
python script.py --trilinos
# or
FIPY_SOLVERS=trilinos python script.py
```

**Characteristics:**
- Parallel with MPI
- Advanced preconditioners
- Good for specific problem types
- Requires Trilinos installation

### PyAMGX

GPU-accelerated solver:

```bash
python script.py --pyamgx
```

**Characteristics:**
- Runs on NVIDIA GPUs
- Very fast for suitable problems
- Requires CUDA and AMGX

## Solver Selection Guide

**Small problems (< 10k cells):**
- Use SciPy (default)
- Direct solvers (LU) are efficient

**Medium problems (10k - 1M cells):**
- Use PETSc with ILU preconditioner
- Or Trilinos

**Large problems (> 1M cells):**
- Use PETSc with multigrid
- Enable parallel execution

**3D problems:**
- Always use PETSc or Trilinos
- Multigrid preconditioners are effective

**Stiff problems:**
- Use robust solvers (GMRES)
- ILU or multigrid preconditioning

## Common Solver Types

### Direct Solvers

**LU decomposition:**
```python
from fipy.solvers import LUSolver
solver = LUSolver()
```

- Exact solution (within machine precision)
- Memory intensive
- Good for small problems

### Iterative Solvers

**GMRES (Generalized Minimal Residual):**
```python
from fipy.solvers import GMRESSolver
solver = GMRESSolver(tolerance=1e-10, maxIterations=1000)
```

- Robust for non-symmetric systems
- Good for convection-dominated problems
- Memory usage grows with iterations

**CG (Conjugate Gradient):**
```python
from fipy.solvers import CGSolver
solver = CGSolver(tolerance=1e-10, maxIterations=1000)
```

- For symmetric positive-definite systems
- Fast for diffusion problems
- Low memory usage

**BiCGStab (Biconjugate Gradient Stabilized):**
```python
from fipy.solvers import BiCGStabSolver
solver = BiCGStabSolver(tolerance=1e-10, maxIterations=1000)
```

- For non-symmetric systems
- More stable than CG for general problems

## Preconditioners

### ILU (Incomplete LU)

```python
from fipy.solvers import PETScSolver
solver = PETScSolver(
    solver='gmres',
    preconditioner='ilu',
    pc_fill_level=3
)
```

- General-purpose
- Good fill level: 1-5
- Higher fill = better preconditioning, more memory

### Multigrid

```python
from fipy.solvers import PETScSolver
solver = PETScSolver(
    solver='richardson',
    preconditioner='mg',
    pc_mg_type='multiplicative'
)
```

- Excellent for elliptic problems
- Optimal O(n) complexity
- Requires good coarse grid operator

### Jacobi/Gauss-Seidel

```python
solver = PETScSolver(
    solver='gmres',
    preconditioner='jacobi'
)
```

- Simple, low memory
- Slow convergence
- Use as smoother in multigrid

## Convergence Criteria

**Relative tolerance:**
```python
solver = PETScSolver(
    tolerance=1e-10,  # Relative residual tolerance
    absoluteTolerance=1e-15  # Absolute tolerance
)
```

**Maximum iterations:**
```python
solver = PETScSolver(maxIterations=1000)
```

**Divergence tolerance:**
```python
solver = PETScSolver(divergenceTolerance=1e5)
```

## Parallel Execution

### Basic Parallel Run

```bash
# Run with 4 processors
mpirun -np 4 python script.py --petsc

# Run with Trilinos
mpirun -np 4 python script.py --trilinos
```

### Important: OpenMP Threading

**Always set OMP_NUM_THREADS=1 for MPI parallel runs:**

```bash
# Correct way
OMP_NUM_THREADS=1 mpirun -np 4 python script.py --petsc

# Wrong way (severe performance penalty)
mpirun -np 4 python script.py --petsc
```

**Why?**
- PETSc/Trilinos spawn OpenMP threads by default
- Python GIL binds threads to same core
- Causes massive overhead

### Testing Parallel Setup

```bash
# Test parallel configuration
mpirun -np 3 python examples/parallel.py

# Expected output:
# processor 0 of 3 :: 5 cells on processor 0 of 3
# processor 1 of 3 :: 7 cells on processor 1 of 3
# processor 2 of 3 :: 6 cells on processor 2 of 3
```

### Parallel Mesh Partitioning

Automatic for most meshes:

```python
# Grid meshes automatically partitioned
mesh = Grid2D(nx=100, ny=100)

# Gmsh meshes with parallel communicator
from fipy import Gmsh2D
mesh = Gmsh2D(geometry_string, communicator=comm)
```

### Accessing Global Values

```python
# Get values from all processors
global_value = phi.globalValue

# Get value on local processor
local_value = phi.value
```

## Performance Optimization

### Solver Selection

**For diffusion problems:**
```python
# CG with multigrid is optimal
solver = PETScSolver(solver='cg', preconditioner='mg')
```

**For convection-diffusion:**
```python
# GMRES with ILU is robust
solver = PETScSolver(solver='gmres', preconditioner='ilu')
```

**For coupled systems:**
```python
# Use block preconditioners
solver = PETScSolver(
    solver='fgmres',
    preconditioner='fieldsplit',
    pc_fieldsplit_type='multiplicative'
)
```

### Memory Management

**For memory-constrained systems:**
```python
# Use iterative solvers (not direct)
solver = CGSolver(tolerance=1e-8)

# Limit GMRES restart
solver = GMRESSolver(restart=50)
```

### Caching

```bash
# Enable caching for repeated calculations
python script.py --cache

# Disable caching
python script.py --no-cache
```

### Inline Operations

```bash
# Use C-optimized operations
python script.py --inline

# Requires weave package
```

## Troubleshooting

**Solver not converging:**
- Check problem formulation
- Try different solver/preconditioner
- Reduce time step
- Check boundary conditions
- Examine mesh quality

**Poor parallel scaling:**
- Ensure OMP_NUM_THREADS=1
- Check mesh partitioning
- Verify MPI configuration
- Monitor load balance

**Memory issues:**
- Reduce mesh resolution
- Use iterative solvers
- Lower preconditioner fill level
- Enable memory-efficient solvers

**Slow convergence:**
- Try better preconditioner
- Increase solver iterations
- Check problem conditioning
- Consider multigrid

## Monitoring

**Enable solver logging:**
```python
import logging
log = logging.getLogger("fipy")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
log.addHandler(console)
```

**Monitor residuals:**
```bash
PETSC_OPTIONS="-ksp_monitor" python script.py --petsc
```

**Check convergence:**
```python
solver = PETScSolver(tolerance=1e-10, maxIterations=1000)
eq.solve(var=phi, dt=0.001, solver=solver)

# Check solver statistics
print(f"Iterations: {solver.iterations}")
print(f"Residual: {solver.residual}")
```

## Advanced Configuration

### Custom Solver Factory

```python
from fipy.solvers import Solver

def create_custom_solver():
    """Create custom solver configuration."""
    solver = PETScSolver(
        solver='fgmres',
        preconditioner='ilu',
        tolerance=1e-10,
        maxIterations=1000,
        pc_fill_level=3,
        restart=100
    )
    return solver

solver = create_custom_solver()
eq.solve(var=phi, dt=0.001, solver=solver)
```

### Adaptive Solver Selection

```python
def select_solver(problem_size, problem_type):
    """Select appropriate solver based on problem characteristics."""
    if problem_size < 10000:
        return LUSolver()
    elif problem_type == 'diffusion':
        return PETScSolver(solver='cg', preconditioner='mg')
    else:
        return PETScSolver(solver='gmres', preconditioner='ilu')

solver = select_solver(mesh.numberOfCells, 'convection-diffusion')
```
