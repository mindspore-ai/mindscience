# Solver Configuration and Parallel Computing

Guide to configuring solvers and parallel execution in FEniCS.

## PETSc Solvers

### Basic Configuration

```python
import dolfinx as dfx

# PETSc options dictionary
options = {
    "ksp_type": "cg",        # Solver type
    "pc_type": "hypre",      # Preconditioner type
    "ksp_rtol": 1e-10,      # Relative tolerance
    "ksp_atol": 1e-14,      # Absolute tolerance
    "ksp_max_it": 1000       # Maximum iterations
}

# Use in problem
problem = dfx.fem.petsc.LinearProblem(a, L, bcs, u_h, petsc_options=options)
dfx.nls.petsc.solve(problem)
```

### Solver Types

**Direct solvers:**
```python
options = {
    "ksp_type": "preonly",   # Preconditioner only (direct)
    "pc_type": "lu"          # LU decomposition
}
```

**Iterative solvers:**
```python
# CG (Conjugate Gradient) - symmetric positive-definite
options = {"ksp_type": "cg", "pc_type": "hypre"}

# GMRES (Generalized Minimal Residual) - non-symmetric
options = {"ksp_type": "gmres", "pc_type": "ilu"}

# FGMRES (Flexible GMRES) - variable preconditioning
options = {"ksp_type": "fgmres", "pc_type": "ilu"}

# Richardson - multigrid
options = {"ksp_type": "richardson", "pc_type": "mg"}
```

### Preconditioners

**Hypre (BoomerAMG):**
```python
options = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_boomeramg_max_levels": 25,
    "pc_hypre_boomeramg_strong_threshold": 0.5
}
```

**ILU (Incomplete LU):**
```python
options = {
    "ksp_type": "gmres",
    "pc_type": "ilu",
    "pc_factor_levels": 3  # Fill level
}
```

**Multigrid:**
```python
options = {
    "ksp_type": "richardson",
    "pc_type": "mg",
    "pc_mg_type": "multiplicative",
    "pc_mg_cycle_type": "v"
}
```

**Jacobi:**
```python
options = {
    "ksp_type": "cg",
    "pc_type": "jacobi"
}
```

## Block Preconditioners

### Field Split

For mixed problems (Stokes, mixed Poisson):

```python
options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    
    # Velocity block (field 0)
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "hypre",
    
    # Pressure block (field 1)
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "jacobi",
    
    "ksp_rtol": 1e-10
}
```

### Schur Complement

**Full Schur:**
```python
options = {
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_schschur_precond": "lu"
}
```

**Diagonal Schur:**
```python
options = {
    "pc_fieldsplit_schur_fact_type": "diag",
    "pc_fieldsplit_schur_precond": "jacobi"
}
```

## Newton Solver

### Basic Newton

```python
# For nonlinear problems
problem = dfx.nls.petsc.NewtonProblem(a, L, bcs, u_h)
solver = dfx.nls.petsc.NewtonSolver(MPI.COMM_WORLD)

# Set parameters
solver.set_param("rtol", 1e-8)      # Relative tolerance
solver.set_param("atol", 1e-10)     # Absolute tolerance
solver.set_param("max_it", 50)       # Maximum iterations
solver.set_param("relaxation", 1.0)  # Relaxation parameter

# Solve
solver.solve(problem)

# Get convergence info
print(f"Iterations: {solver.iteration}")
print(f"Residual: {solver.residual}")
```

### Line Search

```python
# Enable line search for better convergence
solver.set_param("ls_type", "basic")
solver.set_param("ls_max_its", 20)
solver.set_param("ls_alpha", 1.0)
```

### Error Reporting

```python
# Monitor Newton iterations
solver.set_param("error_on_nonconverge", True)
solver.set_param("error_on_divergence", True)
```

## Parallel Execution

### Basic MPI

```python
from mpi4py import MPI

# Check parallel configuration
print(f"Rank: {MPI.COMM_WORLD.rank}")
print(f"Size: {MPI.COMM_WORLD.size}")

# Create mesh (automatically partitioned)
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
```

### Running in Parallel

```bash
# Run with 4 processors
mpirun -np 4 python script.py

# Run with 8 processors
mpirun -np 8 python script.py --petsc
```

### Parallel Mesh

```python
# Mesh is automatically partitioned
mesh = dfx.mesh.create_unit_square(MPI.COMM_WORLD, 100, 100)

# Get local mesh info
num_cells_local = mesh.topology.index_map(mesh.topology.dim, 0).size_local
num_cells_global = mesh.topology.index_map(mesh.topology.dim, 0).size_global

print(f"Local cells: {num_cells_local}")
print(f"Global cells: {num_cells_global}")
```

### Parallel IO

```python
# Write parallel XDMF
with dfx.io.XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_h)

# Read parallel XDMF
with dfx.io.XDMFFile(MPI.COMM_WORLD, "solution.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    u_h = xdmf.read_function(mesh, V)
```

## Solver Selection Guide

### Problem Type

**Elliptic (Poisson, diffusion):**
```python
options = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": 1e-10
}
```

**Non-symmetric (convection-diffusion):**
```python
options = {
    "ksp_type": "gmres",
    "pc_type": "ilu",
    "ksp_rtol": 1e-10
}
```

**Saddle-point (Stokes, mixed):**
```python
options = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "fieldsplit_0_ksp_type": "cg",
    "fieldsplit_0_pc_type": "hypre",
    "fieldsplit_1_ksp_type": "cg",
    "fieldsplit_1_pc_type": "jacobi",
    "ksp_rtol": 1e-10
}
```

### Problem Size

**Small (< 100k DOFs):**
```python
# Direct solver
options = {
    "ksp_type": "preonly",
    "pc_type": "lu"
}
```

**Medium (100k - 1M DOFs):**
```python
# Iterative with good preconditioner
options = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_rtol": 1e-10
}
```

**Large (> 1M DOFs):**
```python
# Multigrid or AMG
options = {
    "ksp_type": "richardson",
    "pc_type": "mg",
    "ksp_rtol": 1e-10
}
```

## Performance Tuning

### Memory Management

```python
# Reduce memory usage
options = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "pc_hypre_boomeramg_max_coarse_size": 1000000
}
```

### Convergence Monitoring

```python
# Monitor convergence
options = {
    "ksp_type": "cg",
    "pc_type": "hypre",
    "ksp_monitor": True,
    "ksp_monitor_true_residual": True
}
```

### Preconditioner Tuning

```python
# Hypre tuning
options = {
    "pc_type": "hypre",
    "pc_hypre_boomeramg_strong_threshold": 0.5,
    "pc_hypre_boomeramg_max_levels": 25,
    "pc_hypre_boomeramg_grid_sweeps": 3
}
```

## Common Issues

### Divergence

**Issue:** Solver diverges

**Solutions:**
- Reduce tolerance
- Try different preconditioner
- Check problem formulation
- Improve mesh quality

### Slow Convergence

**Issue:** Too many iterations

**Solutions:**
- Use better preconditioner
- Try multigrid
- Check problem conditioning
- Increase tolerance

### Poor Parallel Scaling

**Issue:** Speedup not linear

**Solutions:**
- Check load balancing
- Tune preconditioner
- Reduce communication
- Use appropriate solver

### Memory Issues

**Issue:** Out of memory

**Solutions:**
- Use iterative solvers
- Reduce mesh resolution
- Tune preconditioner memory
- Use matrix-free methods

## Advanced Topics

### Matrix-Free Methods

```python
# For very large problems
# See FEniCS matrix-free demo
```

### Custom Preconditioners

```python
# Define custom preconditioner
# Requires PETSc knowledge
```

### Solver Reuse

```python
# Reuse solver for multiple solves
# Can improve performance
```

### GPU Acceleration

```python
# Requires PETSc with CUDA support
# Configure PETSc for GPU
```

## Resources

- PETSc manual: https://petsc.org/main/documentation/manual/
- Hypre documentation: https://hypre.readthedocs.io/
- FEniCS solver demos: https://docs.fenicsproject.org/dolfinx/v0.10.0.post1/python/demos.html
