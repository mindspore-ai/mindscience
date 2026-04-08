# Cluster Execution

## MPI Parallel Execution

DOLFINx is MPI-first. Always use `mpirun` or `srun` for parallel execution:

```bash
# Single node
srun -n 8 python poisson.py

# Multi-node
srun -n 64 python poisson.py
```

## SLURM Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=fenics_case
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

script_path="${1:-poisson_dolfinx.py}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module purge
# module load python
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate fenicsx-env

srun -n "${SLURM_NTASKS}" python "$script_path"
```

## Thread-MPI Hybrid (Not Recommended)

For DOLFINx, pure MPI is the recommended parallel mode. Thread-MPI (combining
MPI within nodes) can cause issues with BLAS libraries:

```bash
# NOT recommended:
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
```

## Preflight Checklist

Before submitting:

- [ ] Script works on a small case locally
- [ ] Python environment is available on the cluster (conda/env modules)
- [ ] Mesh is partitioned for the target core count
- [ ] Output paths are writable
- [ ] `mpi4py` and `petsc4py` are importable in the target environment

## Mesh Partitioning for MPI

DOLFINx can partition a non-partitioned mesh automatically:

```python
# DOLFINx will partition automatically for the given communicator
mesh = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
```

For existing partitioned meshes:

```python
# Load partitioned mesh
with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
```

## PETSc Options for Parallel

```python
petsc_options = {
    "ksp_type": "fgmres",
    "pc_type": "gamgee",
    "pc_gamgee_technology": "p4est",
    "ksp_max_it": 500,
    "ksp_rtol": 1e-6,
}
```

For multi-node, use `gamgee` with `p4est` for best scaling.

## Output in Parallel

DOLFINx uses MPI-IO for parallel output:

```python
from dolfinx import io

# Binary VTK format (BP4 for better performance)
with io.VTXWriter(mesh.comm, "u.bp", [u], engine="BP4") as f:
    for t in [0.0, 0.1, 0.2]:
        f.write(t)
```

For multi-rank output, all ranks must call `write()` collectively.

## Shared vs Distributed Arrays

DOLFINx uses distributed arrays (conceptually like PETSc vectors):

```python
u = fem.Function(V)
print(f"Local size: {u.x.array_local.size}")
print(f"Global size: {u.x.array.size}")

# Gather to root rank for inspection:
import numpy as np
root = 0
if mesh.comm.rank == root:
    gathered = u.x.array[:]
else:
    gathered = None
gathered = mesh.comm.gather(gathered, root=root)
```

## Weak Scaling Guidance

| Problem Size | DOF | Suggested Ranks | Notes |
|--------------|-----|-----------------|-------|
| Small | <100K | 1-8 | Serial fine, small parallelism |
| Medium | 100K-1M | 8-32 | Good parallel efficiency |
| Large | 1M-10M | 32-128 | Requires good preconditioner |
| Very Large | >10M | 128+ | Use AMG, consider problem decomposition |

## Monitoring Parallel Performance

Add to PETSc options:

```bash
-ksp_monitor_true_residual    # Print residual history
-ksp_converged_reason         # Print convergence info
-pc_view                      # Print preconditioner info
```

Or in Python:

```python
opts = PETSc.Options()
opts["ksp_monitor_true_residual"] = None
```

## Common Cluster Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| ImportError: No module named 'dolfinx' | Wrong Python environment | Load correct conda/env |
| MPI_Init_thread failed | Incompatible threading | Use `mpirun --allow-run-as-root` in containers |
| Segfault at mesh creation | Mesh not accessible on all ranks | Verify mesh path is shared |
| Very slow output | Writing from all ranks to one file | Use VTX with MPI-IO |
| Memory grows with rank count | Storing full arrays per rank | Use distributed arrays correctly |

## Job Arrays for Parameter Sweeps

```bash
#!/bin/bash
#SBATCH --job-name=param_sweep
#SBATCH --array=0-9
#SBATCH --ntasks=8

cd "${SLURM_SUBMIT_DIR:-$PWD}"
PARAM="${SLURM_ARRAY_TASK_ID}"

python run.py --param "$PARAM"
```

## Post-Processing on Cluster

Convert to Paraview format on the cluster:

```bash
# Install paraview-python or use dolfinx.io
python -c "from dolfinx import io; io.VTKFile('output.pvd').write(u)"
```
