# Cluster Execution

## GROMACS MPI/MPI+OpenMP Hybrid

GROMACS supports both pure MPI and hybrid MPI+OpenMP parallelization.

### Pure MPI

```bash
#SBATCH --nodes=2
#SBATCH --ntasks=32

srun -n 32 gmx_mpi mdrun -deffnm md
```

### Hybrid (MPI + OpenMP)

```bash
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun -n "${SLURM_NTASKS}" gmx_mpi mdrun -deffnm md
```

## SLURM Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=gromacs_md
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

deffnm="${1:-md}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module purge
# module load gromacs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun -n "${SLURM_NTASKS}" gmx_mpi mdrun -deffnm "$deffnm"
```

## GPU Execution

Modern GROMACS auto-detects GPUs when compiled with CUDA:

```bash
# Single GPU
gmx mdrun -deffnm md -nb gpu

# Multiple GPUs (one per node)
srun -n 1 gmx mdrun -deffnm md -nb gpu -pme gpu

# Full GPU offload
srun -n 1 gmx mdrun -deffnm md -nb gpu -pme gpu -update gpu
```

## Rank/Thread Layout

For best performance:

| Hardware | Recommended Layout |
|----------|-------------------|
| Single socket, 8 cores | `-ntomp 8` (pure OpenMP) |
| Dual socket, 16 cores | `-ntomp 8 -npme 4` (16 ranks) |
| GPU system | `-nb gpu` with `-npme` auto |
| Multi-node | Use MPI ranks per node, OpenMP within |

## PME Load Balancing

Electrostatics (PME) is computationally expensive. GROMACS can balance PME load:

```bash
gmx mdrun -deffnm md -npme 0  # Let GROMACS decide
```

Or set explicitly (PP:PME ratio):

```bash
# 3:1 ratio for 32 total ranks
srun -n 24 gmx_mpi mdrun -deffnm md -npme 8
```

## Preflight Checklist

Before submitting production:

- [ ] `.tpr` generated from clean `grompp`
- [ ] Topology and structure versions match
- [ ] Output/checkpoint paths are writable
- [ ] Walltime matches expected simulation length
- [ ] Restart files would be saved at checkpoints

## Job Sizing

| System Size | Atoms | Nodes | Ranks | Notes |
|-------------|-------|-------|-------|-------|
| Small (< 50K) | <50,000 | 1 | 8-16 | Single node fine |
| Medium (50K-500K) | 50K-500K | 1-2 | 16-32 | May need PME tuning |
| Large (500K-5M) | 500K-5M | 2-8 | 32-128 | PME load important |
| Very large (>5M) | >5M | 8+ | 128+ | Domain decomposition critical |

**Rule:** Benchmark before large production.

## Output and Restart

### Checkpoint Frequency

In your `.mdp`:

```mdp
; Checkpoint every 1 ns (10000 steps at 2 fs)
nstcalcenergy = 1000
```

### Continuation

```bash
# Continue from checkpoint
gmx mdrun -cpi md.cpt -deffnm md

# Append to existing trajectory
gmx mdrun -cpi md.cpt -append yes -deffnm md
```

### Stage Naming

```
md_0_1.tpr     # Stage 0, segment 1
md_0_2.tpr     # Stage 0, segment 2
md_1_1.tpr     # Stage 1, segment 1
```

## Scratch vs Project Storage

- Put trajectory on **scratch** during run (large files)
- Copy final trajectory to **project storage** when done
- Keep `.tpr`, `.edr`, `.log` on project storage

## Monitoring

Watch the log file:

```bash
tail -f slurm-$SLURM_JOB_ID.out
```

Key indicators:
- "Step X: 1 atoms crashed" — System exploded
- "LINCS warnings" — Constraint issues
- "Energy drift" — Check conservation
- Performance: ns/day reported at end

## Common Batch Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Segmentation fault" | Bad .tpr or memory | Verify .tpr with `gmx check` |
| "File not found" | Wrong directory or path | Check file paths |
| "PME nodes must be >= 0" | Bad -npme setting | Remove -npme flag |
| "One of the atoms crashed" | System exploded | Return to NPT, check structure |
| Very slow performance | Wrong rank layout | Tune -ntomp/-npme |
