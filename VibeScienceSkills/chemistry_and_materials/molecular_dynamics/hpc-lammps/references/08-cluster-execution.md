# Cluster Execution

## MPI Parallel LAMMPS

LAMMPS uses MPI for parallel execution:

```bash
srun -n 32 lmp -in in.case
```

## SLURM Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=lammps_case
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

input_script="${1:-in.case}"
log_file="${2:-log.lammps}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# module purge
# module load lammps

srun -n "${SLURM_NTASKS}" lmp -in "$input_script" -log "$log_file"
```

## GPU-Accelerated LAMMPS

LAMMPS with GPU package:

```bash
#SBATCH --gpus=1

srun -n 1 lmp -in in.case -pk gpu 1 -sf gpu
```

### GPU Settings

```lammps
# In input script:
package gpu 1
suffix gpu
```

Or via command line:

```bash
srun -n 1 lmp -in in.case -pk gpu 1 -sf gpu
```

## Hybrid MPI+OpenMP

For hybrid parallelization:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
srun -n "${SLURM_NTASKS}" lmp -in in.case -sf omp -pk omp "${SLURM_CPUS_PER_TASK}"
```

## Rank Count Guidance

| System Size | Atoms | Ranks | Notes |
|-------------|-------|-------|-------|
| Small | < 10K | 1-8 | Serial fine |
| Medium | 10K-100K | 8-32 | Good scaling |
| Large | 100K-1M | 32-128 | Needs tuning |
| Very large | > 1M | 128+ | Domain decomposition |

**Rule:** Benchmark before large production.

## Domain Decomposition

LAMMPS auto-decomposes for most cases. For control:

```lammps
# Fix decomposition
fix 1 all balance 1000 x 10 10 10
```

Or via command line:

```bash
srun -n 32 lmp -in in.case -partition 8x4
```

## Output and Restart

### Checkpoints

```lammps
write_restart restart.equil.%R
```

LAMMPS auto-saves restart files on signal or at end.

### Continuation

```bash
srun -n 32 lmp -in continue.in
```

```lammps
read_restart restart.equil.10000
# Change settings
fix 1 all npt temp 300.0 350.0 100.0 iso 1.0 1.0 1000.0
run 10000
```

## Scratch Storage

Large dumps should write to scratch:

```bash
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Symlink scratch to local
ln -sf /scratch/${USER}/lammps_output ./

dump 1 all atom 1000 /scratch/${USER}/lammps_output/dump.lammpstrj
```

## Monitoring

Watch the output:

```bash
tail -f slurm-$SLURM_JOB_ID.out
```

Key indicators:

```
Per MPI rank memory (Mbytes) = 125.67
Neighbor list builds = 1000
Dangerous builds = 0
```

Dangerous builds > 0 means atoms moved too far between neighbor rebuilds.

## Common Batch Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Segmentation fault" | Bad input or restart | Verify files |
| "Lost atoms" | Atoms left the box | Check boundary settings |
| "Invalid velocity" | Zero velocity on frozen atoms | Use `group` for thermostatting |
| Very slow | Too many ranks for system | Reduce rank count |
| "GPU not supported" | Package not installed | Load correct LAMMPS module |

## Performance Tuning

| Issue | Solution |
|-------|----------|
| Load imbalance | Use `balance` fix |
| Poor scaling | Reduce OpenMP threads |
| Memory per rank too high | Increase ranks |
| Neighbor builds dangerous | Increase skin or rebuild frequency |
