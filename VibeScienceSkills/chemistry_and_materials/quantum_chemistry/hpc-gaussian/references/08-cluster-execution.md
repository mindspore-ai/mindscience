# Cluster Execution

## Gaussian on HPC

Gaussian uses shared memory parallelism within a node (`%NProcShared`).

For multi-node jobs, Gaussian uses Linda (when available).

## SLURM Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=gaussian_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

input_file="${1:-job.gjf}"
output_file="${2:-job.log}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Set Gaussian environment
# export g16root="/path/to/gaussian/root"
# . "$g16root/g16/bsd/g16.profile"

# Create scratch directory
export GAUSS_SCRDIR="${TMPDIR:-/tmp}/${USER}/gaussian-${SLURM_JOB_ID}"
mkdir -p "$GAUSS_SCRDIR"

# Run Gaussian
g16 < "$input_file" > "$output_file"

# Save checkpoint to working directory
# (if job used %Chk with relative path in scratch)
```

## Shared Memory Parallelism

Set `%NProcShared` to match the CPU allocation:

```gjf
%NProcShared=16
%Mem=16GB
#p B3LYP/6-311G(d,p)

16 CPUs, 1GB per CPU
```

If the job requests 16 CPUs, use `%NProcShared=16`.

## Linda Parallelism (Multi-Node)

For multi-node jobs (if Gaussian is configured for Linda):

```bash
#SBATCH --nodes=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
```

```gjf
%NProcShared=16
%LindaWorkers=node1:16 node2:16 node3:16 node4:16
#p B3LYP/6-311G(d,p)
```

**Note:** Linda multi-node requires:
- Gaussian Linda version installed
- Cluster support for MPI between nodes
- Network file system access

**Fallback:** If Linda is not supported, use shared memory on one node only.

## Memory Per CPU

Rule of thumb: 1 GB per CPU for medium basis sets:

```gjf
%NProcShared=16
%Mem=16GB      # 16 CPUs × 1 GB/CPU
```

For large basis sets (cc-pVQZ, etc.):

```gjf
%NProcShared=16
%Mem=32GB      # 16 CPUs × 2 GB/CPU
```

## Scratch Directory

Gaussian uses scratch space for:
- Checkpoint files during calculation
- Temporary integrals
- Dump files

```bash
# Use fast scratch for Gaussian
export GAUSS_SCRDIR="/scratch/${USER}/gaussian_${SLURM_JOB_ID}"
mkdir -p "$GAUSS_SCRDIR"
```

Set the scratch directory explicitly:

```gjf
%RWF=/scratch/${USER}/gaussian_${SLURM_JOB_ID}/rwf
```

## Preflight Checklist

Before submitting:

- [ ] Input file parses correctly (test with small system)
- [ ] `%Chk` path is writable
- [ ] `%Mem` + `%NProcShared` match the job allocation
- [ ] Scratch space is available and sufficient
- [ ] Gaussian module is loaded

## Job Sizing

| System Size | Atoms | CPUs | Memory | Typical Time |
|------------|-------|------|--------|--------------|
| Small | < 20 | 8-16 | 8-16 GB | min-hours |
| Medium | 20-50 | 16-32 | 16-32 GB | hours-days |
| Large | 50-100 | 32+ | 32-64 GB | days |
| Very large | > 100 | 64+ | 64+ GB | Very long |

## Continuation and Restart

If a job hits walltime:

```gjf
%OldChk=mycalc.chk
%Chk=mycalc_restart.chk
#p B3LYP/6-311G(d,p) Opt

Resume optimization

---
# (continue with appropriate input)
```

Gaussian will read the last geometry from the checkpoint and continue.

## Monitoring

Check output file:

```bash
tail -f slurm-$SLURM_JOB_ID.out
```

Key indicators:

```
Normal Link 1 execution
SCF Done:  E(RB3LYP) =  -76.4087     A.U.
```

For errors:

```
Error termination:
```

## Common Batch Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| `File size limit exceeded` | Output file too large | Reduce print level |
| `Bad memory allocation` | %Mem too high | Reduce %Mem |
| `Cannot open %Chk` | Path issue | Use absolute path |
| `Linda not supported` | Not installed or configured | Use shared memory only |
| `Segmentation fault` | Basis set problem | Check input format |
