## Scheduler And Parallelism

## Purpose

Use this reference when the main question is which scheduler syntax to emit and how large the parallel job should be.

## Scheduler detection

Detect the scheduler family from available commands or environment variables.

| Scheduler | Common commands | Common environment hints |
| --- | --- | --- |
| Slurm | `sbatch`, `squeue`, `sinfo`, `srun` | `SLURM_JOB_ID`, `SLURM_CLUSTER_NAME` |
| PBS Pro or Torque | `qsub`, `qstat`, `pbsnodes` | `PBS_JOBID`, `PBS_NODEFILE` |
| LSF | `bsub`, `bjobs`, `bhosts` | `LSB_JOBID`, `LSB_HOSTS` |

Guardrails:

- do not emit mixed directive styles in one script
- do not assume the scheduler from cluster branding alone
- if auto-detection fails, ask for the scheduler family or treat the run as non-clustered

## Directive mapping

| Resource or concept | Slurm | PBS Pro or Torque | LSF |
| --- | --- | --- | --- |
| job name | `#SBATCH --job-name=<name>` | `#PBS -N <name>` | `#BSUB -J <name>` |
| stdout | `#SBATCH --output=<file>` | `#PBS -o <file>` | `#BSUB -o <file>` |
| stderr | `#SBATCH --error=<file>` | `#PBS -e <file>` | `#BSUB -e <file>` |
| queue or partition | `#SBATCH --partition=<name>` | `#PBS -q <name>` | `#BSUB -q <name>` |
| nodes | `#SBATCH --nodes=<n>` | `#PBS -l select=<n>` or `nodes=<n>` | usually implied by total slots and placement rules |
| MPI ranks | `#SBATCH --ntasks=<n>` | usually derived from `select` and `mpiprocs` | `#BSUB -n <n>` |
| CPUs per rank | `#SBATCH --cpus-per-task=<n>` | `ncpus=<n>` style resource clause | often site-specific resource clause |
| wall time | `#SBATCH --time=HH:MM:SS` | `#PBS -l walltime=HH:MM:SS` | `#BSUB -W HH:MM` |
| memory | `#SBATCH --mem=<size>` or `--mem-per-cpu=<size>` | `#PBS -l mem=<size>` | site-specific, often `-M` or `rusage[mem=...]` |
| launcher | `srun` | `mpiexec` or site wrapper | `mpirun`, `jsrun`, or site wrapper |

Prefer the scheduler's native launcher integration when available and documented.

## Parallel sizing rules of thumb

Sizing should come from problem scale, memory pressure, and known solver behavior. Do not pick rank counts arbitrarily.

### CFD and mesh-based finite volume workflows

Typical starting point:

- target roughly 50,000 to 200,000 cells per MPI rank for production-scale CPU jobs
- stay cautious when ranks fall below about 10,000 cells each unless scaling data proves the communication cost is acceptable

Interpretation:

- too many ranks for too few cells usually produces poor efficiency
- too few ranks may cause memory pressure or unacceptable wall time

### Classical molecular dynamics

Typical starting point:

- simple short-range models often tolerate several hundred to around one thousand atoms per CPU rank
- expensive force fields such as reactive or many-body models usually need fewer atoms per rank

Interpretation:

- atom count alone is not enough; force-field cost and neighbor settings matter
- benchmark one short representative run before scaling out

### FEM and sparse linear algebra workflows

Typical starting point:

- size first from memory, not only raw core count
- use direct solvers only when the matrix and memory budget justify them
- for Krylov workflows, match MPI ranks and threads to the preconditioner and linear algebra backend

Interpretation:

- solver and preconditioner choice can dominate scaling more than mesh count alone
- a smaller job with a better algebra stack often beats a larger poorly tuned run

### Electronic structure workloads

Typical starting point:

- keep rank, thread, k-point, and band parallel choices explicit
- increase parallelism only after checking memory per rank and the solver's internal parallel decomposition

Interpretation:

- many DFT workflows do not scale linearly with raw MPI ranks
- parallel choices must match the code's own decomposition model

## Resource-shape checklist

Before emitting a production script, verify:

1. node count, rank count, and threads per rank are mutually consistent
2. `OMP_NUM_THREADS` and library thread variables match the requested CPUs per task
3. total memory request matches the expected working set
4. GPU counts are explicit if any GPU-enabled path is used
5. the launch command matches the allocation model

## Launcher selection

Portable defaults:

- Slurm: prefer `srun`
- PBS: prefer the site-recommended `mpiexec` or `mpirun`
- LSF: prefer the site-recommended launcher and placement syntax

Do not mix launcher stacks casually. If the application guide requires `mpirun`, align its rank count to scheduler-provided variables.

## Queue strategy

Use the smallest resource request that is still production-safe.

Why:

- smaller jobs often start sooner
- shorter wall times usually improve schedulability
- unrealistic peak requests create avoidable pending states

Recommended habits:

- benchmark on a short representative case
- scale node count only after single-node behavior is understood
- keep one serial or small-rank validation path for regression checks

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| job rejected immediately | wrong directive family or syntax | regenerate for the correct scheduler |
| allocation starts but ranks are missing | launcher mismatch | align launcher choice and rank count |
| job remains pending for a long time | oversized or impossible request | reduce nodes, wall time, or memory and re-evaluate queue choice |
| job runs slower on more nodes | communication dominates | reduce rank count or switch to a better rank-thread balance |
