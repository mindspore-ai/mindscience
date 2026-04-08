---
name: hpc-orchestration
description: Coordinate end-to-end HPC execution workflows across scheduler submission, job monitoring, log tracking, self-healing, and post-processing handoff. Use when a task spans Slurm, PBS, LSF, MPI sizing, queue operations, cluster-safe execution, or multi-stage HPC workflow orchestration.
---

# HPC Orchestration

Use this skill as the repository-level execution layer above solver-specific HPC skills.

## Quick Start

| Step | Task | Reference |
|------|------|-----------|
| 1 | Cluster operations and end-to-end workflow | [03-cluster-operations-manual](references/03-cluster-operations-manual.md) |
| 2 | Coordinating solver skills with orchestration stages | [08-lifecycle-manual](references/08-lifecycle-manual.md) |
| 3 | Choosing scheduler directives or MPI sizing | [01-scheduler-and-parallelism](references/01-scheduler-and-parallelism.md) |
| 4 | Launching work inside a Slurm allocation | [02-slurm-launch-patterns](references/02-slurm-launch-patterns.md) |
| 5 | Environment modules, scratch policy, filesystem behavior | [07-environment-and-storage-hygiene](references/07-environment-and-storage-hygiene.md) |
| 6 | File synchronization, staging, archiving, verification | [06-data-transfer-and-staging](references/06-data-transfer-and-staging.md) |
| 7 | Compilation, package stacks, rebuildability | [04-software-build-and-reproducibility](references/04-software-build-and-reproducibility.md) |
| 8 | Live diagnosis, scaling studies, profiler evidence | [12-interactive-debugging-and-profiling](references/12-interactive-debugging-and-profiling.md) |
| 9 | VS Code Remote SSH, Jupyter, port forwarding | [13-remote-development-and-notebooks](references/13-remote-development-and-notebooks.md) |
| 10 | Apptainer or Singularity-style execution | [05-container-workflows](references/05-container-workflows.md) |
| 11 | Aligning solver skills with repository lifecycle contract | [14-public-protocol](references/14-public-protocol.md) |
| 12 | Scheduler, monitoring, log-tracking failures | [error-recovery](references/error-recovery.md) |

## Additional References

Load these on demand:

| Document | Topic |
|----------|-------|
| [11-tools-and-scripts](references/11-tools-and-scripts.md) | Shared execution tool roles |
| [09-ecosystem-roadmap](references/09-ecosystem-roadmap.md) | Repository-wide coverage goals |
| [10-legacy-template](references/10-legacy-template.md) | Adapting older solver skill drafts into current format |

## Guardrails

- Do not run heavy MPI workloads on login nodes.
- Do not separate queue submission from runtime monitoring in a production workflow.
- Do not resubmit unchanged failing jobs when a solver-specific error dictionary exists.
- Do not choose core counts without a scale-based heuristic.

## Assets

**When to include**: When the skill needs files that will be used in the final output.

**Use cases**: Templates, boilerplate code, batch scripts that get copied or modified.

### Scheduler Templates

| File | Use Case |
|------|----------|
| `assets/templates/slurm-basic.sh` | Basic SLURM batch submission |
| `assets/templates/slurm-array.sh` | SLURM array job |
| `assets/templates/slurm-packed-single-node.sh` | Packed single-node SLURM job |
| `assets/templates/slurm-apptainer.sh` | SLURM with Apptainer container |
| `assets/templates/slurm-perf-report.sh` | SLURM with performance reporting |
| `assets/templates/pbs-basic.sh` | Basic PBS batch submission |
| `assets/templates/lsf-basic.sh` | Basic LSF batch submission |
| `assets/templates/rsync-stage-in.sh` | Rsync stage-in for data transfer |

### Development Templates

| File | Use Case |
|------|----------|
| `assets/templates/jupyter-lab-compute.sh` | Jupyter Lab on compute node |
| `assets/templates/ssh-config-compute-proxy.example` | SSH config for compute proxy |

### Shared Scripts

| File | Use Case |
|------|----------|
| `scripts/hpc_job_submitter.py` | Job submission automation |
| `scripts/hpc_job_monitor.py` | Job monitoring |
| `scripts/hpc_log_tracker.py` | Log tracking |
| `scripts/hpc_slurm_deploy.py` | SLURM deployment automation |

## Reference Summary

All references are used in this skill:

| Document | Topic |
|----------|-------|
| [01-scheduler-and-parallelism](references/01-scheduler-and-parallelism.md) | Scheduler directives, MPI sizing |
| [02-slurm-launch-patterns](references/02-slurm-launch-patterns.md) | Launching work inside SLURM allocation |
| [03-cluster-operations-manual](references/03-cluster-operations-manual.md) | End-to-end cluster workflow |
| [04-software-build-and-reproducibility](references/04-software-build-and-reproducibility.md) | Compilation, package stacks, rebuildability |
| [05-container-workflows](references/05-container-workflows.md) | Apptainer/Singularity execution |
| [06-data-transfer-and-staging](references/06-data-transfer-and-staging.md) | File sync, staging, archiving |
| [07-environment-and-storage-hygiene](references/07-environment-and-storage-hygiene.md) | Environment modules, scratch policy |
| [08-lifecycle-manual](references/08-lifecycle-manual.md) | Solver skills with orchestration stages |
| [09-ecosystem-roadmap](references/09-ecosystem-roadmap.md) | Repository-wide coverage goals |
| [10-legacy-template](references/10-legacy-template.md) | Adapting older solver skill drafts |
| [11-tools-and-scripts](references/11-tools-and-scripts.md) | Shared execution tool roles |
| [12-interactive-debugging-and-profiling](references/12-interactive-debugging-and-profiling.md) | Live diagnosis, scaling studies |
| [13-remote-development-and-notebooks](references/13-remote-development-and-notebooks.md) | VS Code SSH, Jupyter, port forwarding |
| [14-public-protocol](references/14-public-protocol.md) | Repository lifecycle contract |
| [error-recovery](references/error-recovery.md) | Scheduler, monitoring, log-tracking failures |
