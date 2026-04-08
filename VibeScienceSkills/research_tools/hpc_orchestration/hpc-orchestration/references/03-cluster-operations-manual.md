## Cluster Operations Manual

## Purpose

Use this as the primary entry point for end-to-end cluster work. It is the high-level manual for turning solver input generation into a safe, reproducible HPC execution workflow.

Read this first when the task spans more than one of:

- cluster login and environment setup
- data transfer and staging
- interactive debugging
- batch submission
- monitoring and restart logic
- remote development or notebooks
- profiling and performance tuning

## Operating model

Think in four layers:

1. access layer: SSH, MFA, bastions, login nodes
2. workflow layer: project directories, storage tiers, data movement
3. execution layer: scheduler, launch commands, monitoring, restart
4. analysis layer: post-processing, profiling, result retention

Keep these layers separated. Many production failures happen because they are mixed casually.

## Cluster etiquette

Portable defaults:

- login nodes are for editing, compiling light software, transfers, and job submission
- compute nodes are for CPU-heavy, GPU-heavy, or memory-heavy work
- debug queues and interactive allocations are for short diagnosis, not long production
- transfers should use the site's preferred path, often login or dedicated transfer nodes rather than compute nodes

Never assume a permissive login-node policy. If the site policy is unclear, choose the more conservative behavior.

## End-to-end workflow

### 1. Prepare the workspace

Create a stable project layout before any job submission:

```text
project/
|- inputs/
|- runs/
|- logs/
|- post/
\- archive/
```

Use:

- immutable source data in `inputs/`
- one run directory per case under `runs/`
- scheduler and solver logs in predictable locations
- post-processed outputs separate from raw solver products

### 2. Choose storage tiers

Use [environment-and-storage-hygiene.md](environment-and-storage-hygiene.md) for the detailed rules.

Portable split:

- home: lightweight config and scripts
- project or work: durable shared inputs and validated outputs
- scratch: large transient run directories
- node-local scratch: temporary high-IO working space

### 3. Stage data

Use [data-transfer-and-staging.md](data-transfer-and-staging.md) when the workflow touches large files, many files, remote sync, or checksum verification.

Rule of thumb:

- stage in only the inputs needed for the next run
- write heavy transient files to scratch
- stage out validated checkpoints and final outputs

### 4. Build or activate the software environment

Use [software-build-and-reproducibility.md](software-build-and-reproducibility.md) when the environment is not already clean and known.

Preferred order:

1. site module stack
2. site-supported package manager or container
3. user-local build only when needed

Capture:

- module list
- compiler and MPI family
- executable path
- solver version

### 5. Validate a small representative run

Before full production:

- run the smallest meaningful case
- check launch semantics, output paths, and restart files
- confirm the solver can start and write logs
- confirm scratch staging and copy-out logic

Do not use full-node production allocations to discover trivial path or environment mistakes.

### 6. Select the execution mode

Choose between:

- interactive allocation for diagnosis and short experiments
- batch job for production
- job array for many independent homogeneous tasks
- packed single-node subjobs for a small number of short independent tasks

Use:

- [scheduler-and-parallelism.md](scheduler-and-parallelism.md) for resource shape
- [slurm-launch-patterns.md](slurm-launch-patterns.md) for Slurm launch strategy
- [container-workflows.md](container-workflows.md) for Apptainer or Singularity

### 7. Monitor and classify

During execution:

- watch queue state
- tail scheduler and solver logs
- classify failures before changing anything
- only resubmit after a real repair

Use:

- `scripts/hpc_job_monitor.py`
- `scripts/hpc_log_tracker.py`
- [error-recovery.md](error-recovery.md)

### 8. Restart or resume

Prefer restart-aware workflows over blind reruns.

Before resubmitting:

- identify the last good checkpoint
- verify whether the solver needs a restart flag or file rename
- adjust wall time or resource shape only when the evidence supports it

### 9. Post-process and archive

After a successful run:

- move critical outputs to durable storage
- keep enough metadata to reproduce the run
- archive only what is worth retaining
- remove scratch clutter that is no longer needed

## Interactive operations

Use [interactive-debugging-and-profiling.md](interactive-debugging-and-profiling.md) for details.

Portable rules:

- use `salloc` or the site's equivalent for live diagnosis
- use `srun --pty bash` for an interactive shell inside an allocation
- keep interactive sessions short and focused
- leave production-scale profiling to a controlled benchmark case

## Remote development and notebooks

Use [remote-development-and-notebooks.md](remote-development-and-notebooks.md) when the task needs VS Code Remote SSH, Jupyter, port forwarding, or browser-based analysis.

Portable rules:

- do not run long-lived notebook kernels on a login node unless the site allows it explicitly
- launch notebook servers from an interactive or batch allocation when they consume real resources
- keep port forwarding explicit and documented

## Containers

Use [container-workflows.md](container-workflows.md) when environment packaging is the hard part of the problem.

Portable rules:

- let the scheduler own placement
- let the container own the userland
- let host-mounted storage own the mutable data

## Performance work

Use [interactive-debugging-and-profiling.md](interactive-debugging-and-profiling.md) for benchmarking, profiler choice, and evidence capture.

Portable sequence:

1. confirm correctness on a small case
2. benchmark one baseline resource shape
3. vary one dimension at a time
4. retain logs, timings, and resource shape
5. only promote changes that improve time to solution or throughput

## Support and escalation package

When handing off to a cluster admin, teammate, or another agent, include:

- the exact job script
- module list or container image identifier
- launch command
- scheduler job ID
- relevant output logs
- current working directory layout
- what changed between the last failed and last successful run

This package avoids repeated low-signal back-and-forth.

## Daily preflight checklist

Before a serious run, confirm:

1. inputs are versioned or otherwise fixed
2. output directory is correct
3. scratch path exists
4. environment activation is explicit
5. resource request matches the intended parallel model
6. restart behavior is known
7. logs will land in predictable files

## Daily shutdown checklist

After a serious run, confirm:

1. critical outputs were copied off transient storage
2. restart files needed for continuation were retained
3. queue allocations are released
4. notebook servers and port forwards are closed
5. scratch clutter is cleaned up when safe

## Manual map

Load these references on demand:

- [scheduler-and-parallelism.md](scheduler-and-parallelism.md)
- [slurm-launch-patterns.md](slurm-launch-patterns.md)
- [environment-and-storage-hygiene.md](environment-and-storage-hygiene.md)
- [data-transfer-and-staging.md](data-transfer-and-staging.md)
- [software-build-and-reproducibility.md](software-build-and-reproducibility.md)
- [interactive-debugging-and-profiling.md](interactive-debugging-and-profiling.md)
- [remote-development-and-notebooks.md](remote-development-and-notebooks.md)
- [container-workflows.md](container-workflows.md)
- [error-recovery.md](error-recovery.md)
