## Container Workflows For HPC

## Purpose

Use this reference when the workflow needs containerized execution on a cluster, especially with Apptainer or Singularity-style images.

Typical triggers:

- packaging an application stack that is difficult to rebuild on the cluster
- running Python or post-processing tools without polluting the site environment
- carrying a solver userland across clusters while still using the site scheduler

## Portable defaults

- Prefer Apptainer or Singularity-style execution on HPC systems over Docker daemon workflows on compute nodes.
- Treat the image as mostly immutable.
- Keep large mutable outputs on mounted host storage, not inside the image.
- Let the scheduler allocate resources and then launch container commands inside the allocation.

## Image and bind strategy

Recommended split:

- image file: software stack
- bind mounts: project data, scratch, licenses, and runtime outputs
- host scheduler: resource allocation and accounting

Example:

```bash
apptainer exec \
  --bind "$PWD:/work" \
  --bind "${SCRATCH_DIR}:/scratch" \
  image.sif \
  bash -lc 'cd /work && python post.py'
```

Guidelines:

- bind the working directory explicitly
- bind scratch explicitly if the workflow stages there
- avoid assuming the container can see every host path automatically
- keep paths simple and documented

## Slurm plus container launch

Preferred pattern under Slurm:

```bash
srun apptainer exec image.sif solver_command
```

Why:

- Slurm still manages placement and accounting
- the container becomes part of the launched step rather than a separate unmanaged process

## MPI inside containers

MPI is the highest-risk part of containerized HPC execution.

Portable recommendations:

- use the site scheduler for rank launch
- verify whether the site expects host MPI libraries, container MPI libraries, or an ABI-compatible combination
- avoid inventing a mixed MPI stack without testing

Two common models:

### Host launcher with compatible containerized application

```bash
srun apptainer exec --bind "$PWD:/work" image.sif solver_mpi input.in
```

Use this when the site provides the MPI runtime and the application inside the image is compatible with it.

### Container used mainly for serial or threaded tools

```bash
srun -n 1 -c 8 apptainer exec --bind "$PWD:/work" image.sif python analyze.py
```

This is safer than a containerized MPI path and works well for preprocessing and post-processing.

## GPU containers

Use the container runtime's GPU passthrough mode when required.

Portable example:

```bash
srun --gpus-per-task=1 apptainer exec --nv image.sif solver_gpu config.json
```

Checklist:

- request GPUs from Slurm explicitly
- use `--nv` only when the container runtime and cluster support it
- keep CUDA or ROCm expectations aligned with the host driver stack

## Writable state and overlays

Prefer host-mounted writable directories for job outputs.

Use overlays or writable sandbox modes only when the cluster policy allows them and the workflow truly needs package mutation at runtime.

Avoid:

- installing packages into the image during every job
- writing important outputs only into ephemeral container state
- hiding license or scratch paths inside undocumented bind rules

## When containers are a good fit

- Python and analysis toolchains with fragile dependencies
- reproducible preprocessing pipelines
- post-processing or visualization preparation
- solvers whose vendor or site documentation already supports Apptainer execution

## When containers are a poor fit

- untested MPI-heavy production runs
- workflows requiring privileged Docker features
- performance-sensitive runs where filesystem and MPI integration have not been benchmarked

## Operational checklist

Before promoting a containerized workflow:

1. confirm the runtime available on the cluster
2. verify image path and bind mounts
3. run a small `hostname` or version test with `srun apptainer exec`
4. test one small production-like case
5. verify logs and outputs land on host storage

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| container sees no input files | missing bind mount | bind the project or case directory explicitly |
| MPI ranks hang or crash | incompatible host and container MPI stack | reduce to one rank test and align MPI packaging |
| GPU not visible in container | missing GPU passthrough | request GPU resources and use the runtime GPU flag |
| output disappears after job ends | wrote only inside container-private state | write outputs to bound host directories |
| job runs but accounting is confusing | container launched outside Slurm step control | wrap `apptainer exec` with `srun` |
