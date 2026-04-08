## Slurm Launch Patterns

## Purpose

Use this reference when the workflow runs on Slurm and the main question is how to launch work correctly after allocation.

Typical triggers:

- choosing between `srun`, `sbatch`, and application wrappers
- deciding whether to use job arrays or packed subjobs
- mapping MPI ranks, OpenMP threads, and GPU resources
- avoiding oversubscription or idle allocations

## First principles

- `sbatch` requests an allocation and runs a batch script later.
- `salloc` requests an interactive allocation.
- `srun` launches one job step inside an allocation.
- Prefer a single launcher stack. Under Slurm, start from `srun` unless the site or solver documentation explicitly requires another launcher.
- Do not allocate resources with Slurm and then ignore them by launching an unmanaged background process.

## Decision matrix

| Workload shape | Preferred pattern | Why |
| --- | --- | --- |
| One MPI job across one or more nodes | `srun solver ...` | Lets Slurm place ranks and track the step |
| One serial or threaded program | `srun -n 1 -c <threads> solver ...` | Binds CPU resources clearly |
| Many independent cases with the same script | job array | Cleaner queue accounting and simpler retries |
| A few independent single-node subjobs inside one allocation | packed subjobs with multiple `srun --exclusive` steps | Reduces queue wait when tasks are short and homogeneous |
| One MPI job per GPU | `srun --gpus-per-task=1 ...` | Makes task-to-GPU mapping explicit |
| Hybrid MPI + OpenMP | `srun -n <ranks> -c <threads> ...` plus `OMP_NUM_THREADS=<threads>` | Keeps rank and thread geometry aligned |

## Core launch patterns

### Pure MPI

```bash
srun -n 128 solver_mpi input.in
```

Use this when the solver reads rank count from the launcher and does not require an additional wrapper.

### Pure OpenMP or threaded serial

```bash
export OMP_NUM_THREADS=16
srun -n 1 -c 16 solver_threads config.yaml
```

Set `-c` to the same value as `OMP_NUM_THREADS` unless the application guide says otherwise.

### Hybrid MPI + OpenMP

```bash
export OMP_NUM_THREADS=8
srun -n 16 -c 8 solver_hybrid case.dat
```

Checklist:

- ranks times threads should fit inside the allocation
- threaded math libraries should not silently add extra threads
- pinning policy should be checked on performance-sensitive runs

### GPU-bound task layout

```bash
srun -n 4 -c 8 --gpus-per-task=1 solver_gpu input.json
```

Use `--gpus-per-task` when each rank needs one GPU. If one job step uses all GPUs on one node, request them at the job level and keep one clear mapping strategy.

## `srun` vs `mpirun`

Portable default:

- use `srun` first on Slurm
- use `mpirun` only when the site stack or application packaging requires it

Why:

- Slurm tracks `srun` job steps directly
- environment propagation is usually simpler
- rank placement is less likely to drift from the allocation

Use extra caution when mixing launchers:

- do not wrap `srun` inside `mpirun`
- do not use both for the same step
- if a solver guide insists on `mpirun`, make sure rank count matches Slurm variables such as `SLURM_NTASKS`

## Job arrays

Use a job array when many tasks share one script and differ only by index-driven inputs.

Example:

```bash
#SBATCH --array=0-31

case_dir=$(printf "case_%03d" "${SLURM_ARRAY_TASK_ID}")
cd "$case_dir"
srun -n 1 postprocess.sh
```

Good fits:

- parameter sweeps
- many small preprocessing jobs
- embarrassingly parallel post-processing

Avoid arrays when:

- each case needs a different resource shape
- tasks must communicate with each other
- the workflow requires a tightly coordinated multi-stage pipeline inside one allocation

## Packed subjobs inside one allocation

Packed subjobs are useful when you already have one node allocation and want several independent single-node tasks to run concurrently.

Pattern:

```bash
srun --exclusive -N 1 -n 1 task_a.sh &
srun --exclusive -N 1 -n 1 task_b.sh &
wait
```

Use this only when:

- each subjob fits inside one node or a clearly carved subset of the allocation
- the tasks are short enough that separate queue waits would dominate wall time
- you can tolerate all subjobs sharing one batch job lifetime

Risks:

- hidden oversubscription if CPU or GPU counts are not carved explicitly
- harder per-subjob accounting than a true job array
- one failed backgrounded step may leave the batch job running if error handling is weak

Recommended guardrails:

- use `set -euo pipefail`
- record one log per subjob
- use `wait` and capture exit codes
- for CPU jobs, set `-c` per subjob
- for GPU jobs, assign `CUDA_VISIBLE_DEVICES` or use step-level GPU directives

## Resource-shape checklist

Before finalizing a Slurm script, verify:

1. `--nodes`, `--ntasks`, and `--cpus-per-task` are consistent with the solver parallel model.
2. `OMP_NUM_THREADS`, MKL threading, and other library threads do not exceed `--cpus-per-task`.
3. GPU count is explicit through `--gpus`, `--gpus-per-node`, or `--gpus-per-task`.
4. Every launched step fits inside the allocation without overlap unless overlap is intentional.
5. Output logs are separated per step or per array index.

## Workflow recommendations by skill family

- OpenFOAM, SU2, FEniCS MPI, CalculiX MPI, ElmerFEM MPI: prefer one `srun` per solver stage.
- LAMMPS and GROMACS: prefer the launcher recommended by the site build, but under Slurm start from `srun` unless packaging says otherwise.
- Quantum ESPRESSO and VASP: make rank, thread, and k-point or band parallel choices explicit before changing node counts.

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| fewer ranks than requested | launcher mismatch or wrong rank flag | standardize on `srun` or align `mpirun -np` with `SLURM_NTASKS` |
| all cores busy but solver is slower | OpenMP oversubscription | reduce library threads and match `-c` to thread count |
| some GPUs idle | ambiguous task-to-GPU mapping | use `--gpus-per-task` or a clear device assignment strategy |
| backgrounded subjobs interfere | missing `--exclusive` or CPU carving | add explicit step isolation and per-step CPU counts |
| array tasks overwrite outputs | no index-based output naming | include `${SLURM_ARRAY_TASK_ID}` in paths and logs |
