## Environment And Storage Hygiene

## Purpose

Use this reference when the workflow risk is operational rather than solver-specific:

- environment modules and shell startup behavior
- conda or virtual environment activation inside batch jobs
- project storage versus scratch or node-local storage
- parallel filesystem hygiene for many-file workloads

## Shell startup rules

Batch jobs should start cleanly and quietly.

Guardrails:

- do not print banners or debug text from shell startup files that execute in non-interactive shells
- keep `.bashrc` safe for batch use
- put interactive-only behavior behind checks such as `[[ $- == *i* ]]`
- do not assume login-shell initialization order is identical across clusters

Why it matters:

- noisy startup files can corrupt tools that parse command output
- unexpected `module load` or `conda activate` calls create non-reproducible jobs
- startup side effects are difficult to diagnose from scheduler logs alone

## Module environment discipline

Prefer explicit module handling inside the batch script.

Minimal pattern:

```bash
module purge
module load compiler-stack
module load mpi-stack
module load application-module
module list
```

Recommended habits:

- start from `module purge` when reproducibility matters
- use `module spider` or site docs to find the canonical package name
- record `module list` in the job log
- keep compiler, MPI, and application stacks consistent
- avoid mixing unrelated toolchains unless the site explicitly supports it

## Conda and Python environments

If Python tooling is part of the workflow:

- activate environments inside the batch script, not only in the login shell
- keep the activation sequence explicit
- store heavy package caches outside small home quotas if the site recommends it

Portable pattern:

```bash
module purge
module load python
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate my-env
python -V
```

Use `conda run -n my-env ...` when shell activation is brittle.

## Filesystem tiers

Most clusters effectively expose several storage classes:

| Tier | Typical use | Good practice |
| --- | --- | --- |
| home | small configs, scripts, lightweight metadata | keep it small and stable |
| project or work | shared inputs, checkpoints, reusable datasets | organize by project and retain important outputs |
| scratch | large transient run directories | treat as disposable and back up what matters |
| node-local scratch | temporary per-job high-IO workspace | stage in, run, stage out |

Do not assume names like `/scratch` or `/tmp` are portable. Detect what the site provides and express the workflow in terms of a generic scratch tier.

## Node-local storage

Node-local storage is valuable for IO-heavy phases because it avoids repeated traffic to the parallel filesystem.

Good fits:

- decompression and preprocessing of large inputs
- temporary solver scratch files
- merge or sort phases that create many intermediates
- checkpoint staging before a final copy-out

General pattern:

1. create a job-unique working directory on local storage
2. copy or stage the minimal required inputs into it
3. run the high-IO phase there
4. copy back essential results, logs, and checkpoints
5. clean up

Portable skeleton:

```bash
scratch_root="${TMPDIR:-/tmp}"
job_scratch="${scratch_root}/${USER}/job-${SLURM_JOB_ID:-manual}"
mkdir -p "$job_scratch"
cp input.dat "$job_scratch/"
cd "$job_scratch"
```

## Parallel filesystem hygiene

Parallel filesystems are strong at large streaming IO and weaker at uncontrolled metadata storms.

Avoid:

- writing millions of tiny files when a packed format would work
- repeated `ls`, `find`, or polling over huge directories during production runs
- forcing every rank to read the same small file independently if the application can stage or broadcast it
- checkpointing too frequently without evidence that restart risk justifies it

Prefer:

- fewer larger files
- hierarchical run directories instead of giant flat directories
- explicit archive or bundle steps for tiny artifacts
- checkpoint intervals tied to wall time and failure risk

## Staging strategy

Use a simple stage-in and stage-out contract:

- immutable inputs come from project storage
- mutable working files go to scratch or node-local storage
- final outputs and validated checkpoints return to durable storage

When writing a solver workflow, decide up front:

- which files are authoritative inputs
- which files are temporary and disposable
- which checkpoints are restart-critical
- which outputs must be retained for downstream analysis

## Log and output hygiene

Recommended practices:

- separate scheduler logs from solver logs
- include job ID and case name in log filenames
- write heavy solver output into case-specific subdirectories
- rotate or truncate verbose debug logs in long workflows when safe

## Environment capture checklist

For reproducible runs, capture:

- module list
- solver executable path from `which`
- solver version banner when available
- core environment variables such as `OMP_NUM_THREADS`
- current working directory and scratch directory

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| batch job fails before solver launch | shell startup side effects | make non-interactive startup quiet and explicit |
| command found on login node but not in batch | environment only set interactively | move module or conda activation into the script |
| solver spends excessive time in IO | run directory on shared filesystem with many temp files | move temp phase to scratch or node-local storage |
| random file-not-found errors in long jobs | scratch cleanup or mixed staging policy | define one durable source and one transient workspace |
| metadata server overload or slow directory ops | too many small files or frequent polling | bundle outputs and reduce filesystem chatter |
