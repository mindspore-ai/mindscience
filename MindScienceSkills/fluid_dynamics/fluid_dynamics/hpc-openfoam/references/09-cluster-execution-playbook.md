# OpenFOAM Cluster Execution Playbook

## Purpose

Use this reference when an OpenFOAM case is moving from local editing into scheduled cluster execution.

## Pairing rule

Use this playbook together with `hpc-orchestration`.

- this file owns OpenFOAM-specific execution sequencing
- `hpc-orchestration` owns scheduler syntax, storage policy, transfers, and monitoring infrastructure

## Preflight

Before queue submission:

1. run the mesh generator such as `blockMesh` or the imported-mesh workflow
2. run `checkMesh`
3. confirm the solver launch form for this environment (`simpleFoam`-style binary or `foamRun -solver <module>`)
4. confirm all required fields exist under `0/`
5. confirm `decomposeParDict` matches the intended rank count if the job will run in parallel

Do not burn a full allocation discovering a missing field file or a broken mesh.

## Execution sequence

Typical parallel sequence:

1. stage the case to scratch if needed
2. run `decomposePar`
3. launch the solver with `srun` or the site-approved MPI launcher
4. inspect residual and Courant behavior in the log
5. run `reconstructPar` or targeted reconstruction when the workflow needs merged outputs

Typical command structure:

```bash
decomposePar -force
solver_or_module="${1:-simpleFoam}"
if command -v "$solver_or_module" >/dev/null 2>&1; then
  srun "$solver_or_module" -parallel > "log.${solver_or_module}" 2>&1
else
  srun foamRun -solver "$solver_or_module" -parallel > "log.foamRun_${solver_or_module}" 2>&1
fi
reconstructPar
```

For transient cases, keep solver and reconstruction stages distinct so a failing solve does not masquerade as a reconstruction problem.

## Rank-count alignment

OpenFOAM parallel jobs need these values aligned:

- scheduler task count
- `numberOfSubdomains` in `decomposeParDict`
- actual launcher rank count

If one of these drifts, the run is invalid even if the batch job starts.

Portable starting point:

- use `scotch` unless there is a reason to impose a manual decomposition
- choose rank counts from the mesh scale rather than node count alone

## Log signatures worth watching

High-value live checks:

- residuals that stall or blow up
- `Courant Number` growth
- `bounding` messages on turbulence or phase fields
- `floating point exception`
- processor-directory creation and reconstruction errors

For long runs, use `../../hpc-orchestration/scripts/hpc_log_tracker.py` or a lightweight log parser to watch `ExecutionTime`, `Courant Number`, `Initial residual`, and `bounding` lines at regular intervals.

## Storage and output

Recommended layout:

- keep the authoritative case on project storage
- run heavy processor directories on scratch
- copy back reconstructed fields, logs, and selected time directories

Do not archive every transient processor directory by default unless restart policy requires it.

## Restart and continuation

Before a restart:

- identify the last valid time directory
- confirm whether the run should continue from `latestTime`
- confirm processor directories are consistent if restarting in parallel
- only reconstruct before restart when the restart strategy truly needs merged data

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| batch job starts but solver exits immediately | missing field, broken case path, or wrong environment | rerun small preflight commands and inspect the solver binary path |
| parallel run starts with rank mismatch | `numberOfSubdomains` and scheduler tasks differ | align decomposition and allocation |
| transient run diverges late | timestep or numerics too aggressive | inspect `deltaT`, `maxCo`, and fragile schemes before resubmission |
| reconstruction fails after a good solve | inconsistent processor outputs or missing time directories | inspect processor trees before rerunning the solver |
