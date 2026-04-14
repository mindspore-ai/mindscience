# ParaView Cluster Execution Playbook

## Purpose

Use this reference when a ParaView post-processing workflow is moving into scheduled cluster execution.

## Pairing rule

Use this together with `hpc-orchestration`.

- this file owns ParaView-specific batch rendering, extract, and remote-visualization concerns
- `hpc-orchestration` owns scheduler, transfer, remote-development, and tunnel scaffolds

## Preflight

Before production submission:

1. confirm the script or state file resolves all data paths from the real run directory
2. confirm the workflow should use `pvpython`, `pvbatch`, or `pvserver`
3. confirm output paths land on writable storage
4. benchmark one representative extraction or render step
5. keep state-file portability in mind if the workflow will move between machines

Do not use a large allocation to discover a stale absolute path in a state file.

## Batch rendering

Portable starting point:

- serial or light scripted work: `pvpython script.py`
- batch or MPI-capable workflow: `pvbatch script.py`

Use `pvbatch` for non-interactive cluster-side execution. Keep the script self-contained.

## Remote visualization

When the workflow needs live remote interaction:

- start `pvserver` under the scheduler
- connect through a secure tunnel or cluster-approved path
- keep heavy data resident on the cluster side

## Output policy

Recommended habits:

- write large intermediate data to scratch
- stage back only the screenshots, extracts, animations, or saved states you actually need
- keep output names stage-specific and reproducible

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| batch script runs but produces nothing | output path or active-object logic is implicit | make outputs and pipeline handles explicit |
| `pvbatch` workflow fails with connect logic | mixed batch and remote-server models | choose one execution path |
| remote session will not connect | scheduler, tunnel, or port setup is wrong | rebuild the end-to-end `pvserver` connection path |
| state file works locally but not on cluster | absolute paths or missing data remap | rebuild a portable state or remap data paths |
