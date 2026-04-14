# SU2 Cluster Execution Playbook

## Purpose

Use this reference when a SU2 case is moving from config editing into scheduled cluster execution.

## Pairing rule

Use this together with `hpc-orchestration`.

- this file owns SU2-specific launch, convergence monitoring, and restart concerns
- `hpc-orchestration` owns scheduler, storage, transfer, and monitoring scaffolds

## Preflight

Before a production submission:

1. confirm the mesh file and marker names match the `.cfg`
2. confirm the solver family and physical model are stage-correct
3. run a small validation case or a short iteration-count smoke test
4. confirm restart and output settings are intentional
5. confirm the intended parallel launch count is reasonable for the mesh size

Do not use a long queue allocation to discover a broken mesh-marker mapping.

## Launch strategy

Portable starting point:

- serial smoke test: `SU2_CFD case.cfg`
- MPI production run: `srun SU2_CFD case.cfg`

Keep config, mesh, and launcher aligned. If the site packaging requires `mpirun`, use one launcher path consistently.

## Logs worth watching

High-value checks:

- immediate mesh import failure
- marker mismatch
- residual stagnation or divergence
- CFL or timestep instability in transient workflows
- restart files not being created or consumed as expected

## Storage and output

Recommended habits:

- keep authoritative mesh and config on project storage
- run heavy intermediate outputs on scratch if needed
- keep history and restart artifacts in predictable per-case paths

If history and restart naming are ambiguous, continuation becomes error-prone quickly.

## Restart and continuation

Before continuation:

- confirm whether the workflow should read a restart solution
- confirm output frequency is sufficient for recovery
- keep one explicit naming policy for restart files and stage directories

Do not change major physics options and restart from an incompatible prior state casually.

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| job starts but exits on mesh import | wrong mesh path or marker mismatch | validate mesh and marker sections before resubmission |
| run diverges after queue submission | CFL or numerics too aggressive | reduce aggressiveness before changing scale |
| restart does not continue correctly | `RESTART_SOL`, `RESTART_FILENAME`, or `RESTART_ITER` inconsistent across runs | verify all three are intentional and unchanged between the completed run and the restart run; check that the restart file actually exists at the expected path |
| scaling is poor | too many ranks for the mesh | reduce ranks and benchmark again |
