---
name: hpc-quantum-espresso-cluster-execution
description: Cluster execution playbook for Quantum ESPRESSO including preflight checks, stage-aware execution, launch strategy, and failure pattern triage
type: reference
---

# Quantum ESPRESSO Cluster Execution Playbook

## Purpose

Use this reference when a Quantum ESPRESSO workflow is moving from input preparation into scheduled cluster execution.

## Pairing rule

Use this together with `hpc-orchestration`.

- this file owns `pw.x`-specific execution, restart, and stage-handoff concerns
- `hpc-orchestration` owns scheduler, storage, transfer, and monitoring scaffolds

## Preflight

Before production submission:

1. confirm the workflow stage: SCF, relax, NSCF, DOS, or bands
2. confirm pseudopotentials resolve from the actual run directory or pseudo path
3. confirm `prefix` and `outdir` are intentional and writable
4. confirm the executable path and version
5. benchmark one short representative run shape before scaling out

Do not use a large allocation to discover a broken `prefix`, pseudo path, or namelist typo.

## Stage-aware execution

Typical sequence:

1. SCF or relax
2. optional continuation or tighter run
3. NSCF, DOS, or bands post-ground-state stages

Keep each stage in a dedicated directory or with an explicit artifact handoff policy for:

- `prefix`
- `outdir`
- charge density artifacts
- k-point changes between stages

## Launch strategy

Portable starting point:

- let the scheduler allocate resources
- launch `pw.x` with `srun` unless the site packaging requires another MPI launcher
- keep rank and thread geometry explicit

Example shape:

```bash
export OMP_NUM_THREADS=1
srun -n 32 pw.x -in scf.in > scf.out 2>&1
```

Do not change both parallel shape and SCF stabilization controls at the same time unless it is a deliberate scaling study.

## Logs worth watching

High-value checks:

- immediate parse failure
- pseudopotential path errors
- SCF non-convergence
- relaxation instability
- inconsistent `prefix` or `outdir` handoff between stages

## Storage and output

Recommended habits:

- keep stage directories separate
- retain output logs and the key restart or charge artifacts
- use scratch for heavy transient directories when the site policy suggests it

Many QE operational failures come from `prefix` or `outdir` drift rather than physics choices.

## Restart and continuation

Before continuation:

- confirm the new stage should reuse the prior `prefix` and artifacts
- confirm `outdir` points to the intended data
- keep a copy of the last good stage output before overwriting

Do not mix unrelated stages into one directory casually.

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| job starts and exits immediately | bad input or pseudo path | validate namelists and pseudo resolution in the real run directory |
| SCF behavior changes after resubmission | `outdir` or `prefix` drifted | inspect handoff variables before tuning numerics |
| bands or DOS stage looks inconsistent | ground-state handoff was not trustworthy | rebuild from a validated SCF or NSCF sequence |
| scaling is poor | job shape does not match workload decomposition | benchmark a smaller shape and adjust ranks or threads |
