# VASP Cluster Execution Playbook

## Purpose

Use this reference when a VASP workflow is moving from input preparation into scheduled cluster execution.

## Pairing rule

Use this together with `hpc-orchestration`.

- this file owns VASP-specific execution, continuation, and handoff concerns
- `hpc-orchestration` owns scheduler, environment, transfers, and monitoring scaffolds

## Preflight

Before production submission:

1. confirm `INCAR`, `POSCAR`, `KPOINTS`, and `POTCAR` are mutually consistent
2. confirm the stage intent: relax, static, DOS, or bands
3. confirm the executable path and version
4. confirm whether restart files from the previous stage such as `WAVECAR` or `CHGCAR` are intentionally present
5. benchmark one short representative job shape before scaling out

Do not use a large allocation to discover a malformed four-file input set.

## Stage-aware execution

Typical stage sequence:

1. relaxation
2. static
3. DOS or bands

Keep each stage in its own directory or with an explicit handoff policy for:

- `CONTCAR` to next-stage `POSCAR`
- `CHGCAR`
- `WAVECAR`
- copied or regenerated `KPOINTS`

## Launch strategy

Portable starting point:

- let the scheduler allocate resources
- launch VASP with `srun` unless the site packaging requires another MPI launcher
- keep rank and thread geometry explicit

Example shape:

```bash
export OMP_NUM_THREADS=1
srun -n 32 vasp_std > vasp.out 2>&1
```

Do not change both resource shape and INCAR convergence controls at the same time unless you are deliberately running a scaling study.

## Logs worth watching

High-value checks:

- immediate parsing or setup failure
- SCF non-convergence
- ionic relaxation stalling
- `NELM` exhaustion
- missing or stale handoff files between stages

## Storage and output

Recommended habits:

- keep each stage output in its own directory
- retain `OUTCAR`, `OSZICAR`, `vasp.out`, and the intended restart files
- move heavy or transient outputs through scratch if the site storage policy requires it

The most expensive VASP operational failures are often handoff mistakes, not syntax mistakes.

## Restart and continuation

Before continuation:

- decide whether to reuse `WAVECAR` or `CHGCAR`
- verify the structure and k-point intent still match the new stage
- keep a copy of the last good stage outputs before overwriting them

Do not assume every stage should inherit every file from the previous one.

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| job starts and exits immediately | broken input set or wrong executable path | validate the four-file set and the actual VASP binary |
| SCF behavior changed unexpectedly after resubmission | restart or handoff files drifted | inspect `CHGCAR`, `WAVECAR`, and stage intent |
| relaxation wastes queue time | job shape and convergence controls are both changing | separate physics tuning from scaling changes |
| DOS or bands results look inconsistent | static-stage handoff was not trustworthy | rebuild from a validated static setup |
