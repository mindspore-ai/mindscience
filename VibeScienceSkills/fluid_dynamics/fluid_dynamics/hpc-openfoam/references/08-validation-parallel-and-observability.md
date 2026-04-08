# OpenFOAM Validation, Parallel, And Observability

## Contents

- Validation sequence
- Residual and termination controls
- Function objects and post-processing
- Parallel execution rules

## Validation sequence

Use this order:

1. generate or import the mesh
2. run `checkMesh`
3. verify field files and patch names
4. verify `fvSchemes` and `fvSolution`
5. do a short dry-start or limited startup run before a long production run

If `checkMesh` reports quality problems, review numerics before spending time on solver tuning.

Solver launch guardrail:

- resolve executable form before the dry-start:
  - direct binary (for example `simpleFoam`) when available
  - otherwise `foamRun -solver <moduleName>` on modular OpenFOAM installations

## Residual and termination controls

OpenFOAM supports additional termination logic through `residualControl` in `fvSolution`.

Use it deliberately:

- for steady SIMPLE-style runs, use residual thresholds to stop when initial residuals are low enough
- for transient PIMPLE-style runs, use residual control together with outer-corrector logic

Residual interpretation matters:

- `tolerance` is the absolute stop criterion
- `relTol` is the relative stop criterion
- changing the initial field changes the normalized residual history, so compare runs carefully

Do not compare residual curves from different initializations as if they were directly equivalent.

## Function objects and post-processing

Function objects can run:

- inside the `functions` sub-dictionary of `system/controlDict`
- through `postProcess`
- through solver execution with `-postProcess`

Use function objects when you want structured runtime data instead of scraping ad hoc logs.

High-value categories:

- field diagnostics
- forces and moments
- sampling and probes
- solver-performance reporting
- continuity diagnostics

Representative function objects from the official docs:

- `probes`
- `forceCoeffs`
- `solverInfo`

Useful observability patterns:

- use solver-performance reporting when diagnosing poor linear-solver behavior
- use continuity diagnostics when mass conservation is suspect
- use force or sampling objects instead of storing every large field if only integral outputs are needed

Function-object outputs commonly land in `postProcessing/` and can be chained into later analysis.

## Parallel execution rules

Parallel execution is a contract between:

- `decomposeParDict`
- the actual MPI rank count
- any reconstruction or post-processing step

Rules:

- keep `numberOfSubdomains` equal to the intended rank count
- prefer `scotch` for complex or irregular domains
- after changing decomposition settings, regenerate decomposition before rerunning
- if decomposition changed, do not trust stale `processor*` directories

For production monitoring:

- store the chosen rank count in the case summary
- record which decomposition method was used
- keep function-object output paths stable across reruns so automated log parsers can follow them
