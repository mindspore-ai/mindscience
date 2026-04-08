# OpenFOAM `fvSolution` And Residual Control

## Contents

- `fvSolution` role
- Solver blocks
- SIMPLE, PISO, and PIMPLE controls
- `residualControl`
- Practical termination patterns

## `fvSolution` role

`fvSolution` controls:

- linear solver choice
- tolerances
- algorithm loops
- relaxation factors
- residual-based termination logic

It is the main execution-control dictionary for algebraic solution behavior.

## Solver blocks

Typical block structure:

- `solvers`
- `SIMPLE` or `PISO` or `PIMPLE`
- `relaxationFactors`

Map fields to solver types consistently:

- pressure-like equations -> symmetric-friendly choices such as `PCG` or `GAMG`
- velocity or turbulence equations -> asymmetric-friendly choices such as `PBiCGStab` or `smoothSolver`

## SIMPLE, PISO, and PIMPLE controls

Use these intuitions:

- `SIMPLE`: steady-state pressure-velocity coupling
- `PISO`: transient pressure-velocity correction
- `PIMPLE`: transient workflow with outer correction

High-value controls:

- `nCorrectors`
- `nOuterCorrectors`
- `nNonOrthogonalCorrectors`

Do not inflate all of them at once. Increase the one that matches the actual failure mode.

## `residualControl`

The official case-termination guide describes two main uses.

For steady SIMPLE-style runs:

- terminate when initial residuals fall below specified thresholds

For transient PIMPLE-style runs:

- use `residualControl` together with `nOuterCorrectors` to control outer-loop convergence

Representative official pattern for steady cases:

```text
residualControl
{
    p           1e-2;
    "(Ux Uy)"   1e-4;
    "(k|epsilon|omega)" 1e-4;
}
```

Representative official pattern for transient outer-loop control includes:

- explicit `nOuterCorrectors`
- a field block with `relTol 0` and absolute `tolerance`

## Practical termination patterns

Use residual termination when:

- steady-state convergence matters more than a fixed iteration budget
- transient outer loops should stop when field solves are good enough

Do not treat residual thresholds as universal truth across unrelated cases. They are case-dependent engineering controls, not physics invariants.
