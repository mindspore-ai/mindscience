# OpenFOAM Case Setup

## Contents

- Core directory layout
- Boundary and field consistency
- `controlDict` checklist
- Mesh validation rules

## Core directory layout

Treat a standard case as three coupled layers:

- `0/`: initial and boundary fields such as `U`, `p`, `k`, `epsilon`, `omega`, `nut`, or phase fraction fields
- `constant/`: mesh-independent physics such as transport properties, thermophysical properties, turbulence properties, and sometimes the mesh
- `system/`: execution control, numerics, and decomposition settings

Do not change the solver family without checking all three layers.

## Boundary and field consistency

Keep patch names consistent across:

- the mesh boundary definition
- every field file in `0/`
- function objects and probes that reference patches

Common patch types to use deliberately:

- `wall`
- `patch`
- `symmetryPlane`
- `empty` for strictly 2D front/back planes
- `cyclic`
- `wedge` for axisymmetric wedge cases

Common boundary condition building blocks:

- `fixedValue` for prescribed inlet velocity or known scalar values
- `zeroGradient` for outlet-like extrapolation or natural scalar flux
- `noSlip` for wall velocity
- `inletOutlet` for mixed outflow or recirculation-sensitive scalar behavior

When a turbulence model is enabled, add the matching turbulence fields instead of only `U` and `p`.

Typical field expectations:

- laminar incompressible: `U`, `p`
- `kEpsilon`: `U`, `p`, `k`, `epsilon`, `nut`
- `kOmegaSST`: `U`, `p`, `k`, `omega`, `nut`
- VOF multiphase: `U`, `p_rgh` or `p`, `alpha.<phase>`

## `controlDict` checklist

Always verify these entries:

- `application`
- `startFrom`
- `startTime` when `startFrom startTime`
- `stopAt`
- `endTime`
- `deltaT`
- `writeControl`
- `writeInterval`

Add these deliberately when stability matters:

- `adjustTimeStep yes;`
- `maxCo <value>;`
- `maxAlphaCo <value>;` for VOF or similar interface-capturing runs
- `purgeWrite <N>;` for long transient runs
- `runTimeModifiable true;` when mid-run tuning is expected

Use smaller startup timesteps for fresh meshes, sharp transients, or multiphase interfaces.
If `writeControl` is `adjustableRunTime`, keep timestep adaptivity enabled so the solver can honor the requested output cadence.

## Mesh validation rules

Run `checkMesh` before trusting a new case.

Use these practical decisions:

- if the mesh is strongly non-orthogonal, avoid `orthogonal` normal-gradient handling
- if skewness and non-orthogonality are nontrivial, prefer corrected or limited-corrected Laplacian treatment
- if a block mesh is intended to be 2D, keep one cell in the thin direction and use `empty` on the paired patches

Treat mesh warnings as first-class debugging input. A numerics change without a mesh check is often wasted work.
