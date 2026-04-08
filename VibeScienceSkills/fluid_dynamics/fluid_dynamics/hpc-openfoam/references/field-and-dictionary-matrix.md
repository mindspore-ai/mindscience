# OpenFOAM Field And Dictionary Matrix

## Contents

- Solver-to-field matrix
- Dictionary responsibility matrix
- Stability-control matrix

## Solver-to-field matrix

Use this as a fast compatibility table.

| Solver family | Flow class | Core fields | Extra required setup |
| --- | --- | --- | --- |
| `simpleFoam` | steady incompressible | `U`, `p` | turbulence fields if turbulent |
| `pimpleFoam` | transient incompressible | `U`, `p` | timestep control, outer-corrector logic |
| `interFoam` | transient multiphase | `U`, `p`, `p_rgh`, `alpha.<phase>` | interface and hydrostatic-pressure handling |
| `buoyantSimpleFoam` | steady buoyant heat transfer | `U`, `p`, `p_rgh`, `T` | thermophysical setup |
| `buoyantPimpleFoam` | transient buoyant heat transfer | `U`, `p`, `p_rgh`, `T` | thermophysical setup plus transient controls |

If the chosen solver family and field set do not match this table, repair that mismatch before debugging numerics.

## Dictionary responsibility matrix

Use this table to avoid editing the wrong file.

| Concern | Primary file(s) |
| --- | --- |
| runtime control, write cadence, timestep | `system/controlDict` |
| discretization schemes | `system/fvSchemes` |
| linear solvers, algorithm loops, residual control | `system/fvSolution` |
| mesh topology from blocks | `system/blockMeshDict` |
| turbulence model family | `constant/turbulenceProperties` |
| thermophysical models | `constant/thermophysicalProperties` (OpenFOAM 11+) or `constant/thermophysicalModels` (older versions); multiphase uses `constant/phaseProperties` |
| parallel decomposition | `system/decomposeParDict` |
| start-time fields and patch values | `0/` |

## Stability-control matrix

Use these first-response controls:

| Symptom | First file to inspect | High-value controls |
| --- | --- | --- |
| Courant blow-up | `system/controlDict` | `deltaT`, `adjustTimeStep`, `maxCo`, `maxAlphaCo` |
| solver divergence on poor mesh | `system/fvSchemes` | bounded convection, corrected Laplacians |
| slow or erratic algebraic solves | `system/fvSolution` | solver type, `tolerance`, `relTol`, relaxation |
| turbulence field bounding | `0/` and `system/fvSchemes` | turbulence BCs, bounded transport schemes |
| parallel crash | `system/decomposeParDict` | `numberOfSubdomains`, decomposition method |
