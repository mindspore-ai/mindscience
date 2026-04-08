# OpenFOAM Solver Selection

## Contents

- Solver family map
- Pressure conventions
- File obligations by solver family
- Early selection checklist

## Solver family map

Pick the solver from the physics first. Do not start from a tutorial filename and retrofit the physics later.

Use this map (physics family first, executable form second):

- `simpleFoam` or `foamRun -solver incompressibleFluid`
  - steady-state
  - incompressible
  - turbulence-capable
  - pressure variable is kinematic pressure `p`
  - `fvSchemes` should use `steadyState` time treatment
- `pimpleFoam` or `foamRun -solver incompressibleFluid`
  - transient
  - incompressible
  - turbulence-capable
  - pressure variable is kinematic pressure `p`
  - algorithm is `PIMPLE`
- `interFoam` or `foamRun -solver incompressibleVoF`
  - transient
  - incompressible
  - two immiscible phases
  - hydrostatic-pressure-aware
  - mandatory fields include `p`, `p_rgh`, and `U`
- `rhoPimpleFoam` or `foamRun -solver fluid`
  - transient
  - compressible
  - heat-transfer capable
  - mandatory fields include `p`, `U`, and `T`
- `buoyantPimpleFoam` or `foamRun -solver fluid`
  - transient
  - buoyant heat transfer
  - mandatory fields include `p`, `p_rgh`, `U`, and `T`

Inference from the official solver pages:

- if the run is steady incompressible RANS, default toward `simpleFoam`
- if the run is transient incompressible, default toward `pimpleFoam`
- if free-surface or two-phase interface tracking is required, default toward `interFoam`
- if thermodynamics and density changes are core physics, move into a compressible or buoyant family

Launch compatibility check before run:

- if legacy solver binary exists (`simpleFoam`, `pimpleFoam`, `interFoam`), direct launch is valid
- if binary is missing but `foamRun` exists, launch with the module solver (`incompressibleFluid`, `incompressibleVoF`, `fluid`)
- if `foamRun` reports missing `lib<solver>.so`, do not proceed until the module name is corrected

## Pressure conventions

This is a frequent failure point.

Use these pressure interpretations:

- incompressible solvers like `simpleFoam` and `pimpleFoam` commonly use kinematic pressure `p`
- hydrostatic formulations such as `interFoam` and buoyant families may require both `p` and `p_rgh`
- compressible heat-transfer families usually require thermodynamic pressure `p` plus temperature `T`

Do not copy a `p_rgh`-based case into a pure incompressible kinematic-pressure solver without remapping the field set.

## File obligations by solver family

Minimum expectations:

- all cases:
  - `system/controlDict`
  - `system/fvSchemes`
  - `system/fvSolution`
  - initialized field files in the start-time directory
- turbulence-enabled cases:
  - `constant/turbulenceProperties`
  - turbulence transport fields that match the selected model
- thermodynamic cases:
  - thermophysical model files in `constant/`
- multiphase cases:
  - phase-fraction fields and hydrostatic-pressure-aware setup
- parallel runs:
  - `system/decomposeParDict`

OpenFOAM case structure is always cross-directory. If you change solver family, review `0/`, `constant/`, and `system/` together.

## Early selection checklist

Before generating any files, answer:

1. Is the case steady or transient?
2. Is density effectively constant?
3. Is there one phase or multiple phases?
4. Is temperature or buoyancy materially important?
5. Is the solver expected to run in parallel?

If any of those answers are still unknown, stop and resolve them. Wrong solver-family choice creates cascading mistakes in field names, pressure conventions, and numerics.
