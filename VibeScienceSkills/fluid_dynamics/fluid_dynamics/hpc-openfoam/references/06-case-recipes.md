# OpenFOAM Case Recipes

## Contents

- Internal incompressible pipe or duct flow
- External aerodynamic RANS case
- Free-surface or tank-like case
- Recipe selection checklist

## Internal incompressible pipe or duct flow

Default pattern:

- solver family: `simpleFoam` for steady RANS, `pimpleFoam` for transient
  - OpenFOAM-11 modular fallback: `foamRun -solver incompressibleFluid`
- fields:
  - `U`
  - `p`
  - turbulence fields matching the model, if turbulent
- common patches:
  - inlet
  - outlet
  - walls
- boundary logic:
  - inlet velocity prescribed
  - outlet pressure reference prescribed
  - walls use no-slip velocity and turbulence wall treatment as needed

Practical numerics:

- start with conservative convection schemes
- use under-relaxation on steady RANS startup
- add probes or pressure-drop monitoring instead of storing every field if the goal is hydraulic performance

## External aerodynamic RANS case

Default pattern:

- solver family: often `simpleFoam` for steady attached-flow screening, `pimpleFoam` if transients matter
  - OpenFOAM-11 modular fallback: `foamRun -solver incompressibleFluid`
- fields:
  - `U`
  - `p`
  - turbulence fields
- common patches:
  - far-field or inlet
  - outlet
  - body wall
  - side/top/bottom symmetry or far-field style patches

High-value observability:

- `forceCoeffs` for lift and drag
- `yPlus` for near-wall audit
- `probes` if wake or pressure history matters

Common failure mode:

- overconstrained far field with both pressure and velocity fixed in conflicting ways

## Free-surface or tank-like case

Default pattern:

- solver family: `interFoam`
  - OpenFOAM-11 modular fallback: `foamRun -solver incompressibleVoF`
- fields:
  - `U`
  - `p_rgh`
  - `p`
  - `alpha.<phase>`
- controls:
  - explicit `maxCo`
  - explicit `maxAlphaCo`
  - conservative startup timestep

Typical concerns:

- interface compression and timestep sensitivity
- hydrostatic pressure consistency
- outlet and atmosphere boundary treatment

Do not import a single-phase incompressible template and add `alpha` afterwards. The pressure treatment and field set are different from the start.

## Recipe selection checklist

Pick a recipe by answering:

1. Is the flow internal or external?
2. Is the run steady or transient?
3. Is there one phase or an interface?
4. Is the goal field visualization, integral force, pressure drop, or another reduced metric?

If the user goal is mostly a reduced metric, bias the case toward good function-object instrumentation and minimal necessary field output.
