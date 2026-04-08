# OpenFOAM Turbulence Boundary Recipes

## Contents

- Model-family matching
- Near-wall treatment choices
- Field obligations by model family
- Practical startup recipes

## Model-family matching

Choose the turbulence model family before writing boundary files.

Practical family map:

- laminar:
  - no turbulence transport fields
- RAS `kEpsilon` family:
  - expect `k`, `epsilon`, and `nut`
- RAS `kOmegaSST` family:
  - expect `k`, `omega`, and `nut`
- LES family:
  - expect LES model controls and the corresponding subgrid treatment

Do not mix the field set from one family with the model selection of another.

## Near-wall treatment choices

Inference from the official turbulence and wall-function documentation:

- wall treatment is part of the model design, not a cosmetic boundary choice
- wall functions are appropriate when the mesh is not wall-resolved
- low-Re or wall-resolved workflows require different near-wall assumptions

Use this decision pattern:

- coarse RANS wall-adjacent mesh -> wall-function workflow
- genuinely wall-resolved low-`yPlus` mesh -> low-Re or wall-resolved workflow

Do not promise wall-resolved fidelity when the mesh and boundary conditions are still set up for wall functions.

## Field obligations by model family

Typical expectations:

- `kEpsilon` startup:
  - inlet: provide physically plausible `k` and `epsilon`
  - wall: use wall-appropriate `nut` and `epsilon` handling rather than generic fixed values
- `kOmegaSST` startup:
  - inlet: provide `k` and `omega`
  - wall: use wall-appropriate `omega` and viscosity treatment

The exact patch types depend on version and library availability, but the key rule is stable:

- use boundary-condition types intended for turbulence near-wall treatment
- do not leave turbulence wall patches as generic placeholders such as unconstrained `zeroGradient` unless the model and patch semantics explicitly support it

## Practical startup recipes

For a first stable RANS startup:

1. choose the turbulence family
2. generate the matching field set
3. use conservative inlet turbulence values if the user did not provide them
4. use the model's intended wall treatment
5. start with conservative numerics before raising spatial order

Failure signatures that usually indicate turbulence-BC mismatch:

- repeated `bounding k`
- repeated `bounding epsilon`
- repeated `bounding omega`
- continuity errors that grow while residuals look erratic

In those cases, inspect turbulence boundary conditions before touching every numerical setting in the case.
