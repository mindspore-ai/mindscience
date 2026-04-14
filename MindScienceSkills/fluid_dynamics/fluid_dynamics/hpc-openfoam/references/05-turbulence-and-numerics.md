# OpenFOAM Turbulence And Numerics

## Contents

- Turbulence model matching
- Scheme selection
- Linear solver mapping
- Parallel decomposition choices

## Turbulence model matching

Select the turbulence family in `constant/turbulenceProperties` or the equivalent versioned file, then make the field files agree with it.

Use this map:

- `simulationType laminar`: no turbulence transport fields
- `simulationType RAS` with `kEpsilon`: require `k`, `epsilon`, and `nut`
- `simulationType RAS` with `kOmegaSST`: require `k`, `omega`, and `nut`
- `simulationType LES`: add the LES-specific model settings and confirm the resolved-field requirements for the chosen model

If the field set and turbulence model disagree, fix that mismatch before chasing numerical settings.

## Scheme selection

Use conservative startup numerics on new or unstable cases.

Safe first-pass pattern:

- time derivative: `Euler` or the steady-state form required by the solver
- convection: `Gauss upwind` for the most fragile phase of stabilization
- gradients: `Gauss linear` or limited variants
- Laplacians: corrected or limited corrected on imperfect meshes

Upgrade to higher-order convection only after the case is stable and residuals behave reasonably.

Do not combine:

- aggressive high-order convection
- poor mesh quality
- large startup timesteps

That combination is a common route to `Floating point exception`.

## Linear solver mapping

Map solver type to matrix character.

Typical pairings:

- pressure-like symmetric systems: `PCG` or `GAMG`
- velocity and turbulence equations: `PBiCGStab` or `smoothSolver`

For closed domains with no fixed pressure boundary, add a reference pressure cell and value to avoid a singular pressure system.

Use under-relaxation deliberately on steady-state RANS cases:

- pressure usually lower than velocity
- turbulence fields often need conservative relaxation during startup

Algorithm notes:

- `SIMPLE` is the steady-state pressure-velocity loop
- `PISO` is a transient coupling strategy with corrector loops
- `PIMPLE` blends outer correction with transient handling for harder transient cases

Increase `nNonOrthogonalCorrectors` only when mesh quality demands it. It is not a substitute for a bad boundary setup or an unsafe timestep.

Useful control intuition:

- `nCorrectors`: inner pressure-velocity correction count
- `nOuterCorrectors`: outer coupling loops, mainly relevant for `PIMPLE`
- `nNonOrthogonalCorrectors`: extra pressure correction passes for non-orthogonal meshes

Decision pattern:

- clean mesh and routine transient flow -> keep `nNonOrthogonalCorrectors` low
- harder transient coupling -> increase `nOuterCorrectors` before inflating many unrelated settings
- bad mesh alone is not a reason to overcomplicate all solver controls; first stabilize schemes and timestep

## Parallel decomposition choices

Use `decomposeParDict` as an execution contract, not a suggestion.

Preferred method choices:

- `scotch`: default for complex or unstructured domains
- `hierarchical`: good for structured meshes and topology-aware splits
- `simple`: only when a manual Cartesian split is intentional

Keep `numberOfSubdomains` equal to the planned MPI rank count. If it changes, regenerate the decomposition before rerunning the solver.
