# OpenFOAM Error Recovery

## Contents

- CFL and timestep failures
- Floating point exceptions
- Pressure and bounded-field problems
- Parallel mismatch failures

## CFL and timestep failures

Failure signatures:

- `Courant Number mean`
- `maxCo`
- rapid divergence in the first few timesteps

Recovery actions:

1. Lower `deltaT`.
2. Enable `adjustTimeStep`.
3. Set explicit `maxCo`.
4. For multiphase cases, also cap `maxAlphaCo`.

If the user asks for a fast run, explain the stability tradeoff instead of silently keeping an unsafe timestep.

## Floating point exceptions

Failure signatures:

- `Floating point exception`
- `sigFpe`
- fields becoming `NaN`

Recovery actions:

1. Downgrade convection schemes to bounded or upwind variants.
2. Reduce relaxation aggressiveness.
3. Re-run `checkMesh`.
4. Increase non-orthogonal correction only if mesh quality justifies it.

Do not keep pushing the same scheme set after repeated `NaN` failures.

## Pressure and bounded-field problems

Failure signatures:

- `solution singularity`
- `bounding k`
- `bounding epsilon`
- continuity errors that grow instead of decay

Recovery actions:

1. Verify pressure reference handling for enclosed domains.
2. Bound turbulence transport with conservative convection schemes.
3. Check boundary conditions for missing or contradictory pressure and turbulence values.
4. Loosen startup tolerances only as a temporary stabilization step.

If a scalar goes negative repeatedly, inspect both the boundary conditions and the transport scheme before touching solver tolerances again.

## Parallel mismatch failures

Failure signatures:

- `processor boundary mismatch`
- `decomposeParDict`
- MPI fatal errors right after launch

Recovery actions:

1. Make `numberOfSubdomains` match the MPI rank count.
2. Remove stale `processor*` directories after changing decomposition settings.
3. Re-run `decomposePar`.
4. Prefer `scotch` if a manual split is producing empty or imbalanced partitions.
