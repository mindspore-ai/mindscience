# OpenFOAM Boundary Condition Playbook

## Contents

- Patch naming contract
- Patch-type decisions
- Common field-condition patterns
- Failure patterns to catch early

## Patch naming contract

Treat patch names as a global schema.

Each patch name must agree across:

- the mesh boundary definition
- every field file in the start-time directory
- any function object or sampling setup that references the patch

If the mesh says `inlet` and one field file says `Inlet`, the case is already broken.

## Patch-type decisions

Choose the patch type from geometry, not from the desired field value.

Common geometric patch types:

- `wall`
- `patch`
- `symmetryPlane`
- `empty`
- `cyclic`
- `wedge`

Use them as follows:

- use `empty` only for strictly 2D front/back patches with one cell in the thin direction
- use `wedge` only for proper axisymmetric wedge setups
- use `cyclic` only when the patches are truly paired periodic boundaries

Do not use `symmetryPlane` as a convenience substitute for a wall or outlet.

## Common field-condition patterns

These are practical patterns, not universal laws. Apply them only when the physics matches.

For velocity-like fields:

- inlet with prescribed inflow: often `fixedValue`
- solid wall: usually `noSlip` or another wall-appropriate formulation
- outlet where flux direction may reverse: often a mixed formulation such as `pressureInletOutletVelocity`

For pressure-like fields:

- outlet with reference pressure: often `fixedValue`
- inlet when pressure is not prescribed: often `zeroGradient`
- hydrostatic formulations: verify whether the solver expects `p`, `p_rgh`, or both

For transported scalars:

- outlet recirculation-sensitive behavior often benefits from an inlet/outlet style condition
- wall conditions must match whether the scalar is fixed, insulated, or wall-function-based

For turbulence fields:

- keep wall treatment consistent with the turbulence model family
- if wall functions are intended, do not leave turbulence patch fields as generic placeholders
- if low-Re resolution is intended, do not silently swap in wall-function behavior

Inference from the official boundary-condition documentation:

- the boundary dictionary is field-specific
- the same mesh patch can legitimately use different field condition types for `U`, `p`, `k`, `omega`, and `T`

## Failure patterns to catch early

Stop and repair immediately if you see any of these:

- a patch appears in the mesh but is missing from a field file
- a cyclic pair is defined geometrically but treated as an inlet or outlet in fields
- a 2D case uses `patch` instead of `empty` on front/back planes
- a turbulence model is enabled but corresponding turbulence fields are missing
- the user asks for a pressure-driven flow but the selected conditions overconstrain both pressure and velocity at both ends

Boundary-condition inconsistency is one of the highest-leverage checks in the whole skill. It is usually cheaper to fix before touching numerics.
