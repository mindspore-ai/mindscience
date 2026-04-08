# OpenFOAM Heat-Transfer And Compressible Cases

## Contents

- When to leave incompressible templates
- Buoyant and thermophysical field sets
- Pressure conventions in heated flows
- Practical startup rules

## When to leave incompressible templates

Move out of a plain incompressible case when:

- density change matters
- heat transfer is a primary part of the problem
- buoyancy drives the flow
- the solver family explicitly requires thermophysical models

Do not bolt temperature onto an incompressible template without checking solver-family requirements.

## Buoyant and thermophysical field sets

The official `buoyantPimpleFoam` guide lists mandatory fields:

- `p`
- `p_rgh`
- `U`
- `T`

It also expects:

- `constant/turbulenceProperties`
- heat-transfer configuration under `constant/thermophysicalModels`
- optional `constant/fvOptions`

Inference for the skill:

- if the user asks for buoyant heat transfer, assume both thermal and hydrostatic pressure setup must be reviewed
- if the case uses a buoyant solver, `p_rgh` is a first-class field, not an optional extra

## Pressure conventions in heated flows

Heated and buoyant cases often use:

- `p`: pressure
- `p_rgh`: pressure minus hydrostatic contribution

Do not collapse them into a single incompressible-pressure interpretation.

Common migration failure:

- copying a `simpleFoam` case and adding `T`, while ignoring thermophysical models and hydrostatic-pressure handling

## Practical startup rules

For compressible or buoyant startup:

1. choose the correct solver family first
2. write the required thermal and pressure fields together
3. add thermophysical models before tuning numerics
4. use conservative startup timesteps and coupling controls
5. treat buoyancy-driven transients as transient from the beginning

If a heated case is failing, check field set and solver family before changing every boundary condition.
