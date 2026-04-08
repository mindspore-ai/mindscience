# OpenFOAM Function Object Recipes

## Contents

- Why function objects matter
- Core recipes
- Logging and output layout
- Selection rules

## Why function objects matter

Function objects are the cleanest way to expose machine-readable diagnostics from a case.

They can be:

- embedded in `system/controlDict`
- run via `postProcess`
- executed during a solver run with `-postProcess`

Prefer them to ad hoc log scraping when the target quantity is already supported.

## Core recipes

High-value official function objects include:

- `probes`
  - use for time histories at fixed spatial points
  - useful for transient monitoring and control-signal extraction
- `forceCoeffs`
  - use for lift, drag, and moment coefficients on selected patches
  - useful for airfoils, bluff bodies, and any integrated load analysis
- `solverInfo`
  - use to inspect linear-solver behavior and convergence patterns
  - useful when the case runs but linear iterations are unexpectedly expensive or erratic
- `yPlus`
  - use to assess near-wall resolution for turbulence workflows
  - useful when deciding whether wall functions or wall-resolved treatment are credible

## Logging and output layout

Function-object output commonly lands in `postProcessing/`.

Practical rules:

- keep object names stable so external tooling can locate outputs across reruns
- log only what is decision-relevant; avoid enabling many heavy diagnostics by default
- treat probe coordinates and monitored patches as part of the case contract

If the user asks for integrated loads, do not store every full field unless they also asked for detailed flow-field inspection.

## Selection rules

Use this map:

- pointwise transient monitor -> `probes`
- aerodynamic or hydrodynamic coefficient history -> `forceCoeffs`
- solver iteration diagnostics -> `solverInfo`
- wall-resolution audit -> `yPlus`

When a result can be computed by a function object, prefer that route over inventing a custom parser.
