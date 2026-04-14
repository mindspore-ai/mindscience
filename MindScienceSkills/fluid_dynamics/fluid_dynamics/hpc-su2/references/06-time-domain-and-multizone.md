# SU2 Time-Domain And Multizone Manual

## Contents

- transient workflow
- windowed convergence
- restart in unsteady runs
- multizone basics

## Transient workflow

The official docs define transient control through:

- `TIME_DOMAIN=YES`
- `TIME_MARCHING`
- `TIME_STEP`
- `MAX_TIME`
- `TIME_ITER`
- `INNER_ITER`

Use transient mode only when those controls are explicitly coherent.

## Windowed convergence

The official Solver Setup docs provide a dedicated windowed convergence workflow for periodic or statistically periodic runs.

Key controls:

- `WINDOW_CAUCHY_CRIT`
- `CONV_WINDOW_FIELD`
- `CONV_WINDOW_CAUCHY_ELEMS`
- `CONV_WINDOW_CAUCHY_EPS`
- `WINDOW_START_ITER`
- `WINDOW_FUNCTION`

Use this when coefficient stabilization over time matters more than a raw residual threshold.

## Restart in unsteady runs

For transient restart:

- enable `RESTART_SOL=YES`
- set `RESTART_ITER`
- keep output naming and iteration logic consistent

Do not restart an unsteady run without recording what physical time or iteration the restart corresponds to.

## Multizone basics

The official multizone docs describe two distinct patterns:

- **`MULTIZONE=YES`** — single-physics case partitioned into multiple geometric zones (e.g. a multi-component domain with zone-interface coupling). Uses one solver per zone and `MARKER_ZONE_INTERFACE` to define zone boundaries.
- **Coupled multiphysics** — uses a main config with `MATH_PROBLEM=COUPLED` or solver selection for multiphysics, combined with `CONFIG_LIST` pointing to per-physics config files. Each config drives a different solver (e.g. fluid + structural).

These are separate workflows. Do not add `MULTIZONE=YES` to a coupled multiphysics config or vice versa.
