# SU2 Output, Restart, And History Manual

## Contents

- output files
- history and screen output
- restart logic
- dry-run introspection

## Output files

The official Custom Output docs expose `OUTPUT_FILES` as the primary control surface.

High-value output options include:

- `RESTART`
- `RESTART_ASCII` or `CSV`
- `PARAVIEW`
- `PARAVIEW_ASCII`
- `SURFACE_CSV`
- `SURFACE_PARAVIEW`

Choose outputs from the post-processing goal rather than enabling all formats.

## History and screen output

Use history and screen output to track:

- residuals
- coefficients
- custom monitored quantities

The official docs also note that available fields depend on the active solver, and `SU2_CFD -d <cfg>` can list them.

## Restart logic

The official restart docs describe:

- `RESTART_SOL=YES`
- `RESTART_ITER` for unsteady continuation
- restart filename and write frequency controls

If restart continuity matters, keep `RESTART_FILENAME` and output write frequency stable across reruns.

## Dry-run introspection

`SU2_CFD -d` is a high-value inspection mode.

Use it to:

- list available output fields
- check parsed marker names
- inspect solver-aware config behavior without launching the full solve
