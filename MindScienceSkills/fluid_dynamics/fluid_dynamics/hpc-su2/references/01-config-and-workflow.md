# SU2 Config And Workflow Manual

## Contents

- Core file model
- Standard execution flow
- Solver-family selection
- Output and restart logic

## Core file model

The SU2 config file is the main control surface.

High-value artifacts:

- mesh file
- `.cfg` configuration
- flow and convergence outputs
- optional restart files

Treat the `.cfg` as the main declarative contract for the run.

## Standard execution flow

Practical pipeline:

1. choose solver family and physics
2. prepare mesh and verify marker names
3. write `.cfg`
4. run `SU2_CFD`
5. inspect convergence history and outputs

Use dedicated preprocess or conversion commands only when the mesh source format requires it.

## Solver-family selection

Pick the solver from physics:

- compressible flow -> compressible family
- incompressible flow -> incompressible family
- coupled multiphysics or partitioned workflows -> multizone or coupled tooling

Do not start from a random tutorial config and force unrelated physics into it.

## Output and restart logic

Plan outputs deliberately:

- convergence history
- restart files if continuation matters
- surface and volume output as needed

Keep naming stable when iterative case repair is expected.
