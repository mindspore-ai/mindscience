# SU2 Error Recovery

## Contents

- mesh and marker failures
- config incompatibilities
- convergence failures
- output interpretation issues

## Mesh and marker failures

If the run fails early:

1. verify the mesh file is readable
2. verify marker names match the config
3. verify dimension and physics assumptions match the mesh

## Config incompatibilities

Typical failure class:

- keyword combination incompatible with solver family
- missing required thermodynamic or flow-state data
- wrong marker usage for the chosen boundary condition

Fix the physics-to-config mapping before micro-tuning numerical controls.

## Convergence failures

If residuals stall or blow up:

1. verify the solver family and boundary setup
2. verify freestream or inlet state consistency
3. reduce aggressiveness in startup controls if available
4. inspect mesh quality and wall treatment assumptions

## Output interpretation issues

If output files exist but results look wrong:

1. verify monitored markers
2. verify reference values and coefficient interpretation
3. verify the intended surface is actually included in reporting
