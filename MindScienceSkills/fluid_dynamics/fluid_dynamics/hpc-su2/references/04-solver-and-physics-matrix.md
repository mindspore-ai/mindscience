# SU2 Solver And Physics Matrix

## Contents

- solver-family matrix
- steady versus transient controls
- turbulence and wall-treatment notes

## Solver-family matrix

The official Solver Setup docs list the main solver families.

| `SOLVER` | Problem family |
| --- | --- |
| `EULER` | compressible inviscid |
| `NAVIER_STOKES` | compressible viscous |
| `RANS` | compressible turbulent |
| `INC_EULER` | incompressible inviscid |
| `INC_NAVIER_STOKES` | incompressible viscous |
| `INC_RANS` | incompressible turbulent |
| `HEAT_EQUATION` | heat equation |
| `ELASTICITY` | structural elasticity |
| `MULTIPHYSICS` | coupled multiphysics (fluid + structure, etc.) via per-physics config files and `CONFIG_LIST`; distinct from `MULTIZONE` which partitions a single physics by geometry |

If `SOLVER` does not match the requested physics, stop there and fix it before editing markers or numerics.

## Steady versus transient controls

From the official docs:

- steady state uses `TIME_DOMAIN=NO`
- transient uses `TIME_DOMAIN=YES`
- transient runs may need `TIME_MARCHING`, `TIME_STEP`, `TIME_ITER`, and `INNER_ITER`

Do not leave transient-only controls half-configured in a supposed steady config.

## Turbulence and wall-treatment notes

The official marker docs note that turbulence boundary options differ by model.

Practical rule:

- if the case is RANS, revisit inlet turbulence data and wall treatment explicitly
- if switching from Euler to Navier-Stokes or RANS, revisit wall markers rather than assuming the old wall type still applies
