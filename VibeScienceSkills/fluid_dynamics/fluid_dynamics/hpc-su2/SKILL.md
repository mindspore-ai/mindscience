---
name: hpc-su2
description: SU2 open-source CFD solver for compressible and incompressible flow simulations. Supports Euler, Navier-Stokes, RANS, multiphysics, and shape optimization. Use for external aerodynamics, turbomachinery, multiphase flows, and design optimization.
---

# HPC-SU2

SU2 is an open-source CFD solver centered on the `.cfg` configuration file. It handles compressible and incompressible flows, turbulence modeling, and multiphysics simulations.

## Scientific Applications

| Application | Use Case |
|------------|----------|
| **External Aerodynamics** | Airfoil analysis, vehicle drag/lift |
| **Turbomachinery** | Compressor, turbine, fan simulations |
| **Multiphysics** | Fluid-structure interaction, heat transfer |
| **Shape Optimization** | Discrete adjoint, design variables |
| **Incompressible Flows** | Hydrodynamics, HVAC, marine |
| **Windowed Convergence** | Periodic flows, time-averaged coefficients |

## Key Concepts

### Solver Families
| `SOLVER` | Physics |
|----------|---------|
| `EULER` | Compressible inviscid |
| `NAVIER_STOKES` | Compressible viscous |
| `RANS` | Compressible turbulent |
| `INC_EULER` | Incompressible inviscid |
| `INC_NAVIER_STOKES` | Incompressible viscous |
| `INC_RANS` | Incompressible turbulent |
| `MULTIPHYSICS` | Coupled fluid + structure |

### Execution Commands
| Command | Purpose |
|---------|---------|
| `SU2_CFD` | Primary solver |
| `SU2_DEF` | Mesh deformation (shape optimization) |
| `SU2_DOT` | Discrete adjoint |
| `SU2_CFD -d <cfg>` | Dry-run introspection |

### Boundary Markers
| Marker | Use Case |
|--------|----------|
| `MARKER_FAR` | Farfield boundary |
| `MARKER_INLET` | Inlet with velocity/thermodynamic state |
| `MARKER_OUTLET` | Pressure outlet |
| `MARKER_WALL` | No-slip wall |
| `MARKER_HEATFLUX` | Heat flux wall (compressible) |
| `MARKER_MONITORING` | Force/moment monitoring |

## Workflow

```
1. Config and workflow → [references/01-config-and-workflow.md]
2. Marker and boundaries → [references/02-marker-and-boundary-matrix.md]
3. Convergence and execution → [references/03-convergence-and-execution.md]
4. Solver and physics selection → [references/04-solver-and-physics-matrix.md]
5. Output and restart → [references/05-output-restart-and-history.md]
6. Time domain and multizone → [references/06-time-domain-and-multizone.md]
7. Cluster execution → [references/08-cluster-execution.md]
8. Error diagnosis → [references/error-recovery.md]
```

## Config and Workflow

See [references/01-config-and-workflow.md](references/01-config-and-workflow.md) for:
- Config file as main control surface
- Standard execution flow
- Solver family selection
- Output and restart logic

## Marker and Boundary Matrix

See [references/02-marker-and-boundary-matrix.md](references/02-marker-and-boundary-matrix.md) for:
- Marker-name contract (mesh must match config)
- Boundary categories (inlet, outlet, wall, farfield)
- Marker-selection matrix for physical roles

## Convergence and Execution

See [references/03-convergence-and-execution.md](references/03-convergence-and-execution.md) for:
- Execution commands (SU2_CFD, SU2_DEF, SU2_DOT)
- Convergence controls and monitoring
- Dry-run introspection with `-d` flag
- Multizone notes

## Solver and Physics Matrix

See [references/04-solver-and-physics-matrix.md](references/04-solver-and-physics-matrix.md) for:
- Solver family matrix
- Steady vs transient controls
- Turbulence and wall treatment

## Output, Restart, and History

See [references/05-output-restart-and-history.md](references/05-output-restart-and-history.md) for:
- Output file formats (RESTART, PARAVIEW, CSV)
- History and screen output
- Restart logic
- Dry-run introspection

## Time Domain and Multizone

See [references/06-time-domain-and-multizone.md](references/06-time-domain-and-multizone.md) for:
- Transient workflow controls
- Windowed convergence for periodic flows
- Unsteady restart
- Multizone vs coupled multiphysics distinction

## Cluster Execution

See [references/08-cluster-execution.md](references/08-cluster-execution.md) for:
- SLURM/PBS job submission
- Launch strategy (serial vs MPI)
- Restart and continuation
- Storage and output staging

## Error Recovery

See [references/error-recovery.md](references/error-recovery.md) for diagnosis of:
- Mesh and marker failures
- Config incompatibilities
- Convergence failures
- Output interpretation issues

## Templates

Template files in [assets/templates/](assets/templates/) serve as starting points:

| Template | Purpose |
|----------|---------|
| [`incompressible_steady.cfg`](assets/templates/incompressible_steady.cfg) | Steady incompressible viscous (INC_NAVIER_STOKES) |
| [`compressible_external_aero.cfg`](assets/templates/compressible_external_aero.cfg) | Compressible RANS external aerodynamics |
| [`unsteady_windowed.cfg`](assets/templates/unsteady_windowed.cfg) | Unsteady with windowed convergence |
| [`su2-cfd-slurm.sh`](assets/templates/su2-cfd-slurm.sh) | SLURM job submission script |

## Skill Decision Map

```
User Requirements
├─ Physics Type
│  ├─ Compressible inviscid → EULER
│  ├─ Compressible viscous → NAVIER_STOKES
│  ├─ Compressible turbulent → RANS + SST
│  ├─ Incompressible viscous → INC_NAVIER_STOKES
│  └─ Incompressible turbulent → INC_RANS
├─ Flow Regime
│  ├─ Steady → TIME_DOMAIN=NO (default)
│  └─ Unsteady → TIME_DOMAIN=YES + TIME_STEP
├─ Boundary Setup
│  ├─ Farfield → MARKER_FAR
│  ├─ Inlet → MARKER_INLET
│  ├─ Outlet → MARKER_OUTLET
│  └─ Wall → MARKER_WALL or MARKER_HEATFLUX
└─ Outputs
   ├─ Restart → RESTART file
   ├─ Visualization → PARAVIEW
   └─ Surface data → SURFACE_CSV
```

## Guardrails

### Must Verify
- [ ] Mesh marker names match config exactly
- [ ] Solver family matches requested physics
- [ ] Transient controls are coherent (not half-configured)
- [ ] Output and restart settings are intentional

### Never Do
- Do not invent config keywords outside documented options
- Do not mix steady and transient controls
- Do not use MARKER_WALL for compressible RANS heat flux
- Do not restart from incompatible prior state

## Required Output

Always report:
- Solver family and physics
- Mesh and marker assumptions
- Key config sections
- Convergence status and output files
