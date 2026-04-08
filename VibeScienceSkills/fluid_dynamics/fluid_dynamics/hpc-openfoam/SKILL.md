---
name: hpc-openfoam
description: Generate, review, debug, and recover OpenFOAM case files for CFD workflows. Use when working with OpenFOAM dictionaries, case structure, turbulence fields, boundary conditions, decomposition, numerics, or OpenFOAM runtime errors.
---

# HPC OpenFOAM

OpenFOAM is an open-source Computational Fluid Dynamics (CFD) toolbox widely used in scientific research and engineering for simulating fluid flow, heat transfer, turbulence, and multiphase phenomena. It serves applications ranging from aerospace and automotive aerodynamics to environmental engineering, energy systems, and academic research in fluid mechanics.

## Quick Start

| Step | Task | Reference |
|------|------|-----------|
| 1 | Anatomy of an OpenFOAM case | [01-case-setup](references/01-case-setup.md) |
| 2 | Choosing the right solver | [02-solver-selection](references/02-solver-selection.md) |
| 3 | Mapping physical boundaries to fields | [03-boundary-condition-playbook](references/03-boundary-condition-playbook.md) |
| 4 | Turbulence model and wall treatment | [04-turbulence-bc-recipes](references/04-turbulence-bc-recipes.md) |
| 5 | Schemes, algorithms, decomposition | [05-turbulence-and-numerics](references/05-turbulence-and-numerics.md) |
| 6 | Canonical case templates | [06-case-recipes](references/06-case-recipes.md) |
| 7 | Forces, probes, diagnostics | [07-function-object-recipes](references/07-function-object-recipes.md) |
| 8 | Validation and post-processing | [08-validation-parallel-and-observability](references/08-validation-parallel-and-observability.md) |
| 9 | Scheduler-backed execution | [09-cluster-execution-playbook](references/09-cluster-execution-playbook.md) |
| - | Mesh generation, vertex ordering | [mesh-and-blockmeshdict-manual](references/mesh-and-blockmeshdict-manual.md) |
| - | Thermophysical, buoyant setups | [heat-transfer-and-compressible-cases](references/heat-transfer-and-compressible-cases.md) |
| - | Algorithm loops, solver blocks | [fvsolution-and-residual-control](references/fvsolution-and-residual-control.md) |
| - | Solver-to-field mapping | [field-and-dictionary-matrix](references/field-and-dictionary-matrix.md) |
| - | Crashes, divergence, warnings | [error-recovery](references/error-recovery.md) |

## Skill Map

```
                    OPENFOAM CASE WORKFLOW
    +--------------------------------------------+
    |  Case-Setup                                |
    |  ----------------------------------------  |
    |  Solver-Selection  |  Boundary-Conditions  |
    |  ----------------- | --------------------  |
    |  Turbulence-BC     |  Numerics-Schemes     |
    |  ----------------- | --------------------  |
    |  Case-Recipes      |  Function-Objects     |
    |  ----------------------------------------  |
    |  Validation-Parallel  |  Cluster-Execution |
    |  ----------------------------------------  |
    |  Error-Recovery                            |
    +--------------------------------------------+
```

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Steady or transient? | `02-solver-selection` | simpleFoam vs pimpleFoam |
| Incompressible or compressible? | `02-solver-selection` | rhoPimpleFoam for compressible |
| Single-phase or multiphase? | `02-solver-selection` | interFoam for VOF |
| Laminar or turbulent? | `04-turbulence-bc-recipes` | RAS/LES model selection |
| Wall treatment? | `04-turbulence-bc-recipes` | wallFunction vs resolved |
| Parallel decomposition? | `08-validation-parallel-and-observability` | scotch vs manual |
| SCF not converging? | `error-recovery` | Check mesh, schemes, BCs |

## Work Sequence

1. Classify the case first: steady or transient, incompressible or compressible,
   single-phase or multiphase, laminar or turbulent.
2. Generate the minimum consistent file set across `0/`, `constant/`, and `system/`.
   Do not edit one layer in isolation if it changes the required fields elsewhere.
3. Match solver family and fields:
   - `simpleFoam` or `foamRun -solver incompressibleFluid`: steady incompressible;
     expect `U`, `p`, and turbulence fields if `RAS`.
   - `pimpleFoam` or `foamRun -solver incompressibleFluid` with transient/PIMPLE settings:
     transient incompressible; review timestep control and outer correctors.
   - `interFoam` or `foamRun -solver incompressibleVoF`: multiphase; control both
     `maxCo` and `maxAlphaCo`.
4. Validate mesh and numerics before a long run:
   - run `blockMesh` or the mesh generator
   - run `checkMesh`
   - refuse to keep orthogonal-only schemes on poor-quality meshes
5. Keep parallel settings aligned:
   - make `numberOfSubdomains` match the intended MPI rank count
   - prefer `scotch` for complex geometries unless the user requests a manual layout
6. Resolve executable compatibility before launch:
   - if `simpleFoam`/`pimpleFoam`/`interFoam` exists, it is valid to run directly
   - otherwise prefer `foamRun -solver <moduleName>` and verify the module loads

## Guardrails

- Do not invent dictionary keys, patch types, or solver names.
- Do not use turbulence fields that do not match the chosen model family.
- Do not keep aggressive second-order convection schemes during first-pass
  stabilization on a fragile case.
- Do not treat `checkMesh` warnings as optional if the log is already diverging.

## Assets

**When to include**: When the skill needs files that will be used in the final output.

**Use cases**: Templates, boilerplate code, batch scripts that get copied or modified.

### Case Templates

| Template | Use Case | Key Files |
|---------|----------|-----------|
| `assets/templates/simplefoam-laminar/` | Steady incompressible, laminar | `0/U`, `0/p`, `constant/transportProperties` |
| `assets/templates/simplefoam-turbulent/` | Steady incompressible, RAS kEpsilon | `0/U`, `0/p`, `0/k`, `0/epsilon`, `0/nut` |
| `assets/templates/pimplefoam-minimal/` | Transient incompressible, PIMPLE | Full PIMPLE-controlled transient setup |
| `assets/templates/interfoam-minimal/` | Transient multiphase VOF (water/air) | `0/U`, `0/p`, `0/p_rgh`, `0/alpha.water`, `constant/phaseProperties` |

### Scheduler Scripts

| Template | Use Case |
|---------|----------|
| `assets/templates/openfoam-parallel-slurm.sh` | SLURM batch script for parallel OpenFOAM runs |
| `assets/templates/openfoam-parallel-pbs.sh` | PBS batch script for parallel OpenFOAM runs |

## Reference Summary

All references are used in this skill:

| Document | Topic |
|----------|-------|
| [01-case-setup](references/01-case-setup.md) | Core directory layout, boundary/field consistency, controlDict checklist |
| [02-solver-selection](references/02-solver-selection.md) | Solver family map, pressure conventions, file obligations by solver |
| [03-boundary-condition-playbook](references/03-boundary-condition-playbook.md) | Patch naming contract, patch-type decisions, field-condition patterns |
| [04-turbulence-bc-recipes](references/04-turbulence-bc-recipes.md) | Model-family matching, near-wall treatment, field obligations |
| [05-turbulence-and-numerics](references/05-turbulence-and-numerics.md) | Scheme selection, linear solver mapping, decomposition |
| [06-case-recipes](references/06-case-recipes.md) | Internal duct flow, external aerodynamics, free-surface cases |
| [07-function-object-recipes](references/07-function-object-recipes.md) | Probes, forceCoeffs, solverInfo, yPlus, logging patterns |
| [08-validation-parallel-and-observability](references/08-validation-parallel-and-observability.md) | Validation sequence, residual controls, function objects, parallel rules |
| [09-cluster-execution-playbook](references/09-cluster-execution-playbook.md) | Scheduler execution, rank-count alignment, log signatures, restart |
| [mesh-and-blockmeshdict-manual](references/mesh-and-blockmeshdict-manual.md) | Mesh generation workflow, blockMeshDict anatomy, vertex ordering |
| [heat-transfer-and-compressible-cases](references/heat-transfer-and-compressible-cases.md) | Buoyant/thermophysical field sets, pressure conventions |
| [fvsolution-and-residual-control](references/fvsolution-and-residual-control.md) | Solver blocks, SIMPLE/PISO/PIMPLE controls, residualControl |
| [field-and-dictionary-matrix](references/field-and-dictionary-matrix.md) | Solver-to-field matrix, dictionary responsibilities |
| [error-recovery](references/error-recovery.md) | CFL failures, FPE, pressure problems, parallel mismatches |

## Error Recovery

Consult `references/error-recovery.md` for:

- Mesh quality issues
- Divergence and stability problems
- Boundary condition errors
- Parallel decomposition failures
- Restart and checkpoint handling

## Outputs

Produce a short case summary that states:

- solver and physics family
- required fields and dictionaries touched
- validation commands run or still needed
- stability risks and the next recovery step if the case is failing
