---
name: hpc-lammps
description: Generate, review, debug, and recover LAMMPS molecular dynamics input scripts and deck assemblies. Use when working with LAMMPS command ordering, data files, force fields, ensembles, neighbor settings, thermo output, cluster execution, or common runtime errors such as lost atoms and non-numeric pressure.
---

# HPC LAMMPS

LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) is a molecular dynamics code used across computational materials science, chemistry, biophysics, and soft matter research — from ab initio-inspired metallic alloys to coarse-grained polymer melts. This skill treats LAMMPS deck authoring as a strict staged workflow.

## Quick Start

| Step | Task | Reference |
|------|------|-----------|
| 1 | Command ordering (eight-stage law) | [01-command-order](references/01-command-order.md) |
| 2 | Unit system selection and timestep sizing | [02-units-and-timestep](references/02-units-and-timestep.md) |
| 3 | Atom styles, data file format, type mapping | [03-atom-types-and-data](references/03-atom-types-and-data.md) |
| 4 | Pair/bond/angle styles (LJ, EAM, Tersoff, ReaxFF) | [04-force-fields](references/04-force-fields.md) |
| 5 | Neighbor list skin, rebuild frequency, stability | [05-neighbor-settings](references/05-neighbor-settings.md) |
| 6 | NVE/NVT/NPT, Langevin, barostat damping | [06-thermostats-barostats](references/06-thermostats-barostats.md) |
| 7 | Thermo output, dumps, computes, restart files | [07-output-and-analysis](references/07-output-and-analysis.md) |
| 8 | SLURM batch, MPI/GPU/Hybrid execution | [08-cluster-execution](references/08-cluster-execution.md) |
| - | Diagnose and fix runtime errors | [error-recovery](references/error-recovery.md) |

## Skill Map

```
                     INPUT SCRIPT AUTHORING
     ┌──────────────────────────────────────────┐
     │  01-Command-Order (The Eight-Stage Law)   │
     │  ────────────────────────────────────    │
     │  02-Units-And-Timestep │ 03-Atom-Types   │
     │  ────────────────────────────────────│────│
     │  04-Force-Fields │ 05-Neighbor-Settings  │
     │  ────────────────────────────────────│────│
     │  06-Thermostats-Barostats │ 07-Output     │
     │  ────────────────────────────────────│────│
     │  08-Cluster-Execution │ Error-Recovery    │
     └──────────────────────────────────────────────┘
```

## Key Decision Points

| Question | Reference | Summary |
|----------|-----------|---------|
| Which unit system? | `02-units-and-timestep` | metal, real, lj — affects everything downstream |
| Structure source? | `03-atom-types-and-data` | `read_data` vs `create_box + create_atoms` |
| Which force field? | `04-force-fields` | LJ, EAM, Tersoff, ReaxFF, or hybrid |
| Which ensemble? | `06-thermostats-barostats` | NVE, NVT, NPT, Langevin, or deforming box |
| Timestep and constraints? | `02-units-and-timestep` | Must match unit system and constraints |
| Neighbor skin and rebuild? | `05-neighbor-settings` | Critical for hot, dense, or deforming runs |
| Output and analysis plan? | `07-output-and-analysis` | Thermo variables, dump frequency, restart strategy |
| Cluster execution? | `08-cluster-execution` | MPI rank count, GPU, SLURM batch |

## Guardrails

- Do not place `read_data` before `units` and `atom_style`.
- Do not copy a timestep value from one unit system to another.
- Do not launch NPT on a fragile fresh structure without an equilibration stage (minimize → NVT → NPT).
- Do not ignore neighbor-list settings on hot or highly deforming runs.
- Do not use two full integrators (e.g., two `nve` fixes) on the same atoms.
- Do not mix `read_data` with `create_box`/`create_atoms` in the same deck.

## Outputs

Always report:

- unit system (metal / real / lj / ...)
- structure source and atom type count
- force field and pair style with file or parameters used
- ensemble, thermostat/barostat choices, and damping parameters
- log-derived failure mode when repairing a script

## Template Files

**When to include**: When the skill needs files that will be used in the final output.

**Use cases**: Templates, boilerplate code, batch scripts that get copied or modified.

| File | Use Case |
|------|----------|
| `assets/templates/lj_fluid_minimal.in` | Reduced LJ fluid — minimal example with NVE ensemble |
| `assets/templates/eam_metal_minimal.in` | EAM metallic system — Cu with NVT and energy minimization |
| `assets/templates/lammps-mpi-slurm.sh` | SLURM batch wrapper for MPI-parallel LAMMPS execution |

## Reference Summary

All references are used in this skill:

| Document | Topic |
|----------|-------|
| [01-command-order](references/01-command-order.md) | The eight-stage command ordering law |
| [02-units-and-timestep](references/02-units-and-timestep.md) | Unit system selection and timestep sizing |
| [03-atom-types-and-data](references/03-atom-types-and-data.md) | Atom styles, data file format, type mapping |
| [04-force-fields](references/04-force-fields.md) | Pair/bond/angle styles: LJ, EAM, Tersoff, ReaxFF |
| [05-neighbor-settings](references/05-neighbor-settings.md) | Neighbor list skin, rebuild frequency, stability |
| [06-thermostats-barostats](references/06-thermostats-barostats.md) | NVE/NVT/NPT, Langevin, barostat damping |
| [07-output-and-analysis](references/07-output-and-analysis.md) | Thermo, dumps, computes, restart files |
| [08-cluster-execution](references/08-cluster-execution.md) | SLURM batch, MPI/GPU/Hybrid execution |
| [error-recovery](references/error-recovery.md) | Lost atoms, NaN pressure, potential errors, neighbor failures |
