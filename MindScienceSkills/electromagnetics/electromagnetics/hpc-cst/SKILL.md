---
name: hpc-cst
description: Build, review, debug, and automate CST Studio Suite electromagnetic simulation workflows. Use when working with CST Microwave Studio, antenna design, RF/microwave components, EMC/EMI analysis, transient and frequency domain solvers, parameter sweeps, optimization, or HPC cluster execution.
---

# CST Studio Suite HPC Skill

CST Studio Suite is a high-performance electromagnetic simulation platform used
for antenna design, microwave devices, EMC/EMI analysis, and signal integrity.
This skill covers the complete workflow from project setup through cluster execution.

## Quick Start

### Typical Workflow
1. Create or import project structure — see [references/01-project-structure.md](references/01-project-structure.md)
2. Select appropriate solver — see [references/02-solver-selection.md](references/02-solver-selection.md)
3. Configure mesh and boundary conditions — see [references/03-mesh-control.md](references/03-mesh-control.md) and [references/04-boundary-conditions.md](references/04-boundary-conditions.md)
4. Set up excitations and ports — see [references/05-excitations-ports.md](references/05-excitations-ports.md)
5. Run parameter sweeps or optimization — see [references/06-parameter-sweep.md](references/06-parameter-sweep.md)
6. Postprocess results — see [references/07-postprocessing.md](references/07-postprocessing.md)
7. Submit to HPC cluster — see [references/08-cluster-execution.md](references/08-cluster-execution.md)
8. Handle errors — see [references/error-recovery.md](references/error-recovery.md)

## Skill Map

```
User Requirements
├─ Project Setup
│  └─ File types, model organization, VBA macros → 01-project-structure.md
├─ Solver Selection
│  ├─ Wideband antenna → Transient Solver → 02-solver-selection.md
│  ├─ High-Q filter → Frequency Domain + Eigenmode → 02-solver-selection.md
│  └─ Large scattering problem → Integral Equation → 02-solver-selection.md
├─ Mesh Configuration
│  ├─ Hexahedral mesh (Transient) → 03-mesh-control.md
│  ├─ Tetrahedral mesh (Frequency Domain) → 03-mesh-control.md
│  └─ Adaptive refinement → 03-mesh-control.md
├─ Boundary Conditions
│  ├─ Open/PML (antenna radiation) → 04-boundary-conditions.md
│  ├─ PEC/PMC (waveguide walls) → 04-boundary-conditions.md
│  ├─ Periodic (array/FSS) → 04-boundary-conditions.md
│  └─ Symmetry (reduce domain) → 04-boundary-conditions.md
├─ Excitations & Ports
│  ├─ Waveguide port (WR-90, microstrip) → 05-excitations-ports.md
│  ├─ Discrete port (lumped feed) → 05-excitations-ports.md
│  └─ Plane wave (RCS) → 05-excitations-ports.md
├─ Parameter Sweep & Optimization
│  ├─ Parameter sweep → 06-parameter-sweep.md
│  ├─ Local/global optimization → 06-parameter-sweep.md
│  └─ DOE, sensitivity analysis → 06-parameter-sweep.md
├─ Postprocessing
│  ├─ S-parameters (Touchstone export) → 07-postprocessing.md
│  ├─ Far-field patterns → 07-postprocessing.md
│  └─ SAR, energy results → 07-postprocessing.md
└─ Cluster Execution
   ├─ SLURM/PBS job scripts → 08-cluster-execution.md
   ├─ MPI/SMP/GPU parallelism → 08-cluster-execution.md
   └─ Checkpoint/restart → 08-cluster-execution.md
```

## Reference Documents

| Document | Content |
|----------|---------|
| [references/01-project-structure.md](references/01-project-structure.md) | Project file types (.cst, .m3d, .res), model organization, VBA macro structure |
| [references/02-solver-selection.md](references/02-solver-selection.md) | Transient, Frequency Domain, Eigenmode, Integral Equation, Asymptotic solvers |
| [references/03-mesh-control.md](references/03-mesh-control.md) | Hexahedral/tetrahedral mesh, adaptive refinement, convergence criteria |
| [references/04-boundary-conditions.md](references/04-boundary-conditions.md) | Open, PEC, PMC, periodic, symmetry boundaries; PML settings |
| [references/05-excitations-ports.md](references/05-excitations-ports.md) | Waveguide, discrete, plane wave ports; S-parameter matrix |
| [references/06-parameter-sweep.md](references/06-parameter-sweep.md) | Parameter sweeps, local/global optimization, DOE, sensitivity analysis |
| [references/07-postprocessing.md](references/07-postprocessing.md) | S-parameters, field visualization, far-field, SAR, energy results |
| [references/08-cluster-execution.md](references/08-cluster-execution.md) | SLURM/PBS job scripts, MPI/SMP/GPU parallelism, checkpoint/restart |
| [references/error-recovery.md](references/error-recovery.md) | Solver convergence, mesh errors, port issues, license/cluster failures |

## Decision Guide

| Question | Reference | Key Decision |
|----------|-----------|--------------|
| Which solver for wideband antenna? | [02-solver-selection](references/02-solver-selection.md) | Transient Solver |
| Which solver for high-Q filter? | [02-solver-selection](references/02-solver-selection.md) | Frequency Domain + Eigenmode |
| Mesh too coarse or too fine? | [03-mesh-control](references/03-mesh-control.md) | Adaptive mesh refinement |
| Setting up radiation boundaries? | [04-boundary-conditions](references/04-boundary-conditions.md) | Open boundary with PML |
| Adding waveguide excitation? | [05-excitations-ports](references/05-excitations-ports.md) | Waveguide port with TE10 |
| Running parametric sweep? | [06-parameter-sweep](references/06-parameter-sweep.md) | Parameter sweep or optimizer |
| Extracting S-parameters? | [07-postprocessing](references/07-postprocessing.md) | Touchstone export |
| Running on HPC cluster? | [08-cluster-execution](references/08-cluster-execution.md) | SLURM/PBS batch script |
| Solver not converging? | [error-recovery](references/error-recovery.md) | Mesh refinement or solver change |

## Template Files

Template files in `assets/templates/` are ready-to-use starting points for common workflows:

| Template | Type | Purpose | Reference |
|----------|------|---------|-----------|
| [assets/templates/waveguide_port.cst](assets/templates/waveguide_port.cst) | VBA macro | Waveguide port setup (WR-90 example) | [05-excitations-ports.md](references/05-excitations-ports.md) |
| [assets/templates/antenna_sparameter.cst](assets/templates/antenna_sparameter.cst) | VBA macro | Microstrip patch antenna with discrete feed | [05-excitations-ports.md](references/05-excitations-ports.md), [07-postprocessing.md](references/07-postprocessing.md) |
| [assets/templates/cst_slurm.sh](assets/templates/cst_slurm.sh) | Batch script | SLURM submission for CST jobs | [08-cluster-execution.md](references/08-cluster-execution.md) |

**When to use templates:**
- Start from `waveguide_port.cst` when modeling waveguide components
- Start from `antenna_sparameter.cst` when designing patch antennas
- Use `cst_slurm.sh` as the base for any HPC cluster submission

## Guardrails

- Never use Transient solver for narrowband high-Q resonant structures — use Frequency Domain or Eigenmode
- Never ignore mesh convergence warnings — results may be inaccurate
- Never set open boundaries closer than λ/4 to the structure
- Never run large simulations without enabling checkpoint/restart
- Never use PEC boundary on surfaces that should radiate

## Output Standard

Always report at completion:

```
- Solver type and frequency range
- Mesh cells and convergence status
- Boundary conditions applied
- Key excitations and ports
- S-parameters (S11, S21) or far-field results (gain, efficiency)
- Next steps or recommendations
```
