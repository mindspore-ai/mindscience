---
name: hpc-cp2k
description: CP2K quantum chemistry and molecular dynamics software. Supports DFT, MD, AIMD, and various electronic structure methods. Use for periodic systems, ab initio molecular dynamics, and hybrid DFT calculations.
---

# HPC-CP2K Skill

CP2K is an open-source quantum chemistry and molecular dynamics software package, supporting DFT, MD, AIMD, and various electronic structure methods.

## Quick Start

### Typical Workflow
1. Prepare molecular structure file (.xyz or .pdb)
2. Write CP2K input file (.inp) — see [references/01-input-structure.md](references/01-input-structure.md)
3. Select appropriate DFT functional and basis set — see [references/02-dft-methods.md](references/02-dft-methods.md)
4. Choose calculation type (energy, optimization, or MD) — see [references/03-molecular-dynamics.md](references/03-molecular-dynamics.md) or [references/04-geometry-optimization.md](references/04-geometry-optimization.md)
5. Submit job to HPC cluster — see [references/05-cluster-execution.md](references/05-cluster-execution.md)
6. Analyze output results and handle errors — see [references/error-recovery.md](references/error-recovery.md)

### Minimal Example
```bash
# Single Point Energy
cp2k.psmp -i energy.inp -o energy.out
```

## Skill Map

```
User Requirements
├─ Energy Calculations
│  ├─ Single Point Energy → ENERGY/FORCE_EVAL
│  └─ Energy Decomposition → ENERGY_DECOMPOSITION
├─ Structure Optimization
│  ├─ Geometry Optimization → GEO_OPT (see references/04-geometry-optimization.md)
│  └─ Transition State Search → BAND/NEB
├─ Molecular Dynamics
│  ├─ AIMD → MD (Born-Oppenheimer) (see references/03-molecular-dynamics.md)
│  ├─ CPMD → MD (Car-Parrinello) (see references/03-molecular-dynamics.md)
│  └─ Classical MD → MD (Classical force fields)
├─ Electronic Structure
│  ├─ DFT → GPW/GAPW (see references/02-dft-methods.md)
│  ├─ Semi-empirical → XTB/PM3/AM1
│  └─ Post-HF → RI-MP2/RPA/Double-hybrid
└─ Spectroscopic Properties
   ├─ Vibrational Spectrum → VIBRATIONAL_ANALYSIS
   └─ Electronic Spectrum → TDDFT/RT-TDDFT
```

## Reference Documents

| Document | Purpose |
|----------|---------|
| [references/01-input-structure.md](references/01-input-structure.md) | Input file syntax, GLOBAL/FORCE_EVAL/MOTION sections |
| [references/02-dft-methods.md](references/02-dft-methods.md) | Functional selection, basis sets, cutoff energy, k-points |
| [references/03-molecular-dynamics.md](references/03-molecular-dynamics.md) | AIMD, CPMD, ensembles, thermostats, barostats |
| [references/04-geometry-optimization.md](references/04-geometry-optimization.md) | Optimizers, convergence criteria, TS search |
| [references/05-cluster-execution.md](references/05-cluster-execution.md) | SLURM/PBS scripts, parallel strategies, checkpoint restart |
| [references/error-recovery.md](references/error-recovery.md) | SCF convergence, memory, geometry, parallel errors |

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Calculation type? | `ENERGY/GEO_OPT/MD` | Based on research target |
| DFT functional? | `PBE/B3LYP/PBE0` | PBE for solids, B3LYP for molecules |
| Basis set? | `DZVP/TZVP/QZVP` | Balance precision and efficiency |
| Cutoff energy? | `200-600 Ry` | 300 Ry is default value |
| k-point mesh? | `Gamma/Monkhorst-Pack` | Gamma for molecules, k-points for periodic |
| Parallel strategy? | `MPI/OpenMP/GPU` | Based on system scale |

## Guardrails

### Must Check
- [ ] Input file syntax is correct (&END tags must be paired)
- [ ] Coordinate units are correct (default: Angstrom)
- [ ] Pseudopotential and basis set are matched
- [ ] Cutoff energy is sufficient (perform convergence test)
- [ ] k-point settings are reasonable (for periodic systems)

### Common Errors
1. **SCF not converged**: Increase MAX_SCF, use SMEAR method
2. **Insufficient memory**: Reduce cutoff energy or use sparse matrix
3. **Geometry optimization failed**: Check initial structure, adjust step size
4. **Low parallel efficiency**: Adjust process/thread ratio

## Outputs

Always report:

- Calculation type and method (DFT functional, basis set)
- Cutoff energy and k-point settings
- SCF convergence status
- Key output files (.out, .ener, .xyz)
- Performance metrics (wall time, parallel efficiency)

## Template Files

Template files in `assets/templates/` are ready-to-use input files and job scripts:

| Template | Description | Reference |
|----------|-------------|-----------|
| [assets/templates/sp_dft.inp](assets/templates/sp_dft.inp) | DFT single point energy calculation | [01-input-structure.md](references/01-input-structure.md), [02-dft-methods.md](references/02-dft-methods.md) |
| [assets/templates/geo_opt.inp](assets/templates/geo_opt.inp) | Geometry optimization | [04-geometry-optimization.md](references/04-geometry-optimization.md) |
| [assets/templates/aimd.inp](assets/templates/aimd.inp) | Ab initio molecular dynamics | [03-molecular-dynamics.md](references/03-molecular-dynamics.md) |
| [assets/templates/cpmd.inp](assets/templates/cpmd.inp) | Car-Parrinello molecular dynamics | [03-molecular-dynamics.md](references/03-molecular-dynamics.md) |
| [assets/templates/cp2k_slurm.sh](assets/templates/cp2k_slurm.sh) | SLURM submission script | [05-cluster-execution.md](references/05-cluster-execution.md) |

## Error Recovery

Consult [references/error-recovery.md](references/error-recovery.md) for structured diagnosis of:

- SCF convergence failures
- Memory allocation errors
- Geometry optimization failures
- Parallel execution issues
- Input file parsing errors
