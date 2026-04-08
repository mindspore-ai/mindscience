---
name: hpc-orca
description: ORCA quantum chemistry program. Supports DFT, TDDFT, MP2, CCSD, and spectroscopy calculations. Use for electronic structure calculations, geometry optimization, and spectral simulations.
---

# HPC-ORCA Skill

ORCA is a quantum chemistry program supporting DFT, TDDFT, frequency calculations, spectrum simulations, and advanced electronic structure methods (MP2, CCSD, CASSCF).

## Quick Start

### Typical Workflow
1. Write ORCA input file (keyword line, module settings, coordinates) — see [references/01-input-structure.md](references/01-input-structure.md)
2. Select DFT functional and dispersion correction — see [references/02-dft-methods.md](references/02-dft-methods.md)
3. Choose basis set for your system — see [references/03-basis-sets.md](references/03-basis-sets.md)
4. Set up geometry optimization or transition state search — see [references/04-optimization.md](references/04-optimization.md)
5. Run frequency calculation to validate stationary point — see [references/05-frequency.md](references/05-frequency.md)
6. For excited states / spectroscopy — see [references/06-tddft.md](references/06-tddft.md)
7. Add implicit solvent if needed — see [references/07-solvation.md](references/07-solvation.md)
8. For advanced features (ONIOM, relativistic, open-shell) — see [references/08-advanced.md](references/08-advanced.md)
9. Submit to HPC cluster — see [references/09-cluster-execution.md](references/09-cluster-execution.md)
10. Handle errors — see [references/error-recovery.md](references/error-recovery.md)

## Skill Map

```
User Requirements
├─ Input File Authoring
│  ├─ Keyword line structure (! method basis OPT FREQ) → 01-input-structure.md
│  ├─ Module settings (%maxcore, %pal, %geom) → 01-input-structure.md
│  └─ Coordinate input (xyz, z-matrix, external) → 01-input-structure.md
├─ Method Selection
│  ├─ DFT functional (B3LYP, PBE0, wB97X-D, M06-2X) → 02-dft-methods.md
│  ├─ Dispersion correction (D3BJ, D4) → 02-dft-methods.md
│  ├─ RI acceleration (RIJCOSX, RI-J) → 02-dft-methods.md
│  └─ Wavefunction methods (MP2, DLPNO-CCSD(T)) → 02-dft-methods.md
├─ Basis Set Selection
│  └─ def2, Pople, cc-pVnZ, ECP → 03-basis-sets.md
├─ Geometry Optimization
│  ├─ OPT / OPT FREQ → 04-optimization.md
│  ├─ OptTS (transition state) → 04-optimization.md
│  ├─ NEB-TS → 04-optimization.md
│  └─ Convergence criteria → 04-optimization.md
├─ Frequency & Thermochemistry
│  ├─ FREQ / FREQ RAMAN → 05-frequency.md
│  ├─ Thermodynamic corrections → 05-frequency.md
│  └─ Stationary point verification → 05-frequency.md
├─ Spectroscopy
│  ├─ TDDFT (UV-Vis) → 06-tddft.md
│  ├─ ECD (Electronic Circular Dichroism) → 06-tddft.md
│  └─ Excited state optimization → 06-tddft.md
├─ Solvent Effects
│  └─ CPCM, SMD implicit solvent → 07-solvation.md
├─ Advanced Features
│  ├─ ONIOM (QM/MM) → 08-advanced.md
│  ├─ Relativistic (ZORA, DKH2) → 08-advanced.md
│  ├─ Open-shell / broken symmetry → 08-advanced.md
│  └─ NBO, cube file generation → 08-advanced.md
└─ HPC Execution
   ├─ SLURM/PBS job scripts → 09-cluster-execution.md
   ├─ %pal nprocs (parallel) → 09-cluster-execution.md
   ├─ %maxcore (memory) → 09-cluster-execution.md
   └─ Checkpoint restart (MOREAD) → 09-cluster-execution.md
```

## Reference Documents

| Document | Content |
|----------|---------|
| [references/01-input-structure.md](references/01-input-structure.md) | Input file anatomy: keyword line, %module blocks, coordinate input formats, output files |
| [references/02-dft-methods.md](references/02-dft-methods.md) | Functional classification (LDA/GGA/meta-GGA/hybrid/range-separated/double-hybrid), dispersion, RI acceleration |
| [references/03-basis-sets.md](references/03-basis-sets.md) | def2, Pople, cc-pVnZ, aug-cc-pVnZ, ECP basis sets and selection guide |
| [references/04-optimization.md](references/04-optimization.md) | OPT, OptTS, NEB-TS, constraints, convergence criteria |
| [references/05-frequency.md](references/05-frequency.md) | FREQ, RAMAN, thermodynamic corrections, stationary point verification, isotope effects |
| [references/06-tddft.md](references/06-tddft.md) | TDDFT, UV-Vis, ECD, excited state optimization, spin-flip TDDFT |
| [references/07-solvation.md](references/07-solvation.md) | CPCM, SMD implicit solvent, solvent keywords, pKa calculation |
| [references/08-advanced.md](references/08-advanced.md) | ONIOM, ZORA, DKH2, open-shell, broken symmetry, NBO, cube files |
| [references/09-cluster-execution.md](references/09-cluster-execution.md) | SLURM/PBS scripts, %pal nprocs, %maxcore, checkpoint restart, job monitoring |
| [references/error-recovery.md](references/error-recovery.md) | SCF, geometry optimization, memory, basis set, parallel errors; recovery workflows |

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Quick geometry optimization? | [02-dft-methods.md](references/02-dft-methods.md) | `B3LYP/def2-SVP + RIJCOSX` — balance speed and precision |
| High-accuracy energy? | [02-dft-methods.md](references/02-dft-methods.md) | `DLPNO-CCSD(T)/def2-TZVP` — high precision, low computational cost |
| Large molecular system? | [03-basis-sets.md](references/03-basis-sets.md) | `def2-SVP` for pre-optimization, then `def2-TZVP` |
| Excited state calculation? | [06-tddft.md](references/06-tddft.md) | `TDDFT/wB97X-D/def2-TZVP` |
| Transition state search? | [04-optimization.md](references/04-optimization.md) | `OptTS` or `NEB-TS` |
| Spectrum simulation? | [05-frequency.md](references/05-frequency.md) + [06-tddft.md](references/06-tddft.md) | `FREQ` for IR, `TDDFT` for UV-Vis |
| Solvation effects? | [07-solvation.md](references/07-solvation.md) | `CPCM(solvent)` or `SMD(solvent)` |
| Running on HPC cluster? | [09-cluster-execution.md](references/09-cluster-execution.md) | SLURM with `%pal nprocs` |

## Guardrails

- Do not run large calculations in root directory — use dedicated run directories.
- Do not ignore SCF convergence warnings — see [references/error-recovery.md](references/error-recovery.md)
- Do not use incompatible functional/basis set combinations — consult [references/02-dft-methods.md](references/02-dft-methods.md) and [references/03-basis-sets.md](references/03-basis-sets.md)
- ORCA input is case-insensitive; multiplicity = 2S+1 (closed-shell = 1, radical = 2)
- Always run `FREQ` after `OPT` to verify stationary point (no imaginary freq = minimum, 1 imaginary = TS)

## Outputs

Always report:

- Calculation type and method (functional, basis set)
- Charge and multiplicity
- SCF convergence status
- Key output files (.out, .xyz, .gbw, .hess)
- Total energy and optimized geometry (if applicable)

## Template Files

Template files in `assets/templates/` are ready-to-use starting scaffolds that can be copied and modified:

| Template | Type | Use Case | Reference |
|----------|------|---------|-----------|
| [assets/templates/sp_dft.inp](assets/templates/sp_dft.inp) | Input (.inp) | Single point DFT energy | [01-input-structure.md](references/01-input-structure.md), [02-dft-methods.md](references/02-dft-methods.md), [03-basis-sets.md](references/03-basis-sets.md) |
| [assets/templates/opt_freq.inp](assets/templates/opt_freq.inp) | Input (.inp) | Geometry optimization + frequency | [04-optimization.md](references/04-optimization.md), [05-frequency.md](references/05-frequency.md), [07-solvation.md](references/07-solvation.md) |
| [assets/templates/tddft_uvvis.inp](assets/templates/tddft_uvvis.inp) | Input (.inp) | TDDFT UV-Vis spectrum | [06-tddft.md](references/06-tddft.md), [07-solvation.md](references/07-solvation.md) |
| [assets/templates/neb_ts.inp](assets/templates/neb_ts.inp) | Input (.inp) | NEB-TS transition state search | [04-optimization.md](references/04-optimization.md), [02-dft-methods.md](references/02-dft-methods.md) |
| [assets/templates/orca_slurm.sh](assets/templates/orca_slurm.sh) | Batch script | SLURM submission for ORCA jobs | [09-cluster-execution.md](references/09-cluster-execution.md) |

## Error Recovery

Consult [references/error-recovery.md](references/error-recovery.md) for structured diagnosis of:

- SCF convergence failures (damping, level shift, initial guess)
- Geometry optimization failures (trust radius, Hessian, optimizer choice)
- Memory allocation errors (%maxcore, %pal nprocs)
- Basis set errors (spelling, availability)
- Parallel execution issues (OpenMPI, environment)
- Checkpoint and restart from .gbw file
