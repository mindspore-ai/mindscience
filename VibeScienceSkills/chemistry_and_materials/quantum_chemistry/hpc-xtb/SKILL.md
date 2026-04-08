---
name: hpc-xtb
description: XTB extended tight-binding semi-empirical methods. Fast quantum chemistry with GFN-xTB methods. Use for rapid geometry optimization, conformational search, and high-throughput screening.
---

# HPC-XTB Skill

XTB (Extended Tight Binding) is a semi-empirical quantum chemistry method developed by Grimme, providing fast and reliable molecular structure and energy calculations.

## Quick Start

| Step | Task | Reference |
|------|------|-----------|
| 1 | Choose method (GFN2-xTB/GFN1-xTB/GFN-FF) | [01-methods](references/01-methods.md) |
| 2 | Select calculation type (SP/Opt/Hess/MD) | [02-calculation-types](references/02-calculation-types.md) |
| 3 | Configure solvation model (GBSA) | [02-solvation](references/02-solvation.md) |
| 4 | Conformational search with CREST | [03-crest](references/03-crest.md) |
| 5 | Conformer generation and screening | [03-crest-conformer](references/03-crest-conformer.md) |
| - | Diagnose and fix runtime errors | [error-recovery](references/error-recovery.md) |

## Skill Map

```
User Requirements
├─ Energy Calculations
│  ├─ Single Point Energy → --sp
│  └─ Energy Decomposition → --pop
├─ Structure Optimization
│  ├─ Geometry Optimization → --opt
│  └─ Transition State Search → --gfnff-ts
├─ Molecular Dynamics
│  ├─ MD Simulation → --md
│  └─ Metadynamics → --metadyn
├─ Spectroscopy
│  ├─ Vibrational Spectrum → --hess
│  └─ Electronic Spectrum → --vipea
└─ Advanced Applications
   ├─ Solvation → --gbsa
   ├─ Conformational Search → --crest
   └─ ONIOM → Combine with other software
```

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Method? | `01-methods` | GFN2-xTB (default), GFN1-xTB, GFN-FF |
| Charge? | `--chrg` | Molecular charge |
| Spin? | `--uhf` | Open-shell systems |
| Solvent? | `02-solvation` | GBSA for solution environment |
| Parallel? | `-P` | Multi-core parallel |
| Precision? | `--etol/--gtol` | Convergence criteria |

## Guardrails

### Must Check
- [ ] Molecular coordinates are correct (units: Angstrom)
- [ ] Charge settings are correct
- [ ] Spin settings are correct (for open-shell systems)
- [ ] Solvent model selection is reasonable
- [ ] Output files exist after calculation

### Common Errors
1. **SCF not converged**: Increase iteration count
2. **Unreasonable structure**: Check initial coordinates
3. **Insufficient memory**: Reduce parallel processes
4. **Solvent error**: Check solvent name

## Outputs

Always report:

- Calculation type and method (GFN2-xTB/GFN1-xTB)
- Charge and spin settings
- SCF convergence status
- Key output files (.out, .xyz, .hess)
- Total energy and optimized geometry (if applicable)

## Assets

**When to include**: When the skill needs files that will be used in the final output.

**Use cases**: Templates, boilerplate code, batch scripts that get copied or modified.

| File | Use Case |
|------|----------|
| `assets/templates/sp_xtb.sh` | Single point energy calculation |
| `assets/templates/opt_xtb.sh` | Geometry optimization |
| `assets/templates/hess_xtb.sh` | Frequency calculation |
| `assets/templates/md_xtb.sh` | MD simulation |
| `assets/templates/crest_slurm.sh` | CREST conformational search |

## Reference Summary

All references are used in this skill:

| Document | Topic |
|----------|-------|
| [01-methods](references/01-methods.md) | GFN2-xTB, GFN1-xTB, GFN-FF methods |
| [02-calculation-types](references/02-calculation-types.md) | SP, Opt, Hess, MD calculation types |
| [02-solvation](references/02-solvation.md) | GBSA solvation model |
| [03-crest](references/03-crest.md) | CREST conformational search |
| [03-crest-conformer](references/03-crest-conformer.md) | Conformer generation and screening |
| [error-recovery](references/error-recovery.md) | SCF convergence, structure errors, memory, solvent |

## Error Recovery

Consult `references/error-recovery.md` for structured diagnosis of:

- SCF convergence failures
- Structure and coordinate errors
- Memory allocation errors
- Solvent model issues

## Minimal Example

```bash
# Single point energy calculation
xtb molecule.xyz --sp

# Geometry optimization
xtb molecule.xyz --opt

# Frequency calculation
xtb molecule.xyz --hess
```
