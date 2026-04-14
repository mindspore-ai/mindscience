---
name: hpc-pyscf
description: PySCF Python quantum chemistry library. Supports HF, DFT, CC, CI, and multi-reference methods. Use for developing custom quantum chemistry workflows and research applications.
---

# HPC-PySCF Skill

PySCF is an open-source Python quantum chemistry software package supporting DFT, HF, and post-HF methods (MP2, CCSD, CASSCF, CASPT2, FCI, etc.).

## Quick Start

| Step | Task | Reference |
|------|------|-----------|
| 1 | Molecule definition (atoms, basis, symmetry) | [01-molecule-definition](references/01-molecule-definition.md) |
| 2 | Select calculation method (HF/DFT/MP2/CCSD) | [02-calculation-methods](references/02-calculation-methods.md) |
| 3 | Excited states (TD-DFT, CASSCF, EOM-CC) | [03-excited-states](references/03-excited-states.md) |
| 4 | Advanced features (solvation, periodic, relativistic) | [04-advanced-features](references/04-advanced-features.md) |
| - | Diagnose and fix runtime errors | [error-recovery](references/error-recovery.md) |

## Skill Map

```
User Requirements
├─ Molecule Definition       → 01-molecule-definition
├─ Energy Calculations
│  ├─ HF Energy → RHF/UHF/ROHF
│  ├─ DFT Energy → RKS/UKS
│  └─ Post-HF → MP2/CCSD/CCSD(T)/CI
├─ Structure Optimization
│  ├─ Geometry Optimization → PySCF + PyBerny
│  └─ Transition State → PySCF + ASE
├─ Electronic Structure
│  ├─ Single Reference → HF/DFT/CC
│  ├─ Multi-Reference → CASSCF/CASPT2/FCI
│  └─ Open-Shell → UHF/UKS/ROHF
├─ Spectroscopy
│  ├─ Vibrational Spectrum → Hessian
│  ├─ Electronic Spectrum → TD-DFT/EOM-CCSD
│  └─ NMR → GIAO
└─ Advanced Applications
   ├─ Solvation → PCM/COSMO
   ├─ Periodic Systems → k-point calculations
   ├─ Relativistic → X2C/DKH
   ├─ FCI Extensions → SHCI/DMRG
   └─ Custom Methods → Flexible Python API
```

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Calculation method? | `02-calculation-methods` | Based on precision requirements |
| Basis set? | `01-molecule-definition` | STO-3G/6-31G/cc-pVDZ/cc-pVTZ — balance precision and efficiency |
| Spin treatment? | `02-calculation-methods` | RHF/UHF/ROHF for closed-shell/open-shell/half-open-shell |
| Correlation method? | `02-calculation-methods` | MP2/CCSD/CCSD(T) for dynamic correlation |
| Multi-reference? | `03-excited-states` | CASSCF/CASPT2 for static correlation/excited states |
| Solvent model? | `04-advanced-features` | PCM/COSMO/SMD for solution environment |

## Guardrails

### Must Check
- [ ] Molecular coordinates are correct (units: Angstrom)
- [ ] Basis set and element are matched
- [ ] Spin multiplicity is correct
- [ ] Charge settings are correct
- [ ] Convergence criteria are reasonable

### Common Errors
1. **SCF not converged**: Use DIIS, level shift
2. **Insufficient memory**: Reduce basis set or use density fitting
3. **Symmetry error**: Disable symmetry or check structure
4. **Integration error**: Increase integration precision

## Outputs

Always report:

- Calculation method and basis set
- SCF convergence status
- Total energy and geometry (if optimized)
- Key output and data structures
- Computation time and memory usage

## Assets

**When to include**: When the skill needs files that will be used in the final output.

**Use cases**: Templates, boilerplate code, batch scripts that get copied or modified.

| File | Use Case |
|------|----------|
| `assets/templates/sp_hf.py` | HF single point energy calculation |
| `assets/templates/sp_dft.py` | DFT single point energy calculation |
| `assets/templates/mp2_ccsd.py` | Post-HF correlation calculations |
| `assets/templates/casscf.py` | CASSCF multi-reference calculation |
| `assets/templates/tddft.py` | TD-DFT excited state calculation |
| `assets/templates/pyscf_slurm.sh` | SLURM submission script |

## Reference Summary

All references are used in this skill:

| Document | Topic |
|----------|-------|
| [01-molecule-definition](references/01-molecule-definition.md) | Atoms, basis set, symmetry, charge/spin |
| [02-calculation-methods](references/02-calculation-methods.md) | HF, DFT, MP2, CCSD, CCSD(T), CI methods |
| [03-excited-states](references/03-excited-states.md) | TD-DFT, CASSCF, CASPT2, EOM-CCSD |
| [04-advanced-features](references/04-advanced-features.md) | Solvation, periodic, relativistic, FCI extensions |
| [error-recovery](references/error-recovery.md) | SCF convergence, memory, symmetry, integration errors |

## Error Recovery

Consult `references/error-recovery.md` for structured diagnosis of:

- SCF convergence failures
- Memory allocation errors
- Symmetry and integration errors
- Input and API errors

## Minimal Example

```python
from pyscf import gto, scf

mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='cc-pvdz')
mf = scf.RHF(mol)
energy = mf.kernel()
print(f"Energy: {energy} Ha")
```
