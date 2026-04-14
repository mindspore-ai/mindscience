---
name: hpc-psi4
description: Psi4 open-source quantum chemistry software. Python-driven, supports HF, DFT, MP2, CCSD, and many-body methods. Use for high-accuracy electronic structure calculations.
---

# HPC-PSI4 Skill

Psi4 is an open-source quantum chemistry software package driven by Python, focused on high-precision electronic structure calculations, supporting DFT, MP2, CCSD, CCSD(T), and various post-HF methods.

## Quick Start

### Typical Workflow
1. Install Psi4: `conda install psi4` or `pip install psi4`
2. Write Python input script (memory, molecule, options, energy call) — see [references/01-input-structure.md](references/01-input-structure.md)
3. Select calculation method (SCF, DFT, post-HF) — see [references/02-calculation-methods.md](references/02-calculation-methods.md)
4. Choose basis set — see [references/03-basis-sets.md](references/03-basis-sets.md)
5. For advanced features (symmetry, PCM, open-shell, CASSCF) — see [references/03-advanced-features.md](references/03-advanced-features.md)
6. Run and monitor calculation
7. Analyze output results
8. Handle errors — see [references/error-recovery.md](references/error-recovery.md)

## Skill Map

```
User Requirements
├─ Input File Authoring
│  ├─ Python API (psi4.set_memory, psi4.geometry) → 01-input-structure.md
│  ├─ Molecule definition (charge, multiplicity, symmetry) → 01-input-structure.md
│  └─ Options (basis, scf_type, e_convergence) → 01-input-structure.md
├─ Calculation Methods
│  ├─ SCF (RHF/UHF/ROHF) → 02-calculation-methods.md
│  ├─ DFT (B3LYP, PBE0, M06-2X, wB97X-D) → 02-calculation-methods.md
│  ├─ Post-HF (MP2, CCSD, CCSD(T)) → 02-calculation-methods.md
│  └─ Excited states (TD-DFT) → 02-calculation-methods.md
├─ Geometry & Frequencies
│  ├─ optimize() → 02-calculation-methods.md
│  └─ frequency() → 02-calculation-methods.md
├─ Basis Set Selection
│  └─ Pople, Dunning, Ahlrichs (def2) → 03-basis-sets.md
├─ Advanced Features
│  ├─ Symmetry (c1, cs, c2v, d2h) → 03-advanced-features.md
│  ├─ Solvation (PCM) → 03-advanced-features.md
│  ├─ Open-shell (UHF, UKS) → 03-advanced-features.md
│  ├─ Multi-reference (CASSCF) → 03-advanced-features.md
│  └─ MPI parallel (psi4 -n 8) → 03-advanced-features.md
└─ Error Recovery
   ├─ SCF convergence → error-recovery.md
   ├─ Memory allocation → error-recovery.md
   └─ Basis set / geometry errors → error-recovery.md
```

## Reference Documents

| Document | Content |
|----------|---------|
| [references/01-input-structure.md](references/01-input-structure.md) | Python API, molecule definition format, charge/multiplicity/symmetry, common options, SCF types |
| [references/02-calculation-methods.md](references/02-calculation-methods.md) | RHF/UHF/ROHF, DFT functionals, MP2/CCSD/CCSD(T), optimize(), frequency(), TD-DFT |
| [references/03-basis-sets.md](references/03-basis-sets.md) | Pople (6-31G*), Dunning (cc-pVnZ, aug-cc-pVnZ), Ahlrichs (def2), mixed basis sets |
| [references/03-advanced-features.md](references/03-advanced-features.md) | Symmetry, PCM solvation, open-shell, CASSCF, MPI parallel, output control |
| [references/error-recovery.md](references/error-recovery.md) | SCF convergence (DIIS, level shift), memory, basis set, geometry optimization, CCSD errors |

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Calculation method? | [02-calculation-methods.md](references/02-calculation-methods.md) | SCF/DFT/MP2/CCSD based on precision needs |
| Basis set? | [03-basis-sets.md](references/03-basis-sets.md) | `cc-pVDZ` for routine, `def2-tzvp` for balanced, `aug-cc-pVTZ` for anions |
| Reference state? | [02-calculation-methods.md](references/02-calculation-methods.md) | RHF (closed-shell), UHF/ROHF (open-shell) |
| Correlation method? | [02-calculation-methods.md](references/02-calculation-methods.md) | MP2 (medium), CCSD/CCSD(T) (high accuracy) |
| Memory allocation? | [01-input-structure.md](references/01-input-structure.md) | `psi4.set_memory('16 GB')` based on system size |
| Parallel strategy? | [03-advanced-features.md](references/03-advanced-features.md) | `psi4 -n 8` for MPI multi-core |

## Guardrails

- Do not invent method names or options — consult [references/01-input-structure.md](references/01-input-structure.md) and [references/02-calculation-methods.md](references/02-calculation-methods.md)
- Molecular coordinates must be correct (units: Angstrom) and basis set must match elements
- Spin multiplicity must be correct (2S+1; closed-shell = 1, radical = 2)
- Always set `psi4.set_memory()` before the molecule definition
- For high-accuracy energies, use density fitting (`scf_type: df`) to reduce memory

## Outputs

Always report:

- Calculation method and basis set
- SCF convergence status
- Total energy and geometry (if optimized)
- Key output files (.dat, .grad)
- Computation time and resource usage

## Template Files

Template files in `assets/templates/` are ready-to-use starting scaffolds that can be copied and modified:

| Template | Type | Use Case | Reference |
|----------|------|---------|-----------|
| [assets/templates/sp_scf.py](assets/templates/sp_scf.py) | Python script | Hartree-Fock single point energy | [01-input-structure.md](references/01-input-structure.md), [02-calculation-methods.md](references/02-calculation-methods.md) |
| [assets/templates/sp_dft.py](assets/templates/sp_dft.py) | Python script | DFT single point energy (B3LYP/PBE0/M06-2X) | [02-calculation-methods.md](references/02-calculation-methods.md), [03-basis-sets.md](references/03-basis-sets.md) |
| [assets/templates/mp2_ccsd.py](assets/templates/mp2_ccsd.py) | Python script | MP2, CCSD, CCSD(T) correlation calculations | [02-calculation-methods.md](references/02-calculation-methods.md), [03-advanced-features.md](references/03-advanced-features.md) |
| [assets/templates/opt_freq.py](assets/templates/opt_freq.py) | Python script | Geometry optimization + frequency | [02-calculation-methods.md](references/02-calculation-methods.md), [03-advanced-features.md](references/03-advanced-features.md) |
| [assets/templates/psi4_slurm.sh](assets/templates/psi4_slurm.sh) | Batch script | SLURM submission for Psi4 jobs (8-core MPI) | [03-advanced-features.md](references/03-advanced-features.md) |

## Error Recovery

Consult [references/error-recovery.md](references/error-recovery.md) for structured diagnosis of:

- **SCF convergence failures** — DIIS, level shift, different initial guess
- **Memory allocation errors** — increase `psi4.set_memory()`, use `scf_type: df`
- **Basis set errors** — check spelling, use supported basis sets
- **Geometry optimization failures** — relax `g_convergence`, change optimizer
- **CCSD convergence failures** — increase `cc_maxiter`, adjust `r_convergence`
