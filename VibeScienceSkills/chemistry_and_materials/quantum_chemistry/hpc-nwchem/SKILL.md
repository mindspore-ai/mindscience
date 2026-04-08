---
name: hpc-nwchem
description: NWChem ab initio computational chemistry software for quantum chemistry and molecular dynamics on HPC systems. Supports DFT, HF, MP2, CCSD(T), TDDFT, CASSCF, and periodic boundary conditions. Use for electronic structure calculations, geometry optimization, vibrational analysis, and QM/MM simulations.
---

# HPC-NWChem

NWChem is an open-source high-performance computational chemistry software package for quantum chemistry and molecular dynamics on large-scale HPC systems.

## Scientific Applications

| Application | Use Case |
|------------|----------|
| **Electronic Structure** | DFT (B3LYP, PBE, PBE0, M06-2x), HF, MP2, CCSD, CCSD(T), CASSCF |
| **Geometry Optimization** | Energy minimization, transition state search |
| **Vibrational Analysis** | Frequency calculations, IR/Raman spectra |
| **Spectroscopy** | TDDFT for electronic excitations, UV-Vis |
| **Thermochemistry** | Energy predictions, zero-point energy, thermal corrections |
| **QM/MM** | Hybrid quantum mechanics/molecular mechanics |
| **Periodic Systems** | Band structure, crystals, surfaces |

## Key Concepts

### Calculation Methods
| Method | Type | Use Case |
|--------|------|----------|
| HF | Single reference | Hartree-Fock baseline |
| DFT | Single reference | Ground state properties, efficiency |
| MP2 | Single reference | Electron correlation, non-covalent |
| CCSD(T) | Single reference | High-accuracy thermochemistry |
| CASSCF | Multi-reference | Excited states, bond breaking |
| TDDFT | Excited state | Electronic spectra |

### Basis Sets
- Pople: 6-31G*, 6-311++G**
- Dunning: cc-pVDZ, cc-pVTZ, cc-pVQZ
- Ahlrichs: def2-SVP, def2-TZVP, def2-QZVP

### Output Files
| File | Description |
|------|-------------|
| `.out` | Main output file |
| `.chk` | Checkpoint file for restarts |
| `.db` | Database file |
| `*.movecs` | Molecular orbital coefficients |

## Workflow

1. Prepare input file → [references/01-input-structure.md]
2. Select calculation method → [references/02-calculation-methods.md]
3. Configure advanced features → [references/03-advanced-features.md]
4. Set up parallel execution → [references/04-parallel-computing.md]
5. Submit to HPC cluster
6. Diagnose errors → [references/error-recovery.md]

## Input Structure

See [references/01-input-structure.md](references/01-input-structure.md) for:
- Geometry specification (xyz, z-matrix)
- Basis set definitions and libraries
- Task directives (energy, optimize, frequencies)
- Memory and disk settings

## Calculation Methods

See [references/02-calculation-methods.md](references/02-calculation-methods.md) for:
- SCF convergence and options
- DFT functionals (LDA, GGA, meta-GGA, hybrid)
- MP2 and CCSD(T) correlation
- TDDFT for excited states
- Frequency calculations

## Advanced Features

See [references/03-advanced-features.md](references/03-advanced-features.md) for:
- Open-shell systems (UHF/UKS)
- Solvent models (COSMO, SMD)
- Periodic boundary conditions (BAND)
- QM/MM coupling

## Parallel Computing

See [references/04-parallel-computing.md](references/04-parallel-computing.md) for:
- MPI parallel execution
- Memory configuration
- SLURM/PBS/LSF submission scripts
- Scalability considerations

## Error Recovery

See [references/error-recovery.md](references/error-recovery.md) for diagnosis of:
- SCF convergence failures
- Memory allocation errors
- Geometry optimization failure
- Basis set errors
- Parallel execution issues

## Templates

Template files in [assets/templates/](assets/templates/) serve as starting points:

| Template | Purpose |
|----------|---------|
| [`sp_dft.nw`](assets/templates/sp_dft.nw) | DFT single point energy with B3LYP/6-31G* |
| [`opt_freq.nw`](assets/templates/opt_freq.nw) | Geometry optimization + frequency analysis |
| [`mp2.nw`](assets/templates/mp2.nw) | MP2 correlation energy calculation |
| [`tddft.nw`](assets/templates/tddft.nw) | TDDFT excited states (10 roots) |
| [`nwchem_slurm.sh`](assets/templates/nwchem_slurm.sh) | SLURM job submission script |

## Skill Decision Map

```
User Requirements
├─ Energy Calculations
│  ├─ HF Energy → SCF module
│  ├─ DFT Energy → DFT with functional selection
│  └─ Post-HF → MP2 or CCSD(T)
├─ Structure Optimization
│  ├─ Geometry → DRIVER directive
│  └─ Frequencies → FREQUENCIES task
├─ Electronic Structure
│  ├─ Ground state → HF/DFT
│  ├─ Excited state → TDDFT
│  └─ Multi-reference → CASSCF
├─ Spectroscopy
│  ├─ Vibrational → FREQUENCIES
│  └─ Electronic → TDDFT
├─ Periodic Systems
│  └─ Band structure → BAND module
└─ QM/MM
   └─ Hybrid → QMMM module
```

## Guardrails

### Must Verify
- [ ] Basis set is appropriate for the element
- [ ] Memory settings are adequate
- [ ] Task directive matches desired calculation
- [ ] SCF convergence criteria are appropriate

### Never Do
- Do not use small basis sets for final energies
- Do not ignore SCF convergence warnings
- Do not set unrealistic convergence criteria
- Do not mix basis set libraries carelessly

## Required Output

Always report:
- Calculation method and basis set
- Memory and parallel configuration
- SCF convergence status
- Total energy
- Optimized geometry (if applicable)
