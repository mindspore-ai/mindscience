---
name: hpc-vasp
description: Build, review, debug, and automate VASP first-principles workflows. Use when working with VASP input sets such as INCAR, POSCAR, KPOINTS, and POTCAR; when choosing SCF, relaxation, static, DOS, or band-structure stages; or when fixing convergence, symmetry, cutoff, and k-point issues.
---

# HPC VASP Skill

Treat VASP as a staged workflow built around a coherent four-file input set.

## Quick Start

### Typical Workflow
1. Acquire or prepare structure (CIF, POSCAR) — see [references/10-structure-preparation.md](references/10-structure-preparation.md)
2. Generate four-file input set (INCAR, POSCAR, KPOINTS, POTCAR) — see [references/01-input-set-manual.md](references/01-input-set-manual.md)
3. Select workflow stage (relax, static, DOS, bands) — see [references/02-stage-and-parameter-matrix.md](references/02-stage-and-parameter-matrix.md)
4. Configure INCAR tags for the stage — see [references/07-incar-tag-matrix.md](references/07-incar-tag-matrix.md)
5. Set pseudopotential, k-points, and convergence controls — see [references/03-pseudopotential-kpoints-and-convergence.md](references/03-pseudopotential-kpoints-and-convergence.md)
6. For DOS or band-structure workflows — see [references/06-dos-and-band-workflows.md](references/06-dos-and-band-workflows.md)
7. Configure restart and handoff between stages — see [references/09-restarts-spin-and-wavefunction-files.md](references/09-restarts-spin-and-wavefunction-files.md) and [references/12-workflow-handoff-matrix.md](references/12-workflow-handoff-matrix.md)
8. Submit to HPC cluster — see [references/04-cluster-execution-playbook.md](references/04-cluster-execution-playbook.md)
9. Handle errors — see [references/05-error-recovery.md](references/05-error-recovery.md) and [references/error-pattern-dictionary.md](references/error-pattern-dictionary.md)

## Skill Map

```
User Requirements
├─ Input Set Authoring
│  ├─ Four-file model (INCAR, POSCAR, KPOINTS, POTCAR) → 01-input-set-manual.md
│  ├─ Species order discipline, coordinate modes → 08-poscar-species-and-structure.md
│  └─ INCAR tag selection per stage → 07-incar-tag-matrix.md
├─ Workflow Stage Selection
│  ├─ Stage map (relax, static, DOS, bands) → 02-stage-and-parameter-matrix.md
│  ├─ INCAR controls per stage (IBRION, NSW, ISMEAR, SIGMA) → 07-incar-tag-matrix.md
│  └─ DOS / band-structure workflows → 06-dos-and-band-workflows.md
├─ Convergence & Cutoffs
│  ├─ POTCAR coherence and ENCUT selection → 03-pseudopotential-kpoints-and-convergence.md
│  ├─ K-point strategy (Gamma, Monkhorst-Pack, line-mode) → 03-pseudopotential-kpoints-and-convergence.md
│  ├─ Metallic vs insulating smearing → 03-pseudopotential-kpoints-and-convergence.md
│  └─ Material-specific parameters (DFT+U, magnetism) → 11-material-specific-parameters.md
├─ Structure Preparation
│  ├─ Download from Materials Project → 10-structure-preparation.md
│  ├─ Build with ASE (bulk, surface, molecule) → 10-structure-preparation.md
│  ├─ Format conversion (CIF → POSCAR) → 10-structure-preparation.md
│  └─ Generate VASP inputs with qvasp → 10-structure-preparation.md
├─ Restart & Handoff
│  ├─ ISTART/ICHARG logic, CONTCAR → POSCAR handoff → 09-restarts-spin-and-wavefunction-files.md
│  └─ Stage artifact and restart-file matrix → 12-workflow-handoff-matrix.md
└─ HPC Cluster Execution
   ├─ SLURM/PBS job scripts → 04-cluster-execution-playbook.md
   ├─ Stage-aware execution and artifact discipline → 04-cluster-execution-playbook.md
   └─ Restart and continuation → 04-cluster-execution-playbook.md
```

## Reference Documents

| Document | Content |
|----------|---------|
| [references/01-input-set-manual.md](references/01-input-set-manual.md) | Four-file model (INCAR, POSCAR, KPOINTS, POTCAR), file responsibility matrix, stage-to-directory workflow |
| [references/02-stage-and-parameter-matrix.md](references/02-stage-and-parameter-matrix.md) | Stage map (relax, static, DOS, bands), INCAR matrix, KPOINTS strategy |
| [references/03-pseudopotential-kpoints-and-convergence.md](references/03-pseudopotential-kpoints-and-convergence.md) | POTCAR coherence, metallic vs insulating logic, convergence troubleshooting |
| [references/04-cluster-execution-playbook.md](references/04-cluster-execution-playbook.md) | SLURM/PBS scripts, preflight checks, stage-aware execution, srun vasp_std, restart/continuation |
| [references/05-error-recovery.md](references/05-error-recovery.md) | Input-set mismatches, SCF failures, ionic relaxation failures, workflow handoff failures |
| [references/06-dos-and-band-workflows.md](references/06-dos-and-band-workflows.md) | DOS workflow, band-structure workflow, k-point mode differences, stage ordering |
| [references/07-incar-tag-matrix.md](references/07-incar-tag-matrix.md) | Stage-to-INCAR matrix, electronic-class-to-smearing (ISMEAR/SIGMA), relaxation controls (IBRION, NSW, ISIF) |
| [references/08-poscar-species-and-structure.md](references/08-poscar-species-and-structure.md) | POSCAR structure, species order discipline, coordinate modes (direct/Cartesian), ISIF implications |
| [references/09-restarts-spin-and-wavefunction-files.md](references/09-restarts-spin-and-wavefunction-files.md) | WAVECAR/CHGCAR/CONTCAR, ISTART/ICHARG logic, ISPIN/MAGMOM, practical continuation rules |
| [references/10-structure-preparation.md](references/10-structure-preparation.md) | Materials Project download, ASE structure building, format conversion, qvasp toolkit |
| [references/11-material-specific-parameters.md](references/11-material-specific-parameters.md) | Magnetic materials (ISPIN/MAGMOM), DFT+U (LDAUU/LDAUL), metals (ISMEAR/SIGMA), KSPACING |
| [references/12-workflow-handoff-matrix.md](references/12-workflow-handoff-matrix.md) | Stage-artifact matrix, restart-file matrix, KPOINTS handoff matrix |
| [references/error-pattern-dictionary.md](references/error-pattern-dictionary.md) | Structured failure signatures: VASP_SPECIES_ORDER_MISMATCH, VASP_METAL_SMEARING_MISMATCH, VASP_RELAXATION_TOO_AGGRESSIVE, etc. |

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Which workflow stage? | [02-stage-and-parameter-matrix.md](references/02-stage-and-parameter-matrix.md) | ionic relaxation (IBRION=2), static SCF (IBRION=-1, NSW=0), DOS, bands |
| What INCAR tags for this stage? | [07-incar-tag-matrix.md](references/07-incar-tag-matrix.md) | Per-stage INCAR tag lookup |
| Metal or insulator? | [03-pseudopotential-kpoints-and-convergence.md](references/03-pseudopotential-kpoints-and-convergence.md) | ISMEAR=0 (insulator), ISMEAR=1-2 (metal) |
| ENCUT value? | [10-structure-preparation.md](references/10-structure-preparation.md) | 1.2–1.3× max ENMAX in POTCAR |
| K-point mode? | [03-pseudopotential-kpoints-and-convergence.md](references/03-pseudopotential-kpoints-and-convergence.md) | Gamma (supercells), Monkhorst-Pack (bulk), line-mode (bands) |
| Material magnetic? | [11-material-specific-parameters.md](references/11-material-specific-parameters.md) | ISPIN=2, MAGMOM initial values per element |
| DFT+U needed? | [11-material-specific-parameters.md](references/11-material-specific-parameters.md) | For transition metal oxides, LDA+U with appropriate U values |
| Running on HPC cluster? | [04-cluster-execution-playbook.md](references/04-cluster-execution-playbook.md) | srun vasp_std, preflight checks, stage directory discipline |

## Guardrails

- Do not invent INCAR tags from other DFT codes — consult [references/07-incar-tag-matrix.md](references/07-incar-tag-matrix.md)
- Do not set `ENCUT` below 1.2× maximum `ENMAX` in POTCAR — insufficient cutoff causes convergence instability
- Do not use `ISPIN=1` for potentially magnetic materials without checking spin polarization
- Do not leave `ISMEAR=1` (metal smearing) for an insulator or semiconductor — see [references/07-incar-tag-matrix.md](references/07-incar-tag-matrix.md)
- Do not run DOS or bands before geometry and ground-state SCF are trustworthy — see [references/06-dos-and-band-workflows.md](references/06-dos-and-band-workflows.md)
- Do not change `prefix` or `outdir` between related stages (QE analog; VASP equivalent: CONTCAR, CHGCAR, WAVECAR discipline) — see [references/09-restarts-spin-and-wavefunction-files.md](references/09-restarts-spin-and-wavefunction-files.md) and [references/12-workflow-handoff-matrix.md](references/12-workflow-handoff-matrix.md)

## Outputs

Summarize:

- workflow stage (relax, static, DOS, bands)
- INCAR intent and key tags
- KPOINTS strategy and mode
- POTCAR and species assumptions
- expected key outputs (CONTCAR, OUTCAR, DOSCAR, etc.)
- next stage handoff plan

## Template Files

Template files in `assets/templates/` are ready-to-use starting scaffolds that can be copied and modified:

| Template | Type | Use Case | Reference |
|----------|------|---------|-----------|
| [assets/templates/INCAR_relax](assets/templates/INCAR_relax) | INCAR | Ionic relaxation (IBRION=2, NSW=100, ISIF=3) | [07-incar-tag-matrix.md](references/07-incar-tag-matrix.md), [02-stage-and-parameter-matrix.md](references/02-stage-and-parameter-matrix.md) |
| [assets/templates/INCAR_static](assets/templates/INCAR_static) | INCAR | Static SCF (IBRION=-1, NSW=0, ISMEAR=-5) | [07-incar-tag-matrix.md](references/07-incar-tag-matrix.md), [06-dos-and-band-workflows.md](references/06-dos-and-band-workflows.md) |
| [assets/templates/INCAR_bands](assets/templates/INCAR_bands) | INCAR | Band structure (ICHARG=11, line-mode KPOINTS follow-on) | [06-dos-and-band-workflows.md](references/06-dos-and-band-workflows.md), [09-restarts-spin-and-wavefunction-files.md](references/09-restarts-spin-and-wavefunction-files.md) |
| [assets/templates/KPOINTS_gamma](assets/templates/KPOINTS_gamma) | KPOINTS | Gamma-only mesh for large supercells | [03-pseudopotential-kpoints-and-convergence.md](references/03-pseudopotential-kpoints-and-convergence.md) |
| [assets/templates/KPOINTS_mp_6x6x6](assets/templates/KPOINTS_mp_6x6x6) | KPOINTS | Monkhorst-Pack 6×6×6 mesh | [03-pseudopotential-kpoints-and-convergence.md](references/03-pseudopotential-kpoints-and-convergence.md) |
| [assets/templates/POSCAR_si](assets/templates/POSCAR_si) | POSCAR | Si diamond structure (conventional cell) | [08-poscar-species-and-structure.md](references/08-poscar-species-and-structure.md) |
| [assets/templates/vasp-standard-slurm.sh](assets/templates/vasp-standard-slurm.sh) | Batch script | SLURM submission for VASP jobs | [04-cluster-execution-playbook.md](references/04-cluster-execution-playbook.md) |

## Utility Scripts

Scripts in `scripts/` assist with structure acquisition and VASP input generation:

| Script | Use Case | Reference |
|--------|---------|-----------|
| [scripts/download_from_mp.py](scripts/download_from_mp.py) | Download CIF from Materials Project by material ID or formula | [10-structure-preparation.md](references/10-structure-preparation.md) |
| [scripts/build_structure_ase.py](scripts/build_structure_ase.py) | Build bulk crystals, surfaces, molecules with ASE | [10-structure-preparation.md](references/10-structure-preparation.md) |
| [scripts/convert_structure.py](scripts/convert_structure.py) | Convert structure files (CIF, XYZ, POSCAR, etc.) using pymatgen | [10-structure-preparation.md](references/10-structure-preparation.md) |
| [scripts/read_with_ase.py](scripts/read_with_ase.py) | Read and convert structure files using ASE | [10-structure-preparation.md](references/10-structure-preparation.md) |
| [scripts/generate_vasp_inputs.sh](scripts/generate_vasp_inputs.sh) | Generate INCAR, KPOINTS from existing POSCAR/POTCAR using qvasp | [10-structure-preparation.md](references/10-structure-preparation.md) |

## Error Recovery

Consult these documents for structured diagnosis:

- [references/05-error-recovery.md](references/05-error-recovery.md) — Input-set mismatches, SCF failures, ionic relaxation failures, workflow handoff failures
- [references/error-pattern-dictionary.md](references/error-pattern-dictionary.md) — Structured failure signatures: VASP_SPECIES_ORDER_MISMATCH, VASP_STAGE_TAG_MISMATCH, VASP_METAL_SMEARING_MISMATCH, VASP_CUTOFF_OR_KMESH_TOO_WEAK, VASP_RELAXATION_TOO_AGGRESSIVE, VASP_WRONG_RELAXATION_SCOPE, VASP_DOS_WITH_WRONG_KPOINT_MODE, VASP_BANDS_WITH_WRONG_HANDOFF
