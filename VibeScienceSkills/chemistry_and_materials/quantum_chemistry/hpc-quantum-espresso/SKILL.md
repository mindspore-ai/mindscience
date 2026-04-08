---
name: hpc-quantum-espresso
description: Build, review, debug, and automate Quantum ESPRESSO workflows. Use when working with `pw.x` input files, pseudopotentials, k-point setup, SCF/relax/bands workflows, plane-wave cutoffs, occupations, or Quantum ESPRESSO execution and convergence issues.
---

# HPC Quantum ESPRESSO Skill

Treat Quantum ESPRESSO as a staged first-principles workflow centered on namelist-based input files.

## Quick Start

### Typical Workflow
1. Select workflow stage (SCF → relax → NSCF/bands) — see [references/01-pwx-workflow.md](references/01-pwx-workflow.md)
2. Write `pw.x` input with correct namelists, atomic species, pseudopotentials, and k-points — see [references/02-input-and-parameter-matrix.md](references/02-input-and-parameter-matrix.md)
3. Choose pseudopotential family and k-point strategy — see [references/04-pseudopotential-kpoints-and-occupations.md](references/04-pseudopotential-kpoints-and-occupations.md)
4. Set `ecutwfc`/`ecutrho` and SCF mixing parameters — see [references/06-cutoff-and-mixing-matrix.md](references/06-cutoff-and-mixing-matrix.md)
5. Configure `conv_thr`, `ion_dynamics`, and convergence thresholds — see [references/07-convergence-thresholds.md](references/07-convergence-thresholds.md)
6. For bands or DOS post-processing — see [references/05-restarts-and-postprocessing.md](references/05-restarts-and-postprocessing.md)
7. Submit to HPC cluster with proper `prefix`/`outdir` discipline — see [references/08-cluster-execution.md](references/08-cluster-execution.md)
8. Handle errors — see [references/error-recovery.md](references/error-recovery.md)

## Skill Map

```
User Requirements
├─ Input File Authoring
│  ├─ Workflow stages (SCF → relax → NSCF → bands) → 01-pwx-workflow.md
│  ├─ Namelist structure (&CONTROL, &SYSTEM, &ELECTRONS, &IONS, &CELL) → 02-input-and-parameter-matrix.md
│  ├─ ATOMIC_SPECIES, ATOMIC_POSITIONS, K_POINTS cards → 02-input-and-parameter-matrix.md
│  └─ Coordinate and cell format (crystal, angstrom, ibrav) → 02-input-and-parameter-matrix.md
├─ Pseudopotentials & K-Points
│  ├─ Pseudo family coherence and file paths → 04-pseudopotential-kpoints-and-occupations.md
│  ├─ K-point modes (automatic, gamma, crystal_b) → 04-pseudopotential-kpoints-and-occupations.md
│  └─ Occupations and smearing (fixed, smearing, degauss) → 04-pseudopotential-kpoints-and-occupations.md
├─ Cutoffs & Mixing
│  ├─ ecutwfc and ecutrho selection → 06-cutoff-and-mixing-matrix.md
│  ├─ Mixing strategy (mixing_beta, mixing_mode) → 06-cutoff-and-mixing-matrix.md
│  └─ First-response troubleshooting → 06-cutoff-and-mixing-matrix.md
├─ Convergence & Dynamics
│  ├─ SCF convergence (conv_thr) → 07-convergence-thresholds.md
│  ├─ Relaxation (relax, vc-relax) → 07-convergence-thresholds.md
│  ├─ Ionic/cell dynamics (ion_dynamics, cell_dynamics) → 07-convergence-thresholds.md
│  └─ Stage selection (SCF/relax/bands workflow) → 03-scf-relax-bands-and-convergence.md
├─ Restarts & Post-Processing
│  ├─ Clean restarts (prefix, outdir discipline) → 05-restarts-and-postprocessing.md
│  ├─ NSCF and bands handoff → 05-restarts-and-postprocessing.md
│  └─ DOS and post-processing workflow → 05-restarts-and-postprocessing.md
└─ HPC Cluster Execution
   ├─ SLURM/PBS job scripts → 08-cluster-execution.md
   ├─ Stage-aware execution and artifact handoff → 08-cluster-execution.md
   ├─ srun pw.x launch strategy → 08-cluster-execution.md
   └─ Restart and continuation → 08-cluster-execution.md
```

## Reference Documents

| Document | Content |
|----------|---------|
| [references/01-pwx-workflow.md](references/01-pwx-workflow.md) | Core workflow stages (SCF → relax → NSCF → bands), input file structure, execution pattern |
| [references/02-input-and-parameter-matrix.md](references/02-input-and-parameter-matrix.md) | Namelist responsibility matrix, ATOMIC_SPECIES, ATOMIC_POSITIONS, K_POINTS, CELL_PARAMETERS |
| [references/03-scf-relax-bands-and-convergence.md](references/03-scf-relax-bands-and-convergence.md) | Stage-selection map, SCF convergence, relaxation workflow, bands workflow |
| [references/04-pseudopotential-kpoints-and-occupations.md](references/04-pseudopotential-kpoints-and-occupations.md) | Pseudo family coherence, k-point modes, occupations/smearing, metallic vs insulating logic |
| [references/05-restarts-and-postprocessing.md](references/05-restarts-and-postprocessing.md) | Clean restart rules, prefix/outdir discipline, NSCF/bands handoff, post-processing workflow |
| [references/06-cutoff-and-mixing-matrix.md](references/06-cutoff-and-mixing-matrix.md) | ecutwfc/ecutrho, mixing parameters (mixing_beta, mixing_mode), first-response troubleshooting |
| [references/07-convergence-thresholds.md](references/07-convergence-thresholds.md) | conv_thr, ion_dynamics, cell_dynamics, relaxation controls, numerical stability |
| [references/08-cluster-execution.md](references/08-cluster-execution.md) | SLURM/PBS scripts, preflight checks, stage-aware execution, srun pw.x, restart/continuation |
| [references/error-recovery.md](references/error-recovery.md) | Input parsing failures, SCF convergence, pseudopotential/species mismatches, downstream failures |

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Which calculation type? | [03-scf-relax-bands-and-convergence.md](references/03-scf-relax-bands-and-convergence.md) | SCF (ground state), relax (optimize coords), vc-relax (optimize cell), bands (band structure) |
| What namelists are needed? | [02-input-and-parameter-matrix.md](references/02-input-and-parameter-matrix.md) | &CONTROL, &SYSTEM, &ELECTRONS, &IONS, &CELL by concern |
| Which k-point mode? | [04-pseudopotential-kpoints-and-occupations.md](references/04-pseudopotential-kpoints-and-occupations.md) | automatic (uniform grid), gamma (Gamma-only), crystal_b (band path) |
| Fixed occupation or smearing? | [04-pseudopotential-kpoints-and-occupations.md](references/04-pseudopotential-kpoints-and-occupations.md) | fixed (insulators), smearing (metals) |
| What cutoffs to use? | [06-cutoff-and-mixing-matrix.md](references/06-cutoff-and-mixing-matrix.md) | ecutwfc (wavefunction), ecutrho (charge density, typically 4× ecutwfc) |
| conv_thr value? | [07-convergence-thresholds.md](references/07-convergence-thresholds.md) | 1.0d-6 to 1.0d-8 for production; 1.0d-5 for quick checks |
| How to handle SCF oscillation? | [06-cutoff-and-mixing-matrix.md](references/06-cutoff-and-mixing-matrix.md) | Reduce mixing_beta to 0.1–0.3, check metallic behavior |
| Running on HPC cluster? | [08-cluster-execution.md](references/08-cluster-execution.md) | srun pw.x, preflight checks, prefix/outdir discipline |

## Guardrails

- Do not invent namelist keys from other DFT codes — consult [references/02-input-and-parameter-matrix.md](references/02-input-and-parameter-matrix.md)
- Do not mix pseudopotential families without verifying cutoff compatibility
- Do not run bands or DOS before a converged SCF ground state is established — see [01-pwx-workflow.md](references/01-pwx-workflow.md) and [05-restarts-and-postprocessing.md](references/05-restarts-and-postprocessing.md)
- Do not change `prefix` or `outdir` between related stages — see [05-restarts-and-postprocessing.md](references/05-restarts-and-postprocessing.md)
- Do not use a bands-style k-point path as a uniform SCF sampling grid — see [04-pseudopotential-kpoints-and-occupations.md](references/04-pseudopotential-kpoints-and-occupations.md)
- Do not tighten `conv_thr` by more than one order at a time when debugging — see [07-convergence-thresholds.md](references/07-convergence-thresholds.md)

## Outputs

Summarize:

- workflow stage (SCF, relax, vc-relax, bands, NSCF, DOS)
- pseudopotential family and species setup
- k-point strategy and density
- cutoffs (`ecutwfc`, `ecutrho`) and convergence controls
- key outputs (`.save/` directory, wavefunction, charge density)
- `prefix` and `outdir` paths

## Template Files

Template files in `assets/templates/` are ready-to-use starting scaffolds that can be copied and modified:

| Template | Type | Use Case | Reference |
|----------|------|---------|-----------|
| [assets/templates/scf_si.in](assets/templates/scf_si.in) | pw.x input | SCF ground-state calculation (Si) | [01-pwx-workflow.md](references/01-pwx-workflow.md), [02-input-and-parameter-matrix.md](references/02-input-and-parameter-matrix.md), [04-pseudopotential-kpoints-and-occupations.md](references/04-pseudopotential-kpoints-and-occupations.md) |
| [assets/templates/relax_si.in](assets/templates/relax_si.in) | pw.x input | Geometry relaxation (Si) | [03-scf-relax-bands-and-convergence.md](references/03-scf-relax-bands-and-convergence.md), [07-convergence-thresholds.md](references/07-convergence-thresholds.md) |
| [assets/templates/bands_si.in](assets/templates/bands_si.in) | pw.x input | Band structure calculation (Si) | [05-restarts-and-postprocessing.md](references/05-restarts-and-postprocessing.md), [04-pseudopotential-kpoints-and-occupations.md](references/04-pseudopotential-kpoints-and-occupations.md) |
| [assets/templates/qe-pwx-slurm.sh](assets/templates/qe-pwx-slurm.sh) | Batch script | SLURM submission for pw.x jobs | [08-cluster-execution.md](references/08-cluster-execution.md) |

## Error Recovery

Consult [references/error-recovery.md](references/error-recovery.md) for structured diagnosis of:

- **Input parsing failures** — namelist syntax, section order, species labels
- **SCF convergence failures** — occupations/smearing, k-points, cutoffs, mixing parameters
- **Pseudopotential and species mismatches** — pseudo path, label coherence, cutoff incompatibility
- **Downstream workflow failures** — relax, vc-relax, bands stages; prefix/outdir drift
