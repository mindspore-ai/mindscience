---
name: hpc-gaussian
description: Build, review, debug, and automate Gaussian quantum chemistry workflows. Use when working with Gaussian input files, Link 0 directives, route sections, charge and multiplicity, basis sets, SCF or optimization or frequency jobs, checkpoint handoff, or Gaussian execution and restart issues.
---

# HPC Gaussian Skill

Treat Gaussian as a staged quantum-chemistry workflow built around a valid input deck and explicit checkpoint policy.

## Quick Start

### Typical Workflow
1. Write input file (Link0, route, title, charge/mult, geometry) — see [references/01-input-file-structure.md](references/01-input-file-structure.md)
2. Select method and basis set — see [references/02-method-selection.md](references/02-method-selection.md)
3. Choose job type (SP, Opt, Freq, Scan, IRC) — see [references/03-job-types.md](references/03-job-types.md)
4. Handle SCF convergence if needed — see [references/04-scf-convergence.md](references/04-scf-convergence.md)
5. Set up geometry, charge, and multiplicity — see [references/05-geometries-charges.md](references/05-geometries-charges.md)
6. Configure checkpoint/restart for multi-step workflows — see [references/06-checkpoint-restart.md](references/06-checkpoint-restart.md)
7. Add solvent model if needed — see [references/07-solvent-models.md](references/07-solvent-models.md)
8. Submit to HPC cluster — see [references/08-cluster-execution.md](references/08-cluster-execution.md)
9. Handle errors — see [references/error-recovery.md](references/error-recovery.md)

## Skill Map

```
User Requirements
├─ Input Deck Authoring
│  ├─ Link0 directives (%Chk, %Mem, %NProcShared) → 01-input-file-structure.md
│  ├─ Route section (method/basis/job keyword) → 01-input-file-structure.md
│  ├─ Charge, multiplicity, geometry format → 05-geometries-charges.md
│  └─ Method selection (HF, DFT, MP2, CCSD(T)) → 02-method-selection.md
├─ Job Types
│  ├─ Single Point (SP) → 03-job-types.md
│  ├─ Geometry Optimization (Opt) → 03-job-types.md
│  ├─ Frequency (Freq) → 03-job-types.md
│  ├─ Scan / IRC / Transition State → 03-job-types.md
│  └─ Excited States (TD-DFT) → 03-job-types.md
├─ SCF & Convergence
│  ├─ SCF convergence strategies → 04-scf-convergence.md
│  └─ Guess methods (Core, TCore, Mix) → 04-scf-convergence.md
├─ Solvent Effects
│  └─ PCM, SMD, CPCM models → 07-solvent-models.md
├─ Checkpoint & Restart
│  ├─ %Chk, %OldChk, Guess=Read → 06-checkpoint-restart.md
│  ├─ Formatted checkpoint (fChk) → 06-checkpoint-restart.md
│  └─ Multi-step Link1 workflows → 06-checkpoint-restart.md
└─ HPC Cluster Execution
   ├─ SLURM/PBS job scripts → 08-cluster-execution.md
   ├─ Shared memory (NProcShared) → 08-cluster-execution.md
   └─ Scratch directory management → 08-cluster-execution.md
```

## Reference Documents

| Document | Content |
|----------|---------|
| [references/01-input-file-structure.md](references/01-input-file-structure.md) | Input file anatomy: Link0 directives, route section, title, charge/mult, Cartesian/Z-matrix geometry |
| [references/02-method-selection.md](references/02-method-selection.md) | HF, DFT functionals (B3LYP, ωB97X-D, M06-2X), basis sets (6-311G, cc-pVDZ), MP2, CCSD(T) |
| [references/03-job-types.md](references/03-job-types.md) | SP, Opt, Freq, Scan, IRC, TD-DFT, composite methods (CBS-QB3), job sequencing |
| [references/04-scf-convergence.md](references/04-scf-convergence.md) | SCF guess methods, DIIS, level-shifting, stability checks, functional recommendations |
| [references/05-geometries-charges.md](references/05-geometries-charges.md) | Charge/mult format, Cartesian/Z-matrix, conformer search, TS starting geometries |
| [references/06-checkpoint-restart.md](references/06-checkpoint-restart.md) | %Chk/%OldChk, Guess=Read, Link1 multi-step, formchk, cube files |
| [references/07-solvent-models.md](references/07-solvent-models.md) | PCM, SMD, CPCM; solvent selection; non-equilibrium solvation |
| [references/08-cluster-execution.md](references/08-cluster-execution.md) | SLURM/PBS scripts, %NProcShared, %Mem, scratch directory, Linda multi-node |
| [references/error-recovery.md](references/error-recovery.md) | Input, SCF, optimization, frequency, checkpoint, runtime errors; recovery workflow |

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Single point, opt, or freq? | [03-job-types.md](references/03-job-types.md) | SP, Opt, Freq, or combined job types |
| Which method and basis? | [02-method-selection.md](references/02-method-selection.md) | HF, DFT, MP2, CCSD(T) with appropriate basis |
| SCF convergence issues? | [04-scf-convergence.md](references/04-scf-convergence.md) | SCF=QC, Guess=TCore, or damping strategies |
| Solvent effects needed? | [07-solvent-models.md](references/07-solvent-models.md) | SCRF=PCM or SCRF=SMD |
| Checkpoint or restart? | [06-checkpoint-restart.md](references/06-checkpoint-restart.md) | %Chk, %OldChk, or Guess=Read |

## Guardrails

- Do not invent Gaussian keywords or Link 0 directives — consult [01-input-file-structure.md](references/01-input-file-structure.md)
- Do not treat charge and multiplicity as cosmetic metadata — they must match the electron count.
- Do not run restart-sensitive workflows without an explicit checkpoint policy — see [06-checkpoint-restart.md](references/06-checkpoint-restart.md)
- Do not copy route sections from unrelated systems without checking method, basis, and job intent.
- Do not run Freq on a non-optimized geometry.

## Outputs

Always report:

- job type (SP, Opt, Freq, etc.)
- method and basis selection
- charge and multiplicity
- checkpoint or restart policy
- expected key outputs (.log, .chk, .fchk) and next stage

## Template Files

Template files in `assets/templates/` are ready-to-use starting scaffolds that can be copied and modified:

| Template | Type | Use Case | Reference |
|----------|------|---------|-----------|
| [assets/templates/sp_water.gjf](assets/templates/sp_water.gjf) | Input (.gjf) | Single point energy of water | [01-input-file-structure.md](references/01-input-file-structure.md), [02-method-selection.md](references/02-method-selection.md) |
| [assets/templates/opt_freq_water.gjf](assets/templates/opt_freq_water.gjf) | Input (.gjf) | Geometry optimization + frequency | [03-job-types.md](references/03-job-types.md), [06-checkpoint-restart.md](references/06-checkpoint-restart.md) |
| [assets/templates/gaussian-g16-slurm.sh](assets/templates/gaussian-g16-slurm.sh) | Batch script | SLURM submission for Gaussian 16 | [08-cluster-execution.md](references/08-cluster-execution.md) |

## Error Recovery

Consult [references/error-recovery.md](references/error-recovery.md) for structured diagnosis of:

- Input file errors (Link0, route, charge/mult)
- SCF convergence failures
- Optimization failures
- Frequency calculation errors
- Restart and checkpoint issues
- Runtime errors (memory, scratch, segmentation)
