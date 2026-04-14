---
name: hpc-gromacs
description: GROMACS molecular dynamics simulation package for proteins, lipids, polymers, and biomolecular systems. Supports system building, energy minimization, equilibration (NVT/NPT), production MD, and trajectory analysis. Use for MD workflow automation, .mdp parameter lookup, and HPC cluster execution.
---

# HPC GROMACS

GROMACS is a molecular dynamics (MD) package for simulating proteins, lipids, polymers, and biomolecular systems at scale. It implements staged workflows: system build → energy minimization → equilibration → production MD → analysis.

## Scientific Applications

| Application | Use Case |
|------------|----------|
| **Protein MD** | Folding, binding, conformational dynamics |
| **Lipid Bilayers** | Membrane simulations, drug permeability |
| **Polymer Simulation** | Polymer physics, rheology |
| **Free Energy** | Alchemical transformations, binding affinities |
| **Enhanced Sampling** | Umbrella sampling, metadynamics |
| **Analysis** | RMSD, RMSF, RDF, hydrogen bonds, SASA |

## Key Concepts

### Force Fields
- AMBER (protein, nucleic acids)
- CHARMM (proteins, lipids, carbohydrates)
- OPLS-AA (proteins)
- MARTINI (coarse-grained)

### Water Models
- SPC, SPC/E, TIP3P, TIP4P, TIP5P

### Ensembles
- NVT (canonical)
- NPT (isothermal-isobaric)
- NPH (isoenthalpic)

### Output Files
| File | Description |
|------|-------------|
| `.tpr` | Portable binary run input |
| `.trr` | Full precision trajectory |
| `.xtc` | Compressed trajectory |
| `.edr` | Energy file |
| `.gro`, `.pdb` | Coordinates |

## Workflow

```
1. Build system → [references/02-system-building.md]
2. Energy minimization → [references/03-minimization.md]
3. Equilibration (NVT → NPT) → [references/04-equilibration.md]
4. Production MD → [references/05-production-md.md]
5. Trajectory analysis → [references/06-analysis.md]
6. Parameter lookup → [references/07-mdp-reference.md]
7. Cluster submission → [references/08-cluster-execution.md]
8. Error diagnosis → [references/error-recovery.md]
```

See [references/01-workflow-overview.md](references/01-workflow-overview.md) for full pipeline anatomy.

## System Building

See [references/02-system-building.md](references/02-system-building.md) for:
- pdb2gmx structure processing
- Solvation and ion addition (genbox, genion)
- Force field and water model selection

## Energy Minimization

See [references/03-minimization.md](references/03-minimization.md) for:
- Steepest descent and conjugate gradient
- Convergence criteria (emtol, Fmax)
- Troubleshooting warnings

## Equilibration

See [references/04-equilibration.md](references/04-equilibration.md) for:
- NVT equilibration (temperature coupling)
- NPT equilibration (pressure coupling)
- Thermostats: v-rescale, Nose-Hoover, berendsen
- Barostats: Parrinello-Rahman, Berendsen

## Production MD

See [references/05-production-md.md](references/05-production-md.md) for:
- Timestep selection (2 fs with LINCS)
- PME electrostatics
- Trajectory output controls
- GPU and MPI+OpenMP parallelization

## Analysis

See [references/06-analysis.md](references/06-analysis.md) for:
- RMSD, RMSF (structural stability)
- Radial distribution function (RDF)
- Hydrogen bond analysis
- Principal component analysis (PCA)
- Free energy calculations (grompp, wham)

## MDP Reference

See [references/07-mdp-reference.md](references/07-mdp-reference.md) for complete .mdp parameter descriptions organized by category.

## Cluster Execution

See [references/08-cluster-execution.md](references/08-cluster-execution.md) for:
- SLURM, PBS, LSF submission scripts
- MPI+OpenMP hybrid parallelization
- GPU acceleration
- Multi-node scaling

## Error Recovery

See [references/error-recovery.md](references/error-recovery.md) for diagnosis of:
- grompp validation failures
- mdrun crashes andsegfaults
- Pressure/temperature instability
- LINCS warnings

## Templates

Template files in [assets/templates/](assets/templates/) serve as starting points:

| Template | Purpose |
|----------|---------|
| [`em.mdp`](assets/templates/em.mdp) | Energy minimization (steepest descent) |
| [`nvt.mdp`](assets/templates/nvt.mdp) | NVT equilibration with V-rescale thermostat |
| [`npt.mdp`](assets/templates/npt.mdp) | NPT equilibration with Parrinello-Rahman barostat |
| [`md_prod.mdp`](assets/templates/md_prod.mdp) | Production MD with PME electrostatics |
| [`gromacs-mdrun-slurm.sh`](assets/templates/gromacs-mdrun-slurm.sh) | SLURM job submission script |

## Skill Decision Map

```
User Requirements
├─ System Preparation
│  ├─ Structure cleaning → pdb2gmx
│  ├─ Solvation → genbox
│  └─ Ions → genion
├─ Energy Minimization
│  ├─ Steepest descent → emtol, emstep
│  └─ Conjugate gradient → cg integrator
├─ Equilibration
│  ├─ NVT → tcoupl, tau-t, ref-t
│  └─ NPT → pcoupl, tau-p, ref-p
├─ Production MD
│  ├─ Small molecules → 2 fs timestep, LINCS
│  ├─ Constraints → h-bonds, all-bonds
│  └─ Electrostatics → PME
└─ Analysis
   ├─ Structural → RMSD, RMSF, radius of gyration
   ├─ Thermodynamic → energy, pressure, density
   └─ Dynamic → diffusion, autocorrelation
```

## Guardrails

### Must Verify
- [ ] Force field and water model are compatible
- [ ] .mdp parameters match intended ensemble
- [ ] grompp validation passes before mdrun
- [ ] Timestep is appropriate for constraints

### Never Do
- Do not skip equilibration phases
- Do not mix force fields within a system
- Do not ignore grompp warnings
- Do not use cutoffs incompatible with the force field

## Required Output

Always report:
- Force field and water model used
- .mdp stage and ensemble
- Generated run artifacts (.tpr, trajectory files)
- Analysis results requested
- Convergence status (EM, equilibration)
