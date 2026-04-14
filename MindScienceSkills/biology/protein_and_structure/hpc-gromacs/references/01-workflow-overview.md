# MD Simulation Pipeline

## The GROMACS Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 0: System Preparation                                │
│  pdb2gmx → editconf → solvate → genion → grompp           │
├─────────────────────────────────────────────────────────────┤
│  STAGE 1: Energy Minimization (EM)                          │
│  gmx grompp → gmx mdrun (10-500 ps equivalent)             │
├─────────────────────────────────────────────────────────────┤
│  STAGE 2: Equilibration (NVT)                               │
│  gmx grompp → gmx mdrun (100-500 ps)                       │
├─────────────────────────────────────────────────────────────┤
│  STAGE 3: Equilibration (NPT)                              │
│  gmx grompp → gmx mdrun (100-500 ps)                       │
├─────────────────────────────────────────────────────────────┤
│  STAGE 4: Production MD                                     │
│  gmx grompp → gmx mdrun (ns to μs scale)                   │
├─────────────────────────────────────────────────────────────┤
│  STAGE 5: Analysis                                          │
│  gmx energy, gmx rms, gmx trjconv, etc.                    │
└─────────────────────────────────────────────────────────────┘
```

## Never Skip Stages

Each stage exists for a reason:

- **EM**: Remove steric clashes and find a valid starting volume
- **NVT**: Stabilize temperature at the target value
- **NPT**: Stabilize pressure (and thus density) before production
- **Production**: Data collection with validated system

Skipping stages leads to unstable production runs or invalid results.

## Command Reference

| Command | Purpose |
|---------|---------|
| `gmx pdb2gmx` | Convert PDB to GROMACS topology |
| `gmx editconf` | Define box, center molecule |
| `gmx solvate` | Add solvent molecules |
| `gmx grompp` | Preprocess: .mdp + .top + .gro → .tpr |
| `gmx mdrun` | Execute the simulation |
| `gmx energy` | Extract energies from .edr |
| `gmx trjconv` | Convert/filter trajectory |
| `gmx rms` | RMSD/RMSF calculation |
| `gmx rdf` | Radial distribution function |

## File Formats

| Format | Role | Produced By |
|--------|------|-------------|
| `.pdb` | Input structure | Experiment or builder |
| `.gro` | Coordinates + box | editconf, solvate, grompp |
| `.top` | Topology | pdb2gmx |
| `.itp` | Included topologies | pdb2gmx |
| `.mdp` | Run parameters | User |
| `.tpr` | Portable run input | grompp |
| `.trr` | Full precision trajectory | mdrun |
| `.xtc` | Compressed trajectory | mdrun |
| `.edr` | Energy file | mdrun |
| `.log` | Run log | mdrun |
| `.cpt` | Checkpoint (restart) | mdrun |

## Stage Naming Convention

Keep each stage in its own directory with a clear prefix:

```
system/
├── 00_input/           # Original PDB
├── 01_em/              # Minimization
├── 02_nvt/             # NVT equilibration
├── 03_npt/             # NPT equilibration
├── 04_prod/            # Production MD
└── 05_analysis/        # Analysis scripts
```

## Checking Stage Outputs

Always verify after each stage:

```bash
# After EM: check energy, should be negative and declining
gmx energy -f em.edr -o em_energy.xvg

# After NVT/NPT: check temperature/pressure is stable
gmx energy -f nvt.edr -o temperature.xvg

# After production: check trajectory quality
gmx check -f md.trr
```

## When Something Goes Wrong

| Stage Failed | Likely Cause | Fix |
|-------------|--------------|-----|
| EM | Bad starting structure, clashes | Increase IM steps, change minimizer |
| NVT | Temperature not converging | Extend time, check velocities |
| NPT | Pressure oscillations | Tune tau-p, use slower coupling |
| Production | Exploding system | Return to NPT, check for bad contacts |

## Timestep Guidance

| Constraints | Max Timestep | Notes |
|------------|-------------|-------|
| H-bonds constrained (all) | 2 fs | Standard production |
| H-bonds constrained (bonds to H) | 2 fs | Default LINCS |
| No constraints | 0.5-1 fs | For flexible water |
| Water constrained | 2 fs | Standard |

Never use dt > 2 fs with standard MD setup.
