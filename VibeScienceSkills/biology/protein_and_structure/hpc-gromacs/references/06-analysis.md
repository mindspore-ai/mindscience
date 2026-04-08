# Analysis

## Analysis Philosophy

Analysis comes **after** the simulation, using the trajectory files.

**Always** process trajectory before analysis:
```bash
gmx trjconv -f md.xtc -s md.tpr -o md_pbc.xtc -pbc mol -ur compact
```

## Essential Pre-Processing

### Center and PBC

```bash
# Center protein in box, wrap water
gmx trjconv -f md.xtc -s md.tpr -o md_centered.xtc -center yes -pbc mol -ur compact
```

Select `1` (Protein) for centering group, `0` (Whole system) for output.

### Fitting (for RMSD)

```bash
# Fit to reference (usually first frame or minimized structure)
gmx trjconv -f md_centered.xtc -s md.tpr -o md_fit.xtc -fit rot+trans
```

Select `1` (Backbone) for fitting, `0` (Whole system) for output.

## Common Analyses

### Energy Analysis

```bash
# Extract energies
gmx energy -f md.edr -o energy.xvg

# Select desired energy terms
# Common: Potential, Kinetic, Temperature, Pressure, Density, Bond, Angle, Proper-Dih., LJ-14, Coulomb-14, LJ (SR), Coulomb (SR)
```

### RMSD (Root Mean Square Deviation)

```bash
# Backbone RMSD vs time
gmx rms -s md.tpr -f md_fit.xtc -o rmsd.xvg -tu ns

# Select: 4 (Backbone) for both fitting and output
```

Interpretation:
- < 0.1-0.2 nm: Stable, well-folded
- > 0.3-0.5 nm: Significant conformational change
- Monotonic increase: Unfolding or denaturation

### RMSF (Root Mean Square Fluctuation)

```bash
gmx rmsf -s md.tpr -f md_fit.xtc -o rmsf.xvg -res

# Select: 3 (CA) for output
```

Interpretation:
- High RMSF regions = flexible loops, disordered regions
- Low RMSF regions = structured, stable cores

### Radius of Gyration

```bash
gmx gyrate -s md.tpr -f md_fit.xtc -o gyrate.xvg

# Select: 1 (Protein)
```

Interpretation:
- Stable Rg: Well-folded protein
- Increasing Rg: Unfolding or expansion
- Decreasing Rg: Compaction

### Hydrogen Bonds

```bash
gmx hbond -f md_fit.xtc -s md.tpr -num hbond_num.xvg -g hbond.log
```

### SASA (Solvent Accessible Surface Area)

```bash
gmx sasa -s md.tpr -f md_fit.xtc -o sasa.xvg -tu ns

# Select: 1 (Protein)
```

### RMSD per Residue (Distance Matrix)

```bash
gmx rmsdist -s md.tpr -f md_fit.xtc -o rmsdist.xvg
```

### Ramachandran Plot

For backbone dihedrals:

```bash
gmx rama -s md.tpr -f md_fit.xtc -o rama.xvg
```

## Trajectory Manipulation

### Extract Frames

```bash
# Extract frames at 1 ns intervals
gmx trjconv -f md.xtc -s md.tpr -o frame_0ns.pdb -dump 0
gmx trjconv -f md.xtc -s md.tpr -o frame_100ns.pdb -dump 100000

# Extract every 10th frame
gmx trjconv -f md.xtc -s md.tpr -o frames_10ns.xtc -dt 10000
```

### Split Trajectory by Residue

```bash
# Split by chain or domain
gmx trj_select -f md.xtc -s md.tpr -on select.ndx -select "resname CHAIN_A"
```

### Cluster Analysis

```bash
gmx cluster -s md.tpr -f md_fit.xtc -o clusters.xpm -g clusters.log -dist clusters_dist.xvg

# RMSD clustering with 0.2 nm cutoff
gmx cluster -s md.tpr -f md_fit.xtc -cl clusters.pdb -cutoff 0.2 -method gromos
```

## Free Energy Analysis

### PME Interaction Energy

```bash
gmx energy -f md.edr -o pair_energy.xvg
# Select: LJ-14, Coulomb-14, LJ (SR), Coulomb (SR)
```

### MM-PBSA Binding Free Energy (post-processing)

Requires separate scripts (gmx_MMPBSA, APBS):

```
gmx_MMPBSA -f md.xtc -s md.tpr -cs md.gro -ci index.ndx -cg 1 13
```

## Common Analysis Pitfalls

| Pitfall | Problem | Fix |
|---------|---------|-----|
| Using unprocessed trajectory | "Planets", broken molecules | Always use `gmx trjconv -pbc mol` |
| Wrong reference structure | RMSD off by whole system motion | Fit before RMSD |
| Not discarding equilibration | Averages skewed by startup | Discard first ~10-20% of trajectory |
| Wrong group selected | Garbage output | Verify group selection |
| Different trajectories | Inconsistent analysis | Use same processed trajectory |
