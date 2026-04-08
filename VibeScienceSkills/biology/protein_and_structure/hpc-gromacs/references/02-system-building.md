# System Building

## The Build Sequence

```
1. gmx pdb2gmx     → Generate .top and .gro from PDB
2. gmx editconf    → Define simulation box
3. gmx solvate     → Add water molecules
4. gmx genion      → Replace water with ions (if needed)
5. gmx grompp      → Build .tpr for minimization
```

## Step 1: pdb2gmx

```bash
gmx pdb2gmx -f input.pdb -o processed.gro -p system.top
```

Interactive prompts:
1. Select force field (e.g., 15 = CHARMM36)
2. Select water model (matching force field)

**Force field must match water model.** CHARMM36 uses TIP3P, not SPC/E.

### Supported Residues

If your PDB has non-standard residues:
1. Check if the force field supports it
2. Add parameters manually to the .top/.itp
3. Or use `gmx pdb2gmx` with `-ter` flag for terminal groups

### Protonation States

Set protonation explicitly if needed:

```bash
gmx pdb2gmx -f input.pdb -o processed.gro -p system.top -his
```

Prompts for HIS protonation: HID, HIE, or HIP.

## Step 2: Define Box

```bash
gmx editconf -f processed.gro -o boxed.gro -bt cubic -d 1.0
```

| Box Type | Flag | Notes |
|----------|------|-------|
| Cubic | `cubic` | Fast, most common |
| Dodecahedron | `dodecahedron` | Better for globular proteins |
| Octahedron | `octahedron` | Sometimes used |
| Rhombic dodecahedron | `rhombic` | For membrane simulations |

`-d 1.0` = 1.0 nm distance from solute to box edge.

**Minimum distance:** Never use `-d 0`. Use at least 0.8-1.0 nm to avoid
periodic image interactions.

## Step 3: Solvate

```bash
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p system.top
```

`-cs spc216.gro` = generic 3-point water (works for TIP3P, SPC/E).

The `-p system.top` flag updates the atom count in the topology.

## Step 4: Add Ions

### Generate tpr for genion

```bash
# Create an ion MDMPS with generic settings
gmx grompp -f ions.mdp -c solvated.gro -p system.top -o ions.tpr
```

### Replace solvent with ions

```bash
gmx genion -s ions.tpr -o solvated_ions.gro -p system.top -pname NA -nname CL -neutral
```

Interactive prompt: select the solvent group (SOL).

`-neutral` adds enough ions to neutralize the system charge.

### Common Ion Concentration

For physiological salt (0.15 M):

```bash
gmx genion -s ions.tpr -o solvated_ions.gro -p system.top \
    -pname NA -nname CL -neutral \
    -conc 0.15
```

**Note:** `-conc` adds additional ions beyond neutralization.

## System Building Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| pdb2gmx fails on residue | Force field doesn't support it | Add parameters manually |
| Wrong residue names | PDB naming convention mismatch | Remap names to GROMACS convention |
| Water inside protein | Solvation before fixing | Use `-try` flag or fix structure |
| Charge not integer | System charge not neutralized | Add ions to neutralize |
| Segfault in genion | Bad tpr or box | Re-run grompp with better input |

## Membrane Protein Systems

For embedded systems in a bilayer:

```
1. Prepare membrane (.gro, .top)
2. Prepare protein
3. Use gmx insert-molecules or gmx membrane-embed
4. Solvate
5. Add ions
```

This is more complex and requires specialized tools (CHARMM-GUI, insane.py).

## Water Model Compatibility

| Force Field | Compatible Water Models |
|------------|----------------------|
| CHARMM36 | TIP3P |
| AMBER99SB-ILDN | TIP3P, SPC/E |
| OPLS-AA | TIP3P, SPC/E |
| GROMOS 54A7 | SPC water |

Using an incompatible water model produces invalid results.
