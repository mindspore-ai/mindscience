# Force Fields

## Pair Style Overview

| Style Family | Examples | Use Case |
|-------------|----------|----------|
| Lennard-Jones | `lj/cut`, `lj/charmm/coul/long` | Simple fluids, noble gases |
| EAM/MEAM | `eam`, `meam` | Metallic systems |
| Tersoff | `tersoff` | Covalent materials (Si, C) |
| ReaxFF | `reaxff` | Reactive chemistry |
| Buckingham | `buck`, `buck/coul/long` | Ionic materials |
| Morse | `morse` | Bonds in some metals |

## Lennard-Jones Family

### Basic LJ

```lammps
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
```

Format: `pair_coeff type1 type2 epsilon sigma cutoff`

### LJ with Long-Range Dispersion (for energetics)

```lammps
pair_style lj/cut/coul/long 2.5
pair_coeff * * 1.0 1.0
kspace_style pppm 1.0e-4
```

## EAM (Embedded Atom Method)

For metallic systems:

```lammps
pair_style eam
pair_coeff * * Cu_u3.eam Cu
```

The potential file (e.g., `Cu_u3.eam`) must be in the working directory or
have its path specified.

### MEAM (Modified EAM)

```lammps
pair_style meam
pair_coeff * * library.meam Cu Cu.meam Cu
```

## Tersoff

For covalent materials:

```lammps
pair_style tersoff
pair_coeff * * SiC.tersoff Si C
```

Tersoff is for Si, C, Ge, and some compounds. The parameter file must match
the elements present.

## ReaxFF

For reactive force fields:

```lammps
pair_style reaxff
pair_coeff * * ffield.reaxff C H O N
fix 1 all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff
```

ReaxFF requires charge equilibration (QEq). It is computationally expensive.

## Buckingham (Ionic)

```lammps
pair_style buck/coul/long 2.5
pair_coeff * * 12000.0 0.15 300.0
kspace_style pppm 1.0e-4
```

## Hybrid Pair Styles

Mix different pair styles for different interactions:

```lammps
pair_style hybrid lj/cut 2.5 eam
pair_coeff * * lj/cut 1.0 1.0 2.5
pair_coeff 1 2 eam CuNi.eam
```

**Rule:** Only use hybrid when physically justified, not as a workaround.

## Bond, Angle, Dihedral Styles

For molecular systems:

### Bonds

```lammps
bond_style harmonic
bond_coeff 1 300.0 1.2

# In data file:
# bond type atom1 atom2
1 1 1 2
```

### Angles

```lammps
angle_style harmonic
angle_coeff 1 50.0 109.47
```

### Dihedrals

```lammps
dihedral_style opls
dihedral_coeff 1 0.0 1.0 3
```

### Impropers

```lammps
improper_style harmonic
improper_coeff 1 10.0 0.0
```

## KSpace (Long-Range Electrostatics)

For charged or polar systems:

```lammps
pair_style lj/cut/coul/long 2.5
pair_coeff * * 1.0 1.0
kspace_style pppm 1.0e-4
kspace_modify gewald 0.2
```

| Method | Notes |
|--------|-------|
| `pppm` | Particle-Particle Particle-Mesh Ewald |
| `ewald` | Standard Ewald (slower than PPPM) |
| `pppm/gpu` | GPU-accelerated PPPM |

## Force Field Compatibility

| Force Field Type | LAMMPS Pair Style | Notes |
|-----------------|-------------------|-------|
| Lennard-Jones | `lj/cut` | Simple, fast |
| Buckingham | `buck`, `buck/coul/long` | Ionic crystals |
| CHARMM | `charmm` | Proteins, organic |
| AMBER | `lj/cut/coul/long` | Similar to CHARMM |
| OPLS | `oplsaa` | Organic, drugs |
| EAM | `eam`, `meam` | Metals |

## Common Force Field Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Cannot open potential file" | Wrong path | Check file exists and path |
| "Incorrect args for pair coefficients" | Wrong format | Match potential file format |
| "Substitution level 0" | Typo in potential | Check filename spelling |
| "Invalid weight" | EAM file mismatch | Verify element names |
| "Dihedral type not supported" | Style mismatch | Check supported dihedral styles |
