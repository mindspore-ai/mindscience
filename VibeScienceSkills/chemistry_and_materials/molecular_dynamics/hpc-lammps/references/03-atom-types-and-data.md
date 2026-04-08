# Atom Types and Data Files

## Atom Style

Choose based on what physics you need:

| Style | Attributes | Use When |
|-------|-----------|----------|
| `atomic` | x, v | Simple LJ, neutrals |
| `charge` | x, v, q | Electrostatics |
| `bond` | x, v, q, bonds | Molecules with bonds |
| `angle` | + angle | Molecules with angles |
| `full` | + dihedrals, impropers | Biomolecules, organic |
| `ellipsoid` | + shape, quaternion | Anisotropic particles |
| `body` | + shape, inertia, quaternian | Flexible bodies |

```lammps
atom_style charge    # For charged systems
atom_style full      # For biomolecules (AMBER-style)
```

## Data File Format

LAMMPS data files have sections:

```data
# Title line
LAMMPS Description

# Atoms
4 atoms
2 atom types

# Bonds (if atom_style includes bonds)
0 bonds
0 bond types

# Angles
0 angles
0 angle types

# Dihedrals
0 dihedrals
0 dihedral types

# Impropers
0 impropers
0 improper types

# Extra bond-per-atom, angle-per-atom, etc.
1 extra bond per atom
0 extra angle per atom
0 extra dihedral per atom

# Pair coefficients (if not in pair_coeff)
1 pair coeffs
1 pair types

# Bond coefficients (if needed)
0 bond coeffs

# Angle coefficients (if needed)
0 angle coeffs

# Masses
Masses

1 63.55  # Cu

# Atom class labels (optional)
```
## Pair Coeff vs Data File Coeff

LAMMPS can read pair coefficients from the data file OR set them in the input:

```lammps
# In input script:
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5

# OR in data file under "Pair Coeffs":
# pair_coeff
# 1 1.0 1.0 2.5
```

**Rule:** If defined in both places, the input script wins.

## Box and Coordinates

```
# Box bounds (required)
xlo xhi
0.0 100.0

ylo yhi
0.0 100.0

zlo zhi
0.0 100.0

# Atoms
# id type x y z
1 1 5.0 5.0 5.0
2 1 10.0 5.0 5.0
...
```

## Generating Data Files

### Convert from PDB

Use `topotools` (VMD plugin) or `moltemplate`:

```bash
# In VMD:
package require topotools
topo readlammpsdata system.data full
```

### Use moltemplate

```bash
moltemplate.sh -atomstyle full system.lt
```

### Packmol + Topotools

```bash
packmol < system.inp
vmd system.pdb
# Use topotools to convert to LAMMPS data
```

## Atom Type Mapping

The atom types in the data file must match `pair_coeff` assignments:

```lammps
# Data file: atom type 1 = Cu, type 2 = Ni

pair_style eam
pair_coeff 1 2 CuNi.eam    # type 1 and 2 use this file
pair_coeff 1 1 Cu.eam      # type 1 uses Cu
pair_coeff 2 2 Ni.eam      # type 2 uses Ni
```

**Rule:** Check that atom type numbers in data match pair_coeff references.

## Molecule Data

For molecular systems with bonds:

```data
# In Atoms section:
# id mol charge type x y z

1 1 0.0 1 5.0 5.0 5.0
2 1 0.0 1 5.5 5.0 5.0
...

# Bonds section:
# id type atom1 atom2
1 1 1 2
```

The molecule ID (mol) groups atoms into molecules for fixes that need
molecular topology.

## Verifying Data Files

```bash
# Use LAMMPS to read and verify
lmp -in verify_input.in
```

In the input:

```lammps
read_data system.data
write_data system_check.data
```

Compare `system_check.data` to the original.

## Common Data File Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Invalid atom type" | Type number > declared types | Fix type count or numbering |
| "Bond atom missing" | Bond references non-existent atom | Check bond section |
| "Coords don't match box" | Atom outside box | Remap coordinates |
| "No angles but style needs them" | atom_style mismatch | Change atom_style |
| "Incorrect # of atoms" | Count mismatch | Verify counts match |
