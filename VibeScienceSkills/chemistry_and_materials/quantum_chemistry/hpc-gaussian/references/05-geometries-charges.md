# Geometries, Charges, and Multiplicities

## Charge and Multiplicity

Format: `charge multiplicity`

```gjf
0 1    # Neutral closed-shell singlet
-1 1   # Anion singlet
+1 2   # Cation doublet (radical)
```

### Common Cases

| System | Charge | Multiplicity | Electrons |
|--------|--------|--------------|-----------|
| Water (closed shell) | 0 | 1 | 10 |
| Ammonia radical cation | +1 | 2 | 10 |
| OH radical | 0 | 2 | 9 |
| Carbon radical | 0 | 3 | 6 |
| Methane (closed shell) | 0 | 1 | 10 |
| Iron (III) cation | +3 | 5 | 23 (d⁵) |

### Open Shell Systems

```gjf
# For doublet (one unpaired electron):
0 2

# For triplet (two unpaired electrons):
0 3
```

### Anions

Anions can be weakly bound. Use diffuse functions:

```gjf
# aug-cc-pVDZ for anions
#p B3LYP/aug-cc-pVDZ
```

## Geometry Input

### Cartesian Coordinates

```gjf
0 1
O   0.0  0.0  0.0
H   0.96  0.0  0.0
H  -0.24  0.93  0.0
```

Units are in Angstroms by default.

### Z-Matrix

Internal coordinates (bonds, angles, dihedrals):

```gjf
0 1
C
H 1 1.09
H 1 1.09 2 109.5
H 1 1.09 2 109.5 3 120.0
H 1 1.09 2 109.5 3 -120.0
```

### Mixed Coordinates

```gjf
0 1
C          0.0  0.0  0.0
H 1 1.09   0.0  0.0  0.0
H 1 1.09 2 109.5
```

## Generating Good Starting Geometries

### From PDB

1. Extract coordinates from PDB
2. Convert to Gaussian format
3. Optimize before expensive calculations

### Fragment-Based Construction

Build molecule from fragments:

```gjf
#p B3LYP/6-31G(d,p) Geom=ModRedundant

Fragment-based build

0 1
C
N 1 1.47
O 2 1.23 1 122.0
```

## Conformational Searching

Before frequency calculation on a molecule with multiple conformers:

```gjf
#p B3LYP/6-31G(d,p) Opt=ModRedundant

Conformational search

0 1
...initial geometry...

1 2 S 60.0  # Rotate bond 1-2 by 60 degrees
1 2 S 120.0
1 2 S 180.0
```

Or use:
- CREST (conformer-rotamer ensemble sampling)
- MacroModel
- Molecular dynamics + clustering

## Transition States

TS geometries are difficult to guess. Strategies:

1. **Scan along reaction coordinate**
2. **Use known similar TS**
3. **Optimize from approximate saddle point**

```gjf
#p B3LYP/6-31G(d,p) Opt=TS
```

After TS optimization, **always verify with frequency** (one imaginary frequency).

## Handling Multiple Molecules

For multiple molecules in one calculation:

```gjf
%Chk=complex.chk
%Mem=16GB
%NProcShared=8
#p B3LYP/6-311G(d,p) SCRF=(PCM,Solvent=Water)

Complex + 2 waters

0 1
...first molecule...

-1 1
...second molecule (ion)...

0 1
...third molecule (water)...
```

Separate molecules with blank lines.

## Fragments in Gaussian

For large systems, use ONIOM (QM/MM):

```gjf
#p B3LYP/6-31G(d,p) ONIOM=EmbedCharge

ONIOM calculation

0 1
X 0.0 0.0 0.0
... geometry with atom types: H=3, C=6, O=8, N=7 ...

Link1
%OldChk=complex.chk
%Chk=oniom.chk
#p B3LYP/6-311G(d,p) Geom=AllCheck

High layer: first 50 atoms
```

## Common Geometry Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Z-matrix atom index invalid` | Z-matrix references wrong atom | Check atom indices |
| `Multiplicity violation` | Charge/multiplicity doesn't match electrons | Verify charge/multiplicity |
| `Too many variables` | Z-matrix with circular dependencies | Use Cartesian or freeze variables |
| `Optimization failed` | Bad starting geometry | Use better starting structure |
