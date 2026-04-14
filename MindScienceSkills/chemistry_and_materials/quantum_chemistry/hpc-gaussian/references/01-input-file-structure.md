# Input File Structure

## Input File Anatomy

A Gaussian input file has five parts:

```
%Link0 directives         ← Runtime and file directives
#route section            ← Method, basis, job type
                           ← (blank line)
title                     ← Short description
                           ← (blank line)
charge multiplicity       ← e.g., 0 1
geometry (Z-matrix or Cartesian)
                           ← (blank line, optional additional sections)
```

## Link 0 Directives

Link 0 directives start with `%` and control runtime settings:

```gjf
%Chk=checkpoint.chk           # Checkpoint file
%Mem=16GB                      # Memory per CPU
%NProcShared=16                # Shared memory parallelism
%OldChk=old_checkpoint.chk    # Restart from previous checkpoint
```

### Common Link 0 Directives

| Directive | Purpose | Example |
|-----------|---------|---------|
| `%Chk` | Checkpoint file | `%Chk=calc.chk` |
| `%OldChk` | Read from previous checkpoint | `%OldChk=prev.chk` |
| `%Mem` | Memory per processor | `%Mem=8GB` |
| `%NProcShared` | Number of shared memory CPUs | `%NProcShared=16` |
| `%LindaWorkers` | Multi-node workers | `%LindaWorkers=node1 node2` |
| `%RWF` | Read-write file path | `%RWF=/scratch/rwf` |

## Route Section

Route section starts with `#` and describes the calculation:

```gjf
#p B3LYP/6-311G(d,p) Opt Freq
```

| Part | Meaning |
|------|---------|
| `#` | Required start |
| `p` | Print level (p = verbose) |
| `B3LYP` | Exchange-correlation functional |
| `/6-311G(d,p)` | Basis set |
| `Opt` | Optimization job |
| `Freq` | Frequency calculation |

## Job Types

| Keyword | Meaning | Use |
|---------|---------|-----|
| `SP` | Single point energy | Energy on a fixed geometry |
| `Opt` | Geometry optimization | Find minimum |
| `Freq` | Frequency calculation | Vibrational analysis |
| `Opt=ModRedundant` | Partial optimization | Optimize specific coordinates |
| `Scan` | Potential energy scan | Relaxed scan along coordinates |
| `IRC` | Intrinsic reaction coordinate | Reaction path following |
| `Guess=TCore` | Initial guess | For difficult SCF |

## Title Section

```gjf
Water B3LYP/6-311G(d,p) optimization
```

Keep it short and informative.

## Charge and Multiplicity

```gjf
0 1          # Neutral singlet (closed shell)
0 2          # Neutral doublet (radical)
-1 1        # Anion singlet
+1 2        # Cation doublet (open shell)
```

Format: `charge spin_multiplicity`

| Multiplicity | Meaning | Electrons |
|-------------|---------|-----------|
| 1 | Singlet (all paired) | Even number |
| 2 | Doublet (one unpaired) | Odd number |
| 3 | Triplet | Odd number |
| 4 | Quartet | Odd number |

## Geometry Format

### Cartesian Coordinates

```gjf
0 1
O   0.0  0.0  0.0
H   0.95  0.0  0.0
H  -0.24  0.92  0.0
```

### Z-Matrix

```gjf
O
H 1 0.96
H 1 0.96 2 104.5
```

## Basis Set Notation

| Notation | Meaning |
|----------|---------|
| `6-31G` | Split-valence, no polarization |
| `6-311G` | Triple-zeta valence |
| `6-311G(d,p)` | Triple-zeta + d on heavy, p on H |
| `cc-pVDZ` | Correlation-consistent, double-zeta |
| `aug-cc-pVDZ` | cc-pVDZ + diffuse functions |

## Complete Example

```gjf
%Chk=water_opt.chk
%Mem=8GB
%NProcShared=8
#p B3LYP/6-311G(d,p) Opt Freq

Water optimization and frequency

0 1
O   0.000000  -0.075792   0.000000
H   0.866812   0.601353   0.000000
H  -0.866812   0.601353   0.000000
```
