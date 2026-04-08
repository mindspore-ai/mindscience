# FEFF Input File Structure

## Basic Structure

```
TITLE Task description

CONTROL ...
PRINT ...

EDGE K
TARGET 1

RMAX 6.0
SCF 7.0

ATOMS
  Atomic coordinates
END
```

## Keyword Description

| Keyword | Description | Example |
|--------|------|------|
| TITLE | Task description | `TITLE Cu K-edge` |
| EDGE | Absorption edge | `EDGE K`, `EDGE L3` |
| TARGET | Absorber atom index | `TARGET 1` |
| RMAX | Cluster radius (A) | `RMAX 6.0` |
| SCF | SCF parameter | `SCF 7.0` |
| ATOMS | Atomic coordinates | `ATOMS...END` |

## Absorption Edges

| Edge | Description |
|----|------|
| K | K edge (1s) |
| L1 | L1 edge (2s) |
| L2 | L2 edge (2p1/2) |
| L3 | L3 edge (2p3/2) |
| M | M edge |

## ATOMS Format

```
ATOMS
  Index  Element  X  Y  Z  ipot
  ...
END
```

## Atomic Coordinate Format

```
ATOMS
  Index  Element  X  Y  Z
  0  Cu   0.0000   0.0000   0.0000
  1  Cu   2.5560   0.0000   0.0000
END
```

- Index starts from 0
- First atom is the absorber atom
- Coordinates in Angstrom (default) or Bohr (optional)

## ipot Definition

| ipot | Description |
|------|------|
| 0 | Absorber atom |
| 1 | First coordination shell |
| 2 | Second coordination shell |
| ... | Other atoms |

## Common Issues

1. Incorrect coordinate units
2. Improper ipot settings
3. Cluster too small
