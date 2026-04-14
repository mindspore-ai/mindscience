---
name: hpc-quantum-espresso-input-parameter-matrix
description: Namelist responsibility matrix, species and pseudopotential mapping, and k-point and cutoff parameters for Quantum ESPRESSO
type: reference
---

# Quantum ESPRESSO Input And Parameter Matrix

## Contents

- Namelist responsibility matrix
- Species and pseudopotential matrix
- k-point and cutoff matrix

## Namelist responsibility matrix

| Concern | Primary section |
| --- | --- |
| run type, prefix, IO paths | `&CONTROL` |
| lattice, species counts, cutoffs, occupations | `&SYSTEM` |
| SCF and diagonalization convergence | `&ELECTRONS` |
| ionic motion | `&IONS` |
| variable-cell controls | `&CELL` |

## Species and pseudopotential matrix

| Concern | Primary section |
| --- | --- |
| species labels, masses, pseudo filenames | `ATOMIC_SPECIES` |
| coordinates and coordinate mode | `ATOMIC_POSITIONS` |
| lattice vectors when needed | `CELL_PARAMETERS` |

Keep species labels consistent across all sections.

## K-point and cutoff matrix

| Concern | Primary section or key |
| --- | --- |
| Brillouin-zone sampling | `K_POINTS` |
| wavefunction cutoff | `ecutwfc` |
| charge-density cutoff | `ecutrho` |
| occupations and smearing | `occupations`, `smearing`, `degauss` |

If the pseudopotential family implies cutoff expectations, reflect them in `ecutwfc` and `ecutrho` rather than guessing generic values.
