# VASP Stage And Parameter Matrix

## Contents

- Stage map
- High-value INCAR matrix
- KPOINTS strategy matrix

## Stage map

| Goal | Typical stage |
| --- | --- |
| geometry optimization | ionic relaxation |
| accurate total energy | static SCF |
| density of states | DOS workflow after trustworthy SCF |
| band path | band-structure workflow after trustworthy SCF |

Do not reuse one INCAR unchanged across all stages.

## High-value INCAR matrix

| Concern | Typical tags |
| --- | --- |
| electronic algorithm and iterations | `ALGO`, `NELM`, `EDIFF` |
| ionic relaxation | `IBRION`, `NSW`, `ISIF`, `EDIFFG` |
| smearing | `ISMEAR`, `SIGMA` |
| spin treatment | `ISPIN`, `MAGMOM` |
| cutoffs and precision | `ENCUT`, `PREC` |

## KPOINTS strategy matrix

| Goal | Typical KPOINTS mode |
| --- | --- |
| routine SCF on periodic bulk | Monkhorst-Pack or Gamma-centered mesh |
| large supercell or Gamma-only workflow | Gamma-only file |
| band path | line-mode path |

If the stage is DOS or band structure, the k-point strategy is part of the workflow definition, not an optional extra.
