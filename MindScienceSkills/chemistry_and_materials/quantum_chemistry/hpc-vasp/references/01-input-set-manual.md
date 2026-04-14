# VASP Input Set Manual

## Contents

- Four-file model
- File responsibility matrix
- Stage-to-directory workflow

## Four-file model

Treat the canonical VASP input as:

- `INCAR`
- `POSCAR`
- `KPOINTS`
- `POTCAR`

The official VASP wiki presents these four files as the core run surface. A valid workflow keeps them mutually consistent.

## File responsibility matrix

| Concern | Primary file |
| --- | --- |
| run mode, electronic and ionic controls | `INCAR` |
| lattice, species order, coordinates | `POSCAR` |
| Brillouin-zone sampling | `KPOINTS` |
| pseudopotential datasets | `POTCAR` |

Species order must stay aligned between `POSCAR` and `POTCAR`.

## Stage-to-directory workflow

Practical stage ordering:

1. relaxation
2. optional second relaxation or tighter static
3. DOS or band-structure stage

Keep stage directories explicit and copy forward only the data that should persist, such as relaxed structure or charge density when appropriate.
