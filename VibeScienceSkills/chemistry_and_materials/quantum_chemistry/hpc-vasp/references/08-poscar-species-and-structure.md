# VASP POSCAR, Species, And Structure Manual

## Contents

- POSCAR structure
- species order discipline
- coordinate modes
- cell and relaxation implications

## POSCAR structure

The VASP wiki describes `POSCAR` as the structural input containing:

- title line
- scaling factor
- lattice vectors
- species list or element counts, depending on format conventions
- counts per species
- coordinate mode
- atomic positions

Treat `POSCAR` as the authoritative structural state for the stage.

## Species order discipline

High-value rule from the VASP wiki:

- species order in `POSCAR` must match the order of concatenated datasets in `POTCAR`

If those orders diverge, the run is physically misassigned even if parsing succeeds.

## Coordinate modes

Typical position modes:

- direct/fractional coordinates
- Cartesian coordinates

Do not edit coordinates without checking the active mode first.

## Cell and relaxation implications

When geometry optimization is active:

- lattice vectors and positions can both matter
- `ISIF` determines whether ions only, ions plus cell shape, or other structural degrees of freedom can relax

If the user wants only internal coordinates relaxed, do not leave a full cell-relaxation setup in place.
