# FEFF Structure Input

## CIF Conversion

1. Import CIF using Artemis
2. Select absorber atom
3. Export feff.inp

## Structure Optimization

- Optimize using VASP/Quantum ESPRESSO
- Export optimized coordinates
- Convert to FEFF format

## Cluster Radius Selection

- RMAX controls the cluster size
- Larger RMAX includes more atoms but increases computation
- Typical values: 5-10 Angstrom

## Coordinate Units

- Default: Angstrom
- Optional: Bohr
