# XTB Calculation Types

## Single Point Energy Calculation

```bash
xtb molecule.xyz --sp
```

Output files:
- `.energy`: Total energy
- `.charges`: Atomic charges
- `.wbo`: Wiberg bond orders
- `.dipole`: Dipole moment

## Geometry Optimization

```bash
# Standard optimization
xtb molecule.xyz --opt

# Transition state optimization
xtb molecule.xyz --gfnff-ts

# Constrained optimization
xtb molecule.xyz --opt --input constraints.inp
```

Output files:
- `xtbopt.xyz`: Optimized structure
- `xtbopt.log`: Optimization log

## Frequency Calculation

```bash
xtb molecule.xyz --hess
```

Output files:
- `.vibspectrum`: Vibrational frequencies
- `.hessian`: Hessian matrix
- `.g98.out`: Gaussian format output

## Molecular Dynamics

```bash
# NVT ensemble
xtb molecule.xyz --md --time 100 --temp 300

# NPT ensemble
xtb molecule.xyz --md --mdtemp 300 --mdpress 1.0
```

Parameter descriptions:
- `--time`: Simulation time (ps)
- `--temp`: Temperature (K)
- `--step`: Time step (fs)

## Solvation

```bash
# GBSA solvation model
xtb molecule.xyz --sp --gbsa water
```

Supported solvents:
- water, acetone, acetonitrile
- dmf, dmso, ethanol
- methanol, thf, toluene